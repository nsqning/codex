use std::fs::File;
use std::fs::{self};
use std::io::Read;
use std::io::Write;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use clap::Parser;
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::header::AUTHORIZATION;
use reqwest::header::HOST;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderName;
use reqwest::header::HeaderValue;
use serde::Serialize;
use serde_json::json;
use tiny_http::Header;
use tiny_http::Method;
use tiny_http::Request;
use tiny_http::Response;
use tiny_http::Server;
use tiny_http::StatusCode;

mod chat_completions;
mod dump;
mod read_api_key;
use chat_completions::ChatCompletionsBridgeConfig;
use chat_completions::bridge_responses_to_chat_completions;
use dump::ExchangeDumper;
use read_api_key::read_auth_header_from_stdin;

/// CLI arguments for the proxy.
#[derive(Debug, Clone, Parser)]
#[command(name = "responses-api-proxy", about = "Minimal OpenAI responses proxy")]
pub struct Args {
    /// Port to listen on. If not set, an ephemeral port is used.
    #[arg(long)]
    pub port: Option<u16>,

    /// Path to a JSON file to write startup info (single line). Includes {"port": <u16>}.
    #[arg(long, value_name = "FILE")]
    pub server_info: Option<PathBuf>,

    /// Enable HTTP shutdown endpoint at GET /shutdown
    #[arg(long)]
    pub http_shutdown: bool,

    /// Absolute URL the proxy should forward requests to (defaults to OpenAI).
    #[arg(long, default_value = "https://api.openai.com/v1/responses")]
    pub upstream_url: String,

    /// Absolute Chat Completions URL to bridge Responses requests to.
    #[arg(long, value_name = "URL")]
    pub chat_completions_upstream_url: Option<String>,

    /// Override the Host header used for --chat-completions-upstream-url.
    #[arg(long, value_name = "HOST")]
    pub chat_completions_host_header: Option<String>,

    /// Override the model sent to the Chat Completions upstream.
    #[arg(long, value_name = "MODEL")]
    pub chat_completions_model: Option<String>,

    /// Request field used for translated tool declarations.
    #[arg(long, default_value = "toolCalls")]
    pub chat_completions_tool_field: String,

    /// User-Agent header to send to the Chat Completions upstream.
    #[arg(long, value_name = "VALUE")]
    pub chat_completions_user_agent: Option<String>,

    /// Value for the DeepSeek-compatible `thinking` request field.
    #[arg(long, default_value_t = false)]
    pub chat_completions_thinking: bool,

    /// Maximum completion tokens sent to the Chat Completions upstream.
    #[arg(long, default_value_t = 1000)]
    pub chat_completions_max_tokens: u32,

    /// Temperature sent to the Chat Completions upstream.
    #[arg(long, default_value_t = 0.0)]
    pub chat_completions_temperature: f64,

    /// Ask the Chat Completions upstream for streaming chunks before normalizing them.
    #[arg(long, default_value_t = false)]
    pub chat_completions_stream_upstream: bool,

    /// Directory where request/response dumps should be written as JSON.
    #[arg(long, value_name = "DIR")]
    pub dump_dir: Option<PathBuf>,
}

#[derive(Serialize)]
struct ServerInfo {
    port: u16,
    pid: u32,
}

struct ForwardConfig {
    mode: ForwardMode,
}

enum ForwardMode {
    Responses(ResponsesForwardConfig),
    ChatCompletions(ChatCompletionsBridgeConfig),
}

struct ResponsesForwardConfig {
    upstream_url: Url,
    host_header: HeaderValue,
}

/// Entry point for the library main, for parity with other crates.
pub fn run_main(args: Args) -> Result<()> {
    let auth_header = read_auth_header_from_stdin()?;

    let mode = if let Some(chat_completions_upstream_url) = args.chat_completions_upstream_url {
        let upstream_url = Url::parse(&chat_completions_upstream_url)
            .context("parsing --chat-completions-upstream-url")?;
        let host_header = host_header_for_url(
            &upstream_url,
            args.chat_completions_host_header.as_deref(),
        )?;
        let user_agent = args
            .chat_completions_user_agent
            .as_deref()
            .map(HeaderValue::from_str)
            .transpose()
            .context("constructing User-Agent header from --chat-completions-user-agent")?;

        ForwardMode::ChatCompletions(ChatCompletionsBridgeConfig {
            upstream_url,
            host_header,
            model: args.chat_completions_model,
            tool_field: args.chat_completions_tool_field,
            user_agent,
            thinking: args.chat_completions_thinking,
            max_tokens: args.chat_completions_max_tokens,
            temperature: args.chat_completions_temperature,
            stream_upstream: args.chat_completions_stream_upstream,
        })
    } else {
        let upstream_url = Url::parse(&args.upstream_url).context("parsing --upstream-url")?;
        let host_header = host_header_for_url(&upstream_url, None)?;

        ForwardMode::Responses(ResponsesForwardConfig {
            upstream_url,
            host_header,
        })
    };

    let forward_config = Arc::new(ForwardConfig {
        mode,
    });
    let dump_dir = args
        .dump_dir
        .map(ExchangeDumper::new)
        .transpose()
        .context("creating --dump-dir")?
        .map(Arc::new);

    let (listener, bound_addr) = bind_listener(args.port)?;
    if let Some(path) = args.server_info.as_ref() {
        write_server_info(path, bound_addr.port())?;
    }
    let server = Server::from_listener(listener, None)
        .map_err(|err| anyhow!("creating HTTP server: {err}"))?;
    let client = Arc::new(
        Client::builder()
            // Disable reqwest's 30s default so long-lived response streams keep flowing.
            .timeout(None::<Duration>)
            .build()
            .context("building reqwest client")?,
    );

    eprintln!("responses-api-proxy listening on {bound_addr}");

    let http_shutdown = args.http_shutdown;
    for request in server.incoming_requests() {
        let client = client.clone();
        let forward_config = forward_config.clone();
        let dump_dir = dump_dir.clone();
        std::thread::spawn(move || {
            if http_shutdown && request.method() == &Method::Get && request.url() == "/shutdown" {
                let _ = request.respond(Response::new_empty(StatusCode(200)));
                std::process::exit(0);
            }

            if let Err(e) = forward_request(
                &client,
                auth_header,
                &forward_config,
                dump_dir.as_deref(),
                request,
            ) {
                eprintln!("forwarding error: {e}");
            }
        });
    }

    Err(anyhow!("server stopped unexpectedly"))
}

fn host_header_for_url(upstream_url: &Url, override_host: Option<&str>) -> Result<HeaderValue> {
    let host = match override_host {
        Some(host) => host.to_string(),
        None => match (upstream_url.host_str(), upstream_url.port()) {
            (Some(host), Some(port)) => format!("{host}:{port}"),
            (Some(host), None) => host.to_string(),
            _ => return Err(anyhow!("upstream URL must include a host")),
        },
    };
    HeaderValue::from_str(&host).context("constructing Host header from upstream URL")
}

fn bind_listener(port: Option<u16>) -> Result<(TcpListener, SocketAddr)> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port.unwrap_or(0)));
    let listener = TcpListener::bind(addr).with_context(|| format!("failed to bind {addr}"))?;
    let bound = listener.local_addr().context("failed to read local_addr")?;
    Ok((listener, bound))
}

fn write_server_info(path: &Path, port: u16) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let info = ServerInfo {
        port,
        pid: std::process::id(),
    };
    let mut data = serde_json::to_string(&info)?;
    data.push('\n');
    let mut f = File::create(path)?;
    f.write_all(data.as_bytes())?;
    Ok(())
}

fn forward_request(
    client: &Client,
    auth_header: &'static str,
    config: &ForwardConfig,
    dump_dir: Option<&ExchangeDumper>,
    mut req: Request,
) -> Result<()> {
    // Only allow POST /v1/responses exactly, no query string.
    let method = req.method().clone();
    let url_path = req.url().to_string();
    let allow = method == Method::Post && url_path == "/v1/responses";

    if !allow {
        let resp = Response::new_empty(StatusCode(403));
        let _ = req.respond(resp);
        return Ok(());
    }

    // Read request body
    let mut body = Vec::new();
    let reader = req.as_reader();
    reader.read_to_end(&mut body)?;

    let exchange_dump = dump_dir.and_then(|dump_dir| {
        dump_dir
            .dump_request(&method, &url_path, req.headers(), &body)
            .map_err(|err| {
                eprintln!("responses-api-proxy failed to dump request: {err}");
                err
            })
            .ok()
    });

    // Build headers for upstream, forwarding everything from the incoming
    // request except Authorization (we replace it below).
    let mut headers = HeaderMap::new();
    for header in req.headers() {
        let name_ascii = header.field.as_str();
        let lower = name_ascii.to_ascii_lowercase();
        if lower.as_str() == "authorization" || lower.as_str() == "host" {
            continue;
        }

        let header_name = match HeaderName::from_bytes(lower.as_bytes()) {
            Ok(name) => name,
            Err(_) => continue,
        };
        if let Ok(value) = HeaderValue::from_bytes(header.value.as_bytes()) {
            headers.append(header_name, value);
        }
    }

    // As part of our effort to to keep `auth_header` secret, we use a
    // combination of `from_static()` and `set_sensitive(true)`.
    let mut auth_header_value = HeaderValue::from_static(auth_header);
    auth_header_value.set_sensitive(true);
    headers.insert(AUTHORIZATION, auth_header_value);

    match &config.mode {
        ForwardMode::Responses(config) => {
            forward_responses_request(client, config, exchange_dump, headers, body, req)
        }
        ForwardMode::ChatCompletions(config) => {
            bridge_responses_to_chat_completions(
                client,
                config,
                exchange_dump,
                headers,
                body,
                req,
            )
        }
    }
}

fn forward_responses_request(
    client: &Client,
    config: &ResponsesForwardConfig,
    exchange_dump: Option<dump::ExchangeDump>,
    mut headers: HeaderMap,
    body: Vec<u8>,
    req: Request,
) -> Result<()> {
    headers.insert(HOST, config.host_header.clone());

    let upstream_resp = client
        .post(config.upstream_url.clone())
        .headers(headers)
        .body(body)
        .send()
        .context("forwarding request to upstream")?;

    // We have to create an adapter between a `reqwest::blocking::Response`
    // and a `tiny_http::Response`. Fortunately, `reqwest::blocking::Response`
    // implements `Read`, so we can use it directly as the body of the
    // `tiny_http::Response`.
    let status = upstream_resp.status();
    let mut response_headers = Vec::new();
    for (name, value) in upstream_resp.headers().iter() {
        // Skip headers that tiny_http manages itself.
        if matches!(
            name.as_str(),
            "content-length" | "transfer-encoding" | "connection" | "trailer" | "upgrade"
        ) {
            continue;
        }

        if let Ok(header) = Header::from_bytes(name.as_str().as_bytes(), value.as_bytes()) {
            response_headers.push(header);
        }
    }

    let content_length = upstream_resp.content_length().and_then(|len| {
        if len <= usize::MAX as u64 {
            Some(len as usize)
        } else {
            None
        }
    });

    let response_body: Box<dyn Read + Send> = if let Some(exchange_dump) = exchange_dump {
        let headers = upstream_resp.headers().clone();
        Box::new(exchange_dump.tee_response_body(status.as_u16(), &headers, upstream_resp))
    } else {
        Box::new(upstream_resp)
    };

    let response = Response::new(
        StatusCode(status.as_u16()),
        response_headers,
        response_body,
        content_length,
        None,
    );

    let _ = req.respond(response);
    Ok(())
}

pub(crate) fn respond_with_json_error(req: Request, status: StatusCode, message: &str) {
    let body = json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
        }
    })
    .to_string();
    let response = Response::from_string(body).with_status_code(status);
    let response = match Header::from_bytes(&b"content-type"[..], &b"application/json"[..]) {
        Ok(header) => response.with_header(header),
        Err(_) => response,
    };
    let _ = req.respond(response);
}
