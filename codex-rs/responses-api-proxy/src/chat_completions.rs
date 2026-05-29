use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::Cursor;
use std::io::Read;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::header::ACCEPT;
use reqwest::header::CONTENT_LENGTH;
use reqwest::header::CONTENT_TYPE;
use reqwest::header::HOST;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderValue;
use reqwest::header::USER_AGENT;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use tiny_http::Header;
use tiny_http::Request;
use tiny_http::Response;
use tiny_http::StatusCode;

use crate::dump::ExchangeDump;
use crate::respond_with_json_error;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct ChatCompletionsBridgeConfig {
    pub(crate) upstream_url: Url,
    pub(crate) host_header: HeaderValue,
    pub(crate) model: Option<String>,
    pub(crate) tool_field: String,
    pub(crate) user_agent: Option<HeaderValue>,
    pub(crate) thinking: bool,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: f64,
    pub(crate) stream_upstream: bool,
}

struct BridgeRequestContext {
    custom_tool_names: HashSet<String>,
    requested_model: String,
}

#[derive(Debug, Default)]
struct ChatCompletionOutput {
    id: Option<String>,
    model: Option<String>,
    content: String,
    tool_calls: Vec<ChatToolCall>,
    usage: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatToolCall {
    id: Option<String>,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

pub(crate) fn bridge_responses_to_chat_completions(
    client: &Client,
    config: &ChatCompletionsBridgeConfig,
    exchange_dump: Option<ExchangeDump>,
    mut headers: HeaderMap,
    body: Vec<u8>,
    req: Request,
) -> Result<()> {
    let (chat_request, request_context) = match build_chat_completion_request(&body, config) {
        Ok(request) => request,
        Err(err) => {
            respond_with_json_error(req, StatusCode(400), &err.to_string());
            return Ok(());
        }
    };

    headers.insert(HOST, config.host_header.clone());
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
    headers.remove(CONTENT_LENGTH);
    if let Some(user_agent) = config.user_agent.as_ref() {
        headers.insert(USER_AGENT, user_agent.clone());
    }

    let upstream_body = serde_json::to_vec(&chat_request)?;
    let upstream_resp = client
        .post(config.upstream_url.clone())
        .headers(headers)
        .body(upstream_body)
        .send()
        .context("forwarding bridged request to chat completions upstream")?;

    let upstream_status = upstream_resp.status();
    let upstream_headers = upstream_resp.headers().clone();
    let upstream_body = upstream_resp
        .text()
        .context("reading chat completions upstream response")?;

    if !upstream_status.is_success() {
        return respond_with_body(
            exchange_dump,
            upstream_status.as_u16(),
            &upstream_headers,
            upstream_body.into_bytes(),
            req,
        );
    }

    let normalized_body =
        match normalize_chat_completion_response(&upstream_body, &request_context) {
            Ok(body) => body,
            Err(err) => {
                respond_with_json_error(req, StatusCode(502), &err.to_string());
                return Ok(());
            }
        };

    let mut response_headers = HeaderMap::new();
    response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response_headers.insert("cache-control", HeaderValue::from_static("no-cache"));

    respond_with_body(
        exchange_dump,
        200,
        &response_headers,
        normalized_body,
        req,
    )
}

fn respond_with_body(
    exchange_dump: Option<ExchangeDump>,
    status_code: u16,
    headers: &HeaderMap,
    body: Vec<u8>,
    req: Request,
) -> Result<()> {
    let mut response_headers = Vec::new();
    for (name, value) in headers {
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

    let content_length = body.len();
    let response_body: Box<dyn Read + Send> = if let Some(exchange_dump) = exchange_dump {
        Box::new(exchange_dump.tee_response_body(status_code, headers, Cursor::new(body)))
    } else {
        Box::new(Cursor::new(body))
    };

    let response = Response::new(
        StatusCode(status_code),
        response_headers,
        response_body,
        Some(content_length),
        None,
    );

    let _ = req.respond(response);
    Ok(())
}

fn build_chat_completion_request(
    body: &[u8],
    config: &ChatCompletionsBridgeConfig,
) -> Result<(Value, BridgeRequestContext)> {
    let responses_request: Value =
        serde_json::from_slice(body).context("parsing Responses request body")?;
    let model = config
        .model
        .clone()
        .or_else(|| {
            responses_request
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .ok_or_else(|| anyhow!("Responses request is missing `model`"))?;

    let mut messages = Vec::new();
    if let Some(instructions) = responses_request
        .get("instructions")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        messages.push(json!({
            "role": "system",
            "content": instructions,
        }));
    }

    match responses_request.get("input") {
        Some(Value::Array(items)) => {
            for item in items {
                append_response_item_as_chat_messages(item, &mut messages);
            }
        }
        Some(Value::String(text)) => {
            messages.push(json!({
                "role": "user",
                "content": text,
            }));
        }
        Some(other) => {
            messages.push(json!({
                "role": "user",
                "content": value_to_text(other),
            }));
        }
        None => {}
    }

    if messages.is_empty() {
        return Err(anyhow!("Responses request did not contain any chat messages"));
    }

    let mut custom_tool_names = HashSet::new();
    let tools = responses_request
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| convert_responses_tools(tools, &mut custom_tool_names))
        .unwrap_or_default();

    let mut request = Map::new();
    request.insert("model".to_string(), Value::String(model.clone()));
    request.insert("temperature".to_string(), json!(config.temperature));
    request.insert("max_tokens".to_string(), json!(config.max_tokens));
    request.insert("frequency_penalty".to_string(), json!(0));
    request.insert("presence_penalty".to_string(), json!(0));
    request.insert("stop".to_string(), Value::Null);
    request.insert("stream".to_string(), json!(config.stream_upstream));
    request.insert("thinking".to_string(), json!(config.thinking));
    request.insert("messages".to_string(), Value::Array(messages));
    if !tools.is_empty() {
        request.insert(config.tool_field.clone(), Value::Array(tools));
    }

    Ok((
        Value::Object(request),
        BridgeRequestContext {
            custom_tool_names,
            requested_model: model,
        },
    ))
}

fn append_response_item_as_chat_messages(item: &Value, messages: &mut Vec<Value>) {
    let item_type = item.get("type").and_then(Value::as_str);
    match item_type {
        Some("message") => {
            let role = item
                .get("role")
                .and_then(Value::as_str)
                .map(chat_role)
                .unwrap_or("user");
            let content = item
                .get("content")
                .map(content_items_to_text)
                .unwrap_or_default();
            if !content.trim().is_empty() {
                messages.push(json!({
                    "role": role,
                    "content": content,
                }));
            }
        }
        Some("function_call") => {
            let Some(name) = item.get("name").and_then(Value::as_str) else {
                return;
            };
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(next_call_id);
            let arguments = item
                .get("arguments")
                .map(arguments_to_string)
                .unwrap_or_else(|| "{}".to_string());
            messages.push(json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }],
            }));
        }
        Some("custom_tool_call") => {
            let Some(name) = item.get("name").and_then(Value::as_str) else {
                return;
            };
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(next_call_id);
            let input = item.get("input").map(value_to_text).unwrap_or_default();
            messages.push(json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json!({"input": input}).to_string(),
                    },
                }],
            }));
        }
        Some("function_call_output") | Some("custom_tool_call_output") => {
            let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
                return;
            };
            let content = item
                .get("output")
                .map(function_output_to_text)
                .unwrap_or_default();
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }));
        }
        Some("tool_search_output") => {
            let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
                return;
            };
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": item.to_string(),
            }));
        }
        Some("reasoning" | "web_search_call" | "image_generation_call" | "compaction")
        | Some("compaction_trigger" | "context_compaction")
        | None => {}
        Some(_) => {}
    }
}

fn chat_role(role: &str) -> &str {
    match role {
        "developer" => "system",
        "assistant" | "system" | "tool" | "user" => role,
        _ => "user",
    }
}

fn convert_responses_tools(
    tools: &[Value],
    custom_tool_names: &mut HashSet<String>,
) -> Vec<Value> {
    let mut chat_tools = Vec::new();
    for tool in tools {
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => {
                if let Some(chat_tool) = convert_function_tool(tool, None) {
                    chat_tools.push(chat_tool);
                }
            }
            Some("namespace") => {
                let namespace_name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
                let namespace_description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(namespace_tools) = tool.get("tools").and_then(Value::as_array) {
                    for namespace_tool in namespace_tools {
                        let description_prefix = (!namespace_description.is_empty())
                            .then_some(namespace_description);
                        if let Some(chat_tool) = convert_function_tool_with_namespace(
                            namespace_tool,
                            namespace_name,
                            description_prefix,
                        ) {
                            chat_tools.push(chat_tool);
                        }
                    }
                }
            }
            Some("custom") => {
                if let Some(name) = tool.get("name").and_then(Value::as_str) {
                    custom_tool_names.insert(name.to_string());
                    let description = tool
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    chat_tools.push(chat_function_tool(
                        name,
                        description,
                        json!({
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "Raw input for the custom tool.",
                                },
                            },
                            "required": ["input"],
                            "additionalProperties": false,
                        }),
                    ));
                }
            }
            Some("tool_search") => {
                chat_tools.push(chat_function_tool(
                    "tool_search",
                    tool.get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("Search available tools."),
                    tool.get("parameters").cloned().unwrap_or_else(|| json!({})),
                ));
            }
            Some("web_search" | "image_generation") | None => {}
            Some(_) => {}
        }
    }
    chat_tools
}

fn convert_function_tool(tool: &Value, name_override: Option<String>) -> Option<Value> {
    let name = name_override.or_else(|| {
        tool.get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
    })?;
    let description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let parameters = tool.get("parameters").cloned().unwrap_or_else(|| json!({}));
    Some(chat_function_tool(&name, description, parameters))
}

fn convert_function_tool_with_namespace(
    tool: &Value,
    namespace_name: &str,
    description_prefix: Option<&str>,
) -> Option<Value> {
    let tool_name = tool.get("name").and_then(Value::as_str)?;
    let name = if namespace_name.is_empty() || tool_name.starts_with(namespace_name) {
        tool_name.to_string()
    } else {
        format!("{namespace_name}{tool_name}")
    };
    let description = match (
        description_prefix,
        tool.get("description").and_then(Value::as_str),
    ) {
        (Some(prefix), Some(description)) if !description.is_empty() => {
            format!("{prefix}\n\n{description}")
        }
        (Some(prefix), _) => prefix.to_string(),
        (_, Some(description)) => description.to_string(),
        (None, None) => String::new(),
    };
    let parameters = tool.get("parameters").cloned().unwrap_or_else(|| json!({}));
    Some(chat_function_tool(&name, &description, parameters))
}

fn chat_function_tool(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    })
}

fn normalize_chat_completion_response(
    body: &str,
    request_context: &BridgeRequestContext,
) -> Result<Vec<u8>> {
    let output = if looks_like_sse(body) {
        parse_streaming_chat_completion(body)?
    } else {
        parse_unary_chat_completion(body)?
    };

    Ok(render_responses_sse(&output, request_context).into_bytes())
}

fn parse_unary_chat_completion(body: &str) -> Result<ChatCompletionOutput> {
    let value: Value = serde_json::from_str(body).context("parsing chat completions response")?;
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| anyhow!("chat completions response did not contain choices[0]"))?;
    let message = choice
        .get("message")
        .or_else(|| choice.get("delta"))
        .ok_or_else(|| anyhow!("chat completions choice did not contain message"))?;

    Ok(ChatCompletionOutput {
        id: value.get("id").and_then(Value::as_str).map(str::to_string),
        model: value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        content: message
            .get("content")
            .map(content_items_to_text)
            .unwrap_or_default(),
        tool_calls: extract_tool_calls(message),
        usage: value.get("usage").cloned(),
    })
}

fn parse_streaming_chat_completion(body: &str) -> Result<ChatCompletionOutput> {
    let mut output = ChatCompletionOutput::default();
    let mut partial_tool_calls: BTreeMap<u64, PartialToolCall> = BTreeMap::new();

    for data in sse_data_messages(body) {
        if data == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(&data)
            .with_context(|| format!("parsing chat completions stream chunk: {data}"))?;
        if output.id.is_none() {
            output.id = value.get("id").and_then(Value::as_str).map(str::to_string);
        }
        if output.model.is_none() {
            output.model = value
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        if value.get("usage").is_some() {
            output.usage = value.get("usage").cloned();
        }

        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for choice in choices {
            let Some(delta) = choice.get("delta").or_else(|| choice.get("message")) else {
                continue;
            };
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                output.content.push_str(content);
            }
            merge_tool_call_deltas(delta, &mut partial_tool_calls);
        }
    }

    output.tool_calls = partial_tool_calls
        .into_values()
        .filter_map(|partial| {
            Some(ChatToolCall {
                id: partial.id,
                name: partial.name?,
                arguments: if partial.arguments.is_empty() {
                    "{}".to_string()
                } else {
                    partial.arguments
                },
            })
        })
        .collect();

    Ok(output)
}

fn looks_like_sse(body: &str) -> bool {
    body.trim_start().starts_with("data:") || body.trim_start().starts_with("event:")
}

fn sse_data_messages(body: &str) -> Vec<String> {
    body.replace("\r\n", "\n")
        .split("\n\n")
        .filter_map(|event| {
            let mut data_lines = Vec::new();
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    data_lines.push(data.trim_start());
                }
            }
            (!data_lines.is_empty()).then(|| data_lines.join("\n"))
        })
        .collect()
}

fn merge_tool_call_deltas(delta: &Value, partial_tool_calls: &mut BTreeMap<u64, PartialToolCall>) {
    let Some(tool_calls) = delta
        .get("tool_calls")
        .or_else(|| delta.get("toolCalls"))
        .and_then(Value::as_array)
    else {
        return;
    };

    for (fallback_index, tool_call) in tool_calls.iter().enumerate() {
        let index = tool_call
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(fallback_index as u64);
        let partial = partial_tool_calls.entry(index).or_default();
        if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
            partial.id = Some(id.to_string());
        }
        if let Some(function) = tool_call.get("function") {
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                partial.name = Some(name.to_string());
            }
            if let Some(arguments) = function.get("arguments") {
                partial.arguments.push_str(&arguments_to_string(arguments));
            }
        }
    }
}

fn extract_tool_calls(message: &Value) -> Vec<ChatToolCall> {
    let Some(tool_calls) = message
        .get("tool_calls")
        .or_else(|| message.get("toolCalls"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };

    tool_calls
        .iter()
        .filter_map(|tool_call| {
            let function = tool_call.get("function")?;
            let name = function.get("name").and_then(Value::as_str)?;
            Some(ChatToolCall {
                id: tool_call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                name: name.to_string(),
                arguments: function
                    .get("arguments")
                    .map(arguments_to_string)
                    .unwrap_or_else(|| "{}".to_string()),
            })
        })
        .collect()
}

fn render_responses_sse(
    output: &ChatCompletionOutput,
    request_context: &BridgeRequestContext,
) -> String {
    let response_id = output.id.clone().unwrap_or_else(next_response_id);
    let response_model = output
        .model
        .as_deref()
        .unwrap_or(request_context.requested_model.as_str());
    let mut sse = String::new();

    write_sse_event(
        &mut sse,
        "response.created",
        json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "model": response_model,
                "status": "in_progress",
            },
        }),
    );

    if !output.content.is_empty() {
        let message_id = next_item_id("msg");
        write_sse_event(
            &mut sse,
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "id": message_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                },
            }),
        );
        write_sse_event(
            &mut sse,
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "item_id": message_id,
                "delta": output.content,
            }),
        );
        write_sse_event(
            &mut sse,
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": message_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": output.content}],
                },
            }),
        );
    }

    for tool_call in &output.tool_calls {
        let call_id = tool_call.id.clone().unwrap_or_else(next_call_id);
        let item_id = next_item_id("fc");
        let item = if request_context.custom_tool_names.contains(&tool_call.name) {
            json!({
                "type": "custom_tool_call",
                "id": item_id,
                "call_id": call_id,
                "name": tool_call.name,
                "input": custom_tool_input(&tool_call.arguments),
            })
        } else {
            json!({
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            })
        };
        write_sse_event(
            &mut sse,
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "item": item,
            }),
        );
    }

    write_sse_event(
        &mut sse,
        "response.completed",
        json!({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "model": response_model,
                "status": "completed",
                "usage": output.usage.as_ref().map(normalize_usage).unwrap_or(Value::Null),
                "end_turn": output.tool_calls.is_empty(),
            },
        }),
    );

    sse
}

fn write_sse_event(out: &mut String, event: &str, data: Value) {
    let _ = writeln!(out, "event: {event}");
    let _ = writeln!(out, "data: {data}");
    out.push('\n');
}

fn normalize_usage(usage: &Value) -> Value {
    let input_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let cached_tokens = usage
        .get("prompt_cache_hit_tokens")
        .or_else(|| usage.get("cached_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(input_tokens + output_tokens);

    json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens,
    })
}

fn custom_tool_input(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("input")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| arguments.to_string())
}

fn content_items_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("content").and_then(Value::as_str))
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(_) => value
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| value_to_text(value)),
        Value::Null | Value::Bool(_) | Value::Number(_) => value_to_text(value),
    }
}

fn function_output_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(_) => content_items_to_text(value),
        Value::Object(_) | Value::Bool(_) | Value::Number(_) | Value::Null => value_to_text(value),
    }
}

fn arguments_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        _ => value.to_string(),
    }
}

fn next_response_id() -> String {
    next_item_id("resp")
}

fn next_call_id() -> String {
    next_item_id("call")
}

fn next_item_id(prefix: &str) -> String {
    let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    format!("{prefix}_{timestamp_ms}_{sequence}")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use reqwest::Url;
    use reqwest::header::HeaderValue;
    use serde_json::json;

    use super::BridgeRequestContext;
    use super::ChatCompletionsBridgeConfig;
    use super::ChatCompletionOutput;
    use super::ChatToolCall;
    use super::build_chat_completion_request;
    use super::normalize_chat_completion_response;
    use super::render_responses_sse;

    fn config() -> ChatCompletionsBridgeConfig {
        ChatCompletionsBridgeConfig {
            upstream_url: Url::parse("http://127.0.0.1:8000/v1/chat/completions").unwrap(),
            host_header: HeaderValue::from_static("summarizer-a.wbx2.com"),
            model: Some("deepseek-r1-32b".to_string()),
            tool_field: "toolCalls".to_string(),
            user_agent: None,
            thinking: false,
            max_tokens: 1000,
            temperature: 0.0,
            stream_upstream: false,
        }
    }

    #[test]
    fn builds_chat_completion_request_from_responses_payload() {
        let body = json!({
            "model": "gpt-5",
            "instructions": "Be direct.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "calculator",
                    "arguments": "{\"expression\":\"1+2\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "3"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "calculator",
                    "description": "Evaluate arithmetic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            ]
        });

        let (actual, context) =
            build_chat_completion_request(body.to_string().as_bytes(), &config())
                .expect("build request");

        assert_eq!(context.requested_model, "deepseek-r1-32b");
        assert_eq!(
            actual,
            json!({
                "model": "deepseek-r1-32b",
                "temperature": 0.0,
                "max_tokens": 1000,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": null,
                "stream": false,
                "thinking": false,
                "messages": [
                    {"role": "system", "content": "Be direct."},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": "{\"expression\":\"1+2\"}"
                            }
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "3"}
                ],
                "toolCalls": [{
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate arithmetic.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"}
                            },
                            "required": ["expression"]
                        }
                    }
                }]
            })
        );
    }

    #[test]
    fn normalizes_unary_chat_response_to_responses_sse() {
        let body = json!({
            "id": "chatcmpl-1",
            "model": "deepseek-r1-32b",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5
            }
        });
        let context = BridgeRequestContext {
            custom_tool_names: Default::default(),
            requested_model: "deepseek-r1-32b".to_string(),
        };

        let actual = String::from_utf8(
            normalize_chat_completion_response(&body.to_string(), &context)
                .expect("normalize response"),
        )
        .expect("utf8");

        assert!(actual.contains("event: response.created"));
        assert!(actual.contains("event: response.output_item.added"));
        assert!(actual.contains("event: response.output_text.delta"));
        assert!(actual.contains("event: response.output_item.done"));
        assert!(actual.contains("event: response.completed"));
        assert!(actual.contains("\"delta\":\"hello\""));
        assert!(actual.contains("\"input_tokens\":2"));
    }

    #[test]
    fn renders_function_call_item_from_chat_tool_call() {
        let context = BridgeRequestContext {
            custom_tool_names: Default::default(),
            requested_model: "deepseek-r1-32b".to_string(),
        };
        let output = ChatCompletionOutput {
            id: Some("chatcmpl-tool".to_string()),
            model: Some("deepseek-r1-32b".to_string()),
            content: String::new(),
            tool_calls: vec![ChatToolCall {
                id: Some("call_1".to_string()),
                name: "calculator".to_string(),
                arguments: "{\"expression\":\"1+2\"}".to_string(),
            }],
            usage: None,
        };

        let actual = render_responses_sse(&output, &context);

        assert!(actual.contains("\"type\":\"function_call\""));
        assert!(actual.contains("\"call_id\":\"call_1\""));
        assert!(actual.contains("\"name\":\"calculator\""));
        assert!(actual.contains("\"end_turn\":false"));
    }
}
