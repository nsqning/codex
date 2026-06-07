$ErrorActionPreference = "Stop"

$codexRoot = $PSScriptRoot
$maRoot = Resolve-Path -LiteralPath (Join-Path $codexRoot "..\multiple_agents")
$env:MA_CODEX_DEEPSEEK_CODEX_ROOT = $codexRoot
$launcher = Join-Path $maRoot "scripts\codex-deepseek.ps1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Multiple Agents codex-deepseek launcher was not found at $launcher"
}

& $launcher @args
exit $LASTEXITCODE
