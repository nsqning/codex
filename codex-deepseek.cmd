@echo off
setlocal

set "CODEX_ROOT=%~dp0"
set "MA_ROOT=%CODEX_ROOT%..\multiple_agents"
set "MA_CODEX_DEEPSEEK_CODEX_ROOT=%CODEX_ROOT%"

if exist "%MA_ROOT%\scripts\codex-deepseek.cmd" (
  call "%MA_ROOT%\scripts\codex-deepseek.cmd" %*
  exit /b %ERRORLEVEL%
)

echo Multiple Agents codex-deepseek launcher was not found at "%MA_ROOT%\scripts\codex-deepseek.cmd". 1>&2
exit /b 1
