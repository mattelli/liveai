//! System tools (Python `src/tools/system.py`): platform info, date/time,
//! working directory, shell commands and the graceful-exit signal.

use crate::common::cprint;
use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use anyhow::Result;
use chrono::{Datelike, Local, Timelike};
use std::sync::Arc;

/// Exits the program gracefully.
/// Useful when user asks to quit or exit the application.
/// (Python raises SystemExit; the Rust session loop observes the flag.)
pub async fn exit_gracefully(
    _ctx: Arc<ToolContext>,
    _args: serde_json::Value,
) -> Result<ToolOutput> {
    cprint("\nExiting gracefully. Bye! 👋", "reset", false, "\n");
    Ok(ToolOutput::exit())
}

/// Returns system information such as OS, CPU.
/// Useful when user asks for system status or performance.
pub fn system_info() -> String {
    sysinfo_snapshot()
}

fn sysinfo_snapshot() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let family = std::env::consts::FAMILY;
    let host = std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| {
        // Best-effort processor info on non-Windows shells
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| arch.to_string())
    });
    format!("{family} {os} ({arch}), Processor: {host}")
}

/// Returns the current date and time.
/// Useful when user asks for time, date, or 'what day is it?'.
pub fn current_date_time() -> String {
    // Python: %A, %B %d, %Y at %I:%M %p, %Z (UTC)
    let now = chrono::Utc::now();
    let local = now.with_timezone(&Local);
    let tz_abbrev = chrono::Utc::now().format("%Z").to_string();
    format!(
        "{}, {} {:02}, {} at {:02}:{:02} {:?}, {}",
        local.format("%A"),
        local.format("%B"),
        local.day(),
        local.year(),
        local.format("%I").to_string().trim_start_matches('0'),
        local.minute(),
        local.format("%p"),
        tz_abbrev
    )
}

/// Returns the current working directory.
/// Useful when user asks for getting current directory files and directories,
/// or when user asks for launching commands.
pub fn get_working_directory() -> String {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|e| format!("Error fetching system info: {e}"))
}

/// Run a command with a subprocess and wait for it to terminate.
/// It returns either the command output or error as string.
/// Useful when user asks to work with files, directories,
/// or to execute commands for example:
///     git, wget, curl, cd, mkdir, mv, cp, find, grep, python, ping
/// and so on so fourth.
pub async fn run_command(cmd: &str, timeout_s: f64, cwd: Option<&std::path::Path>) -> String {
    let mut command = if cfg!(windows) {
        let mut c = tokio::process::Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    } else {
        let mut c = tokio::process::Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    };
    command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }
    let mut process = match command.spawn() {
        Ok(p) => p,
        Err(e) => return format!("Error starting command: {e}"),
    };
    let timed_out = tokio::time::timeout(
        std::time::Duration::from_secs_f64(timeout_s.max(0.001)),
        process.wait(),
    )
    .await;
    match timed_out {
        Err(_) => {
            let _ = process.kill().await;
            format!("Command timed out after {timeout_s}s")
        }
        Ok(Err(e)) => format!("Error running command: {e}"),
        Ok(Ok(status)) => {
            // collect the buffered outputs after exit
            let output = process.wait_with_output().await;
            let stderr = match &output {
                Ok(o) => String::from_utf8_lossy(&o.stderr).to_string(),
                Err(e) => format!("{e}"),
            };
            if !status.success() {
                return format!("Return code [{}]: {stderr}", status.code().unwrap_or(-1));
            }
            match output {
                Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
                Err(e) => format!("Error reading output: {e}"),
            }
        }
    }
}

// ------------------------------------------------------------- registry

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "system_info".into(),
            description: "Returns system information such as OS, CPU.\nUseful when user asks for system status or performance.".into(),
            schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            handler: Arc::new(|_ctx, _args| {
                Box::pin(async move {
                    log_call!("system_info");
                    Ok(ToolOutput::text(system_info()))
                })
            }),
        },
        ToolDef {
            name: "current_date_time".into(),
            description: "Returns the current date and time.\nUseful when user asks for time, date, or 'what day is it?'".into(),
            schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            handler: Arc::new(|_ctx, _args| {
                Box::pin(async move {
                    log_call!("current_date_time");
                    Ok(ToolOutput::text(current_date_time()))
                })
            }),
        },
        ToolDef {
            name: "get_working_directory".into(),
            description: "Returns the current working directory.\nUseful when user asks for getting current directory files and directories,\nor when user asks for launching commands.".into(),
            schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            handler: Arc::new(|_ctx, _args| {
                Box::pin(async move {
                    log_call!("get_working_directory");
                    Ok(ToolOutput::text(get_working_directory()))
                })
            }),
        },
        ToolDef {
            name: "run_command".into(),
            description: "Run a command with a subprocess and wait for it to terminate.\nIt returns either the command output or error as string.\nUseful when user asks to work with files, directories,\nor to execute commands for example:\n    git, wget, curl, cd, mkdir, mv, cp, find, grep, python, ping\nand so on so fourth.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout_s": {"type": "number", "default": 30.0},
                    "cwd": {"type": "string"}
                },
                "required": ["cmd"]
            }),
            handler: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let cmd = args.get("cmd").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                    let timeout_s = args.get("timeout_s").and_then(|v| v.as_f64()).unwrap_or(30.0);
                    let cwd = args.get("cwd").and_then(|v| v.as_str()).map(std::path::PathBuf::from);
                    log_call!("run_command", "cmd" => cmd, "timeout_s" => timeout_s, "cwd" => cwd.clone().map(|p| p.to_string_lossy().to_string()).unwrap_or_default());
                    Ok(ToolOutput::text(run_command(&cmd, timeout_s, cwd.as_deref()).await))
                })
            }),
        },
    ]
}

/// `exit_gracefully` — Python keeps it as the **last** entry of
/// `_default_tools`, so the registry assembly pushes it separately.
pub fn exit_gracefully_tool() -> ToolDef {
    ToolDef {
        name: "exit_gracefully".into(),
        description:
            "Exits the program gracefully.\nUseful when user asks to quit or exit the application."
                .into(),
        schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
        handler: Arc::new(|ctx, args| Box::pin(async move { exit_gracefully(ctx, args).await })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_time_format() {
        let text = current_date_time();
        // "Monday, June 10, 2024 at 9:30 AM, UTC" shape
        assert!(text.contains(" at "));
        assert!(text.contains(", 20") || text.contains(", 20"));
    }

    #[test]
    fn working_directory() {
        let cwd = get_working_directory();
        assert!(!cwd.is_empty());
    }

    #[tokio::test]
    async fn run_command_echo() {
        let out = run_command("echo hello", 5.0, None).await;
        assert!(out.trim() == "hello" || out.contains("hello"), "got: {out}");
    }

    #[tokio::test]
    async fn run_command_timeout() {
        let out = run_command("ping -n 5 127.0.0.1", 0.1, None).await;
        assert!(out.contains("timed out"), "got: {out}");
    }

    #[test]
    fn system_info_string() {
        let info = system_info();
        assert!(info.contains("Processor:"));
    }
}
