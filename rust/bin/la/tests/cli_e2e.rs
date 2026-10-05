//! CLI e2e (PLAN §10.3): `la --prompt "hi" --no-tools --config <temp>` against
//! a fake OpenAI-compatible server. Asserts exit code 0, the streamed
//! response on stdout and the turn-end context line — the full binary path
//! (arg parsing → config load → brain → session → printer).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};

/// Minimal fake server: `/v1/models` metadata + one scripted SSE chat reply.
fn spawn_fake_server(events: Vec<String>) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap().to_string();
    let handle = std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                continue;
            }
            let is_models = request_line.contains("GET /v1/models");
            // drain headers (+ body for POST)
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                    break;
                }
                if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
            }
            if content_length > 0 {
                let mut body = vec![0u8; content_length];
                reader.read_exact(&mut body).ok();
            }
            if is_models {
                let body = serde_json::json!({
                    "object": "list",
                    "data": [{
                        "id": "fake-e2e-model",
                        "object": "model",
                        "created": 1_700_000_000u64,
                        "owned_by": "fake",
                        "max_context_length": 4096,
                        "capabilities": ["completion"]
                    }]
                });
                let body = serde_json::to_string(&body).unwrap();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                continue;
            }
            // SSE chat completion: scripted chunks, then [DONE]
            let mut head =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_string();
            let _ = stream.write_all(head.as_bytes());
            head.clear();
            for event in &events {
                let payload = format!("data: {event}\n\n");
                let chunk = format!("{:x}\r\n{}\r\n", payload.len(), payload);
                let _ = stream.write_all(chunk.as_bytes());
            }
            let _ = stream.write_all(b"0\r\n\r\n");
            let _ = stream.flush();
        }
    });
    (addr, handle)
}

fn content_chunk(text: &str) -> String {
    serde_json::json!({
        "id": "chatcmpl-e2e", "object": "chat.completion.chunk", "created": 1, "model": "fake-e2e-model",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
    })
    .to_string()
}

fn finish_chunk(reason: &str) -> String {
    serde_json::json!({
        "id": "chatcmpl-e2e", "object": "chat.completion.chunk", "created": 1, "model": "fake-e2e-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
    })
    .to_string()
}

fn write_config(dir: &std::path::Path, base_url: &str) -> std::path::PathBuf {
    let path = dir.join("config.toml");
    let toml = format!(
        r#"[brain]
system_prompt = "You are a test assistant."
history_path = "chat_history.log"
max_response_tokens = 0.2
temperature = 0.6
api_key_env = "LA_E2E_API_KEY"
base_url = "{base_url}"
model = "fake-e2e-model"

[agent]
bootstrap_prompt = ""
"#
    );
    std::fs::write(&path, toml).unwrap();
    path
}

/// Strip ANSI escape sequences (streamed reply is printed in cyan).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c2 in chars.by_ref() {
                if c2.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn run_la(config: &std::path::Path, extra_args: &[&str]) -> (String, String, Option<i32>) {
    let output = Command::new(env!("CARGO_BIN_EXE_la"))
        .args(["--prompt", "hi", "--config", config.to_str().unwrap()])
        .args(extra_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("LA_E2E_API_KEY", "e2e-key")
        .output()
        .expect("spawn la");
    (
        strip_ansi(&String::from_utf8_lossy(&output.stdout)),
        String::from_utf8_lossy(&output.stderr).to_string(),
        output.status.code(),
    )
}

#[test]
fn single_prompt_turn_streams_reply_and_exits_zero() {
    let (addr, _server) = spawn_fake_server(vec![
        content_chunk("Hello"),
        content_chunk(" from"),
        content_chunk(" the fake server"),
        finish_chunk("stop"),
    ]);
    let dir = tempfile::tempdir().expect("tempdir");
    let config = write_config(dir.path(), &format!("http://{addr}/v1"));

    // SAFETY: test process, single-threaded test body.
    unsafe {
        std::env::set_var("LA_E2E_API_KEY", "e2e-key");
    }

    let (stdout, stderr, code) = run_la(&config, &["--no-tools"]);
    assert!(
        code == Some(0),
        "exit code: {code:?}\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("Hello from the fake server"),
        "streamed reply missing from stdout:\n{stdout}"
    );
    // Python parity: the no-tools path (`model_fragments`) prints no context
    // usage line — only the tool path (`act`) does.
    assert!(
        !stdout.contains("context:"),
        "no-tools turn must not print a context line:\n{stdout}"
    );
    assert!(!stdout.contains("⛔"), "error output:\n{stdout}");
    // keep the server thread alive until the child exited
    drop(_server);
}

#[test]
fn prompt_turn_with_tools_prints_context_line() {
    let (addr, _server) = spawn_fake_server(vec![content_chunk("Answer"), finish_chunk("stop")]);
    let dir = tempfile::tempdir().expect("tempdir");
    let config = write_config(dir.path(), &format!("http://{addr}/v1"));

    unsafe {
        std::env::set_var("LA_E2E_API_KEY", "e2e-key");
    }

    let (stdout, stderr, code) = run_la(&config, &[]); // tools enabled
    assert!(
        code == Some(0),
        "exit code: {code:?}\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("Answer"), "reply missing:\n{stdout}");
    assert!(
        stdout.contains("context:") && stdout.contains("[4k]"),
        "act() rounds must end with the context usage line:\n{stdout}"
    );
    drop(_server);
}
