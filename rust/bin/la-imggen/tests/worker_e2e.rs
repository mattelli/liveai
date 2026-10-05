//! Worker e2e: spawn the built `la-imggen` binary and drive the JSON-lines
//! protocol (PLAN §10.3). Runs against the feature-less build, where every
//! backend request answers with a deterministic error — the round-trip and
//! id-matching logic is what's under test, not the diffusion pipeline.

use la_proto::{Envelope, MainToWorker, T2iParams, WorkerToMain};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

fn spawn_imggen() -> Child {
    Command::new(env!("CARGO_BIN_EXE_la-imggen"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn la-imggen")
}

fn send(child: &mut Child, id: u64, body: &MainToWorker) {
    let stdin = child.stdin.as_mut().expect("stdin");
    let envelope = Envelope::with_id(id, body.clone());
    let line = serde_json::to_string(&envelope).expect("serialize");
    writeln!(stdin, "{line}").expect("write");
    stdin.flush().expect("flush");
}

/// Read one worker message; returns (id, body) or None on EOF.
fn read_response(
    reader: &mut BufReader<std::process::ChildStdout>,
) -> Option<(Option<u64>, WorkerToMain)> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    let id = value.get("id").and_then(|v| v.as_u64());
    let body: WorkerToMain = serde_json::from_value(value).ok()?;
    Some((id, body))
}

fn t2i_params() -> T2iParams {
    T2iParams {
        prompt: "a red apple".into(),
        negative_prompt: None,
        width: 512,
        height: 512,
        seed: -1,
        steps: 20,
        out_file: "unused.png".into(),
    }
}

#[test]
fn unload_without_loaded_model_answers_error_with_matching_id() {
    let mut child = spawn_imggen();
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));
    send(&mut child, 11, &MainToWorker::Unload);
    let (id, body) = read_response(&mut reader).expect("worker response");
    assert_eq!(id, Some(11), "response must carry the request id");
    match body {
        WorkerToMain::Error { message } => {
            assert!(
                message.contains("No image generation model is loaded"),
                "got: {message}"
            );
        }
        other => panic!("expected Error, got: {other:?}"),
    }
    send(&mut child, 12, &MainToWorker::Stop);
    let status = child.wait().expect("wait");
    assert!(status.success(), "stop must exit cleanly, got: {status}");
}

#[test]
fn t2i_without_backend_answers_error() {
    let mut child = spawn_imggen();
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));
    send(&mut child, 1, &MainToWorker::T2i(t2i_params()));
    let (id, body) = read_response(&mut reader).expect("worker response");
    assert_eq!(id, Some(1), "response must carry the request id");
    match body {
        WorkerToMain::Error { message } => {
            assert!(
                message.contains("rebuild with") && message.contains("--features sd"),
                "stub error must carry the rebuild hint, got: {message}"
            );
        }
        other => panic!("expected Error, got: {other:?}"),
    }
    send(&mut child, 2, &MainToWorker::Stop);
    assert!(child.wait().expect("wait").success());
}

#[test]
fn stdin_eof_terminates_the_worker() {
    // Parent-death watchdog (PLAN §3.3): closing stdin ends the worker.
    let mut child = spawn_imggen();
    child.stdin.take().expect("stdin"); // drop -> EOF
    let status = child.wait().expect("wait");
    assert!(status.success(), "EOF must be a clean exit, got: {status}");
}
