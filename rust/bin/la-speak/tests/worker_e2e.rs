//! Worker e2e: spawn the built `la-speak` binary and drive the JSON-lines
//! protocol (PLAN §10.3). Without the `tts` feature the synthesis requests
//! answer with a deterministic rebuild-hint error; `unload`/`stop`/EOF
//! lifecycle is verified against the real process.

use la_proto::{Envelope, MainToWorker, WorkerToMain};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

fn spawn_speak() -> Child {
    Command::new(env!("CARGO_BIN_EXE_la-speak"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn la-speak")
}

fn send(child: &mut Child, id: u64, body: &MainToWorker) {
    let stdin = child.stdin.as_mut().expect("stdin");
    let envelope = Envelope::with_id(id, body.clone());
    let line = serde_json::to_string(&envelope).expect("serialize");
    writeln!(stdin, "{line}").expect("write");
    stdin.flush().expect("flush");
}

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

#[test]
fn speak_without_tts_backend_answers_error_with_matching_id() {
    let mut child = spawn_speak();
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));
    send(
        &mut child,
        5,
        &MainToWorker::Speak {
            text: "Hello world".into(),
            strip: true,
            file: None,
        },
    );
    let (id, body) = read_response(&mut reader).expect("worker response");
    assert_eq!(id, Some(5), "response must carry the request id");
    match body {
        WorkerToMain::Error { message } => {
            assert!(
                message.contains("without tts support") && message.contains("--features tts"),
                "stub error must carry the rebuild hint, got: {message}"
            );
        }
        other => panic!("expected Error, got: {other:?}"),
    }
    send(&mut child, 6, &MainToWorker::Stop);
    assert!(
        child.wait().expect("wait").success(),
        "stop must exit cleanly"
    );
}

#[test]
fn unload_without_generator_is_a_no_op_done() {
    // Python `unload_speech_generator` without a loaded generator: a no-op
    // success (unlike imggen, which errors "No image generation model is
    // loaded"); the worker answers Done with an empty path.
    let mut child = spawn_speak();
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));
    send(&mut child, 7, &MainToWorker::Unload);
    let (id, body) = read_response(&mut reader).expect("worker response");
    assert_eq!(id, Some(7));
    match body {
        WorkerToMain::Done { path, seed, .. } => {
            assert_eq!(path, "");
            assert_eq!(seed, None);
        }
        other => panic!("expected Done, got: {other:?}"),
    }
    send(&mut child, 8, &MainToWorker::Stop);
    assert!(child.wait().expect("wait").success());
}

#[test]
fn stdin_eof_terminates_the_worker() {
    let mut child = spawn_speak();
    child.stdin.take().expect("stdin"); // drop -> EOF
    let status = child.wait().expect("wait");
    assert!(status.success(), "EOF must be a clean exit, got: {status}");
}
