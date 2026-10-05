//! Agent integration tests against a hand-rolled fake OpenAI-compatible
//! server (SSE streaming + tool calls + `/models` metadata), mirroring the
//! behaviors validated by `tests/test_live_ai.py` + `tests/test_main.py`.

use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
};
use la_core::agent::{Agent, AgentLike, model_info_string};
use la_core::brain::response_token_cap;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

// ------------------------------------------------------------------ fakes

/// A tiny HTTP server: routes
/// - `GET  /v1/models` and `/v1/models/{id}` (metadata with extras)
/// - `POST /v1/chat/completions` (SSE stream scripted per request body)
struct FakeServer;

impl FakeServer {
    fn spawn(scripts: Vec<(String, Vec<String>)>) -> (String, tokio::sync::oneshot::Receiver<()>) {
        use std::collections::VecDeque;
        let scripts = Arc::new(std::sync::Mutex::new(VecDeque::from(scripts)));
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let scripts_for_model = scripts.clone();
        std::thread::spawn(move || {
            let tx_ref = tx;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let _ = tx_ref;
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                // read request head
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                    continue;
                }
                let is_models = request_line.contains("GET /v1/models");
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                }
                if content_length > 0 {
                    let mut body = vec![0u8; content_length];
                    use std::io::Read;
                    reader.read_exact(&mut body).ok();
                }
                if is_models {
                    let body = serde_json::json!({
                        "object": "list",
                        "data": [{
                            "id": "test-model",
                            "object": "model",
                            "created": 1_700_000_000u64,
                            "owned_by": "fake-org",
                            "max_context_length": 4096,
                            "capabilities": ["completion", "tools"]
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
                // chat completion: pick the next script
                let script = scripts_for_model
                    .lock()
                    .unwrap()
                    .pop_front()
                    .map(|(_, events)| events)
                    .unwrap_or_default();
                let mut head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_string();
                let _ = stream.write_all(head.as_bytes());
                head.clear();
                for event in script {
                    let payload = format!("data: {event}\n\n");
                    let chunk = format!("{:x}\r\n{}\r\n", payload.len(), payload);
                    let _ = stream.write_all(chunk.as_bytes());
                    let _ = stream.flush();
                }
                let _ = stream.write_all(b"0\r\n\r\n");
                let _ = stream.flush();
            }
        });
        (addr, rx)
    }
}

fn content_chunk(text: &str) -> String {
    serde_json::json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": text}}]
    })
    .to_string()
}

fn tool_call_chunk(
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> String {
    serde_json::json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": index,
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments}
            }]}
        }]
    })
    .to_string()
}

fn finish_chunk(reason: &str) -> String {
    serde_json::json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
    })
    .to_string()
}

// --------------------------------------------------------------- helpers

fn test_config(base_url: &str) -> la_core::Config {
    let mut config = la_core::Config::default();
    config.brain.model = "test-model".into();
    config.brain.base_url = Some(base_url.to_string());
    config.brain.api_key_env = Some("LA_TEST_KEY".into());
    config.brain.system_prompt = "You are a helpful assistant".into();
    config.brain.max_response_tokens = 0.0;
    // SAFETY: test startup, single-threaded access to this key
    unsafe {
        std::env::set_var("LA_TEST_KEY", "test-key");
    }
    config
}

/// Count characters of the text content of a message list (context usage).
fn content_chars(messages: &[ChatCompletionRequestMessage]) -> usize {
    messages
        .iter()
        .map(|m| la_core::brain::OpenAICore::message_content(m).len())
        .sum()
}

// ------------------------------------------------------------------ tests

#[tokio::test]
async fn streaming_plain_response_keeps_history() {
    let (addr, _shutdown) = FakeServer::spawn(vec![(
        String::new(),
        vec![
            content_chunk("Hello"),
            content_chunk(" world"),
            finish_chunk("stop"),
        ],
    )]);
    let config = test_config(&format!("http://{addr}/v1"));
    let agent = Agent::new(&config).await.unwrap();
    agent.model_fragments("hi").await.unwrap();
    let messages = agent.core.messages.lock().unwrap();
    assert_eq!(messages.len(), 3, "history: {messages:?}");
    assert!(matches!(
        &messages[1],
        ChatCompletionRequestMessage::User(_)
    ));
    let content = match &messages[2] {
        ChatCompletionRequestMessage::Assistant(a) => match &a.content {
            Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => t.clone(),
            _ => String::new(),
        },
        _ => String::new(),
    };
    assert_eq!(content, "Hello world");
}

#[tokio::test]
async fn act_round_runs_tool_and_returns_answer() {
    let (addr, _shutdown) = FakeServer::spawn(vec![
        (
            String::new(),
            vec![
                content_chunk("calling"),
                tool_call_chunk(0, Some("call_1"), Some("current_date_time"), Some("{}")),
                finish_chunk("tool_calls"),
            ],
        ),
        (
            String::new(),
            vec![
                content_chunk("It is "),
                content_chunk("today."),
                finish_chunk("stop"),
            ],
        ),
    ]);
    let config = test_config(&format!("http://{addr}/v1"));
    let agent = Agent::new(&config).await.unwrap();
    agent.act("what day is it?").await.unwrap();
    let messages = agent.core.messages.lock().unwrap();
    // system, user, assistant(tool_calls), tool(result), assistant(answer)
    assert_eq!(messages.len(), 5, "history: {messages:?}");
    assert!(matches!(
        &messages[3],
        ChatCompletionRequestMessage::Tool(_)
    ));
}

#[tokio::test]
async fn cancel_flag_lifecycle() {
    // Python parity: cancel_generation sets the pending flag; a fresh turn
    // (act) clears it before streaming, so a *pre-turn* cancel does not skip
    // the turn (Ctrl+C only cancels the in-flight generation).
    let (addr, _shutdown) = FakeServer::spawn(vec![(
        String::new(),
        vec![content_chunk("ok"), finish_chunk("stop")],
    )]);
    let config = test_config(&format!("http://{addr}/v1"));
    let agent = Agent::new(&config).await.unwrap();
    agent.cancel();
    assert!(agent.is_cancelled(), "cancel() sets the pending flag");
    agent.act("do something").await.unwrap();
    assert!(
        !agent.is_cancelled(),
        "a fresh turn clears the pending flag"
    );
    let messages = agent.core.messages.lock().unwrap();
    assert!(
        messages
            .iter()
            .any(|m| matches!(m, ChatCompletionRequestMessage::Assistant(_))),
        "the turn completed normally"
    );
}

#[test]
fn history_repair_pads_and_drops() {
    use async_openai::types::chat::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent,
    };

    let mut messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestMessage::System(
            async_openai::types::chat::ChatCompletionRequestSystemMessage {
                content: async_openai::types::chat::ChatCompletionRequestSystemMessageContent::Text(
                    "sys".into(),
                ),
                name: None,
            },
        ),
        // empty assistant turn (cancelled generation)
        ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
            content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                String::new(),
            )),
            ..Default::default()
        }),
        // trailing unanswered user prompt
        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text("unanswered".into()),
            name: None,
        }),
    ];
    la_core::brain::OpenAICore::repair_history(&mut messages);
    // Python parity: the unanswered user prompt is dropped, the empty
    // assistant turn padded -> [system, assistant(placeholder)]
    assert_eq!(
        messages.len(),
        2,
        "empty assistant padded, unanswered user dropped"
    );
    let text = match &messages[1] {
        ChatCompletionRequestMessage::Assistant(a) => match &a.content {
            Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => t.clone(),
            _ => String::new(),
        },
        _ => String::new(),
    };
    assert_eq!(text, la_core::common::CANCELLED_PLACEHOLDER);
    // the unanswered user prompt was dropped with it
    assert!(
        !messages
            .iter()
            .any(|m| matches!(m, ChatCompletionRequestMessage::User(_)))
    );
}

#[test]
fn response_token_cap_matches_python() {
    assert_eq!(response_token_cap(0.0, 128_000).unwrap(), 128_000);
    assert_eq!(response_token_cap(0.1, 4_096).unwrap(), 409);
    assert_eq!(response_token_cap(0.2, 8_192).unwrap(), 1_638);
    assert!(response_token_cap(1.0, 100).is_err());
    assert!(response_token_cap(-0.5, 100).is_err());
}

#[tokio::test]
async fn model_info_reports_server_metadata() {
    let (addr, _shutdown) = FakeServer::spawn(vec![]);
    let config = test_config(&format!("http://{addr}/v1"));
    let agent = Agent::new(&config).await.unwrap();
    let view = agent.core.view();
    let info = model_info_string(&view, &view.metadata());
    assert!(info.contains("model=test-model"));
    assert!(info.contains("provider=fake-org"));
    assert!(info.contains("context_window=4096 tokens (server-reported)"));
    assert!(info.contains("server_context=max_context_length=4096"));
    assert!(info.contains("capabilities=completion, tools"));
    assert!(info.contains("created=2023-11-14"));
    // the fake server reports capabilities without "vision" -> no vision
    assert!(info.contains("vision=no"));
    assert!(info.contains("vision_source=server"));
}

#[tokio::test]
async fn context_usage_estimate() {
    let (addr, _shutdown) = FakeServer::spawn(vec![]);
    let config = test_config(&format!("http://{addr}/v1"));
    let agent = Agent::new(&config).await.unwrap();
    let base_chars = content_chars(&agent.core.messages.lock().unwrap());
    agent.core.append_user_text(&"x".repeat(400));
    let (pct, window) = agent.core.context_usage();
    assert_eq!(window, 4096);
    // ~ (base + 100 tokens) / 4096 * 100
    assert!(pct > 0.0);
    let _ = base_chars;
}
