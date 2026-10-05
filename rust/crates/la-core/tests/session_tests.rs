//! ConversationSession tests (mirror `tests/test_session.py` and parts of
//! `test_main.py`): commands, EOF handling, tool launcher, generation steps.

use la_core::agent::AgentLike;
use la_core::config::Config;
use la_core::input::ReadOutcome;
use la_core::session::{ConversationSession, PromptSource, ScriptedPrompts};
use la_core::tools::registry::{ToolContext, ToolDef};
use std::sync::{Arc, Mutex};

/// Fake agent (tests/conftest.py FakeAgent parity): records prompts, prints
/// fragments, supports cancellation.
struct FakeAgent {
    acted: Mutex<Vec<String>>,
    should_cancel: std::sync::atomic::AtomicBool,
}

impl FakeAgent {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            acted: Mutex::new(Vec::new()),
            should_cancel: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

impl AgentLike for FakeAgent {
    fn bootstrap(&self) -> Option<String> {
        Some("bootstrap".into())
    }

    fn run_turn<'a>(
        &'a self,
        prompt: &'a str,
        use_tools: bool,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'a>> {
        self.acted.lock().unwrap().push(prompt.to_string());
        if use_tools {
            self.should_cancel
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn cancel(&self) {
        self.should_cancel
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn is_cancelled(&self) -> bool {
        self.should_cancel.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn flush(&self) {}

    fn tools(&self) -> Vec<ToolDef> {
        Vec::new()
    }
}

fn test_context() -> Arc<ToolContext> {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::default();
    config.brain.history_path = dir.path().join("personal.log");
    config.brain.system_prompt = "system".into();
    config.agent.bootstrap_prompt = "bootstrap".into();
    let view = la_core::brain::AgentView {
        client: async_openai::Client::with_config(async_openai::config::OpenAIConfig::new()),
        http: reqwest::Client::new(),
        base_url: "http://localhost".into(),
        model_name: "test".into(),
        temperature: 0.6,
        max_tokens: 100,
        context_window: 1000,
        context_window_source: "config",
        vision: false,
        vision_source: "config",
        messages: Arc::new(Mutex::new(vec![])),
        model_meta: Arc::new(std::sync::RwLock::new(None)),
    };
    // Leak the tempdir for the session lifetime (test process anyway).
    std::mem::forget(dir);
    ToolContext::new(&config, view)
}

/// Test convenience helper; some parity tests below may not use it yet.
#[allow(dead_code)]
fn session_with(
    prompts: Vec<ReadOutcome>,
    use_tools: bool,
    quit_on_interrupt: bool,
) -> ConversationSession {
    let agent = FakeAgent::new();
    let context = test_context();
    // Give the FakeAgent the registry built over this context (empty tools).
    let reader: Arc<Mutex<dyn PromptSource>> = Arc::new(Mutex::new(ScriptedPrompts::new(prompts)));
    ConversationSession::new(agent, reader, context, use_tools, quit_on_interrupt)
}

#[tokio::test]
async fn single_prompt_turn_runs_agent_then_eof() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> = Arc::new(Mutex::new(ScriptedPrompts::new(vec![
        ReadOutcome::Line("hello".into()),
        ReadOutcome::Eof,
    ])));
    let context = test_context();
    let mut session = ConversationSession::new(agent.clone(), reader, context, true, false);
    let code = session.run().await;
    assert_eq!(code, 0);
    assert_eq!(
        *agent.acted.lock().unwrap(),
        vec!["bootstrap".to_string(), "hello".to_string()]
    );
}

#[tokio::test]
async fn commands_quit_and_exit() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> =
        Arc::new(Mutex::new(ScriptedPrompts::new(vec![ReadOutcome::Line(
            "/quit".into(),
        )])));
    let mut session = ConversationSession::new(agent.clone(), reader, test_context(), true, false);
    assert_eq!(session.run().await, 0);
}

#[tokio::test]
async fn exit_alias_quits() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> =
        Arc::new(Mutex::new(ScriptedPrompts::new(vec![ReadOutcome::Line(
            "exit".into(),
        )])));
    let mut session = ConversationSession::new(agent.clone(), reader, test_context(), true, false);
    assert_eq!(session.run().await, 0);
}

#[tokio::test]
async fn eof_ends_session() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> =
        Arc::new(Mutex::new(ScriptedPrompts::new(vec![ReadOutcome::Eof])));
    let mut session = ConversationSession::new(agent.clone(), reader, test_context(), true, false);
    assert_eq!(session.run().await, 0);
}

#[tokio::test]
async fn interrupted_voice_prompt_quits() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> = Arc::new(Mutex::new(ScriptedPrompts::new(vec![
        ReadOutcome::Interrupted,
        ReadOutcome::Line("never reached".into()),
    ])));
    let mut session = ConversationSession::new(agent, reader, test_context(), true, true);
    session.run().await;
    // The second prompt must never be consumed.
    let agent2 = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> = Arc::new(Mutex::new(ScriptedPrompts::new(vec![
        ReadOutcome::Interrupted,
        ReadOutcome::Line("still here".into()),
    ])));
    let mut chat_session =
        ConversationSession::new(agent2.clone(), reader, test_context(), true, false);
    // chat mode: Ctrl+C clears the line and re-prompts
    chat_session.run().await;
    assert_eq!(
        *agent2.acted.lock().unwrap()[1..].to_vec(),
        vec!["still here".to_string()]
    );
}

#[tokio::test]
async fn multi_line_paste_is_not_a_command() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> = Arc::new(Mutex::new(ScriptedPrompts::new(vec![
        ReadOutcome::Line("/quit\nsome more text".into()),
        ReadOutcome::Eof,
    ])));
    let mut session = ConversationSession::new(agent.clone(), reader, test_context(), true, false);
    session.run().await;
    // "/quit\n..." goes to the agent (multi-line paste parity)
    assert!(
        agent
            .acted
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.contains("some more text"))
    );
}

#[tokio::test]
async fn blank_lines_skipped() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> = Arc::new(Mutex::new(ScriptedPrompts::new(vec![
        ReadOutcome::Line("   ".into()),
        ReadOutcome::Eof,
    ])));
    let mut session = ConversationSession::new(agent.clone(), reader, test_context(), true, false);
    session.run().await;
    // only the bootstrap prompt was processed
    assert_eq!(agent.acted.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn clear_command_reseeds_history() {
    let agent = FakeAgent::new();
    let reader: Arc<Mutex<dyn PromptSource>> = Arc::new(Mutex::new(ScriptedPrompts::new(vec![
        ReadOutcome::Line("/clear".into()),
        ReadOutcome::Eof,
    ])));
    let context = test_context();
    context.view.messages.lock().unwrap().push(
        async_openai::types::chat::ChatCompletionRequestMessage::User(
            async_openai::types::chat::ChatCompletionRequestUserMessage {
                content: async_openai::types::chat::ChatCompletionRequestUserMessageContent::Text(
                    "hello".into(),
                ),
                name: None,
            },
        ),
    );
    let mut session = ConversationSession::new(agent, reader, context.clone(), true, false);
    session.run().await;
    // after /clear: system, recall (no personal data), last user message
    let messages = context.view.messages.lock().unwrap();
    assert_eq!(messages.len(), 3);
}
