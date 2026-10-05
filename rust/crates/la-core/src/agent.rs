//! `Brain` + `Agent` — tool-bearing assistant built on top of [`OpenAICore`]
//! (`src/live_ai.py` port).
//!
//! Owns the conversation history (plumbing and repair across cancelled or
//! failed turns), streaming generation with cancellation, the model info tool
//! and the round-based `act()` loop: stream a round, run the requested tools,
//! feed the results back, and repeat until the model answers without tool
//! calls. Cancel (Ctrl+C) closes the in-flight HTTP stream for both streamed
//! responses and act() rounds; partial output is kept in the chat history.

use crate::brain::{DEFAULT_CONTEXT_WINDOW, OpenAICore};
use crate::common::{CANCELLED_PLACEHOLDER, context_line, cprint};
use crate::config::Config;
use crate::printer::StreamPrinter;
use crate::tools::registry::{ToolContext, ToolDef, ToolRegistry};
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionStreamResponseDelta, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionStreamResponse, FunctionCall, FunctionObject,
};
use futures::StreamExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

/// Maximum number of tool-calling rounds per act() (Python `MAX_ACT_ROUNDS`).
pub const MAX_ACT_ROUNDS: usize = 20;

type StreamSlot = futures::stream::BoxStream<
    'static,
    Result<CreateChatCompletionStreamResponse, async_openai::error::OpenAIError>,
>;

/// Speech callback wired to the `la-speak` worker.
type SpeakHook = Arc<dyn Fn(String) + Send + Sync>;

/// Session-bound interface used by `ConversationSession`
/// (Python: the duck-typed agent; tests provide a `FakeAgent`).
pub trait AgentLike: Send + Sync {
    /// The configured bootstrap prompt, if any (run once after the banner).
    fn bootstrap(&self) -> Option<String> {
        None
    }
    /// One agent turn: tool rounds (use_tools) or a plain generation.
    fn run_turn<'a>(
        &'a self,
        prompt: &'a str,
        use_tools: bool,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'a>>;
    /// Stop the in-flight generation as soon as possible (best effort).
    fn cancel(&self);
    /// True while a user cancel (Ctrl+C) is pending for the current turn.
    fn is_cancelled(&self) -> bool;
    /// Flush buffered output and stop the spinner (turn boundary).
    fn flush(&self);
    /// Registered tool definitions (for the `/tools` launcher).
    fn tools(&self) -> Vec<ToolDef>;
    /// `/info` output; None -> "Info unavailable".
    fn model_info(&self) -> Option<String> {
        None
    }
}

/// The tool-bearing agent (Python `Brain` + `Agent` merged).
pub struct Agent {
    pub core: OpenAICore,
    pub stream: Mutex<StreamPrinter>,
    pub cancel_requested: Arc<AtomicBool>,
    pub cancel_notify: Arc<Notify>,
    /// Response fragments of the current round (Python `_fragments_buffer`).
    fragments_buffer: Mutex<Vec<String>>,
    pub registry: Arc<ToolRegistry>,
    /// Strong handle on the tool context (the registry only keeps a Weak).
    pub context: Arc<ToolContext>,
    /// Speech hook (wired by the main binary to the speak worker); settable
    /// after construction via `set_speak`.
    speak: std::sync::RwLock<Option<SpeakHook>>,
    bootstrap_prompt: Option<String>,
}

impl Agent {
    /// Attach the speech callback (called by the main binary after creation).
    pub fn set_speak(&self, speak: SpeakHook) {
        *self.speak.write().unwrap() = Some(speak);
    }
}

/// Accumulated tool call from streamed deltas (Python `_stream_round` slots).
#[derive(Debug, Clone, Default)]
struct ToolCallSlot {
    id: String,
    name: String,
    arguments: String,
}

impl Agent {
    /// Create the agent from the loaded config (client, metadata, tools),
    /// building a default tool context.
    pub async fn new(config: &Config) -> anyhow::Result<Arc<Self>> {
        let mut core = OpenAICore::new(&config.brain)?;
        core.resolve_metadata().await;
        let context = ToolContext::new(config, core.view());
        Self::with_parts(config, core, context).await
    }

    /// Create the agent over a pre-built tool context (the main binary wires
    /// worker handles + ML encoders into it before this call).
    pub async fn with_parts(
        config: &Config,
        core: OpenAICore,
        context: Arc<ToolContext>,
    ) -> anyhow::Result<Arc<Self>> {
        let mut agent = Self {
            core,
            stream: Mutex::new(StreamPrinter::new("cyan")),
            cancel_requested: Arc::new(AtomicBool::new(false)),
            cancel_notify: Arc::new(Notify::new()),
            fragments_buffer: Mutex::new(Vec::new()),
            registry: Arc::new(ToolRegistry::empty()),
            context: context.clone(),
            speak: std::sync::RwLock::new(None),
            bootstrap_prompt: Some(config.agent.bootstrap_prompt.clone()).filter(|s| !s.is_empty()),
        };
        // Build the tool registry bound to the provided context.
        agent.registry = Arc::new(ToolRegistry::build(context.clone()));
        Ok(Arc::new(agent))
    }

    /// The tool-definition list sent to the model (OpenAI function format).
    pub fn tool_definitions(&self) -> Vec<ChatCompletionTools> {
        self.registry
            .tools
            .iter()
            .map(|def| {
                ChatCompletionTools::Function(ChatCompletionTool {
                    function: FunctionObject {
                        name: def.name.clone(),
                        description: Some(def.description.clone()),
                        parameters: Some(def.schema.clone()),
                        strict: None,
                    },
                })
            })
            .collect()
    }

    /// Buffer a fragment and print it (Python `_handle_fragment`).
    fn handle_fragment(&self, content: &str) {
        self.fragments_buffer
            .lock()
            .unwrap()
            .push(content.to_string());
        self.print_fragment(content);
    }

    /// Round-end handler: flush, context line, optional speech, clear buffer
    /// (Python `_handle_round_end`).
    fn handle_round_end(&self, round_index: usize) {
        let _ = round_index;
        let (usage_pct, context_window) = self.core.context_usage();
        {
            let mut stream = self.stream.lock().unwrap();
            stream.flush(); // keep the spinner alive across rounds
        }
        cprint("", "reset", false, "\n");
        context_line(usage_pct, context_window);
        {
            let mut stream = self.stream.lock().unwrap();
            stream.mark_line_clean(); // spinner may resume for the next round
            stream.note("working");
        }
        let response = {
            let mut buffer = self.fragments_buffer.lock().unwrap();
            let response = buffer.join(" ").trim().to_string();
            buffer.clear();
            response
        };
        if !self.cancel_requested.load(Ordering::SeqCst) {
            let speak = self.speak.read().unwrap().clone();
            if let Some(speak) = speak {
                speak(response);
            }
        }
    }

    /// Stream one completion round; return (text, accumulated tool calls).
    async fn stream_round(
        &self,
        tools: &[ChatCompletionTools],
    ) -> anyhow::Result<(String, Vec<ToolCallSlot>)> {
        let stream = self.core.stream_completion(Some(tools.to_vec())).await?;
        let mut boxed: StreamSlot = Box::pin(stream);

        let mut parts: Vec<String> = Vec::new();
        let mut calls: Vec<ToolCallSlot> = Vec::new();
        loop {
            let next = tokio::select! {
                biased;
                item = boxed.next() => item,
                _ = self.cancel_notify.notified() => {
                    if self.cancel_requested.load(Ordering::SeqCst) { None } else { continue }
                },
            };
            let Some(chunk) = next else { break };
            if self.cancel_requested.load(Ordering::SeqCst) {
                break;
            }
            let chunk = chunk?;
            for choice in chunk.choices {
                let delta: &ChatCompletionStreamResponseDelta = &choice.delta;
                if let Some(content) = &delta.content
                    && !content.is_empty()
                {
                    parts.push(content.clone());
                    self.handle_fragment(content);
                }
                for tc in delta.tool_calls.iter().flatten() {
                    let index = tc.index as usize;
                    while calls.len() <= index {
                        calls.push(ToolCallSlot::default());
                    }
                    let slot = &mut calls[index];
                    if let Some(id) = &tc.id {
                        slot.id = id.clone();
                    }
                    if let Some(function) = &tc.function {
                        slot.name
                            .push_str(function.name.as_deref().unwrap_or_default());
                        slot.arguments
                            .push_str(function.arguments.as_deref().unwrap_or_default());
                    }
                }
            }
        }
        // Dropping `boxed` here closes the HTTP stream when cancelled
        // (Python: `stream.close()` in cancel_generation + unwinding loop).
        drop(boxed);
        let calls: Vec<ToolCallSlot> = calls.into_iter().filter(|c| !c.name.is_empty()).collect();
        Ok((parts.concat(), calls))
    }

    /// Print a response fragment to the console (cyan / highlighted code).
    pub fn print_fragment(&self, content: &str) {
        self.stream.lock().unwrap().feed(content);
    }

    /// Execute one tool call and return its string result
    /// (Python `_execute_tool`; errors are returned, not raised).
    async fn execute_tool(&self, call: &ToolCallSlot) -> String {
        let args: serde_json::Value = match serde_json::from_str(if call.arguments.is_empty() {
            "{}"
        } else {
            &call.arguments
        }) {
            Ok(v) => v,
            Err(e) => return format!("Error: malformed arguments for '{}': {e}", call.name),
        };
        match self.registry.get(&call.name) {
            None => format!("Error: unknown tool '{}'", call.name),
            Some(def) => match (def.handler)(self.registry.context(), args).await {
                Ok(output) => output.text,
                Err(e) => format!("Error: {e}"),
            },
        }
    }

    /// Perform an action based on the input prompt; the response may involve
    /// using tools or just generating text (Python `act`).
    pub async fn act(&self, prompt: &str) -> anyhow::Result<()> {
        cprint(
            &format!("\n🤖 [{}]", self.core.model_name),
            "grey",
            false,
            "\n",
        );
        log::info!("Sending to Agent [🤖: {}] ...", self.core.model_name);
        self.core.add_user_prompt(prompt);
        self.cancel_requested.store(false, Ordering::SeqCst);
        self.stream.lock().unwrap().start_spinner("thinking");
        let tools = self.tool_definitions();
        let mut round_index = 0;
        while round_index < MAX_ACT_ROUNDS {
            let (text, tool_calls) = self.stream_round(&tools).await?;
            if !tool_calls.is_empty() && !self.cancel_requested.load(Ordering::SeqCst) {
                // Record the assistant tool-call turn, then the tool results.
                let call_list: Vec<ChatCompletionMessageToolCalls> = tool_calls
                    .iter()
                    .enumerate()
                    .map(|(index, call)| {
                        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                            id: if call.id.is_empty() {
                                format!("call_{index}")
                            } else {
                                call.id.clone()
                            },
                            function: FunctionCall {
                                name: call.name.clone(),
                                arguments: if call.arguments.is_empty() {
                                    "{}".into()
                                } else {
                                    call.arguments.clone()
                                },
                            },
                        })
                    })
                    .collect();
                self.core
                    .messages
                    .lock()
                    .unwrap()
                    .push(ChatCompletionRequestMessage::Assistant(
                        ChatCompletionRequestAssistantMessage {
                            content: if text.is_empty() {
                                None
                            } else {
                                Some(ChatCompletionRequestAssistantMessageContent::Text(text))
                            },
                            tool_calls: Some(call_list),
                            ..Default::default()
                        },
                    ));
                for (index, call) in tool_calls.iter().enumerate() {
                    let result = self.execute_tool(call).await;
                    log::debug!(
                        "Tool {} -> {}",
                        call.name,
                        result.chars().take(200).collect::<String>()
                    );
                    let id = if call.id.is_empty() {
                        format!("call_{index}")
                    } else {
                        call.id.clone()
                    };
                    self.core
                        .messages
                        .lock()
                        .unwrap()
                        .push(ChatCompletionRequestMessage::Tool(
                            ChatCompletionRequestToolMessage {
                                content: ChatCompletionRequestToolMessageContent::Text(result),
                                tool_call_id: id,
                            },
                        ));
                }
                self.handle_round_end(round_index);
                round_index += 1;
                continue;
            }
            let text = if text.is_empty() && self.cancel_requested.load(Ordering::SeqCst) {
                CANCELLED_PLACEHOLDER.to_string()
            } else {
                text
            };
            self.core.append_assistant_text(&text);
            self.handle_round_end(round_index);
            break;
        }
        self.stream.lock().unwrap().stop_spinner();
        Ok(())
    }

    /// Get response fragments as a stream from the model for a prompt
    /// (Python `model_fragments`); tools are not supported in this method.
    pub async fn model_fragments(&self, prompt: &str) -> anyhow::Result<()> {
        log::info!("Sending to Model [🤖: {}] ...", self.core.model_name);
        self.core.add_user_prompt(prompt);
        self.cancel_requested.store(false, Ordering::SeqCst);
        self.stream.lock().unwrap().start_spinner("thinking");
        let mut parts: Vec<String> = Vec::new();
        if let Ok(stream) = self.core.stream_completion(None).await {
            let mut boxed: StreamSlot = Box::pin(stream);
            loop {
                let next = tokio::select! {
                    biased;
                    item = boxed.next() => item,
                    _ = self.cancel_notify.notified() => {
                        if self.cancel_requested.load(Ordering::SeqCst) { None } else { continue }
                    },
                };
                let Some(chunk) = next else { break };
                if self.cancel_requested.load(Ordering::SeqCst) {
                    break;
                }
                let chunk = chunk?;
                for choice in chunk.choices {
                    if let Some(content) = &choice.delta.content
                        && !content.is_empty()
                    {
                        parts.push(content.clone());
                        self.handle_fragment(content);
                    }
                }
            }
            drop(boxed);
        }
        let text = parts.concat();
        if !text.is_empty() {
            self.core.append_assistant_text(&text);
        } else if self.cancel_requested.load(Ordering::SeqCst) {
            self.core.append_assistant_text(CANCELLED_PLACEHOLDER);
        }
        self.flush_stream();
        Ok(())
    }

    /// Flush buffered text and stop the activity spinner (turn boundary).
    pub fn flush_stream(&self) {
        let mut stream = self.stream.lock().unwrap();
        stream.flush();
        stream.stop_spinner();
    }
}

/// The model_info tool body (Python `Brain.model_info`).
pub fn model_info_string(view: &crate::brain::AgentView, metadata: &serde_json::Value) -> String {
    let meta = metadata.as_object().cloned().unwrap_or_default();
    let (usage_pct, context_window) = view.context_usage();
    let mut lines: Vec<String> = vec![
        format!("model={}", view.model_name),
        format!("base_url={} (OpenAI-compatible API)", view.base_url),
    ];
    if let Some(owned_by) = meta.get("owned_by").and_then(|v| v.as_str()) {
        lines.push(format!("provider={owned_by}"));
    }
    if let Some(created) = meta.get("created").and_then(OpenAICore::format_created) {
        lines.push(format!("created={created}"));
    }
    let window_labels = |source: &str| match source {
        "config" => "configured",
        "server" => "server-reported",
        _ => "default (server did not report)",
    };
    lines.push(format!(
        "context_window={} tokens ({})",
        context_window,
        window_labels(view.context_window_source)
    ));
    let mut context_fields: Vec<(String, &serde_json::Value)> = meta
        .iter()
        .filter(|(k, v)| k.to_lowercase().contains("context") && v.is_i64())
        .map(|(k, v)| (k.clone(), v))
        .collect();
    context_fields.sort_by(|a, b| a.0.cmp(&b.0));
    if !context_fields.is_empty() {
        lines.push(format!(
            "server_context={}",
            context_fields
                .iter()
                .map(|(k, v)| format!("{k}={}", v.as_i64().unwrap_or(0)))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let max_fields: Vec<u64> = context_fields
            .iter()
            .filter(|(k, _)| !k.to_lowercase().starts_with("loaded"))
            .filter_map(|(_, v)| v.as_i64().map(|v| v as u64))
            .collect();
        if max_fields.iter().any(|&v| v < context_window) {
            lines.push(
                "warning: configured context_window exceeds the server-reported maximum".into(),
            );
        }
    }
    if let Some(capabilities) = meta.get("capabilities").and_then(|v| v.as_array()) {
        lines.push(format!(
            "capabilities={}",
            capabilities
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let rendered: std::collections::BTreeSet<String> = ["owned_by", "created", "capabilities"]
        .into_iter()
        .map(str::to_string)
        .chain(context_fields.iter().map(|(k, _)| k.clone()))
        .collect();
    for key in meta
        .keys()
        .filter(|k| !rendered.contains(*k))
        .cloned()
        .collect::<Vec<String>>()
    {
        let value = &meta[&key];
        let rendered_value = match value {
            serde_json::Value::Array(items) => items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
                .join(", "),
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        lines.push(format!("{key}={rendered_value}"));
    }
    lines.push(format!("temperature={}", view.temperature));
    lines.push(format!(
        "max_response_tokens={} (per-generation cap)",
        view.max_tokens
    ));
    lines.push(format!("vision={}", if view.vision { "yes" } else { "no" }));
    lines.push(format!("vision_source={}", view.vision_source));
    let message_count = view.messages.lock().unwrap().len();
    lines.push(format!(
        "session_messages={message_count} (context ~{usage_pct:.1}% used)"
    ));
    let _ = DEFAULT_CONTEXT_WINDOW;
    lines.join("\n")
}

impl AgentLike for Agent {
    fn bootstrap(&self) -> Option<String> {
        self.bootstrap_prompt.clone()
    }

    fn run_turn<'a>(
        &'a self,
        prompt: &'a str,
        use_tools: bool,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'a>> {
        if use_tools {
            Box::pin(self.act(prompt))
        } else {
            Box::pin(self.model_fragments(prompt))
        }
    }

    fn cancel(&self) {
        self.cancel_requested.store(true, Ordering::SeqCst);
        self.cancel_notify.notify_waiters();
        // Closing the active HTTP stream happens when the streaming loop
        // breaks and drops the response stream (best-effort server-side stop).
    }

    fn is_cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::SeqCst)
    }

    fn flush(&self) {
        self.flush_stream();
    }

    fn tools(&self) -> Vec<ToolDef> {
        self.registry.tools.clone()
    }

    fn model_info(&self) -> Option<String> {
        let view = self.core.view();
        Some(model_info_string(&view, &view.metadata()))
    }
}
