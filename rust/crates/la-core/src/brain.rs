//! `OpenAICore` — core client for an OpenAI-compatible chat API
//! (`src/live_ai.py::OpenAICore` port).
//!
//! Talks to `[brain] base_url` with `async-openai`, so cloud models
//! (OpenAI, Mistral, Groq, OpenRouter, DeepSeek, Together, ...) and local
//! servers (Ollama, LM Studio) share one implementation: connection handling,
//! model metadata resolution (context window, vision) from the OpenAI models
//! endpoint, and the tool-definition / stream-chunk helpers. The API key is
//! read from the environment variable named by `[brain] api_key_env`
//! (also resolved from a project `.env` file) and never stored in config.toml.

use crate::common::CANCELLED_PLACEHOLDER;
use anyhow::{Context as AnyhowContext, Result, anyhow};
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPart, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent,
};
use chrono::DateTime;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

pub const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;

/// Server metadata keys describing the model's maximum context, in the order
/// they are preferred when several are reported (Python `_CONTEXT_MAX_KEYS`).
const CONTEXT_MAX_KEYS: [&str; 4] = [
    "max_context_length",
    "context_length",
    "context_window",
    "max_model_len",
];

/// Model record keys skipped when flattening (Python `_MODEL_RECORD_SKIP`).
const MODEL_RECORD_SKIP: [&str; 5] = ["id", "object", "root", "parent", "permission"];

/// Load `.env` from the project root; existing environment variables always
/// win over file entries (Python `load_dotenv(override=False)`).
pub fn load_env_file(dotenv_path: Option<PathBuf>) {
    let path = dotenv_path.unwrap_or_else(|| PathBuf::from(".env"));
    // `dotenvy::from_path` does NOT override existing vars (override=False
    // semantics, matching src/live_ai.py `_load_env_file`). A missing file is
    // not an error for the caller.
    let _ = dotenvy::from_path(path).err();
}

/// Read-only view of the Brain shared with the session-bound tools
/// (Python: Brain's property surface used by `src.tools.vision`/`memory`).
/// All fields are shared state (Arcs/clones) so it can be cloned into the
/// tool context without lifetime ties.
#[derive(Clone)]
pub struct AgentView {
    pub client: async_openai::Client<OpenAIConfig>,
    pub http: reqwest::Client,
    pub base_url: String,
    pub model_name: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub context_window: u64,
    pub context_window_source: &'static str,
    pub vision: bool,
    pub vision_source: &'static str,
    pub messages: Arc<Mutex<Vec<ChatCompletionRequestMessage>>>,
    pub model_meta: Arc<RwLock<Option<Value>>>,
}

impl AgentView {
    /// Approximate context usage (no server-side tokenizer available):
    /// `~4 chars per token` estimate over the live history.
    pub fn context_usage(&self) -> (f64, u64) {
        let messages = self.messages.lock().unwrap();
        let chars: usize = messages
            .iter()
            .map(|m| OpenAICore::message_content(m).len())
            .sum();
        let tokens = chars / 4;
        (
            tokens as f64 / self.context_window as f64 * 100.0,
            self.context_window,
        )
    }

    /// The cached server metadata (empty object when unknown).
    pub fn metadata(&self) -> Value {
        self.model_meta
            .read()
            .unwrap()
            .clone()
            .unwrap_or(Value::Null)
    }
}

/// Context-window / vision resolution results.
pub struct ResolvedModel {
    pub context_window: u64,
    pub context_window_source: &'static str,
    pub vision: bool,
    pub vision_source: &'static str,
}

/// The OpenAI-compatible core client.
pub struct OpenAICore {
    pub config: crate::config::BrainCfg,
    pub client: async_openai::Client<OpenAIConfig>,
    pub http: reqwest::Client,
    pub base_url: String,
    pub model_name: String,
    pub api_key: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub context_window: u64,
    pub context_window_source: &'static str,
    pub vision: bool,
    pub vision_source: &'static str,
    pub messages: Arc<Mutex<Vec<ChatCompletionRequestMessage>>>,
    pub model_meta: Arc<RwLock<Option<Value>>>,
}

impl OpenAICore {
    /// Connect to the server configured in `[brain]`.
    pub fn new(config: &crate::config::BrainCfg) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("Missing [brain] base_url"))?;
        let api_key_env = config.api_key_env.clone().unwrap_or_default();
        load_env_file(None); // fill unset variables from .env (real env vars win)
        let api_key = std::env::var(&api_key_env).map_err(|_| {
            anyhow!(
                "API key not found: set the '{api_key_env}' environment variable, \
                 e.g. put '{api_key_env}=sk-...' in the project .env file"
            )
        })?;
        let model = config.model.clone();
        log::info!(
            "🚀 Connecting to remote model '{}' @ {} ...",
            model,
            base_url
        );

        let openai_config = OpenAIConfig::new()
            .with_api_key(api_key.clone())
            .with_api_base(base_url.clone());
        let client = async_openai::Client::with_config(openai_config);
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        let mut core = Self {
            config: config.clone(),
            client,
            http,
            base_url,
            model_name: model,
            api_key,
            temperature: config.temperature as f32,
            max_tokens: 0, // resolved below
            context_window: DEFAULT_CONTEXT_WINDOW,
            context_window_source: "default",
            vision: config.vision.unwrap_or(false),
            vision_source: "config",
            messages: Arc::new(Mutex::new(Vec::new())),
            model_meta: Arc::new(RwLock::new(None)),
        };
        core.messages
            .lock()
            .unwrap()
            .push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(
                        config.system_prompt.clone(),
                    ),
                    name: None,
                },
            ));

        let resolved = core.resolve_model_settings();
        core.context_window = resolved.context_window;
        core.context_window_source = resolved.context_window_source;
        core.vision = resolved.vision;
        core.vision_source = resolved.vision_source;
        core.max_tokens = response_token_cap(config.max_response_tokens, core.context_window)?;
        log::info!("✅ Remote model '{}' connected.", core.model_name);
        Ok(core)
    }

    // ------------------------------------------------------- model metadata

    /// Flatten an SDK model record / raw JSON into a plain scalar map
    /// (Python `_model_record_fields` + filtering).
    fn flatten_record(record: &Value) -> Value {
        let mut meta = serde_json::Map::new();
        if let Value::Object(map) = record {
            for (key, value) in map {
                if MODEL_RECORD_SKIP.contains(&key.as_str()) || value.is_null() {
                    continue;
                }
                if let Value::String(s) = value {
                    if s.is_empty() {
                        continue;
                    }
                    meta.insert(key.clone(), Value::String(s.clone()));
                } else if let Value::Array(list) = value {
                    // keep lists of scalars (capabilities); drop object lists
                    if !list.is_empty()
                        && list.iter().all(|v| {
                            v.is_string()
                                || v.is_i64()
                                || v.is_u64()
                                || v.is_f64()
                                || v.is_boolean()
                        })
                    {
                        meta.insert(key.clone(), Value::Array(list.clone()));
                    }
                } else if value.is_string()
                    || value.is_i64()
                    || value.is_u64()
                    || value.is_f64()
                    || value.is_boolean()
                {
                    meta.insert(key.clone(), value.clone());
                }
                // nested structures skipped
            }
        }
        Value::Object(meta)
    }

    /// Async model metadata fetch (Python `_fetch_model_metadata`).
    ///
    /// `GET /models/{id}` first, then `GET /models` + lookup by id; some
    /// servers (e.g. Together.ai) answer with a bare JSON array — accepted by
    /// the raw fallback, which also preserves server-specific extra fields the
    /// typed SDK drops (LM Studio: max_context_length, capabilities, ...).
    /// Failures degrade silently; success is cached.
    pub async fn fetch_model_metadata_async(&self) -> Value {
        let cached = self.model_meta.read().unwrap().clone();
        if let Some(meta) = cached {
            return meta;
        }
        let mut merged = serde_json::Map::new();

        // 1) typed SDK retrieve
        if let Ok(record) = self.client.models().retrieve(&self.model_name).await {
            merge_record(
                &mut merged,
                &Self::flatten_record(&serde_json::to_value(record).unwrap_or_default()),
            );
        }
        // 2) typed list + lookup by id (some servers only implement the list variant)
        if merged.is_empty()
            && let Ok(list) = self.client.models().list().await
        {
            for model in list.data {
                if model.id == self.model_name {
                    merge_record(
                        &mut merged,
                        &Self::flatten_record(&serde_json::to_value(model).unwrap_or_default()),
                    );
                    break;
                }
            }
        }
        // 3) raw fallback (bare-array servers, extra fields preserved)
        if let Ok(record) = self.find_in_raw_models_list().await {
            merge_record(&mut merged, &record);
        }

        let meta = Value::Object(merged);
        if meta.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
            *self.model_meta.write().unwrap() = Some(meta.clone());
        }
        meta
    }

    /// Raw `GET /models` for servers the typed SDK cannot parse (Python
    /// `_find_in_raw_models_list`). Returns the matching record or Err.
    async fn find_in_raw_models_list(&self) -> Result<Value> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_key)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .context("raw GET /models failed")?
            .error_for_status()
            .context("raw GET /models failed")?;
        let mut data = resp.json::<Value>().await.context("raw GET /models body")?;
        if let Value::Object(map) = &mut data {
            data = map.get("data").cloned().unwrap_or(Value::Null);
        }
        if let Value::Array(items) = data {
            for item in items {
                if let Value::Object(map) = &item
                    && map.get("id").and_then(|v| v.as_str()) == Some(&self.model_name)
                {
                    return Ok(Self::flatten_record(&item));
                }
            }
        }
        Err(anyhow!("model not present in raw /models list"))
    }

    /// Resolve context window + vision from config/server (Python
    /// `_resolve_context_window` / `_resolve_vision_flag`).
    fn resolve_model_settings(&mut self) -> ResolvedModel {
        // Context window: explicit config > server report > default.
        let mut context_window = DEFAULT_CONTEXT_WINDOW;
        let mut context_window_source: &'static str = "default";
        if let Some(configured) = self.config.context_window.filter(|c| *c > 0) {
            context_window = configured;
            context_window_source = "config";
        } else {
            // The metadata fetch needs &mut self (this fn); read cache only.
            let meta = self
                .model_meta
                .read()
                .unwrap()
                .clone()
                .unwrap_or(Value::Null);
            let mut context_fields: Vec<(String, u64)> = meta
                .as_object()
                .map(|m| {
                    m.iter()
                        .filter(|(k, v)| k.to_lowercase().contains("context") && v.is_i64())
                        .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0) as u64))
                        .collect()
                })
                .unwrap_or_default();
            context_fields.sort();
            let mut adopted = false;
            for key in CONTEXT_MAX_KEYS {
                if let Some((_, v)) = context_fields.iter().find(|(k, _)| *k == key) {
                    log::info!(
                        "Model '{}' context window from server ({}): {}",
                        self.model_name,
                        key,
                        v
                    );
                    context_window = *v;
                    context_window_source = "server";
                    adopted = true;
                    break;
                }
            }
            if !adopted
                && let Some((key, v)) = context_fields
                    .iter()
                    .find(|(k, _)| !k.to_lowercase().starts_with("loaded"))
            {
                log::info!(
                    "Model '{}' context window from server ({}): {}",
                    self.model_name,
                    key,
                    v
                );
                context_window = *v;
                context_window_source = "server";
            }
        }

        // Vision: server-reported capabilities are authoritative when present.
        let mut vision = self.config.vision.unwrap_or(false);
        let mut vision_source: &'static str = "config";
        let meta = self
            .model_meta
            .read()
            .unwrap()
            .clone()
            .unwrap_or(Value::Null);
        let capabilities = meta
            .as_object()
            .and_then(|m| m.get("capabilities"))
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            });
        if let Some(caps) = &capabilities
            && !caps.is_empty()
        {
            vision_source = "server";
            vision = caps.iter().any(|c| c.to_lowercase() == "vision");
            log::info!(
                "Model '{}' capabilities (server): {} -> vision={}",
                self.model_name,
                caps.join(", "),
                vision
            );
        }

        ResolvedModel {
            context_window,
            context_window_source,
            vision,
            vision_source,
        }
    }

    /// Populate the metadata cache (call after connecting, before resolution).
    pub async fn resolve_metadata(&mut self) -> Value {
        let meta = self.fetch_model_metadata_async().await;
        let resolved = self.resolve_model_settings();
        self.context_window = resolved.context_window;
        self.context_window_source = resolved.context_window_source;
        self.vision = resolved.vision;
        self.vision_source = resolved.vision_source;
        self.max_tokens = response_token_cap(self.config.max_response_tokens, self.context_window)
            .unwrap_or(self.context_window as u32);
        meta
    }

    // ----------------------------------------------------------- generation

    /// Build the streaming request body from the current history.
    pub fn stream_request(
        &self,
        tools: Option<Vec<async_openai::types::chat::ChatCompletionTools>>,
    ) -> Result<async_openai::types::chat::CreateChatCompletionRequest> {
        let messages = self.messages.lock().unwrap().clone();
        let mut req = async_openai::types::chat::CreateChatCompletionRequestArgs::default();
        req.model(self.model_name.clone())
            .messages(messages)
            .temperature(self.temperature)
            // Legacy name, still accepted by all OpenAI-compatible endpoints.
            .max_tokens(self.max_tokens)
            .stream(true);
        if let Some(tools) = tools {
            req.tools(tools);
        }
        req.build().map_err(|e| anyhow!("request build: {e}"))
    }

    /// Create the SSE completion stream.
    pub async fn stream_completion(
        &self,
        tools: Option<Vec<async_openai::types::chat::ChatCompletionTools>>,
    ) -> Result<async_openai::types::chat::ChatCompletionResponseStream> {
        let req = self.stream_request(tools)?;
        Ok(self.client.chat().create_stream(req).await?)
    }

    /// Create a non-streaming completion (used by the vision tool).
    pub async fn plain_completion(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> Result<async_openai::types::chat::CreateChatCompletionResponse> {
        let mut req = async_openai::types::chat::CreateChatCompletionRequestArgs::default();
        req.model(self.model_name.clone())
            .messages(messages)
            .temperature(self.temperature);
        // (stream deliberately unset)
        Ok(self.client.chat().create(req.build()?).await?)
    }

    // -------------------------------------------------------------- history

    /// The live conversation history (tools may append/pop entries).
    pub fn messages(&self) -> Arc<Mutex<Vec<ChatCompletionRequestMessage>>> {
        self.messages.clone()
    }

    /// Role of a history message.
    pub fn role_of(message: &ChatCompletionRequestMessage) -> &'static str {
        match message {
            ChatCompletionRequestMessage::System(_) => "system",
            ChatCompletionRequestMessage::User(_) => "user",
            ChatCompletionRequestMessage::Assistant(_) => "assistant",
            ChatCompletionRequestMessage::Tool(_) => "tool",
            ChatCompletionRequestMessage::Developer(_) => "developer",
            ChatCompletionRequestMessage::Function(_) => "function",
        }
    }

    pub fn message_content(message: &ChatCompletionRequestMessage) -> String {
        match message {
            ChatCompletionRequestMessage::System(m) => match &m.content {
                ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
                ChatCompletionRequestSystemMessageContent::Array(_) => String::new(),
            },
            ChatCompletionRequestMessage::User(m) => match &m.content {
                ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
                ChatCompletionRequestUserMessageContent::Array(_) => String::new(),
            },
            ChatCompletionRequestMessage::Assistant(m) => match &m.content {
                Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => t.clone(),
                Some(ChatCompletionRequestAssistantMessageContent::Array(parts)) => parts
                    .iter()
                    .map(|part| match part {
                        ChatCompletionRequestAssistantMessageContentPart::Text(t) => t.text.clone(),
                        ChatCompletionRequestAssistantMessageContentPart::Refusal(r) => {
                            r.refusal.clone()
                        }
                    })
                    .collect(),
                _ => String::new(),
            },
            ChatCompletionRequestMessage::Tool(m) => match &m.content {
                ChatCompletionRequestToolMessageContent::Text(t) => t.clone(),
                _ => String::new(),
            },
            _ => String::new(),
        }
    }

    /// Keep the chat history template-valid across cancelled/failed turns
    /// (Python `_repair_history`): pad the empty assistant turn with a minimal
    /// placeholder and drop a trailing unanswered user prompt.
    pub fn repair_history(messages: &mut Vec<ChatCompletionRequestMessage>) {
        while let Some(last) = messages.last() {
            match last {
                ChatCompletionRequestMessage::Assistant(m) => {
                    let content_empty = match &m.content {
                        Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => {
                            t.trim().is_empty()
                        }
                        Some(ChatCompletionRequestAssistantMessageContent::Array(parts)) => {
                            parts.is_empty()
                        }
                        None => true,
                    };
                    if content_empty {
                        messages.pop();
                        messages.push(ChatCompletionRequestMessage::Assistant(
                            ChatCompletionRequestAssistantMessage {
                                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                                    CANCELLED_PLACEHOLDER.into(),
                                )),
                                ..Default::default()
                            },
                        ));
                    }
                    break;
                }
                ChatCompletionRequestMessage::User(_) => {
                    messages.pop(); // unanswered prompt from a failed/aborted turn
                    continue;
                }
                _ => break,
            }
        }
    }

    /// Append a user prompt (after repairing the history).
    pub fn add_user_prompt(&self, prompt: &str) {
        {
            let mut messages = self.messages.lock().unwrap();
            Self::repair_history(&mut messages);
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                    name: None,
                },
            ));
        }
        log::debug!("Prompt:\n{prompt}");
    }

    /// Append plain assistant text.
    pub fn append_assistant_text(&self, text: &str) {
        self.messages
            .lock()
            .unwrap()
            .push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessage {
                    content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                        text.to_string(),
                    )),
                    ..Default::default()
                },
            ));
    }

    /// Append plain user text (no repair; used by the vision tool).
    pub fn append_user_text(&self, text: &str) {
        self.messages
            .lock()
            .unwrap()
            .push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(text.to_string()),
                    name: None,
                },
            ));
    }

    /// Read-only view shared with the tool context.
    pub fn view(&self) -> AgentView {
        AgentView {
            client: self.client.clone(),
            http: self.http.clone(),
            base_url: self.base_url.clone(),
            model_name: self.model_name.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            context_window: self.context_window,
            context_window_source: self.context_window_source,
            vision: self.vision,
            vision_source: self.vision_source,
            messages: self.messages.clone(),
            model_meta: self.model_meta.clone(),
        }
    }

    /// Approximate context usage (no server-side tokenizer available).
    pub fn context_usage(&self) -> (f64, u64) {
        let messages = self.messages.lock().unwrap();
        let chars: usize = messages
            .iter()
            .map(|m| Self::message_content(m).len())
            .sum();
        let tokens = chars / 4; // rough ~4 chars per token estimate
        (
            tokens as f64 / self.context_window as f64 * 100.0,
            self.context_window,
        )
    }

    /// Format the `created` unix timestamp as YYYY-MM-DD.
    pub fn format_created(created: &Value) -> Option<String> {
        let secs = match created {
            Value::Number(n) => n.as_f64()?,
            _ => return None,
        };
        DateTime::from_timestamp(secs as i64, 0).map(|dt| dt.format("%Y-%m-%d").to_string())
    }
}

/// Resolve `[brain] max_response_tokens` into a per-generation cap
/// (Python `_response_token_cap`).
pub fn response_token_cap(max_response_tokens: f64, context_window: u64) -> Result<u32> {
    if max_response_tokens == 0.0 {
        return Ok(context_window as u32);
    }
    if !(0.0..1.0).contains(&max_response_tokens) {
        return Err(anyhow!(
            "brain.max_response_tokens must be a ratio with 0 < ratio < 1, got {max_response_tokens}"
        ));
    }
    Ok((((context_window as f64) * max_response_tokens) as u64).max(1) as u32)
}

fn merge_record(merged: &mut serde_json::Map<String, Value>, record: &Value) {
    if let Value::Object(map) = record {
        for (k, v) in map {
            merged.insert(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python `load_dotenv(override=False)` parity: file entries fill unset
    /// variables, but an existing environment variable always wins.
    #[test]
    fn env_file_fills_unset_and_never_overrides_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".env");
        std::fs::write(
            &path,
            "LA_TEST_UNSET_KEY=from_file\nLA_TEST_SET_KEY=from_file\n",
        )
        .unwrap();
        // SAFETY: single-threaded manipulation of uniquely named test vars.
        unsafe {
            std::env::set_var("LA_TEST_SET_KEY", "from_env");
            std::env::remove_var("LA_TEST_UNSET_KEY");
        }
        load_env_file(Some(path.clone()));
        assert_eq!(
            std::env::var("LA_TEST_SET_KEY").unwrap(),
            "from_env",
            "existing env var must win"
        );
        assert_eq!(
            std::env::var("LA_TEST_UNSET_KEY").unwrap(),
            "from_file",
            "file entry must fill unset var"
        );
        unsafe {
            std::env::remove_var("LA_TEST_SET_KEY");
            std::env::remove_var("LA_TEST_UNSET_KEY");
        }
    }
}
