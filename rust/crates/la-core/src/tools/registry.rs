//! Tool registry for the Live Assistant agent (Python `src.tools` port).
//!
//! Every tool = `{ name, description, JSON-schema parameters, handler }`.
//! The registry order matches `_default_tools` in `src/live_ai.py` (asserted
//! by a snapshot test). Handlers are async and receive the shared
//! [`ToolContext`]; errors are returned as strings (never raised), mirroring
//! the Python tools' behavior of returning `f"Error: {e}"`.

use crate::brain::AgentView;
use crate::config::Config;
use crate::encoders::Encoders;
use crate::workers::Workers;
use std::sync::{Arc, Mutex, Weak};

pub type Handler = Arc<
    dyn Fn(
            Arc<ToolContext>,
            serde_json::Value,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<ToolOutput>>
        + Send
        + Sync,
>;

/// Result of a tool handler (Python tools return strings; the graceful-exit
/// tool additionally signals the session loop to quit).
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub text: String,
    pub exit: bool,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            exit: false,
        }
    }
    pub fn exit() -> Self {
        Self {
            text: String::new(),
            exit: true,
        }
    }
}

/// One registered tool.
#[derive(Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the parameters object (properties + required).
    pub schema: serde_json::Value,
    pub handler: Handler,
}

impl std::fmt::Debug for ToolDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl ToolDef {
    /// The first description line (used by the `/tools` launcher list).
    pub fn summary(&self) -> &str {
        self.description.lines().next().unwrap_or("").trim()
    }

    /// Names of required parameters: schema properties without a "default".
    pub fn required_params(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let Some(props) = self.schema.get("properties").and_then(|v| v.as_object()) else {
            return out;
        };
        // Walk the `required` array in its declared order (the JSON schema /
        // Python signature order). Iterating `properties` instead would yield
        // alphabetically sorted keys (serde_json maps are BTreeMaps), which
        // would reorder the `/tools` launcher prompts (Python parity: the
        // prompts follow the signature order, e.g. `file` before `content`).
        let required: Vec<String> = self
            .schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        for name in &required {
            if let Some(spec) = props.get(name) {
                let type_hint = spec
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("string")
                    .to_string();
                out.push((name.clone(), type_hint));
            }
        }
        out
    }
}

/// Lazily-built shared engine instances (Python module-level caches).
#[derive(Default)]
pub struct ToolCaches {
    pub summarizer: Mutex<Option<Arc<crate::summarizer::Summarizer>>>,
    pub document_query: Mutex<Option<Arc<crate::document_query::DocumentQuery>>>,
    pub image_search: Mutex<Option<Arc<crate::image_search::ImageSearch>>>,
    pub embedding_corpus: Mutex<Option<Arc<crate::embedding_search::EmbeddingCorpus>>>,
}

/// Shared state of the running session, passed to every tool handler.
pub struct ToolContext {
    pub config: Config,
    /// Live conversation history (plain OpenAI-style messages).
    pub view: AgentView,
    /// Worker-process handles (listen/speak/imggen), spawned on demand.
    pub workers: Arc<Workers>,
    /// ML encoders wired by the binary (None -> graceful tool errors).
    pub encoders: Encoders,
    /// Lazily-built shared engine instances.
    pub caches: ToolCaches,
}

impl ToolContext {
    pub fn new(config: &Config, view: AgentView) -> Arc<Self> {
        Arc::new(Self {
            config: config.clone(),
            view,
            workers: Arc::new(Workers::new()),
            encoders: Encoders::default(),
            caches: ToolCaches::default(),
        })
    }

    /// Build a context with custom workers/encoders (used by the main binary).
    pub fn with_parts(
        config: &Config,
        view: AgentView,
        workers: Arc<Workers>,
        encoders: Encoders,
    ) -> Arc<Self> {
        Arc::new(Self {
            config: config.clone(),
            view,
            workers,
            encoders,
            caches: ToolCaches::default(),
        })
    }
}

/// The ordered tool set (Python `_default_tools`).
pub struct ToolRegistry {
    pub tools: Vec<ToolDef>,
    context: Mutex<Weak<ToolContext>>,
}

impl ToolRegistry {
    pub fn empty() -> Self {
        Self {
            tools: Vec::new(),
            context: Mutex::new(Weak::new()),
        }
    }

    /// Build the full registry; tools capture the context weakly.
    pub fn build(context: Arc<ToolContext>) -> Self {
        let mut registry = Self::empty();
        registry.tools = crate::tools::defaults::default_tools(&context);
        *registry.context.lock().unwrap() = Arc::downgrade(&context);
        registry
    }

    /// The bound context (panics when the registry outlived its session —
    /// cannot happen in the app; guards tests).
    pub fn context(&self) -> Arc<ToolContext> {
        self.context
            .lock()
            .unwrap()
            .upgrade()
            .expect("tool registry used after session dropped")
    }

    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| t.name == name)
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }
}

/// Convenience: spawn a blocking job from an async handler.
pub fn spawn_blocking<F, T>(f: F) -> futures::future::BoxFuture<'static, anyhow::Result<T>>
where
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    Box::pin(async move {
        tokio::task::spawn_blocking(f)
            .await
            .map_err(|e| anyhow::anyhow!("join error: {e}"))?
    })
}

/// Helper used by handlers: run a future that returns a String, mapping
/// errors to the Python-style "Error: ..." convention is done by callers.
pub fn boxed<F>(future: F) -> futures::future::BoxFuture<'static, anyhow::Result<ToolOutput>>
where
    F: Future<Output = anyhow::Result<ToolOutput>> + Send + 'static,
{
    Box::pin(future)
}
