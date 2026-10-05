//! Registry assembly (Python `_default_tools` order) + worker-callable tools.

use super::registry::{ToolContext, ToolDef, ToolRegistry};
use std::sync::Arc;

/// Tool set of the agent, in the exact order of Python `_default_tools`
/// (`model_info` first, `exit_gracefully` last).
pub fn default_tools(ctx: &Arc<ToolContext>) -> Vec<ToolDef> {
    let _ = ctx;
    let mut tools: Vec<ToolDef> = Vec::new();
    // 1) model_info (session-bound Brain tool)
    tools.push(model_info_tool());
    // 2) memory
    tools.extend(super::memory::tool_defs());
    // 3) vision
    tools.push(super::vision::tool_def());
    // 4) speech toggle
    tools.push(super::speech::tool_def());
    // 5) summarize / docquery / imgsearch / corpora
    tools.push(super::summarize::tool_def());
    tools.push(super::docquery::tool_def());
    tools.push(super::imgsearch::tool_def());
    tools.extend(super::corpora::tool_defs());
    // 6) capture_image
    tools.push(super::camera::tool_def());
    // 7) system
    tools.extend(super::system::tool_defs());
    // 8) files
    tools.extend(super::files::tool_defs());
    // 9) web
    tools.extend(super::web::tool_defs());
    // 10) archives
    tools.extend(super::archives::tool_defs());
    // 11) finance
    tools.extend(super::finance::tool_defs());
    // 12) image generation
    tools.extend(super::image_gen::tool_defs());
    // 13) exit_gracefully — deliberately the very last tool (Python parity)
    tools.push(super::system::exit_gracefully_tool());
    tools
}

/// `model_info` — session-bound (Python `Brain.model_info`).
fn model_info_tool() -> ToolDef {
    ToolDef {
        name: "model_info".into(),
        description: "Returns detailed info about the loaded model: endpoint, provider and\neverything the server reports through the OpenAI models endpoint\n(ownership, context lengths, capabilities, quantization, ...), plus\nthe current session settings.\nUseful when user ask for information about the loaded model.".into(),
        schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
        handler: Arc::new(|ctx, _args| {
            Box::pin(async move {
                let view = ctx.view.clone();
                let meta = view.metadata();
                Ok(crate::tools::registry::ToolOutput::text(
                    crate::agent::model_info_string(&view, &meta),
                ))
            })
        }),
    }
}

/// Build a registry over a context and verify the Python parity order
/// (used by the snapshot test).
pub fn registry_for_tests() -> ToolRegistry {
    let config = crate::config::Config::default();
    let view = crate::brain::AgentView {
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
        messages: Arc::new(std::sync::Mutex::new(vec![])),
        model_meta: Arc::new(std::sync::RwLock::new(None)),
    };
    ToolRegistry::build(ToolContext::new(&config, view))
}
