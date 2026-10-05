//! Speech tool (Python `src/tools/speech.py` port): toggle the Piper
//! text-to-speech engine for agent responses, routed to the `la-speak`
//! worker process.

use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use std::sync::Arc;

/// Enable or disable speech for Agent responses.
/// When enabled, the Agent will use the Speech Generator.
/// It can be toggled on the fly during the conversation.
pub async fn allow_speech(ctx: Arc<ToolContext>, enable: bool) -> String {
    log_call!("allow_speech", "enable" => enable);
    if enable {
        match ctx.workers.speak_enable().await {
            Ok(_) => "Speech Generator initialized".into(),
            Err(e) => format!("Error: {e}"),
        }
    } else {
        ctx.workers.speak_disable().await;
        "Speech Generator released".into()
    }
}

/// Speak `text` through the session speech generator, when enabled.
/// Errors surface to the session loop like Python's exceptions.
pub async fn speak_if_enabled(ctx: Arc<ToolContext>, text: &str) -> anyhow::Result<()> {
    ctx.workers.speak_if_enabled(text).await
}

pub fn tool_def() -> ToolDef {
    ToolDef {
        name: "allow_speech".into(),
        description: "Enable or disable speech for Agent responses.\nWhen enabled, the Agent will use the Speech Generator.\nIt can be toggled on the fly during the conversation.".into(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {"enable": {"type": "boolean"}},
            "required": ["enable"]
        }),
        handler: Arc::new(|ctx, args| Box::pin(async move {
            let enable = args.get("enable").and_then(|v| v.as_bool()).unwrap_or(false);
            Ok(ToolOutput::text(allow_speech(ctx, enable).await))
        })),
    }
}
