//! Document-summarization tool (Python `src/tools/summarize.py` port).
//! The heavy `Summarizer` is built lazily on first use and cached at context
//! level (loading the embedding model is slow), mirroring the Python
//! module-level cache.

use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use std::sync::Arc;

/// Build (once) and return the shared `Summarizer` from the config + encoders.
fn shared_summarizer(ctx: &ToolContext) -> anyhow::Result<Arc<crate::summarizer::Summarizer>> {
    if let Some(summarizer) = ctx.caches.summarizer.lock().unwrap().clone() {
        return Ok(summarizer);
    }
    let summarizer = Arc::new(crate::summarizer::Summarizer::new(
        crate::summarizer::SummarizerConfig {
            model_path: ctx.config.summarization.model_path.clone(),
            device: ctx.config.summarization.device.clone(),
            ratio: ctx.config.summarization.ratio,
            threshold: ctx.config.summarization.threshold,
            max_sentences: ctx.config.summarization.max_sentences,
        },
        ctx.encoders.text.clone(),
    ));
    *ctx.caches.summarizer.lock().unwrap() = Some(summarizer.clone());
    Ok(summarizer)
}

/// Summarize the content of a document into its most important sentences.
/// Useful when the user asks to summarize, condense, shorten or outline a
/// document. `file` is the path of the document to read; text, PDF, DOCX,
/// XLSX, HTML and image files are supported. `max_sentences` = 0 uses the
/// ratio configured in the `[summarization]` section.
pub async fn summarize_doc(ctx: Arc<ToolContext>, file: &str, max_sentences: usize) -> String {
    log_call!("summarize_doc", "file" => file, "max_sentences" => max_sentences);
    let summarizer = match shared_summarizer(&ctx) {
        Ok(s) => s,
        Err(e) => return format!("Error: {e}"),
    };
    let path = std::path::PathBuf::from(file);
    let result = tokio::task::spawn_blocking(move || {
        summarizer.summarize_file(&path, (max_sentences > 0).then_some(max_sentences))
    })
    .await
    .unwrap_or_else(|e| Err(anyhow::anyhow!("join error: {e}")));
    match result {
        Ok(text) => text,
        Err(e) => format!("Error: {e}"),
    }
}

pub fn tool_def() -> ToolDef {
    ToolDef {
        name: "summarize_doc".into(),
        description: "Summarize the content of a document into its most important sentences.\nUseful when the user asks to summarize, condense, shorten or outline a\ndocument. `file` is the path of the document to read; text, PDF, DOCX,\nXLSX, HTML and image files are supported. `max_sentences` = 0 uses the\nratio configured in the `[summarization]` section.".into(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "max_sentences": {"type": "integer", "default": 0}
            },
            "required": ["file"]
        }),
        handler: Arc::new(|ctx, args| Box::pin(async move {
            let file = args.get("file").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let max_sentences = args.get("max_sentences").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            Ok(ToolOutput::text(summarize_doc(ctx, &file, max_sentences).await))
        })),
    }
}
