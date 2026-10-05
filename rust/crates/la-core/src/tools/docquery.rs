//! In-document search tool (Python `src/tools/docquery.py` port): CrossEncoder
//! scoring of query/passage pairs; the heavy `DocumentQuery` is built lazily
//! on first use and cached at context level.

use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use std::path::PathBuf;
use std::sync::Arc;

fn shared_document_query(
    ctx: &ToolContext,
) -> anyhow::Result<Arc<crate::document_query::DocumentQuery>> {
    if let Some(doc_query) = ctx.caches.document_query.lock().unwrap().clone() {
        return Ok(doc_query);
    }
    let doc_query = Arc::new(crate::document_query::DocumentQuery::new(
        ctx.config.document_query.model_path.clone(),
        ctx.config.document_query.device.clone(),
        ctx.config.document_query.chunk_size,
        ctx.config.document_query.top_k,
        ctx.encoders.cross.clone(),
        None,
    ));
    *ctx.caches.document_query.lock().unwrap() = Some(doc_query.clone());
    Ok(doc_query)
}

/// Search one document for the passages most relevant to a query.
/// Useful when the user asks to find, locate or look up something inside a
/// specific file without building a corpus. Any type supported by the
/// document loader works: text, PDF (results keep page numbers), DOCX, XLSX,
/// HTML, ... A CrossEncoder scores query/passage pairs. `top_k` = 0 uses the
/// value configured in the `[document_query]` section.
pub async fn query_document(
    ctx: Arc<ToolContext>,
    file: &str,
    query: &str,
    top_k: usize,
) -> String {
    log_call!("query_document", "file" => file, "query" => query, "top_k" => top_k);
    if query.trim().is_empty() {
        return "Error: a query is required.".into();
    }
    let doc_query = match shared_document_query(&ctx) {
        Ok(d) => d,
        Err(e) => return format!("Error: {e}"),
    };
    let path = PathBuf::from(file);
    let query_owned = query.to_string();
    tokio::task::spawn_blocking(move || {
        doc_query.query(&path, &query_owned, (top_k > 0).then_some(top_k))
    })
    .await
    .unwrap_or_else(|e| format!("Error: join error: {e}"))
}

pub fn tool_def() -> ToolDef {
    ToolDef {
        name: "query_document".into(),
        description: "Search one document for the passages most relevant to a query.\nUseful when the user asks to find, locate or look up something inside a\nspecific file without building a corpus. Any type supported by the\ndocument loader works: text, PDF (results keep page numbers), DOCX, XLSX,\nHTML, ... A CrossEncoder scores query/passage pairs.\n`top_k` = 0 uses the value configured in the `[document_query]`\nsection.".into(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 0}
            },
            "required": ["file", "query"]
        }),
        handler: Arc::new(|ctx, args| Box::pin(async move {
            let file = args.get("file").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let top_k = args.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            Ok(ToolOutput::text(query_document(ctx, &file, &query, top_k).await))
        })),
    }
}
