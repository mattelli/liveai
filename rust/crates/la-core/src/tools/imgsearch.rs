//! Folder image-search tool (Python `src/tools/imgsearch.py` port): CLIP
//! embeddings rank images by a text query; the heavy `ImageSearch` is built
//! lazily on first use and cached at context level.

use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use std::path::PathBuf;
use std::sync::Arc;

fn shared_image_search(ctx: &ToolContext) -> anyhow::Result<Arc<crate::image_search::ImageSearch>> {
    if let Some(search) = ctx.caches.image_search.lock().unwrap().clone() {
        return Ok(search);
    }
    let search = Arc::new(crate::image_search::ImageSearch::new(
        ctx.config.image_search.model_path.clone(),
        ctx.config.image_search.device.clone(),
        ctx.config.image_search.top_k,
        ctx.encoders.image.clone(),
        ctx.encoders.clip_text.clone(),
        None,
    ));
    *ctx.caches.image_search.lock().unwrap() = Some(search.clone());
    Ok(search)
}

/// Search the images inside a folder for the ones best matching a text query,
/// using the CLIP model. Useful when the user asks to find, locate or show a
/// picture/photo/screenshot in a folder by describing its content. Returns
/// the top_k matches as '<rank>) [<cosine score>] <absolute image path>'
/// lines, best match first; `top_k` = 0 uses the value configured in the
/// `[image_search]` section (default 3).
pub async fn image_search(
    ctx: Arc<ToolContext>,
    folder: &str,
    query: &str,
    top_k: usize,
) -> String {
    log_call!("image_search", "folder" => folder, "query" => query, "top_k" => top_k);
    if query.trim().is_empty() {
        return "Error: a query is required.".into();
    }
    let search = match shared_image_search(&ctx) {
        Ok(s) => s,
        Err(e) => return format!("Error: {e}"),
    };
    let directory = PathBuf::from(folder);
    let query_owned = query.to_string();
    let result = tokio::task::spawn_blocking(move || {
        search.search(&directory, &query_owned, (top_k > 0).then_some(top_k))
    })
    .await
    .unwrap_or_else(|e| Err(anyhow::anyhow!("join error: {e}")));
    match result {
        Err(e) => format!("Error: {e}"),
        Ok(hits) => {
            if hits.is_empty() {
                return format!("No readable images found in '{folder}'.");
            }
            hits.iter()
                .enumerate()
                .map(|(rank, (score, path))| format!("{}) [{score:.3}] {path}", rank + 1))
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

pub fn tool_def() -> ToolDef {
    ToolDef {
        name: "image_search".into(),
        description: "Search the images inside a folder for the ones best matching a text query,\nusing the CLIP model. Useful when the user asks to find, locate or show a\npicture/photo/screenshot in a folder by describing its content. Returns\nthe top_k matches as '<rank>) [<cosine score>] <absolute image path>'\nlines, best match first; `top_k` = 0 uses the value configured in the\n`[image_search]` section (default 3).".into(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "folder": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 0}
            },
            "required": ["folder", "query"]
        }),
        handler: Arc::new(|ctx, args| Box::pin(async move {
            let folder = args.get("folder").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let top_k = args.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            Ok(ToolOutput::text(image_search(ctx, &folder, &query, top_k).await))
        })),
    }
}
