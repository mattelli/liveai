//! Corpus tools (Python `src/tools/corpora.py` port): build and search named
//! corpora (text + images); the heavy `EmbeddingCorpus` is built lazily on
//! first use and cached at context level.

use crate::embedding_search::{CorpusModels, resolve_corpus_dir};
use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use std::path::PathBuf;
use std::sync::Arc;

fn shared_embedding_corpus(
    ctx: &ToolContext,
) -> anyhow::Result<Arc<crate::embedding_search::EmbeddingCorpus>> {
    if let Some(corpus) = ctx.caches.embedding_corpus.lock().unwrap().clone() {
        return Ok(corpus);
    }
    let embedding = &ctx.config.embedding;
    let corpus = Arc::new(crate::embedding_search::EmbeddingCorpus::new(
        embedding.model_path.clone(),
        embedding.device.clone(),
        crate::document_loader::normalize_extensions(&embedding.extensions),
        embedding.chunk_size,
        embedding.top_k,
        embedding.image_model_path.clone(),
        embedding.multimodal,
        crate::document_loader::normalize_extensions(&embedding.image_extensions),
        CorpusModels {
            text: ctx.encoders.text.clone(),
            clip_text: ctx.encoders.clip_text.clone(),
            image: ctx.encoders.image.clone(),
        },
    ));
    *ctx.caches.embedding_corpus.lock().unwrap() = Some(corpus.clone());
    Ok(corpus)
}

/// List the available local corpora (name, mode, record counts, sources, built time).
/// Useful before query_corpus when the user refers to a topic instead of a corpus name.
pub fn list_corpora(ctx: &ToolContext) -> String {
    log_call!("list_corpora");
    let entries = crate::embedding_search::list_corpora(&ctx.config.embedding.corpora_dir);
    if entries.is_empty() {
        return "No corpora available.".into();
    }
    let mut lines = vec![format!("Available corpora ({}):", entries.len())];
    for entry in entries {
        lines.push(format!(
            "- {name}: mode={mode}, records={records} (text {text} / image {image}), sources={sources}, built={built}",
            name = entry["name"].as_str().unwrap_or(""),
            mode = entry["mode"].as_str().unwrap_or(""),
            records = entry["records"].as_i64().unwrap_or(0),
            text = entry["text_records"].as_i64().unwrap_or(0),
            image = entry["image_records"].as_i64().unwrap_or(0),
            sources = entry["sources"].as_i64().unwrap_or(0),
            built = entry["built_at"].as_str().unwrap_or(""),
        ));
    }
    lines.join("\n")
}

/// Build a named corpus from all supported files in a directory.
/// Useful when the user asks to index, embed, or store documents for later
/// semantic search.
pub async fn store_corpus(
    ctx: Arc<ToolContext>,
    directory: &str,
    corpus_name: &str,
    overwrite: bool,
) -> String {
    log_call!("store_corpus", "directory" => directory, "corpus_name" => corpus_name, "overwrite" => overwrite);
    if corpus_name.trim().is_empty() {
        return "Error: a corpus name is required (call list_corpora to see available corpora)."
            .into();
    }
    let corpus_dir = match resolve_corpus_dir(corpus_name, &ctx.config.embedding.corpora_dir) {
        Ok(dir) => dir,
        Err(e) => return format!("Error: {e}"),
    };
    let corpus = match shared_embedding_corpus(&ctx) {
        Ok(c) => c,
        Err(e) => return format!("Error: {e}"),
    };
    let source = PathBuf::from(directory);
    let result = tokio::task::spawn_blocking(move || corpus.store(&source, &corpus_dir, overwrite))
        .await
        .unwrap_or_else(|e| Err(anyhow::anyhow!("join error: {e}")));
    match result {
        Ok(text) => text,
        Err(e) => format!("Error: {e}"),
    }
}

/// Search a named corpus for the content most relevant to a query text.
/// Useful when the user asks a question that should be answered from indexed
/// documents or when looking for a matching image. `kind` restricts results
/// to 'text' or 'image'; leave it empty to search both.
pub async fn query_corpus(
    ctx: Arc<ToolContext>,
    query: &str,
    corpus_name: &str,
    kind: &str,
    top_k: usize,
) -> String {
    log_call!("query_corpus", "query" => query, "corpus_name" => corpus_name, "kind" => kind, "top_k" => top_k);
    if corpus_name.trim().is_empty() {
        return "Error: a corpus name is required.".into();
    }
    let corpus_dir = match resolve_corpus_dir(corpus_name, &ctx.config.embedding.corpora_dir) {
        Ok(dir) => dir,
        Err(e) => return format!("Error: {e}"),
    };
    let corpus = match shared_embedding_corpus(&ctx) {
        Ok(c) => c,
        Err(e) => return format!("Error: {e}"),
    };
    let kind_owned = kind.trim().to_string();
    let query_owned = query.to_string();
    tokio::task::spawn_blocking(move || {
        corpus.query(
            &corpus_dir,
            &query_owned,
            (top_k > 0).then_some(top_k),
            (!kind_owned.is_empty()).then_some(kind_owned.as_str()),
        )
    })
    .await
    .unwrap_or_else(|e| format!("Error: join error: {e}"))
}

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "list_corpora".into(),
            description: "List the available local corpora (name, mode, record counts, sources, built time).\nUseful before query_corpus when the user refers to a topic instead of a corpus name.".into(),
            schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            handler: Arc::new(|ctx, _args| Box::pin(async move {
                Ok(ToolOutput::text(list_corpora(&ctx)))
            })),
        },
        ToolDef {
            name: "store_corpus".into(),
            description: "Build a named corpus from all supported files in a directory.\nUseful when the user asks to index, embed, or store documents for later semantic search.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "corpus_name": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": false}
                },
                "required": ["directory", "corpus_name"]
            }),
            handler: Arc::new(|ctx, args| Box::pin(async move {
                let directory = args.get("directory").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let corpus_name = args.get("corpus_name").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let overwrite = args.get("overwrite").and_then(|v| v.as_bool()).unwrap_or(false);
                Ok(ToolOutput::text(store_corpus(ctx, &directory, &corpus_name, overwrite).await))
            })),
        },
        ToolDef {
            name: "query_corpus".into(),
            description: "Search a named corpus for the content most relevant to a query text.\nUseful when the user asks a question that should be answered from indexed documents\nor when looking for a matching image.\n`kind` restricts results to 'text' or 'image'; leave it empty to search both.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "corpus_name": {"type": "string"},
                    "kind": {"type": "string", "default": ""},
                    "top_k": {"type": "integer", "default": 0}
                },
                "required": ["query", "corpus_name"]
            }),
            handler: Arc::new(|ctx, args| Box::pin(async move {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let corpus_name = args.get("corpus_name").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let kind = args.get("kind").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let top_k = args.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
                Ok(ToolOutput::text(query_corpus(ctx, &query, &corpus_name, &kind, top_k).await))
            })),
        },
    ]
}
