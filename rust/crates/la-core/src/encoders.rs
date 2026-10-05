//! Shared dependency-injection seams for the ML-backed tools
//! (`TextEmbedder`, `ImageEmbedder`, `CrossEncoder`).
//!
//! These traits are the Rust equivalent of the Python test seams
//! (`model_factory`, `cross_encoder_factory`, ...): unit tests inject fakes,
//! the `la-ml` crate (candle, behind features) provides the real encoders and
//! the `la` binary wires them in. Without them the tools fail gracefully with
//! the same kind of message as the Python code when a model is missing.

use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Embed a batch of texts; returns one row per text (unnormalized).
pub trait TextEmbedder: Send + Sync {
    fn encode(&self, texts: &[String]) -> anyhow::Result<Array2<f64>>;
}

/// Embed a batch of images given by path; returns one row per image.
pub trait ImageEmbedder: Send + Sync {
    fn encode_paths(&self, paths: &[PathBuf]) -> anyhow::Result<Array2<f64>>;
}

/// Score `[query, passage]` pairs; one score per pair, higher = better.
pub trait CrossEncoder: Send + Sync {
    fn predict(&self, pairs: &[(String, String)]) -> anyhow::Result<Vec<f64>>;
}

/// A text embedder that also implements the image side (CLIP multimodal).
pub trait MultimodalEmbedder: TextEmbedder + ImageEmbedder {}
impl<T: TextEmbedder + ImageEmbedder> MultimodalEmbedder for T {}

/// The encoder bundle wired by the main binary; `None` entries mean the tool
/// fails gracefully (Python: missing model folder -> friendly error).
#[derive(Clone, Default)]
pub struct Encoders {
    /// `[summarization]`/`[embedding] text` encoder (all-MiniLM-L6-v2).
    pub text: Option<Arc<dyn TextEmbedder>>,
    /// CLIP text encoder (multimodal corpora mode).
    pub clip_text: Option<Arc<dyn TextEmbedder>>,
    /// CLIP image encoder (multimodal corpora mode + image_search).
    pub image: Option<Arc<dyn ImageEmbedder>>,
    /// `[document_query]` cross encoder (ms-marco-TinyBERT-L2).
    pub cross: Option<Arc<dyn CrossEncoder>>,
}

/// True when `model_path` is a directory containing a saved model
/// (Python `is_model_available`).
pub fn is_model_available(model_path: &Path) -> bool {
    const MODEL_MARKERS: [&str; 4] = [
        "modules.json",
        "config_sentence_transformers.json",
        "pytorch_model.bin",
        "model.safetensors",
    ];
    if !model_path.is_dir() {
        return false;
    }
    MODEL_MARKERS
        .iter()
        .any(|marker| model_path.join(marker).exists())
}
