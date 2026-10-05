//! `la-ml` — candle-based encoder implementations for the Live Assistant
//! (Phase-4 tools): sentence embeddings (all-MiniLM-L6-v2), CLIP image
//! embeddings (clip-ViT-B-32) and the TinyBERT cross-encoder
//! (ms-marco-TinyBERT-L2).
//!
//! Everything here is feature-gated (`embeddings`, `clip`, `cross-encoder`)
//! so the default workspace build stays toolchain-free; the `la` binary
//! enables the features with `--features ml`. Models are read from local
//! folders configured in `config.toml` (`.res/...`); weights must be
//! safetensors (convert pytorch_model.bin once — see rust/README.md).
//!
//! Pooling notes (parity with sentence-transformers):
//! - MiniLM: mean pooling over tokens + the sentence-transformers activation
//!   (identity) — `normalize_rows` afterwards is done by `la-core`.
//! - CLIP: text side = CLIP text tower + projection; image side = CLIP vision
//!   tower + projection (one shared space).
//! - Cross-encoder: `logits[0]` with sigmoid (num_labels=1) or softmax class-1
//!   probability (num_labels=2), matching sentence-transformers defaults.

#[cfg(feature = "clip")]
pub mod clip;
#[cfg(feature = "cross-encoder")]
pub mod cross_encoder;
#[cfg(feature = "embeddings")]
pub mod minilm;

/// Resolve a config device string ("cpu"|"cuda"|"auto") into a candle Device.
#[cfg(feature = "embeddings")]
pub fn resolve_device(device: &str) -> candle_core::Device {
    match device.to_lowercase().as_str() {
        "cuda" => candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu),
        _ => candle_core::Device::Cpu, // "auto" on the CPU reference machine
    }
}
