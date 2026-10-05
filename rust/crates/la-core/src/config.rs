//! Configuration (`src/common.py::load_config` port).
//!
//! Reads the **same** `config.toml` used by the Python assistant and resolves
//! path-like values (`*_file`, `*_dir`, `*_path`, plus string list items)
//! relative to the config file's directory. Unknown keys are ignored
//! (additive `[rust]` section allowed; Python ignores it too).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

// --------------------------------------------------------------- sections

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LoggingCfg {
    #[serde(default)]
    pub level: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechDetCfg {
    #[serde(default = "d_speech_det_model_path")]
    pub model_path: PathBuf,
    #[serde(default = "d_silence_timeout_s")]
    pub silence_timeout_s: f64,
    #[serde(default = "d_language")]
    pub language: String,
    #[serde(default = "d_beam_size")]
    pub beam_size: u32,
    #[serde(default = "d_show_levels")]
    pub show_levels: bool,
    #[serde(default = "d_device")]
    pub device: String,
    #[serde(default = "d_compute_type")]
    pub compute_type: String,
    #[serde(default = "d_calibration_chunks")]
    pub calibration_chunks: usize,
    #[serde(default = "d_guard_factor")]
    pub guard_factor: f64,
}

impl Default for SpeechDetCfg {
    fn default() -> Self {
        Self {
            model_path: d_speech_det_model_path(),
            silence_timeout_s: d_silence_timeout_s(),
            language: d_language(),
            beam_size: d_beam_size(),
            show_levels: d_show_levels(),
            device: d_device(),
            compute_type: d_compute_type(),
            calibration_chunks: d_calibration_chunks(),
            guard_factor: d_guard_factor(),
        }
    }
}

fn d_speech_det_model_path() -> PathBuf {
    PathBuf::from(".res/faster-whisper-medium")
}
fn d_silence_timeout_s() -> f64 {
    2.0
}
fn d_language() -> String {
    "en".into()
}
fn d_beam_size() -> u32 {
    5
}
fn d_show_levels() -> bool {
    true
}
fn d_device() -> String {
    "auto".into()
}
fn d_compute_type() -> String {
    "int8".into()
}
fn d_calibration_chunks() -> usize {
    20
}
fn d_guard_factor() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct BrainCfg {
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub history_path: PathBuf,
    #[serde(default)]
    pub vision: Option<bool>,
    #[serde(default)]
    pub context_window: Option<u64>,
    #[serde(default = "d_max_response_tokens")]
    pub max_response_tokens: f64,
    #[serde(default = "d_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: String,
}

impl Default for BrainCfg {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            history_path: PathBuf::from("chat_history.log"),
            vision: None,
            context_window: None,
            max_response_tokens: d_max_response_tokens(),
            temperature: d_temperature(),
            api_key_env: None,
            base_url: None,
            model: String::new(),
        }
    }
}

fn d_max_response_tokens() -> f64 {
    0.2
}
fn d_temperature() -> f64 {
    0.6
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AgentCfg {
    #[serde(default)]
    pub bootstrap_prompt: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechGenCfg {
    #[serde(default)]
    pub model_path: PathBuf,
    #[serde(default = "d_samplerate")]
    pub samplerate: u32,
    #[serde(default = "d_volume")]
    pub volume: f64,
    #[serde(default = "d_speech_speed")]
    pub speech_speed: f64,
    #[serde(default)]
    pub speaker_id: Option<i32>,
    #[serde(default = "d_one_f64")]
    pub noise_scale: f64,
    #[serde(default = "d_one_f64")]
    pub noise_w_scale: f64,
    #[serde(default)]
    pub normalize_audio: bool,
}

impl Default for SpeechGenCfg {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(".res/voices/en_US-hfc_female-medium.onnx"),
            samplerate: d_samplerate(),
            volume: d_volume(),
            speech_speed: d_speech_speed(),
            speaker_id: None,
            noise_scale: d_one_f64(),
            noise_w_scale: d_one_f64(),
            normalize_audio: false,
        }
    }
}

fn d_samplerate() -> u32 {
    22_050
}
fn d_volume() -> f64 {
    1.0
}
fn d_speech_speed() -> f64 {
    1.0
}
fn d_one_f64() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputCfg {
    #[serde(default)]
    pub history_file: Option<PathBuf>,
    #[serde(default = "d_true")]
    pub prompt_toolkit: bool,
    #[serde(default = "d_true")]
    pub auto_suggest: bool,
    #[serde(default = "d_true")]
    pub completion: bool,
    #[serde(default = "d_true")]
    pub explicit_paste: bool,
    #[serde(default)]
    pub multiline: bool,
}

impl Default for InputCfg {
    fn default() -> Self {
        Self {
            history_file: None,
            prompt_toolkit: true,
            auto_suggest: true,
            completion: true,
            explicit_paste: true,
            multiline: false,
        }
    }
}

fn d_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct SummarizationCfg {
    #[serde(default)]
    pub model_path: PathBuf,
    #[serde(default = "d_cpu")]
    pub device: String,
    #[serde(default = "d_ratio")]
    pub ratio: f64,
    #[serde(default = "d_threshold")]
    pub threshold: f64,
    #[serde(default)]
    pub max_sentences: usize,
}

impl Default for SummarizationCfg {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(".res/embedding/all-MiniLM-L6-v2"),
            device: d_cpu(),
            ratio: d_ratio(),
            threshold: d_threshold(),
            max_sentences: 0,
        }
    }
}

fn d_cpu() -> String {
    "cpu".into()
}
fn d_ratio() -> f64 {
    0.1
}
fn d_threshold() -> f64 {
    0.5
}

#[derive(Debug, Clone, Deserialize)]
pub struct DocumentQueryCfg {
    #[serde(default)]
    pub model_path: PathBuf,
    #[serde(default = "d_cpu")]
    pub device: String,
    #[serde(default = "d_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "d_top_k")]
    pub top_k: usize,
}

impl Default for DocumentQueryCfg {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(".res/crossencoder/ms-marco-TinyBERT-L2"),
            device: d_cpu(),
            chunk_size: d_chunk_size(),
            top_k: d_top_k(),
        }
    }
}

fn d_chunk_size() -> usize {
    3
}
fn d_top_k() -> usize {
    5
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageSearchCfg {
    #[serde(default)]
    pub model_path: PathBuf,
    #[serde(default = "d_cpu")]
    pub device: String,
    #[serde(default = "d_top_k3")]
    pub top_k: usize,
}

impl Default for ImageSearchCfg {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(".res/embedding/clip-ViT-B-32"),
            device: d_cpu(),
            top_k: d_top_k3(),
        }
    }
}

fn d_top_k3() -> usize {
    3
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenCfg {
    #[serde(default = "d_t2i_model")]
    pub t2i_model: String,
    #[serde(default = "d_i2i_model")]
    pub i2i_model: String,
    #[serde(default = "d_device")]
    pub device: String,
    #[serde(default = "d_negative_prompt")]
    pub negative_prompt: String,
    #[serde(default = "d_width")]
    pub width: u32,
    #[serde(default = "d_width")]
    pub height: u32,
    #[serde(default = "d_steps")]
    pub steps: u32,
    #[serde(default = "d_seed")]
    pub seed: i64,
    #[serde(default = "d_guidance")]
    pub guidance: f64,
    #[serde(default = "d_denoise")]
    pub denoise: f64,
    #[serde(default = "d_out_dir")]
    pub out_dir: PathBuf,
}

impl Default for ImageGenCfg {
    fn default() -> Self {
        Self {
            t2i_model: d_t2i_model(),
            i2i_model: d_i2i_model(),
            device: d_device(),
            negative_prompt: d_negative_prompt(),
            width: d_width(),
            height: d_width(),
            steps: d_steps(),
            seed: d_seed(),
            guidance: d_guidance(),
            denoise: d_denoise(),
            out_dir: d_out_dir(),
        }
    }
}

fn d_t2i_model() -> String {
    "UnfilteredAI/NSFW-GEN-ANIME".into()
}
fn d_i2i_model() -> String {
    "runwayml/stable-diffusion-v1-5".into()
}
fn d_negative_prompt() -> String {
    "blurry, low quality, deformed, extra limbs, watermark, text".into()
}
fn d_width() -> u32 {
    512
}
fn d_steps() -> u32 {
    50
}
fn d_seed() -> i64 {
    42
}
fn d_guidance() -> f64 {
    5.0
}
fn d_denoise() -> f64 {
    0.5
}
fn d_out_dir() -> PathBuf {
    PathBuf::from(".out_images")
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingCfg {
    #[serde(default)]
    pub model_path: PathBuf,
    #[serde(default)]
    pub image_model_path: Option<PathBuf>,
    #[serde(default = "d_multimodal")]
    pub multimodal: bool,
    #[serde(default = "d_cpu")]
    pub device: String,
    #[serde(default = "d_corpora_dir")]
    pub corpora_dir: PathBuf,
    #[serde(default = "d_extensions")]
    pub extensions: String,
    #[serde(default = "d_image_extensions")]
    pub image_extensions: String,
    #[serde(default = "d_chunk_size5")]
    pub chunk_size: usize,
    #[serde(default = "d_top_k")]
    pub top_k: usize,
}

impl Default for EmbeddingCfg {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(".res/embedding/all-MiniLM-L6-v2"),
            image_model_path: Some(PathBuf::from(".res/embedding/clip-ViT-B-32")),
            multimodal: true,
            device: d_cpu(),
            corpora_dir: d_corpora_dir(),
            extensions: d_extensions(),
            image_extensions: d_image_extensions(),
            chunk_size: d_chunk_size5(),
            top_k: d_top_k(),
        }
    }
}

fn d_multimodal() -> bool {
    true
}
fn d_corpora_dir() -> PathBuf {
    PathBuf::from(".res/corpora")
}
fn d_extensions() -> String {
    "txt, md, markdown, tex, rst, log, csv, html, htm, pdf, docx, xlsx, png, jpg, jpeg, webp, bmp, tiff".into()
}
fn d_image_extensions() -> String {
    "png, jpg, jpeg, webp, bmp, tiff".into()
}
fn d_chunk_size5() -> usize {
    5
}

/// Additive Rust-only overrides (Python ignores the `[rust]` section).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RustCfg {
    /// whisper.cpp ggml model override (file or directory holding a .bin).
    /// Keyed `*_path` so the generic path-resolution rule applies (§6).
    #[serde(default)]
    pub speech_det_model_path: Option<PathBuf>,
}

// ---------------------------------------------------------------- config

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub logging: LoggingCfg,
    #[serde(default)]
    pub speech_det: SpeechDetCfg,
    #[serde(default)]
    pub brain: BrainCfg,
    #[serde(default)]
    pub agent: AgentCfg,
    #[serde(default)]
    pub speech_gen: SpeechGenCfg,
    #[serde(default)]
    pub input: InputCfg,
    #[serde(default)]
    pub summarization: SummarizationCfg,
    #[serde(default)]
    pub document_query: DocumentQueryCfg,
    #[serde(default)]
    pub image_search: ImageSearchCfg,
    #[serde(default)]
    pub image_gen: ImageGenCfg,
    #[serde(default)]
    pub embedding: EmbeddingCfg,
    #[serde(default)]
    pub rust: RustCfg,
}

/// Resolve a path string relative to the config directory.
fn resolve_path(value: &str, base_dir: &Path) -> PathBuf {
    base_dir.join(value)
}

fn is_path_key(key: &str) -> bool {
    key.ends_with("_file") || key.ends_with("_dir") || key.ends_with("_path")
}

/// Recursively resolve path-like values (Python `namespace_from_dict`).
fn resolve_paths(value: &mut serde_json::Value, base_dir: &Path) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                match v {
                    serde_json::Value::String(s) if is_path_key(key) => {
                        let joined = resolve_path(s, base_dir);
                        *v = serde_json::Value::String(joined.to_string_lossy().to_string());
                    }
                    _ => resolve_paths(v, base_dir),
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                if let serde_json::Value::String(s) = item {
                    let joined = resolve_path(s, base_dir);
                    *item = serde_json::Value::String(joined.to_string_lossy().to_string());
                }
            }
        }
        _ => {}
    }
}

/// Convert a TOML document into a JSON value (toml -> serde_json).
fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    match v {
        toml::Value::Boolean(b) => serde_json::json!(*b),
        toml::Value::Integer(i) => serde_json::json!(*i),
        toml::Value::Float(f) => serde_json::json!(*f),
        toml::Value::String(s) => serde_json::json!(*s),
        toml::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(toml_to_json).collect())
        }
        toml::Value::Table(table) => {
            let mut map = serde_json::Map::new();
            for (k, val) in table {
                map.insert(k.clone(), toml_to_json(val));
            }
            serde_json::Value::Object(map)
        }
        toml::Value::Datetime(d) => serde_json::json!(d.to_string()),
    }
}

/// Load `config.toml` and return the typed [`Config`].
///
/// Paths are resolved relative to the config file's parent directory.
pub fn load_config(config_path: impl AsRef<Path>) -> Result<Config> {
    let config_path = config_path.as_ref();
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("Cannot read config file '{}'", config_path.display()))?;
    let config_dir = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let toml_doc: toml::Value = toml::from_str(&raw)
        .with_context(|| format!("Invalid TOML in '{}'", config_path.display()))?;
    let mut value = toml_to_json(&toml_doc);
    resolve_paths(&mut value, &config_dir);
    serde_json::from_value(value).context("Invalid config structure")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_config(content: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn parse_defaults_and_resolve_paths() {
        let (_dir, path) = write_config(
            r#"
[brain]
model = "test-model"
max_response_tokens = 0.2

[speech_det]
model_path = ".res/faster-whisper-medium"

[image_gen]
out_dir = ".out_images"

[embedding]
corpora_dir = ".res/corpora"
extensions = "txt, md"
"#,
        );
        let cfg = load_config(&path).unwrap();
        assert_eq!(cfg.brain.model, "test-model");
        assert_eq!(cfg.brain.max_response_tokens, 0.2);
        assert_eq!(cfg.brain.temperature, 0.6);
        assert_eq!(cfg.speech_det.beam_size, 5);
        assert_eq!(cfg.speech_det.silence_timeout_s, 2.0);
        assert_eq!(cfg.image_gen.width, 512);
        assert_eq!(cfg.image_gen.seed, 42);
        // paths resolved against the config dir
        let base = path.parent().unwrap();
        assert_eq!(
            cfg.speech_det.model_path,
            base.join(".res/faster-whisper-medium")
        );
        assert_eq!(cfg.image_gen.out_dir, base.join(".out_images"));
        assert_eq!(cfg.embedding.corpora_dir, base.join(".res/corpora"));
        // plain string values are NOT treated as paths
        assert_eq!(cfg.embedding.extensions, "txt, md");
        // defaults for missing sections
        assert!(cfg.input.prompt_toolkit);
        assert_eq!(cfg.summarization.ratio, 0.1);
        assert!(cfg.embedding.multimodal);
        assert!(cfg.rust.speech_det_model_path.is_none());
    }

    #[test]
    fn missing_file_errors() {
        let err = load_config("definitely/not/here.toml");
        assert!(err.is_err());
    }

    #[test]
    fn rust_override_section() {
        let (_dir, path) = write_config(
            r#"
[rust]
speech_det_model_path = "res/whisper/ggml-medium.bin"
"#,
        );
        let cfg = load_config(&path).unwrap();
        let base = path.parent().unwrap().to_path_buf();
        assert_eq!(
            cfg.rust.speech_det_model_path.unwrap(),
            base.join("res/whisper/ggml-medium.bin")
        );
    }
}
