//! CLIP embeddings (clip-ViT-B-32) via candle — one shared space for texts
//! and images, mirroring the sentence-transformers multimodal setup.
//!
//! - Text side: CLIP text tower + projection (`clip_text_embeddings`).
//! - Image side: CLIP vision tower + projection (`clip_image_embeddings`).
//!
//! Weights are read from a local `openai/clip-vit-base-patch32`-style folder
//! (`model.safetensors`, `config.json`, `tokenizer.json`); the
//! sentence-transformers wrapper stores its pooling/projection configs on top
//! (`modules.json`/sentence-transformers config) — this implementation applies
//! the standard CLIP text/image projections, which is what the wrapper uses.

#![cfg(feature = "clip")]

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use la_core::encoders::{ImageEmbedder, TextEmbedder};
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

/// CLIP embedder: implements both [`TextEmbedder`] (text tower) and
/// [`ImageEmbedder`] (vision tower).
pub struct ClipEmbedder {
    model: ClipModel,
    tokenizer: Tokenizer,
    max_len: usize,
    device: Device,
    dtype: DType,
}

impl ClipEmbedder {
    /// Load the CLIP model from a local folder.
    pub fn new(model_path: &Path, device: Device) -> Result<Self> {
        let weights = model_path.join("model.safetensors");
        if !weights.exists() {
            return Err(anyhow!(
                "model.safetensors not found in '{}'; store clip-ViT-B-32 there (see rust/README.md)",
                model_path.display()
            ));
        }
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        // ViT-B/32 (the sentence-transformers clip-ViT-B-32 base architecture).
        let config = ClipConfig::vit_base_patch32();
        let max_len = 77;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)? };
        let model = ClipModel::new(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            max_len,
            device,
            dtype: DType::F32,
        })
    }

    fn tokenize(&self, texts: &[String]) -> Result<Tensor> {
        let mut rows: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
        for text in texts {
            let mut ids: Vec<i64> = self
                .tokenizer
                .encode(text.clone(), true)
                .map_err(|e| anyhow!("tokenize: {e}"))?
                .get_ids()
                .iter()
                .take(self.max_len)
                .map(|&i| i as i64)
                .collect();
            ids.resize(self.max_len, 0);
            rows.push(ids);
        }
        let seq = rows.first().map(|r| r.len()).unwrap_or(self.max_len);
        let flat: Vec<i64> = rows.iter().flatten().copied().collect();
        Ok(Tensor::from_iter(flat, &self.device)?.reshape((rows.len(), seq))?)
    }
}

impl TextEmbedder for ClipEmbedder {
    /// Text embeddings in the shared CLIP space.
    fn encode(&self, texts: &[String]) -> Result<Array2<f64>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let tokens = self.tokenize(texts)?;
        let embeddings = self.model.get_text_features(&tokens)?;
        let (rows, cols) = embeddings.dims2()?;
        let flat = embeddings.flatten_all()?.to_vec1::<f32>()?;
        let mut out = Array2::<f64>::zeros((rows, cols));
        for (index, value) in flat.iter().enumerate() {
            out[[index / cols, index % cols]] = *value as f64;
        }
        Ok(out)
    }
}

impl ImageEmbedder for ClipEmbedder {
    /// Image embeddings in the shared CLIP space (vision tower + projection).
    fn encode_paths(&self, paths: &[std::path::PathBuf]) -> Result<Array2<f64>> {
        if paths.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let mut images: Vec<Tensor> = Vec::with_capacity(paths.len());
        let (h, w) = (224usize, 224usize);
        for path in paths {
            let img = image::ImageReader::open(path)?
                .with_guessed_format()?
                .decode()?
                .resize_exact(w as u32, h as u32, image::imageops::FilterType::CatmullRom)
                .to_rgb8();
            let raw = img.into_raw();
            let tensor = Tensor::from_vec(raw, (h, w, 3), &self.device)?
                .permute((2, 0, 1))?
                .to_dtype(self.dtype)?
                .unsqueeze(0)?;
            images.push(tensor);
        }
        let batch = Tensor::cat(&images.iter().collect::<Vec<_>>(), 0)?;
        // CLIP vision forward with the projection head (shared embedding space).
        let embeddings = self.model.get_image_features(&batch)?;
        let (rows, cols) = embeddings.dims2()?;
        let flat = embeddings.flatten_all()?.to_vec1::<f32>()?;
        let mut out = Array2::<f64>::zeros((rows, cols));
        for (index, value) in flat.iter().enumerate() {
            out[[index / cols, index % cols]] = *value as f64;
        }
        Ok(out)
    }
}
