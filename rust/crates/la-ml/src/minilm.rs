//! MiniLM sentence embeddings (all-MiniLM-L6-v2) via candle BERT.
//!
//! Implements [`la_core::encoders::TextEmbedder`]: tokenize with the model's
//! `tokenizer.json`, forward the BERT encoder, mean-pool the last hidden
//! states over the attention mask (sentence-transformers pooling) and return
//! unnormalized rows (`la-core` L2-normalizes afterwards).

#![cfg(feature = "embeddings")]

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use la_core::encoders::TextEmbedder;
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

/// MiniLM sentence embedder.
pub struct MiniLM {
    model: BertModel,
    tokenizer: Tokenizer,
    max_len: usize,
    device: Device,
    dtype: DType,
}

impl MiniLM {
    /// Load the model from a sentence-transformers folder
    /// (`model.safetensors`, `config.json`, `tokenizer.json`).
    pub fn new(model_path: &Path, device: Device) -> Result<Self> {
        let weights = model_path.join("model.safetensors");
        if !weights.exists() {
            return Err(anyhow!(
                "model.safetensors not found in '{}'; convert the model once (see rust/README.md)",
                model_path.display()
            ));
        }
        let tokenizer_file = model_path.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_file).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let config: Config = {
            let text = std::fs::read_to_string(model_path.join("config.json"))?;
            serde_json::from_str(&text).map_err(|e| anyhow!("config.json: {e}"))?
        };
        let max_len = config.max_position_embeddings;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)? };
        let model = BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            max_len,
            device,
            dtype: DType::F32,
        })
    }
}

impl TextEmbedder for MiniLM {
    fn encode(&self, texts: &[String]) -> Result<Array2<f64>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let mut input_ids_rows: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
        let mut attention_rows: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.clone(), true)
                .map_err(|e| anyhow!("tokenize: {e}"))?;
            let ids: Vec<i64> = encoding
                .get_ids()
                .iter()
                .take(self.max_len)
                .map(|&i| i as i64)
                .collect();
            let mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .take(self.max_len)
                .map(|&m| m as i64)
                .collect();
            input_ids_rows.push(ids);
            attention_rows.push(mask);
        }
        let seq_len = input_ids_rows.iter().map(|r| r.len()).max().unwrap_or(0);
        // pad all rows to the same length
        for row in input_ids_rows.iter_mut() {
            row.resize(seq_len, 0);
        }
        for row in attention_rows.iter_mut() {
            row.resize(seq_len, 0);
        }
        let input_ids = Tensor::from_iter(input_ids_rows.iter().flatten().copied(), &self.device)?
            .reshape((texts.len(), seq_len))?;
        let attention_mask =
            Tensor::from_iter(attention_rows.iter().flatten().copied(), &self.device)?
                .reshape((texts.len(), seq_len))?;
        let token_type_ids = Tensor::zeros((texts.len(), seq_len), DType::I64, &self.device)?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        // Mean pooling over tokens with mask == 1 (sentence-transformers default).
        let mask = attention_mask.to_dtype(self.dtype)?.unsqueeze(2)?;
        let sum = (hidden * &mask)?.sum(1)?;
        let count = mask.sum(1)?.clamp(1e-9, 1e9)?;
        let pooled = (sum / count)?;

        let (rows, cols) = pooled.dims2()?;
        let flat = pooled.flatten_all()?.to_vec1::<f32>()?;
        let mut out = Array2::<f64>::zeros((rows, cols));
        for (index, value) in flat.iter().enumerate() {
            out[[index / cols, index % cols]] = *value as f64;
        }
        Ok(out)
    }
}
