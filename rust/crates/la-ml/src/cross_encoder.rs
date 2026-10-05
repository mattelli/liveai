//! TinyBERT cross-encoder (ms-marco-TinyBERT-L2) via candle BERT.
//!
//! Scores `[query, passage]` pairs: forward the BERT encoder over the pair
//! (`query [SEP] passage`), take the `[CLS]` hidden state through the
//! classification head, then apply the sentence-transformers default
//! activation — sigmoid for `num_labels == 1`, softmax class-1 probability
//! for `num_labels == 2`.

#![cfg(feature = "cross-encoder")]

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use la_core::encoders::CrossEncoder;
use std::path::Path;

/// TinyBERT cross-encoder.
pub struct TinyBertCrossEncoder {
    model: BertModel,
    classifier: Linear,
    tokenizer: tokenizers::Tokenizer,
    max_len: usize,
    device: Device,
    num_labels: usize,
}

impl TinyBertCrossEncoder {
    /// Load the cross-encoder from a local folder (`model.safetensors`,
    /// `config.json`, `tokenizer.json`; weights converted to safetensors once).
    pub fn new(model_path: &Path, device: Device) -> Result<Self> {
        let weights = model_path.join("model.safetensors");
        if !weights.exists() {
            return Err(anyhow!(
                "model.safetensors not found in '{}'; convert the model once (see rust/README.md)",
                model_path.display()
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let config: Config = {
            let text = std::fs::read_to_string(model_path.join("config.json"))?;
            serde_json::from_str(&text).map_err(|e| anyhow!("config.json: {e}"))?
        };
        // num_labels comes from the raw config.json (not part of candle's Config)
        let num_labels: usize = {
            let raw: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(model_path.join("config.json"))?)?;
            raw.get("num_labels").and_then(|v| v.as_u64()).unwrap_or(1) as usize
        };
        let max_len = config.max_position_embeddings;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)? };
        let model = BertModel::load(vb.clone(), &config)?;
        let classifier =
            candle_nn::linear_no_bias(config.hidden_size, num_labels, vb.pp("classifier"))?;
        Ok(Self {
            model,
            classifier,
            tokenizer,
            max_len,
            device,
            num_labels,
        })
    }

    fn encode_pair(&self, query: &str, passage: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let text = format!("{query} [SEP] {passage}");
        let encoding = self
            .tokenizer
            .encode(text, true)
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
        Ok((ids, mask))
    }
}

impl CrossEncoder for TinyBertCrossEncoder {
    fn predict(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let mut all_ids: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        let mut all_masks: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        for (query, passage) in pairs {
            let (ids, mask) = self.encode_pair(query, passage)?;
            all_ids.push(ids);
            all_masks.push(mask);
        }
        let seq_len = all_ids.iter().map(|r| r.len()).max().unwrap_or(0);
        for row in all_ids.iter_mut() {
            row.resize(seq_len, 0);
        }
        for row in all_masks.iter_mut() {
            row.resize(seq_len, 0);
        }
        let input_ids = Tensor::from_iter(all_ids.iter().flatten().copied(), &self.device)?
            .reshape((pairs.len(), seq_len))?;
        let attention_mask = Tensor::from_iter(all_masks.iter().flatten().copied(), &self.device)?
            .reshape((pairs.len(), seq_len))?;
        let token_type_ids = Tensor::zeros((pairs.len(), seq_len), DType::I64, &self.device)?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        // [CLS] token is at position 0: narrow along the sequence dim, keep
        // the batch dim, then flatten the head dim into rows.
        let (batch, seq, hidden_size) = hidden.dims3()?;
        let cls = hidden
            .narrow(1, 0, 1)?
            .reshape((batch, seq * hidden_size))?
            .narrow(1, 0, hidden_size)?;
        let logits = self.classifier.forward(&cls)?;

        let logits = logits.to_dtype(DType::F32)?;
        let (rows, cols) = logits.dims2()?;
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut scores = Vec::with_capacity(rows);
        for row in 0..rows {
            let logits_row = &flat[row * cols..(row + 1) * cols];
            let score = if self.num_labels == 1 {
                sigmoid(logits_row[0] as f64)
            } else {
                softmax_second_class(logits_row)
            };
            scores.push(score);
        }
        Ok(scores)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_second_class(logits: &[f32]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&l| ((l as f64) - max as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        0.0
    } else {
        exps[exps.len() - 1] / sum
    }
}
