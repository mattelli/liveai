//! Extractive text summarization (`src/summarizer.py` port).
//!
//! Pipeline: split the input into sentences, embed each sentence, compute the
//! pairwise cosine similarity, run LexRank to score centrality, then return
//! the most central sentences in their original document order. The embedder
//! is injected (Python `model_factory` seam; `la-ml` provides the real one).

use crate::encoders::{TextEmbedder, is_model_available};
use crate::lex_rank::degree_centrality_scores;
use anyhow::Result;
use ndarray::Array2;
use std::path::Path;
use std::sync::Arc;

pub const DEFAULT_RATIO: f64 = 0.1;
pub const DEFAULT_THRESHOLD: f64 = 0.5;
pub const DEFAULT_MAX_SENTENCES: usize = 0; // 0 -> derive the count from the ratio
pub const MAX_INPUT_CHARS: usize = 20_000; // guard against pathological documents

/// Split `text` into sentences (the Python fallback when NLTK data is
/// unavailable — used as the only splitter for parity of behavior without
/// runtime downloads). Python pattern: `(?<=[.!?])\s+` — split at a run of
/// whitespace that directly follows `.`, `!` or `?`. The `regex` crate has
/// no look-behind, so the boundary rule is applied manually.
pub fn split_sentences(text: &str) -> Vec<String> {
    let text = text.trim();
    let bytes = text.as_bytes();
    let mut parts: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            let mut end = i;
            while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            // Only a whitespace run directly after sentence punctuation splits.
            if i > 0 && matches!(bytes[i - 1], b'.' | b'!' | b'?') {
                let part = text[start..i].trim();
                if !part.is_empty() {
                    parts.push(part.to_string());
                }
                start = end;
            }
            i = end;
        } else {
            i += 1;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    parts
}

/// L2-normalize each row; zero rows are left as zeros (norm guarded).
pub fn normalize_rows(matrix: &Array2<f64>) -> Array2<f64> {
    let mut out = matrix.clone();
    for mut row in out.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|v| v / norm);
        }
    }
    out
}

/// Configuration of one summarizer (Python `Summarizer.__init__` params).
#[derive(Debug, Clone)]
pub struct SummarizerConfig {
    pub model_path: std::path::PathBuf,
    pub device: String,
    pub ratio: f64,
    pub threshold: f64,
    pub max_sentences: usize,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from(".res/embedding/all-MiniLM-L6-v2"),
            device: "cpu".into(),
            ratio: DEFAULT_RATIO,
            threshold: DEFAULT_THRESHOLD,
            max_sentences: DEFAULT_MAX_SENTENCES,
        }
    }
}

/// Loads a sentence-embedding model lazily and summarizes text.
pub struct Summarizer {
    pub config: SummarizerConfig,
    embedder: Option<Arc<dyn TextEmbedder>>,
}

impl Summarizer {
    pub fn new(config: SummarizerConfig, embedder: Option<Arc<dyn TextEmbedder>>) -> Self {
        Self { config, embedder }
    }

    fn embedder(&self) -> Result<&Arc<dyn TextEmbedder>> {
        if let Some(embedder) = &self.embedder {
            return Ok(embedder);
        }
        if !is_model_available(&self.config.model_path) {
            return Err(anyhow::anyhow!(
                "Summarization model not found at '{}'. See README.md for how to store the model.",
                self.config.model_path.display()
            ));
        }
        Err(anyhow::anyhow!(
            "Summarization model not wired: build the `la` binary with `--features ml` \
             so the candle-based encoder is available."
        ))
    }

    fn resolve_k(&self, max_sentences: Option<usize>, n_sentences: usize) -> usize {
        let k = match max_sentences {
            Some(k) if k > 0 => k,
            _ => {
                if self.config.max_sentences > 0 {
                    self.config.max_sentences
                } else {
                    ((n_sentences as f64 * self.config.ratio).round() as usize).max(1)
                }
            }
        };
        k.clamp(1, n_sentences)
    }

    /// Summarize the text extracted from a document of any supported type.
    pub fn summarize_file(&self, path: &Path, max_sentences: Option<usize>) -> Result<String> {
        let document = crate::document_loader::load_document(path, None, None);
        if !document.error.is_empty() {
            return Ok(format!(
                "Unable to read '{}': {}",
                document.source, document.error
            ));
        }
        if document.kind == "image" {
            return Ok(format!(
                "No extractable text in '{}' (image file).",
                document.source
            ));
        }
        self.summarize(&document.text, max_sentences)
    }

    /// Return an extractive summary of `text` as a single string.
    pub fn summarize(&self, text: &str, max_sentences: Option<usize>) -> Result<String> {
        let mut text = text.trim().to_string();
        if text.is_empty() {
            return Ok("No text to summarize.".into());
        }
        if text.chars().count() > MAX_INPUT_CHARS {
            text = text.chars().take(MAX_INPUT_CHARS).collect();
        }

        let sentences = split_sentences(&text);
        if sentences.len() <= 1 {
            return Ok(text);
        }

        let k = self.resolve_k(max_sentences, sentences.len());

        let embedder = self.embedder()?;
        let raw = embedder.encode(&sentences)?;
        let normalized = normalize_rows(&raw);
        let similarity = normalized.dot(&normalized.t());
        let scores = degree_centrality_scores(&similarity, Some(self.config.threshold))?;

        // Keep the top-k by centrality, then restore document order.
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(k);
        order.sort_unstable();
        let summary = order
            .iter()
            .map(|&i| sentences[i].clone())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Deterministic fake embedder: one-hot keyword axes (alpha/beta/gamma),
    /// case-insensitive. With sentences [alpha-only, beta-only, alpha+beta,
    /// gamma-only] the similarity graph (threshold 0.5) is a star centered on
    /// the alpha+beta sentence, so the stationary distribution ranks it first
    /// and the remaining sentences tie exactly (bitwise, by symmetry — the
    /// power method keeps them equal), letting the stable sort pick document
    /// order deterministically.
    struct FakeEmbedder;
    impl TextEmbedder for FakeEmbedder {
        fn encode(&self, texts: &[String]) -> Result<Array2<f64>> {
            let mut out = Array2::<f64>::zeros((texts.len(), 3));
            for (i, text) in texts.iter().enumerate() {
                let lower = text.to_lowercase();
                out[[i, 0]] = if lower.contains("alpha") { 1.0 } else { 0.0 };
                out[[i, 1]] = if lower.contains("beta") { 1.0 } else { 0.0 };
                out[[i, 2]] = if lower.contains("gamma") { 1.0 } else { 0.0 };
            }
            Ok(out)
        }
    }

    #[test]
    fn split_sentences_regex() {
        let sentences = split_sentences("One sentence. Two here! A question? Plain");
        assert_eq!(
            sentences,
            vec!["One sentence.", "Two here!", "A question?", "Plain"]
        );
        assert!(split_sentences("").is_empty());
    }

    #[test]
    fn split_sentences_matches_python_lookbehind() {
        // Python `_SENTENCE_BOUNDARY` = `(?<=[.!?])\\s+`: only whitespace
        // directly after `.`, `!` or `?` splits.
        // Whitespace without punctuation before it must NOT split.
        assert_eq!(
            split_sentences("Hello world. Continue here"),
            vec!["Hello world.", "Continue here"]
        );
        assert_eq!(
            split_sentences("No boundary, at all"),
            vec!["No boundary, at all"]
        );
        // A multi-space run after punctuation is one boundary.
        assert_eq!(split_sentences("A.  B"), vec!["A.", "B"]);
        // Newline after punctuation splits too.
        assert_eq!(split_sentences("A.\nB"), vec!["A.", "B"]);
        // Abbreviation-like digits after a dot do not create a boundary
        // (the dot must directly precede the whitespace).
        assert_eq!(
            split_sentences("Version 1.2 works"),
            vec!["Version 1.2 works"]
        );
        // Parts are trimmed and empties dropped (Python list comprehension).
        assert_eq!(split_sentences("  A. B.  "), vec!["A.", "B."]);
    }

    #[test]
    fn normalize_rows_guards_zero() {
        let m = array![[3.0, 4.0], [0.0, 0.0]];
        let normalized = normalize_rows(&m);
        assert!((normalized[[0, 0]] - 0.6).abs() < 1e-9);
        assert!((normalized[[0, 1]] - 0.8).abs() < 1e-9);
        assert!((normalized[[1, 0]]).abs() < 1e-9);
    }

    #[test]
    fn summarize_prefers_central_sentences_in_order() {
        let summarizer = Summarizer::new(SummarizerConfig::default(), Some(Arc::new(FakeEmbedder)));
        let text = "Alpha text here. Beta text there. Alpha and beta text. Gamma text elsewhere.";
        let summary = summarizer.summarize(text, Some(2)).unwrap();
        // sentences[0] (alpha) and sentences[2] (alpha+beta) are the most central
        assert_eq!(summary, "Alpha text here. Alpha and beta text.");
    }

    #[test]
    fn single_sentence_passthrough() {
        let summarizer = Summarizer::new(SummarizerConfig::default(), Some(Arc::new(FakeEmbedder)));
        assert_eq!(
            summarizer.summarize("Just one.", None).unwrap(),
            "Just one."
        );
    }

    #[test]
    fn empty_text_message() {
        let summarizer = Summarizer::new(SummarizerConfig::default(), Some(Arc::new(FakeEmbedder)));
        assert_eq!(
            summarizer.summarize("   ", None).unwrap(),
            "No text to summarize."
        );
    }

    #[test]
    fn max_sentences_bounds() {
        let summarizer = Summarizer::new(SummarizerConfig::default(), Some(Arc::new(FakeEmbedder)));
        let text = "Alpha one. Beta two. Alpha three. Beta four.";
        let k1 = summarizer.summarize(text, Some(1)).unwrap();
        assert!(k1.split_once(". ").is_none() || k1.matches(". ").count() == 1);
    }

    #[test]
    fn model_missing_errors() {
        let summarizer = Summarizer::new(SummarizerConfig::default(), None);
        let err = summarizer.summarize("Alpha. Beta.", None);
        assert!(err.is_err());
    }

    #[test]
    fn summarize_file_reports_unreadable() {
        let summarizer = Summarizer::new(SummarizerConfig::default(), Some(Arc::new(FakeEmbedder)));
        let out = summarizer
            .summarize_file(Path::new("definitely/not/here.pdf"), None)
            .unwrap();
        assert!(out.starts_with("Unable to read"), "got: {out}");
    }

    #[test]
    fn image_files_report_no_text() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("a.png");
        image::DynamicImage::new_rgb8(4, 4).save(&png).unwrap();
        let summarizer = Summarizer::new(SummarizerConfig::default(), Some(Arc::new(FakeEmbedder)));
        let out = summarizer.summarize_file(&png, None).unwrap();
        assert!(out.contains("No extractable text"), "got: {out}");
    }
}
