//! In-document search with a CrossEncoder (`src/document_query.py` port).
//!
//! The document text is split into sentences, up to `chunk_size` consecutive
//! sentences form one passage, then a CrossEncoder scores every
//! `[query, passage]` pair and the `top_k` highest-scoring passages are
//! returned in descending score order. The document is read with
//! [`crate::document_loader`], so any supported file type works. Passages are
//! rebuilt only when the file's mtime changes (per-process cache).

use crate::document_loader::{Document, load_document};
use crate::embedding_search::{DEFAULT_CHUNK_SIZE, build_passages, has_content};
use crate::encoders::CrossEncoder;
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const DEFAULT_TOP_K: usize = 5;

/// One scored passage (text + optional page number).
#[derive(Debug, Clone)]
struct PassageRecord {
    text: String,
    page: Option<u64>,
}

/// Scores passages of one document against a query with a CrossEncoder
/// (Python `DocumentQuery`; `cross_encoder`/`sentence_tokenizer` seams).
pub struct DocumentQuery {
    pub model_path: PathBuf,
    pub device: String,
    pub chunk_size: usize,
    pub top_k: usize,
    cross_encoder: Option<Arc<dyn CrossEncoder>>,
    sentence_tokenizer: Option<SentenceTokenizer>,
    passage_cache: Mutex<HashMap<String, (std::time::SystemTime, Vec<PassageRecord>)>>,
}

/// Sentence-splitting seam (Python injects a callable; tests fake it).
type SentenceTokenizer = Arc<dyn Fn(&str) -> Vec<String> + Send + Sync>;
/// Borrowed view over the tokenizer seam.
type TokenizerFn<'a> = Box<dyn Fn(&str) -> Vec<String> + 'a>;

impl DocumentQuery {
    pub fn new(
        model_path: PathBuf,
        device: String,
        chunk_size: usize,
        top_k: usize,
        cross_encoder: Option<Arc<dyn CrossEncoder>>,
        sentence_tokenizer: Option<SentenceTokenizer>,
    ) -> Self {
        Self {
            model_path,
            device,
            chunk_size: if chunk_size == 0 {
                DEFAULT_CHUNK_SIZE
            } else {
                chunk_size
            },
            top_k: if top_k == 0 { DEFAULT_TOP_K } else { top_k },
            cross_encoder,
            sentence_tokenizer,
            passage_cache: Mutex::new(HashMap::new()),
        }
    }

    fn tokenizer(&self) -> TokenizerFn<'_> {
        match &self.sentence_tokenizer {
            Some(tokenizer) => {
                let tokenizer = tokenizer.clone();
                Box::new(move |text: &str| {
                    tokenizer(text)
                        .into_iter()
                        .filter(|s| !s.trim().is_empty())
                        .collect()
                })
            }
            None => Box::new(|text: &str| crate::summarizer::split_sentences(text)),
        }
    }

    fn cross_encoder(&self) -> Result<&Arc<dyn CrossEncoder>> {
        if let Some(encoder) = &self.cross_encoder {
            return Ok(encoder);
        }
        if !crate::encoders::is_model_available(&self.model_path) {
            return Err(anyhow::anyhow!(
                "Cross-encoder model not found at '{}'. See README.md for how to store the model.",
                self.model_path.display()
            ));
        }
        Err(anyhow::anyhow!(
            "Cross-encoder not wired: build the `la` binary with `--features ml` \
             so the candle-based encoder is available."
        ))
    }

    /// Load `path` and split its text into passage records.
    fn passages(&self, path: &Path) -> Result<Vec<PassageRecord>> {
        let document: Document = load_document(path, None, None);
        if !document.error.is_empty() {
            return Err(anyhow::anyhow!(
                "Unable to read '{}': {}",
                document.source,
                document.error
            ));
        }
        if document.kind == "image" {
            return Err(anyhow::anyhow!(
                "No searchable text in '{}' (image file).",
                document.source
            ));
        }
        let tokenizer = self.tokenizer();
        let mut records = Vec::new();
        // Paginated documents (PDF) expose per-page sections so each passage
        // keeps its real page number.
        let sections: Vec<(Option<u64>, String)> = if document.sections.is_empty() {
            vec![(document.pages, document.text.clone())]
        } else {
            document.sections.clone()
        };
        for (page, section_text) in sections {
            for passage in
                build_passages(&[section_text], self.chunk_size, Some(tokenizer.as_ref()))
            {
                if !has_content(&passage, crate::embedding_search::MIN_ALNUM_CHARS) {
                    continue;
                }
                records.push(PassageRecord {
                    text: passage,
                    page,
                });
            }
        }
        Ok(records)
    }

    /// Rebuild the passage list only when the file's mtime changed.
    fn cached_passages(&self, path: &Path) -> Result<Vec<PassageRecord>> {
        let key = std::fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .to_string();
        let mtime = std::fs::metadata(path)?.modified()?;
        {
            let cache = self.passage_cache.lock().unwrap();
            if let Some((cached_mtime, records)) = cache.get(&key)
                && *cached_mtime == mtime
            {
                return Ok(records.clone());
            }
        }
        let records = self.passages(path)?;
        self.passage_cache
            .lock()
            .unwrap()
            .insert(key, (mtime, records.clone()));
        Ok(records)
    }

    /// Return the `top_k` passages most relevant to `query`.
    pub fn query(&self, path: &Path, query: &str, top_k: Option<usize>) -> String {
        let query = query.trim();
        if query.is_empty() {
            return "No query provided.".into();
        }
        if !path.exists() || !path.is_file() {
            return format!("Unable to read '{}': file not found.", path.display());
        }
        let records = match self.cached_passages(path) {
            Ok(records) => records,
            Err(e) => return format!("{e}"),
        };
        if records.is_empty() {
            return format!(
                "No searchable text found in '{}'.",
                path.file_name().unwrap_or_default().to_string_lossy()
            );
        }

        let k = self.resolve_k(top_k, records.len());
        let inputs: Vec<(String, String)> = records
            .iter()
            .map(|record| (query.to_string(), record.text.clone()))
            .collect();
        let encoder = match self.cross_encoder() {
            Ok(e) => e,
            Err(e) => return format!("{e}"),
        };
        let scores = match encoder.predict(&inputs) {
            Ok(scores) => scores,
            Err(e) => return format!("Error: {e}"),
        };
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(k);

        let mut lines = vec![format!(
            "Top {k} matches for '{query}' in '{}':",
            path.file_name().unwrap_or_default().to_string_lossy()
        )];
        for (rank, &idx) in order.iter().enumerate() {
            let record = &records[idx];
            let line = format!(
                "{}) [{:.3}]{} {}",
                rank + 1,
                scores[idx],
                record
                    .page
                    .map(|p| format!(" (page {p})"))
                    .unwrap_or_default(),
                record.text
            );
            lines.push(line);
        }
        lines.join("\n")
    }

    fn resolve_k(&self, top_k: Option<usize>, n_passages: usize) -> usize {
        // Python `_resolve_k`: the runtime argument wins when > 0, otherwise the
        // configured default applies (already normalized to DEFAULT_TOP_K in
        // `new`), then clamp to [1, n_passages].
        let k = top_k.filter(|k| *k > 0).unwrap_or(self.top_k).max(1);
        k.min(n_passages).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fake cross-encoder: score = count of query tokens in the
    /// passage (so tests can order passages deterministically).
    struct FakeCrossEncoder;
    impl CrossEncoder for FakeCrossEncoder {
        fn predict(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
            Ok(pairs
                .iter()
                .map(|(query, passage)| {
                    let query = query.to_lowercase();
                    let passage = passage.to_lowercase();
                    query
                        .split_whitespace()
                        .filter(|t| passage.contains(t))
                        .count() as f64
                })
                .collect())
        }
    }

    fn document_query() -> DocumentQuery {
        DocumentQuery::new(
            PathBuf::from("fake-model"),
            "cpu".into(),
            1, // one sentence per passage -> one passage per sentence
            3,
            Some(Arc::new(FakeCrossEncoder)),
            None,
        )
    }

    fn write_doc(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn query_ranks_and_caches() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_doc(
            dir.path(),
            "doc.txt",
            "Apples grow on trees. Bananas are yellow fruit. Apples and bananas. Kiwi fruit grows.",
        );
        let dq = document_query();
        let out = dq.query(&path, "apples bananas", None);
        assert!(out.contains("Top 3 matches"), "got: {out}");
        assert!(out.contains("1) [2.000]"), "best passage scores 2: {out}");
        // caching: second query does not fail and stays valid
        let out2 = dq.query(&path, "apples bananas", None);
        assert_eq!(out, out2);
    }

    #[test]
    fn empty_query_message() {
        let dq = document_query();
        assert_eq!(
            dq.query(Path::new("x.txt"), "  ", None),
            "No query provided."
        );
    }

    #[test]
    fn missing_file_message() {
        let dq = document_query();
        assert!(
            dq.query(Path::new("no/such/file.txt"), "q", None)
                .contains("file not found")
        );
    }

    #[test]
    fn image_files_report_no_text() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("img.png");
        image::DynamicImage::new_rgb8(4, 4).save(&png).unwrap();
        let dq = document_query();
        let out = dq.query(&png, "anything", None);
        assert!(out.contains("No searchable text"), "got: {out}");
    }

    #[test]
    fn top_k_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_doc(
            dir.path(),
            "doc.txt",
            "A sentence here. Another sentence. Third one. Fourth.",
        );
        let dq = document_query();
        let out = dq.query(&path, "sentence", Some(99));
        assert!(
            out.contains("Top 4 matches"),
            "clamped to passage count: {out}"
        );
    }

    #[test]
    fn passages_cache_invalidated_by_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_doc(dir.path(), "doc.txt", "Original content sentence one.");
        let dq = document_query();
        let first = dq.query(&path, "original", None);
        assert!(first.contains("Original content"));
        // update the file; the cache must rebuild (mtime change)
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&path, "Completely new words here.").unwrap();
        let second = dq.query(&path, "new", None);
        assert!(second.contains("Completely new"), "got: {second}");
    }
}
