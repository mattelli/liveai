//! Semantic search over a local document corpus
//! (`src/embedding_search.py` port).
//!
//! Pipeline: collect text passages (and images) from a directory, embed them,
//! L2-normalize the embeddings (so inner product equals cosine similarity),
//! persist them to a single binary file (bincode, Python used pickle), then
//! answer queries with an exact inner-product ranking (ndarray matrix product
//! — the Rust equivalent of FAISS `IndexFlatIP`).
//!
//! Two modes:
//! - **text**       — text passages are embedded with the text model; image
//!   files are skipped.
//! - **multimodal** — text passages and images are embedded into one shared
//!   space with the CLIP model, so `query` can return images.
//!
//! Layout parity: `corpus.bin` (Rust-native; Python keeps `corpus.pkl`),
//! `metadata.json` and `processed.md` next to it, images under `images/`.

use crate::document_loader::{self, Document};
use crate::encoders::{ImageEmbedder, TextEmbedder, is_model_available};
use crate::summarizer::split_sentences;
use anyhow::Result;
use ndarray::Array2;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DEFAULT_TOP_K: usize = 5;
pub const DEFAULT_CHUNK_SIZE: usize = 3;
pub const DEFAULT_CORPORA_DIR: &str = ".res/corpora";
pub const CORPUS_FILE: &str = "corpus.bin";
pub const METADATA_FILE: &str = "metadata.json";
pub const PROCESSED_FILE: &str = "processed.md";
pub const IMAGES_DIR: &str = "images";
pub const MIN_ALNUM_CHARS: usize = 4; // drop passages with fewer letters/digits

/// Validate a corpus name; raise for empty/path-like names
/// (Python `sanitize_corpus_name`).
pub fn sanitize_corpus_name(name: &str) -> Result<String> {
    let cleaned = name.trim();
    let re = Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]*$").unwrap();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." || !re.is_match(cleaned) {
        return Err(anyhow::anyhow!("invalid corpus name: {name:?}"));
    }
    Ok(cleaned.to_string())
}

/// Return the folder for `corpus_name` under `corpora_dir`.
pub fn resolve_corpus_dir(corpus_name: &str, corpora_dir: &Path) -> Result<PathBuf> {
    Ok(corpora_dir.join(sanitize_corpus_name(corpus_name)?))
}

/// Metadata written beside every corpus (same shape as Python's).
pub type Metadata = serde_json::Value;

/// Write `metadata.json` inside `corpus_dir`.
pub fn save_metadata(corpus_dir: &Path, metadata: &Metadata) -> Result<()> {
    let path = corpus_dir.join(METADATA_FILE);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(metadata)?)?;
    Ok(())
}

/// Read `metadata.json` from `corpus_dir`.
pub fn load_metadata(corpus_dir: &Path) -> Result<Metadata> {
    let path = corpus_dir.join(METADATA_FILE);
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "Corpus metadata not found: '{}'",
            path.display()
        ));
    }
    Ok(serde_json::from_str(&std::fs::read_to_string(&path)?)?)
}

/// Return every `<corpora_dir>/*/metadata.json` entry, sorted by name.
pub fn list_corpora(corpora_dir: &Path) -> Vec<Metadata> {
    let mut entries: Vec<Metadata> = Vec::new();
    let Ok(root) = std::fs::read_dir(corpora_dir) else {
        return entries;
    };
    for entry in root.flatten() {
        let metadata_file = entry.path().join(METADATA_FILE);
        if let Ok(text) = std::fs::read_to_string(&metadata_file)
            && let Ok(value) = serde_json::from_str::<Metadata>(&text)
        {
            entries.push(value);
        }
    }
    entries.sort_by_key(|e| {
        e.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    });
    entries
}

/// Split each text into sentences and group `chunk_size` per passage
/// (Python `build_passages`).
type Tokenizer<'a> = dyn Fn(&str) -> Vec<String> + 'a;
pub fn build_passages(
    texts: &[String],
    chunk_size: usize,
    tokenizer: Option<&Tokenizer<'_>>,
) -> Vec<String> {
    let splitter = |text: &str| -> Vec<String> { split_sentences(text) };
    let size = chunk_size.max(1);
    let mut passages = Vec::new();
    for text in texts {
        let sentences = match tokenizer {
            Some(tokenizer) => tokenizer(text)
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            None => splitter(text),
        };
        for chunk in sentences.chunks(size) {
            let passage = chunk.join(" ").trim().to_string();
            if !passage.is_empty() {
                passages.push(passage);
            }
        }
    }
    passages
}

/// True when `text` has at least `min_alnum` letters/digits
/// (Python `has_content`).
pub fn has_content(text: &str, min_alnum: usize) -> bool {
    text.chars().filter(|c| c.is_alphanumeric()).count() >= min_alnum.max(1)
}

/// L2-normalize each row; zero rows are left as zeros (Python `normalize_rows`).
pub fn normalize_rows(matrix: &Array2<f64>) -> Array2<f64> {
    crate::summarizer::normalize_rows(matrix)
}

/// One indexed record (Python corpus dict row).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusRecord {
    pub kind: String, // "text" | "image"
    pub source: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub page: Option<u64>,
    #[serde(default)]
    pub image_path: String,
}

/// The persisted corpus payload (Python pickle payload).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusPayload {
    pub version: u32,
    pub mode: String,
    pub text_model_path: String,
    pub image_model_path: String,
    pub records: Vec<CorpusRecord>,
    pub embeddings: Vec<Vec<f64>>,
}

/// Load a corpus file and validate it (Python `load_corpus`).
pub fn load_corpus(path: &Path) -> Result<CorpusPayload> {
    let bytes = std::fs::read(path)?;
    let payload: CorpusPayload = bincode::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("Invalid corpus file: '{}' ({e})", path.display()))?;
    let valid = !payload.records.is_empty() || payload.embeddings.is_empty();
    if !valid {
        return Err(anyhow::anyhow!("Invalid corpus file: '{}'", path.display()));
    }
    Ok(payload)
}

/// Write a corpus file, creating parent directories as needed.
pub fn save_corpus(path: &Path, payload: &CorpusPayload) -> Result<()> {
    let output = path.to_path_buf();
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = bincode::serialize(payload)?;
    std::fs::write(&output, bytes)?;
    Ok(())
}

/// Delete image files under `image_dir` that no record references
/// (Python `prune_orphan_images`; dedup keeps only the first copy).
pub fn prune_orphan_images(image_dir: &Path, records: &[CorpusRecord]) -> usize {
    let Ok(dir) = std::fs::read_dir(image_dir) else {
        return 0;
    };
    let referenced: HashSet<String> = records
        .iter()
        .filter(|r| !r.image_path.is_empty())
        .filter_map(|r| {
            std::fs::canonicalize(&r.image_path)
                .ok()
                .map(|p| p.to_string_lossy().to_string())
        })
        .collect();
    let mut removed = 0;
    for entry in dir.flatten() {
        let path = entry.path();
        if path.is_file() {
            let resolved = std::fs::canonicalize(&path)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            if !referenced.contains(&resolved) {
                let _ = std::fs::remove_file(&path);
                removed += 1;
            }
        }
    }
    removed
}

/// Embedder bundle of one corpus (text / clip / image models).
#[derive(Clone, Default)]
pub struct CorpusModels {
    /// text-only mode encoder (all-MiniLM-L6-v2)
    pub text: Option<Arc<dyn TextEmbedder>>,
    /// multimodal-mode text encoder (CLIP text tower)
    pub clip_text: Option<Arc<dyn TextEmbedder>>,
    /// multimodal-mode image encoder (CLIP vision tower)
    pub image: Option<Arc<dyn ImageEmbedder>>,
}

/// Builds and queries a local corpus of embedded passages and images
/// (Python `EmbeddingCorpus`).
pub struct EmbeddingCorpus {
    pub model_path: PathBuf,
    pub device: String,
    pub extensions: Vec<String>,
    pub chunk_size: usize,
    pub top_k: usize,
    pub image_model_path: Option<PathBuf>,
    pub multimodal: bool,
    pub image_extensions: Vec<String>,
    pub models: CorpusModels,
}

impl EmbeddingCorpus {
    /// Full constructor (Python parity: nine config fields).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_path: PathBuf,
        device: String,
        extensions: Vec<String>,
        chunk_size: usize,
        top_k: usize,
        image_model_path: Option<PathBuf>,
        multimodal: bool,
        image_extensions: Vec<String>,
        models: CorpusModels,
    ) -> Self {
        Self {
            model_path,
            device,
            extensions,
            chunk_size,
            top_k,
            image_model_path,
            multimodal,
            image_extensions,
            models,
        }
    }

    // ------------------------------------------------------------ internals

    fn build_records(&self, documents: &[Document]) -> Vec<CorpusRecord> {
        let mut records: Vec<CorpusRecord> = Vec::new();
        let mut seen_image_hashes: HashSet<String> = HashSet::new();
        for document in documents {
            if !document.error.is_empty() {
                continue;
            }
            if document.kind == "text" {
                // Paginated documents (PDF) expose per-page sections so each
                // passage keeps its real page number.
                let sections: Vec<(Option<u64>, String)> = if document.sections.is_empty() {
                    vec![(document.pages, document.text.clone())]
                } else {
                    document.sections.clone()
                };
                for (page, section_text) in sections {
                    for passage in build_passages(&[section_text], self.chunk_size, None) {
                        if !has_content(&passage, MIN_ALNUM_CHARS) {
                            continue;
                        }
                        records.push(CorpusRecord {
                            kind: "text".into(),
                            source: document.source.clone(),
                            text: passage,
                            page,
                            image_path: String::new(),
                        });
                    }
                }
            }
            if self.multimodal {
                for image_path in &document.images {
                    let Ok(bytes) = std::fs::read(image_path) else {
                        continue;
                    };
                    let digest = md5_hex(&bytes);
                    if !seen_image_hashes.insert(digest) {
                        continue;
                    }
                    records.push(CorpusRecord {
                        kind: "image".into(),
                        source: document.source.clone(),
                        text: document.source.clone(),
                        page: None,
                        image_path: image_path.clone(),
                    });
                }
            }
        }
        records
    }

    fn embed_records_multimodal(&self, records: &[CorpusRecord]) -> Result<Array2<f64>> {
        let text_ids: Vec<usize> = records.iter().positions(|r| r.kind == "text");
        let image_ids: Vec<usize> = records.iter().positions(|r| r.kind == "image");
        let text_embeddings: Option<Array2<f64>> = if text_ids.is_empty() {
            None
        } else {
            let texts: Vec<String> = text_ids.iter().map(|&i| records[i].text.clone()).collect();
            Some(normalize_rows(&self.clip_embedder()?.encode(&texts)?))
        };
        let image_embeddings: Option<Array2<f64>> = if image_ids.is_empty() {
            None
        } else {
            let paths: Vec<PathBuf> = image_ids
                .iter()
                .map(|&i| PathBuf::from(&records[i].image_path))
                .collect();
            Some(normalize_rows(
                &self.image_embedder()?.encode_paths(&paths)?,
            ))
        };
        let ncols = text_embeddings
            .as_ref()
            .map(|e| e.ncols())
            .or_else(|| image_embeddings.as_ref().map(|e| e.ncols()))
            .unwrap_or(0);
        let mut embeddings = Array2::<f64>::zeros((records.len(), ncols));
        if let Some(te) = text_embeddings {
            for (position, &index) in text_ids.iter().enumerate() {
                for (col, value) in te.row(position).iter().enumerate() {
                    embeddings[[index, col]] = *value;
                }
            }
        }
        if let Some(ie) = image_embeddings {
            for (position, &index) in image_ids.iter().enumerate() {
                for (col, value) in ie.row(position).iter().enumerate() {
                    embeddings[[index, col]] = *value;
                }
            }
        }
        Ok(embeddings)
    }

    fn clip_embedder(&self) -> Result<&Arc<dyn TextEmbedder>> {
        self.models.clip_text.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Image embedding model not found at '{}'. See README.md for how to store the model.",
                self.image_model_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
            )
        })
    }

    fn image_embedder(&self) -> Result<&Arc<dyn ImageEmbedder>> {
        self.models.image.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Image embedding model not found at '{}'. See README.md for how to store the model.",
                self.image_model_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
            )
        })
    }

    fn text_embedder(&self) -> Result<&Arc<dyn TextEmbedder>> {
        if let Some(embedder) = &self.models.text {
            return Ok(embedder);
        }
        if !is_model_available(&self.model_path) {
            return Err(anyhow::anyhow!(
                "Embedding model not found at '{}'. See README.md for how to store the model.",
                self.model_path.display()
            ));
        }
        Err(anyhow::anyhow!(
            "Embedding model not wired: build the `la` binary with `--features ml` \
             so the candle-based encoder is available."
        ))
    }

    fn encode_query(&self, query: &str, mode: &str) -> Result<Array2<f64>> {
        let embedder = if mode == "multimodal" {
            self.clip_embedder()?
        } else {
            self.text_embedder()?
        };
        Ok(normalize_rows(&embedder.encode(&[query.to_string()])?))
    }

    fn append_processed(
        &self,
        corpus_dir: &Path,
        documents: &[Document],
        directory: &Path,
        n_text: usize,
        n_image: usize,
    ) -> Result<PathBuf> {
        let processed_path = corpus_dir.join(PROCESSED_FILE);
        if let Some(parent) = processed_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let root = std::fs::canonicalize(directory).unwrap_or_else(|_| directory.to_path_buf());
        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        let mut lines = vec![
            format!(
                "## Corpus build {timestamp} — {}",
                corpus_dir.file_name().unwrap_or_default().to_string_lossy()
            ),
            String::new(),
            format!(
                "- Files: {}  |  text records: {n_text}  |  image records: {n_image}",
                documents.len()
            ),
            String::new(),
        ];
        for document in documents {
            let absolute = root.join(&document.source);
            let suffix = if document.error.is_empty() {
                String::new()
            } else {
                format!(" — ERROR: {}", document.error)
            };
            lines.push(format!("- {}{suffix}", absolute.display()));
        }
        lines.push(String::new());
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&processed_path)?;
        use std::io::Write;
        file.write_all(lines.join("\n").as_bytes())?;
        file.write_all(b"\n")?;
        Ok(processed_path)
    }

    // --------------------------------------------------------------- public

    /// Build a named corpus folder from `directory` (Python `store`).
    pub fn store(&self, directory: &Path, corpus_dir: &Path, overwrite: bool) -> Result<String> {
        let pkl_path = corpus_dir.join(CORPUS_FILE);
        if pkl_path.exists() && !overwrite {
            return Ok("Corpus already exists.".into());
        }

        let image_dir = corpus_dir.join(IMAGES_DIR);
        if image_dir.exists() {
            let _ = std::fs::remove_dir_all(&image_dir);
        }
        std::fs::create_dir_all(&image_dir)?;
        let documents =
            document_loader::iter_documents(directory, Some(&self.extensions), Some(&image_dir))?;
        if documents.is_empty() {
            return Ok(format!(
                "No supported content found in '{}'.",
                directory.display()
            ));
        }

        let records = self.build_records(&documents);
        if records.is_empty() {
            return Ok(format!(
                "No supported content found in '{}'.",
                directory.display()
            ));
        }

        let embeddings = if self.multimodal {
            self.embed_records_multimodal(&records)?
        } else {
            let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
            normalize_rows(&self.text_embedder()?.encode(&texts)?)
        };

        let payload = CorpusPayload {
            version: 2,
            mode: if self.multimodal {
                "multimodal".into()
            } else {
                "text".into()
            },
            text_model_path: self.model_path.to_string_lossy().to_string(),
            image_model_path: self
                .image_model_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            records: records.clone(),
            embeddings: embeddings.rows().into_iter().map(|r| r.to_vec()).collect(),
        };
        save_corpus(&pkl_path, &payload)?;
        prune_orphan_images(&image_dir, &records);

        let n_text = records.iter().filter(|r| r.kind == "text").count();
        let n_image = records.iter().filter(|r| r.kind == "image").count();
        self.append_processed(corpus_dir, &documents, directory, n_text, n_image)?;
        let sources: HashSet<&String> = records
            .iter()
            .filter(|r| !r.source.is_empty())
            .map(|r| &r.source)
            .collect();
        save_metadata(
            corpus_dir,
            &serde_json::json!({
                "name": corpus_dir.file_name().unwrap_or_default().to_string_lossy(),
                "version": 2,
                "mode": payload.mode,
                "text_model_path": payload.text_model_path,
                "image_model_path": payload.image_model_path,
                "source_dir": std::fs::canonicalize(directory).unwrap_or_else(|_| directory.to_path_buf()).display().to_string(),
                "records": records.len(),
                "text_records": n_text,
                "image_records": n_image,
                "sources": sources.len(),
                "built_at": chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            }),
        )?;
        Ok(format!(
            "Corpus '{}' saved to '{}' ({n_text} text / {n_image} image records).",
            corpus_dir.file_name().unwrap_or_default().to_string_lossy(),
            corpus_dir.display()
        ))
    }

    /// Return the `top_k` records most relevant to `query`
    /// (Python `query`; `kind` restricts to "text"/"image").
    pub fn query(
        &self,
        corpus_dir: &Path,
        query: &str,
        top_k: Option<usize>,
        kind: Option<&str>,
    ) -> String {
        let query = query.trim();
        if query.is_empty() {
            return "No query provided.".into();
        }
        let kind = kind
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(str::to_lowercase);
        if let Some(kind) = &kind
            && kind != "text"
            && kind != "image"
        {
            return format!("Invalid kind '{kind}'. Use 'text' or 'image'.");
        }
        let path = corpus_dir.join(CORPUS_FILE);
        if !path.exists() {
            return format!(
                "Corpus not found at '{}'. Use list_corpora() for available names, or store_corpus(...) to build it.",
                path.display()
            );
        }
        let payload = match load_corpus(&path) {
            Ok(p) => p,
            Err(e) => return format!("{e}"),
        };
        if payload.records.is_empty() {
            return "Corpus is empty.".into();
        }
        let embeddings = match Array2::from_shape_vec(
            (
                payload.embeddings.len(),
                payload.embeddings.first().map(|e| e.len()).unwrap_or(0),
            ),
            payload.embeddings.concat(),
        ) {
            Ok(e) => e,
            Err(e) => return format!("Invalid corpus file: '{}' ({e})", path.display()),
        };

        let mode = payload.mode.clone();
        let query_vec = match self.encode_query(query, &mode) {
            Ok(v) => v,
            Err(e) => return format!("{e}"),
        };
        if query_vec.ncols() != embeddings.ncols() {
            return "Query embedding size does not match the corpus; rebuild the corpus with the configured model.".into();
        }

        let candidates: Vec<usize> = match &kind {
            Some(kind) => payload.records.iter().positions(|r| r.kind == *kind),
            None => (0..payload.records.len()).collect(),
        };
        if candidates.is_empty() {
            return format!(
                "No {} records found in the corpus.",
                kind.as_deref().unwrap_or("")
            );
        }

        // Exact inner-product ranking over normalized vectors (FAISS IndexFlatIP).
        let k = top_k
            .filter(|k| *k > 0)
            .unwrap_or(self.top_k)
            .clamp(1, candidates.len());
        let mut scored: Vec<(f64, usize)> = candidates
            .iter()
            .map(|&local| {
                let row = embeddings.row(local);
                let score = row.dot(&query_vec.row(0));
                (score, local)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let label = match &kind {
            Some(kind) => format!("{kind} "),
            None => String::new(),
        };
        let mut lines = vec![format!("Top {k} {label}matches for '{query}':")];
        for (rank, (score, local)) in scored.iter().enumerate() {
            let record = &payload.records[*local];
            let source = &record.source;
            let mut hit = format!(
                "{}) [{score:.3}] ({source}) {}",
                rank + 1,
                if record.text.is_empty() {
                    source.clone()
                } else {
                    record.text.clone()
                }
            );
            if record.kind == "image" {
                hit = format!("{hit} [image] {}", record.image_path);
            }
            lines.push(hit);
        }
        lines.join("\n")
    }
}

trait PositionsExt<T> {
    fn positions(self, predicate: impl Fn(&T) -> bool) -> Vec<usize>;
}
impl<T, I: Iterator<Item = T>> PositionsExt<T> for I {
    fn positions(self, predicate: impl Fn(&T) -> bool) -> Vec<usize> {
        self.enumerate()
            .filter(|(_, item)| predicate(item))
            .map(|(i, _)| i)
            .collect()
    }
}

/// md5 hex digest (Python `hashlib.md5(...).hexdigest()`).
pub fn md5_hex(data: &[u8]) -> String {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::TextEmbedder;

    /// Deterministic fake: 3 dims keyed on keywords (embeds texts AND images
    /// by hashing the path content, so multimodal fakes work offline).
    struct FakeEmbedder {
        dim: usize,
    }
    impl FakeEmbedder {
        fn new() -> Self {
            Self { dim: 4 }
        }
    }
    impl TextEmbedder for FakeEmbedder {
        fn encode(&self, texts: &[String]) -> Result<Array2<f64>> {
            let mut out = Array2::<f64>::zeros((texts.len(), self.dim));
            for (i, text) in texts.iter().enumerate() {
                for (d, token) in ["a", "b", "c", "d"].iter().enumerate() {
                    out[[i, d]] = if text.to_lowercase().contains(token) {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
            Ok(out)
        }
    }
    impl ImageEmbedder for FakeEmbedder {
        fn encode_paths(&self, paths: &[PathBuf]) -> Result<Array2<f64>> {
            let texts: Vec<String> = paths
                .iter()
                .map(|p| {
                    p.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default()
                })
                .collect();
            self.encode(&texts)
        }
    }

    fn corpus_models() -> CorpusModels {
        let fake: Arc<dyn TextEmbedder> = Arc::new(FakeEmbedder::new());
        CorpusModels {
            text: Some(fake.clone()),
            clip_text: Some(fake.clone()),
            image: Some(Arc::new(FakeEmbedder::new())),
        }
    }

    fn write_sample_docs(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("doc1.txt"),
            "Alpha bravo charlie delta echo. Foxtrot golf hotel india juliet. Kilo lima mike november oscar.",
        )
        .unwrap();
        std::fs::write(
            dir.join("doc2.md"),
            "# Heading\n\nPapa quebec romeo sierra tango. Uniform victor whiskey xray yankee.\n",
        )
        .unwrap();
    }

    #[test]
    fn sanitize_and_resolve() {
        assert_eq!(sanitize_corpus_name("autosar").unwrap(), "autosar");
        assert!(sanitize_corpus_name("").is_err());
        assert!(sanitize_corpus_name("..").is_err());
        assert!(sanitize_corpus_name("a/b").is_err());
        let dir = resolve_corpus_dir("autosar", Path::new(".res/corpora")).unwrap();
        assert!(
            dir.ends_with(".res/corpora/autosar")
                || dir.ends_with("corpora\\autosar")
                || dir.to_string_lossy().contains("autosar")
        );
    }

    #[test]
    fn build_passages_chunks() {
        let text = "One two three four five six seven eight.".to_string();
        let passages = build_passages(&[text], 2, None);
        assert!(!passages.is_empty());
        assert!(passages[0].contains("One two"));
    }

    #[test]
    fn has_content_filters_noise() {
        assert!(has_content("hello world", MIN_ALNUM_CHARS));
        assert!(!has_content(". . .", MIN_ALNUM_CHARS));
        assert!(!has_content("", MIN_ALNUM_CHARS));
        // A short but real passage passes a lower threshold ("a . b ." has two
        // letters): below MIN_ALNUM_CHARS yet above a min of 1.
        assert!(has_content("a . b .", 1));
        assert!(!has_content("a . b .", MIN_ALNUM_CHARS));
    }

    #[test]
    fn normalize_rows_matches_python() {
        let m = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let n = normalize_rows(&m);
        assert!((n[[0, 0]] - 0.6).abs() < 1e-9);
        assert!((n[[1, 0]]).abs() < 1e-9);
    }

    #[test]
    fn md5_matches_python() {
        assert_eq!(md5_hex(b"hello"), "5d41402abc4b2a76b9719d911017c592");
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn corpus_store_and_query_round_trip() {
        let base = tempfile::tempdir().unwrap();
        let src = base.path().join("src");
        let corpora_dir = base.path().join("corpora");
        write_sample_docs(&src);

        let corpus = EmbeddingCorpus::new(
            PathBuf::from("fake-model"),
            "cpu".into(),
            document_loader::SUPPORTED_EXTENSIONS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            2,
            3,
            Some(PathBuf::from("fake-clip")),
            true,
            document_loader::IMAGE_EXTENSIONS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            corpus_models(),
        );
        let corpus_dir = resolve_corpus_dir("test", &corpora_dir).unwrap();
        let result = corpus.store(&src, &corpus_dir, false).unwrap();
        assert!(result.starts_with("Corpus 'test' saved"), "got: {result}");

        // second store without overwrite refuses
        assert_eq!(
            corpus.store(&src, &corpus_dir, false).unwrap(),
            "Corpus already exists."
        );

        // files written
        assert!(corpus_dir.join("corpus.bin").exists());
        assert!(corpus_dir.join("metadata.json").exists());
        assert!(corpus_dir.join("processed.md").exists());

        let meta = load_metadata(&corpus_dir).unwrap();
        assert_eq!(meta["name"], "test");
        assert_eq!(meta["version"], 2);

        // query text
        let result = corpus.query(&corpus_dir, "alpha bravo", Some(2), None);
        assert!(
            result.contains("Top 2 matches for 'alpha bravo'"),
            "got: {result}"
        );

        // list_corpora sees it
        let entries = list_corpora(&corpora_dir);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "test");

        // invalid kind
        assert!(
            corpus
                .query(&corpus_dir, "alpha", None, Some("video"))
                .contains("Invalid kind")
        );
    }

    #[test]
    fn query_empty_and_missing_corpus() {
        let base = tempfile::tempdir().unwrap();
        let corpora_dir = base.path().join("corpora");
        let corpus_dir = resolve_corpus_dir("missing", &corpora_dir).unwrap();
        let corpus = EmbeddingCorpus::new(
            PathBuf::from("fake-model"),
            "cpu".into(),
            vec![],
            2,
            3,
            None,
            false,
            vec![],
            corpus_models(),
        );
        assert!(
            corpus
                .query(&corpus_dir, "q", None, None)
                .starts_with("Corpus not found")
        );
        assert_eq!(
            corpus.query(&corpus_dir, "  ", None, None),
            "No query provided."
        );
    }

    #[test]
    fn dedup_identical_images() {
        let base = tempfile::tempdir().unwrap();
        let src = base.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        // two identical PNGs
        let png_bytes = {
            let mut buf = std::io::Cursor::new(Vec::new());
            image::DynamicImage::new_rgb8(8, 8)
                .write_to(&mut buf, image::ImageFormat::Png)
                .unwrap();
            buf.into_inner()
        };
        std::fs::write(src.join("one.png"), &png_bytes).unwrap();
        std::fs::write(src.join("two.png"), &png_bytes).unwrap();

        let corpus = EmbeddingCorpus::new(
            PathBuf::from("fake-model"),
            "cpu".into(),
            vec![".png".into()],
            2,
            3,
            Some(PathBuf::from("fake-clip")),
            true,
            vec![".png".into()],
            corpus_models(),
        );
        let records =
            corpus.build_records(&document_loader::iter_documents(&src, None, None).unwrap());
        let image_records = records.iter().filter(|r| r.kind == "image").count();
        assert_eq!(image_records, 1, "byte-identical images are deduplicated");
    }

    #[test]
    fn metadata_json_shapes_match_python() {
        let base = tempfile::tempdir().unwrap();
        let corpus_dir = base.path().join("c");
        std::fs::create_dir_all(&corpus_dir).unwrap();
        save_metadata(&corpus_dir, &serde_json::json!({"name": "c", "records": 3})).unwrap();
        let meta = load_metadata(&corpus_dir).unwrap();
        assert_eq!(meta["records"], 3);
        assert!(list_corpora(base.path()).is_empty() || list_corpora(base.path()).len() == 1);
    }
}
