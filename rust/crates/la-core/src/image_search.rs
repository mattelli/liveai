//! Image search in a folder with a CLIP model (`src/image_search.py` port).
//!
//! Every image under a directory is embedded with the CLIP model, the text
//! query is embedded into the same shared space, and cosine similarity
//! (L2-normalized vectors) ranks the images. The `top_k` best-matching image
//! paths are returned in descending similarity order. Image embeddings are
//! cached per (path, mtime), so repeated queries over an unchanged folder do
//! not re-encode the images; unreadable or corrupt files are skipped.

use crate::document_loader::IMAGE_EXTENSIONS;
use crate::embedding_search::normalize_rows;
use crate::encoders::{ImageEmbedder, TextEmbedder};
use anyhow::Result;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const DEFAULT_TOP_K: usize = 3;

/// Ranks the images of a folder against a text query with a CLIP model
/// (Python `ImageSearch`; `model_factory` seam).
pub struct ImageSearch {
    pub model_path: PathBuf,
    pub device: String,
    pub top_k: usize,
    pub extensions: Vec<String>,
    /// CLIP vision tower (image side).
    image: Option<Arc<dyn ImageEmbedder>>,
    /// CLIP text tower (query side; shares the embedding space).
    text: Option<Arc<dyn TextEmbedder>>,
    image_cache: Mutex<HashMap<String, (std::time::SystemTime, Vec<f64>)>>,
}

impl ImageSearch {
    pub fn new(
        model_path: PathBuf,
        device: String,
        top_k: usize,
        image: Option<Arc<dyn ImageEmbedder>>,
        text: Option<Arc<dyn TextEmbedder>>,
        extensions: Option<Vec<String>>,
    ) -> Self {
        Self {
            model_path,
            device,
            top_k: if top_k == 0 { DEFAULT_TOP_K } else { top_k },
            extensions: extensions
                .unwrap_or_else(|| IMAGE_EXTENSIONS.iter().map(|s| s.to_string()).collect())
                .into_iter()
                .map(|ext| {
                    let ext = ext.to_lowercase();
                    if let Some(stripped) = ext.strip_prefix('.') {
                        format!(".{stripped}")
                    } else {
                        format!(".{ext}")
                    }
                })
                .collect(),
            image,
            text,
            image_cache: Mutex::new(HashMap::new()),
        }
    }

    fn image_encoder(&self) -> Result<&Arc<dyn ImageEmbedder>> {
        self.image.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Image search model not found at '{}'. See README.md for how to store the model.",
                self.model_path.display()
            )
        })
    }

    fn text_encoder(&self) -> Result<&Arc<dyn TextEmbedder>> {
        self.text.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Image search model not found at '{}'. See README.md for how to store the model.",
                self.model_path.display()
            )
        })
    }

    /// Every supported image file under `directory` (recursively, sorted).
    fn collect_images(&self, directory: &Path) -> Vec<PathBuf> {
        walkdir::WalkDir::new(directory)
            .sort_by_file_name()
            .into_iter()
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_path_buf())
            .filter(|path| {
                path.is_file()
                    && path
                        .extension()
                        .map(|e| {
                            self.extensions
                                .contains(&format!(".{}", e.to_string_lossy().to_lowercase()))
                        })
                        .unwrap_or(false)
            })
            .collect()
    }

    /// Embed one image; `None` when the file cannot be opened or read.
    fn encode_image(&self, path: &Path) -> Option<Vec<f64>> {
        // Decode gate (Python `Image.open(...).convert("RGB")`): unreadable
        // or corrupt files are skipped before they reach the encoder. Cached
        // per (path, mtime) exactly like the embedding itself.
        let key = std::fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .to_string();
        let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
        {
            let cache = self.image_cache.lock().unwrap();
            if let Some((cached_mtime, vector)) = cache.get(&key)
                && *cached_mtime == mtime
            {
                return Some(vector.clone());
            }
        }
        if image::ImageReader::open(path)
            .ok()?
            .with_guessed_format()
            .ok()?
            .decode()
            .is_err()
        {
            return None;
        }
        let vector = self
            .image_encoder()
            .ok()?
            .encode_paths(&[path.to_path_buf()])
            .ok()?
            .row(0)
            .to_vec();
        self.image_cache
            .lock()
            .unwrap()
            .insert(key, (mtime, vector.clone()));
        Some(vector)
    }

    fn resolve_k(&self, top_k: Option<usize>, n_images: usize) -> usize {
        let k = top_k
            .filter(|k| *k > 0)
            .unwrap_or(self.top_k)
            .max(if top_k.is_none() { DEFAULT_TOP_K } else { 0 })
            .max(1);
        k.min(n_images).max(1)
    }

    /// Return the `(score, path)` pairs of the `top_k` best-matching images,
    /// ranked by cosine similarity to the embedded query, descending
    /// (Python `search`).
    pub fn search(
        &self,
        directory: &Path,
        query: &str,
        top_k: Option<usize>,
    ) -> Result<Vec<(f64, String)>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(anyhow::anyhow!("No query provided."));
        }
        if !directory.exists() {
            return Err(anyhow::anyhow!(
                "Directory not found: '{}'",
                directory.display()
            ));
        }
        if !directory.is_dir() {
            return Err(anyhow::anyhow!(
                "Not a directory: '{}'",
                directory.display()
            ));
        }

        let mut kept: Vec<PathBuf> = Vec::new();
        let mut vectors: Vec<Vec<f64>> = Vec::new();
        for path in self.collect_images(directory) {
            if let Some(vector) = self.encode_image(&path) {
                kept.push(path);
                vectors.push(vector);
            }
        }
        if kept.is_empty() {
            return Ok(Vec::new());
        }

        let ncols = vectors.first().map(|v| v.len()).unwrap_or(0);
        let flat: Vec<f64> = vectors.concat();
        let image_vecs = normalize_rows(&Array2::from_shape_vec((kept.len(), ncols.max(1)), flat)?);
        let query_vec = normalize_rows(&self.text_encoder()?.encode(&[query.to_string()])?)
            .row(0)
            .to_vec();

        // cosine similarity on L2-normalized vectors (inner product)
        let mut scored: Vec<(f64, usize)> = image_vecs
            .rows()
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                (
                    row.iter().zip(&query_vec).map(|(a, b)| a * b).sum::<f64>(),
                    index,
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.resolve_k(top_k, kept.len());
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(score, index)| {
                let path = std::fs::canonicalize(&kept[index])
                    .unwrap_or_else(|_| kept[index].clone())
                    .to_string_lossy()
                    .to_string();
                (score, path)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fake: image embedding = keyword flags of the file stem.
    struct FakeImageEmbedder;
    impl ImageEmbedder for FakeImageEmbedder {
        fn encode_paths(&self, paths: &[PathBuf]) -> Result<Array2<f64>> {
            let mut out = Array2::<f64>::zeros((paths.len(), 3));
            for (i, path) in paths.iter().enumerate() {
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
                out[[i, 0]] = if name.contains("dog") { 1.0 } else { 0.0 };
                out[[i, 1]] = if name.contains("cat") { 1.0 } else { 0.0 };
                out[[i, 2]] = if name.contains("snow") { 1.0 } else { 0.0 };
            }
            Ok(out)
        }
    }

    /// Same vector space keyed on the query text.
    struct FakeTextEmbedder;
    impl TextEmbedder for FakeTextEmbedder {
        fn encode(&self, texts: &[String]) -> Result<Array2<f64>> {
            let mut out = Array2::<f64>::zeros((texts.len(), 3));
            for (i, text) in texts.iter().enumerate() {
                let lower = text.to_lowercase();
                out[[i, 0]] = if lower.contains("dog") { 1.0 } else { 0.0 };
                out[[i, 1]] = if lower.contains("cat") { 1.0 } else { 0.0 };
                out[[i, 2]] = if lower.contains("snow") { 1.0 } else { 0.0 };
            }
            Ok(out)
        }
    }

    fn image_search() -> ImageSearch {
        ImageSearch::new(
            PathBuf::from("fake-model"),
            "cpu".into(),
            3,
            Some(Arc::new(FakeImageEmbedder)),
            Some(Arc::new(FakeTextEmbedder)),
            None,
        )
    }

    fn make_image(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        image::DynamicImage::new_rgb8(4, 4).save(&path).unwrap();
        path
    }

    #[test]
    fn search_ranks_dogs_first() {
        let dir = tempfile::tempdir().unwrap();
        make_image(dir.path(), "dog_snow.jpg");
        make_image(dir.path(), "cat_indoor.png");
        make_image(dir.path(), "plain.bmp");
        let search = image_search();
        let hits = search
            .search(dir.path(), "two dogs in the snow", None)
            .unwrap();
        assert_eq!(hits.len(), 3);
        assert!(hits[0].1.ends_with("dog_snow.jpg"), "got: {hits:?}");
        assert!(hits[0].0 > hits[1].0);
    }

    #[test]
    fn empty_query_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(image_search().search(dir.path(), "  ", None).is_err());
    }

    #[test]
    fn missing_directory_errors() {
        let err = image_search()
            .search(Path::new("no/such/dir"), "query", None)
            .unwrap_err();
        assert!(err.to_string().contains("Directory not found"));
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file.txt");
        std::fs::write(&file, "x").unwrap();
        let err = image_search().search(&file, "query", None).unwrap_err();
        assert!(err.to_string().contains("Not a directory"));
    }

    #[test]
    fn corrupt_files_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("corrupt.jpg"), b"not an image").unwrap();
        let search = image_search();
        let hits = search.search(dir.path(), "dog", None).unwrap();
        assert!(hits.is_empty(), "corrupt file must be skipped");
    }

    #[test]
    fn top_k_clamped() {
        let dir = tempfile::tempdir().unwrap();
        make_image(dir.path(), "dog1.png");
        make_image(dir.path(), "dog2.png");
        let hits = image_search().search(dir.path(), "dog", Some(99)).unwrap();
        assert_eq!(hits.len(), 2);
        let hits = image_search().search(dir.path(), "dog", Some(1)).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn image_cache_hit_avoids_reencode() {
        let dir = tempfile::tempdir().unwrap();
        make_image(dir.path(), "dog.png");
        let search = image_search();
        let first = search.search(dir.path(), "dog", None).unwrap();
        let cache_len = search.image_cache.lock().unwrap().len();
        assert_eq!(cache_len, 1);
        let second = search.search(dir.path(), "dog", None).unwrap();
        assert_eq!(first, second);
    }
}
