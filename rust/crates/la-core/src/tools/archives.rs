//! Archive tools (Python `src/tools/archives.py`): extract zip/7z/tar/gz
//! archives and zip folder contents. Pure-Rust codecs (flate2 / bzip2-rs /
//! lzma-rs) keep the build toolchain-free on Windows.

use crate::log_call;
use crate::tools::files::normalize_extensions;
use crate::tools::registry::{ToolDef, ToolOutput};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const ZIP_SUFFIXES: [&str; 1] = [".zip"];
const TAR_SUFFIXES: [&str; 7] = [
    ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz",
];
const SUPPORTED_ARCHIVES: &str = "zip, 7z, tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz and gz";

/// `foo.tar.gz` -> `foo`; `foo.zip` -> `foo`.
fn archive_stem(path: &Path) -> String {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    for suffix in TAR_SUFFIXES
        .iter()
        .chain(ZIP_SUFFIXES.iter())
        .chain([".7z", ".gz"].iter())
    {
        if name.ends_with(suffix) {
            let stem = &path.file_name().unwrap_or_default().to_string_lossy();
            return stem[..stem.len() - suffix.len()].to_string();
        }
    }
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

/// Map an archive path to its extractor: 'zip', 'tar', '7z', 'gz' or ''.
fn archive_kind(path: &Path) -> &'static str {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    if ZIP_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return "zip";
    }
    if TAR_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return "tar";
    }
    if name.ends_with(".7z") {
        return "7z";
    }
    if name.ends_with(".gz") {
        return "gz";
    }
    ""
}

fn is_relative_to(target: &Path, root: &Path) -> bool {
    target.starts_with(root)
}

/// Lexically resolve `.` / `..` components without touching the filesystem
/// (archive entries don't exist yet, so `canonicalize` is unavailable, and on
/// Windows it would also add a verbatim `\\?\` prefix that breaks
/// `starts_with` comparisons against a non-canonicalized destination).
/// Returns `None` when the path escapes above its root (drive-relative `..`).
fn lexical_normalize(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                if !out.pop() {
                    return None; // `..` above the root (escapes the destination)
                }
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    Some(out)
}

/// Extract a zip archive, rejecting entries that escape `dest` (zip slip).
fn extract_zip(path: &Path, dest: &Path) -> anyhow::Result<usize> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let root = lexical_normalize(dest)
        .ok_or_else(|| anyhow::anyhow!("unsafe archive destination '{}'", dest.display()))?;
    let members = archive.len();
    for index in 0..members {
        let name = archive
            .name_for_index(index)
            .unwrap_or_default()
            .to_string();
        // Python parity: `target = (dest / member.filename).resolve()` then
        // `target.is_relative_to(root)` — containment is checked lexically so
        // it also catches absolute entries and `../` traversal before anything
        // is written to disk.
        let target = lexical_normalize(&dest.join(&name))
            .ok_or_else(|| anyhow::anyhow!("unsafe archive entry '{name}'"))?;
        if !is_relative_to(&target, &root) {
            return Err(anyhow::anyhow!("unsafe archive entry '{name}'"));
        }
    }
    archive.extract(dest)?;
    Ok(members)
}

/// Extract a tar archive (tar/tar.gz/tbz2/txz) with a traversal filter
/// (Python `filter='data'`, PEP 706 parity).
fn extract_tar(path: &Path, dest: &Path) -> anyhow::Result<usize> {
    let file = std::fs::File::open(path)?;
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    let reader: Box<dyn Read> = if name.ends_with(".gz") || name.ends_with(".tgz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else if name.ends_with(".bz2") || name.ends_with(".tbz2") {
        Box::new(bzip2_rs::DecoderReader::new(file))
    } else if name.ends_with(".xz") || name.ends_with(".txz") {
        // lzma-rs exposes a one-shot decoder; decompress fully (tar.xz tool
        // archives are moderate) and stream from memory.
        let mut decompressed: Vec<u8> = Vec::new();
        lzma_rs::xz_decompress(&mut std::io::BufReader::new(file), &mut decompressed)?;
        Box::new(std::io::Cursor::new(decompressed))
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(reader);
    // 'data' filter parity: reject entries escaping the destination.
    let entries = archive.entries()?;
    let mut count = 0usize;
    for entry in entries {
        let mut entry = match entry {
            Ok(entry) => entry,
            Err(e) => return Err(anyhow::anyhow!("tar entry: {e}")),
        };
        let path = entry.path()?.to_path_buf();
        if path.is_absolute()
            || path
                .components()
                .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(anyhow::anyhow!("unsafe archive entry '{}'", path.display()));
        }
        let target = dest.join(&path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if entry.header().entry_type().is_file() {
            let mut out = std::fs::File::create(&target)?;
            std::io::copy(&mut entry, &mut out)?;
        } else if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&target)?;
        }
        count += 1;
    }
    Ok(count)
}

/// Extract a 7z archive (pure-Rust `sevenz-rust2`).
fn extract_7z(path: &Path, dest: &Path) -> anyhow::Result<usize> {
    sevenz_rust2::decompress_file(path, dest)
        .map_err(|e| anyhow::anyhow!("7z extraction failed: {e}"))?;
    // count the extracted entries from the destination listing
    Ok(std::fs::read_dir(dest).map(|d| d.count()).unwrap_or(0))
}

/// Decompress a bare `.gz` file (not a tar) into `dest/<stem>`.
fn extract_gz(path: &Path, dest: &Path) -> anyhow::Result<usize> {
    let target = dest.join(archive_stem(path));
    let file = std::fs::File::open(path)?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut output = std::fs::File::create(&target)?;
    std::io::copy(&mut decoder, &mut output)?;
    Ok(1)
}

/// Unzip / unpack a compressed archive and save its content into a folder.
/// Useful when user asks to extract, unzip, unpack or decompress an archive.
/// Supported formats: zip, 7z, tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz and
/// single-file gz. By default the content is extracted into a new folder named
/// after the archive (e.g. `data.zip` -> `data/`); `out_dir` overrides it.
/// A non-empty destination is only written when `overwrite` is True. Returns
/// the destination folder and the number of extracted entries.
pub fn extract_archive(archive: &str, out_dir: Option<&str>, overwrite: bool) -> String {
    log_call!("extract_archive", "archive" => archive, "out_dir" => out_dir.unwrap_or_default(), "overwrite" => overwrite);
    let archive_path = PathBuf::from(archive);
    if !archive_path.exists() {
        return format!("Error: '{}' does not exist!", archive_path.display());
    }
    if archive_path.is_dir() {
        return format!("Error: '{}' is a directory!", archive_path.display());
    }
    let kind = archive_kind(&archive_path);
    if kind.is_empty() {
        return format!(
            "Error: unsupported archive type '{}'. Supported: {SUPPORTED_ARCHIVES}.",
            archive_path
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
        );
    }

    let dest: PathBuf = if kind == "gz" {
        // single decompressed file, extracted beside the archive
        match out_dir {
            Some(dir) => PathBuf::from(dir),
            None => archive_path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
        }
    } else {
        match out_dir {
            Some(dir) => PathBuf::from(dir),
            None => archive_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(archive_stem(&archive_path)),
        }
    };
    if kind == "gz" {
        if dest.join(archive_stem(&archive_path)).exists() && !overwrite {
            return "File already exists.".into();
        }
    } else if dest.exists()
        && std::fs::read_dir(&dest)
            .map(|d| d.count() > 0)
            .unwrap_or(false)
        && !overwrite
    {
        return format!(
            "Destination '{}' is not empty. Use overwrite=True to extract into it.",
            dest.display()
        );
    }
    let _ = std::fs::create_dir_all(&dest);
    let entries = match kind {
        "zip" => extract_zip(&archive_path, &dest),
        "tar" => extract_tar(&archive_path, &dest),
        "7z" => extract_7z(&archive_path, &dest),
        "gz" => extract_gz(&archive_path, &dest),
        _ => unreachable!(),
    };
    match entries {
        Err(e) => format!("Error extracting '{}': {e}", archive_path.display()),
        Ok(entries) => format!(
            "Extracted {entries} entries from '{}' to '{}'",
            archive_path.display(),
            dest.display()
        ),
    }
}

/// Every file under `directory` (recursively, sorted) to pack into a zip.
/// Mirrors `search_pattern_in_files`: hidden entries matched by the exclude
/// glob are skipped, extension filters are optional comma lists.
fn collect_zip_files(
    directory: &Path,
    include_extensions: &[String],
    exclude_extensions: &[String],
    excluded_items: &[String],
    skip_path: &Path,
) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for path in walkdir::WalkDir::new(directory)
        .sort_by_file_name()
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = path.path();
        if !path.is_file() {
            continue;
        }
        if path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
            == skip_path
                .canonicalize()
                .unwrap_or_else(|_| skip_path.to_path_buf())
        {
            continue;
        }
        if excluded_items
            .iter()
            .any(|item| path.iter().any(|c| c == std::ffi::OsStr::new(item)))
        {
            continue;
        }
        let ext = path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        if !include_extensions.is_empty() && !include_extensions.contains(&ext) {
            continue;
        }
        if !exclude_extensions.is_empty() && exclude_extensions.contains(&ext) {
            continue;
        }
        files.push(path.to_path_buf());
    }
    files
}

/// Zip the files of a folder into a zip archive (recursively, keeping the
/// relative paths). Useful when user asks to compress, zip or pack a folder or
/// its files. `include_extensions`/`exclude_extensions` accept `py` or
/// `.py` (empty = all files); `exclude_glob` (default `".*"`) skips
/// hidden entries. A `.zip` suffix is appended to `archive` when missing,
/// and existing archives are only replaced when `overwrite` is True. Returns
/// the archive path and the number of files zipped.
pub fn zip_files(
    directory: &str,
    archive: &str,
    exclude_glob: Option<&str>,
    include_extensions: &str,
    exclude_extensions: &str,
    overwrite: bool,
) -> String {
    log_call!("zip_files", "directory" => directory, "archive" => archive, "exclude_glob" => exclude_glob.unwrap_or_default(), "include_extensions" => include_extensions, "exclude_extensions" => exclude_extensions, "overwrite" => overwrite);
    let search_dir = PathBuf::from(directory);
    if !search_dir.exists() {
        return format!("'{}' does not exist!", search_dir.display());
    }
    if !search_dir.is_dir() {
        return format!("'{}' is not a directory!", search_dir.display());
    }

    let mut archive_path = PathBuf::from(archive);
    if archive_path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase() != "zip")
        .unwrap_or(true)
    {
        let suffix = archive_path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default();
        archive_path.set_file_name(format!(
            "{}{suffix}.zip",
            archive_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        ));
    }
    if archive_path.exists() && !overwrite {
        return "File already exists.".into();
    }

    let excluded_items: Vec<String> = match exclude_glob {
        Some(glob) if !glob.is_empty() => walkdir::WalkDir::new(&search_dir)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .skip(1)
            .filter(|e| crate::tools::files::glob_match(glob, &e.file_name().to_string_lossy()))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect(),
        _ => Vec::new(),
    };
    let files = collect_zip_files(
        &search_dir,
        &normalize_extensions(include_extensions),
        &normalize_extensions(exclude_extensions),
        &excluded_items,
        &archive_path,
    );
    if files.is_empty() {
        return format!("No files to zip in '{}'.", search_dir.display());
    }

    let mut total_bytes = 0u64;
    let result = (|| -> anyhow::Result<()> {
        if let Some(parent) = archive_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&archive_path)?;
        let mut zip = zip::ZipWriter::new(file);
        let options: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for path in &files {
            let arcname = path.strip_prefix(&search_dir)?;
            zip.start_file(arcname.to_string_lossy().to_string(), options)?;
            let bytes = std::fs::read(path)?;
            total_bytes += bytes.len() as u64;
            std::io::Write::write_all(&mut zip, &bytes)?;
        }
        zip.finish()?;
        Ok(())
    })();
    match result {
        Err(e) => format!("Error zipping: {e}"),
        Ok(_) => format!(
            "Zipped {} files ({total_bytes} bytes) from '{}' to '{}'.",
            files.len(),
            search_dir.display(),
            archive_path.display()
        ),
    }
}

// ------------------------------------------------------------- registry

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "extract_archive".into(),
            description: "Unzip / unpack a compressed archive and save its content into a folder.\nUseful when user asks to extract, unzip, unpack or decompress an archive.\nSupported formats: zip, 7z, tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz and\nsingle-file gz. By default the content is extracted into a new folder named\nafter the archive (e.g. `data.zip` -> `data/`); `out_dir` overrides it.\nA non-empty destination is only written when `overwrite` is True. Returns\nthe destination folder and the number of extracted entries.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "archive": {"type": "string"},
                    "out_dir": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": false}
                },
                "required": ["archive"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let archive = args.get("archive").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let out_dir = args.get("out_dir").and_then(|v| v.as_str()).map(str::to_string);
                let overwrite = args.get("overwrite").and_then(|v| v.as_bool()).unwrap_or(false);
                Ok(ToolOutput::text(extract_archive(&archive, out_dir.as_deref(), overwrite)))
            })),
        },
        ToolDef {
            name: "zip_files".into(),
            description: "Zip the files of a folder into a zip archive (recursively, keeping the\nrelative paths). Useful when user asks to compress, zip or pack a folder or\nits files. `include_extensions`/`exclude_extensions` accept `py` or\n`.py` (empty = all files); `exclude_glob` (default `\".*\"`) skips\nhidden entries. A `.zip` suffix is appended to `archive` when missing,\nand existing archives are only replaced when `overwrite` is True. Returns\nthe archive path and the number of files zipped.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "archive": {"type": "string"},
                    "exclude_glob": {"type": "string", "default": ".*"},
                    "include_extensions": {"type": "string", "default": ""},
                    "exclude_extensions": {"type": "string", "default": ""},
                    "overwrite": {"type": "boolean", "default": false}
                },
                "required": ["directory", "archive"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let directory = args.get("directory").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let archive = args.get("archive").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let exclude_glob = args.get("exclude_glob").and_then(|v| v.as_str()).unwrap_or(".*").to_string();
                let include = args.get("include_extensions").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let exclude = args.get("exclude_extensions").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let overwrite = args.get("overwrite").and_then(|v| v.as_bool()).unwrap_or(false);
                Ok(ToolOutput::text(zip_files(&directory, &archive, Some(&exclude_glob), &include, &exclude, overwrite)))
            })),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_stem_and_kind() {
        let dir = tempfile::tempdir().unwrap();
        let tar_gz = dir.path().join("data.tar.gz");
        assert_eq!(archive_stem(&tar_gz), "data");
        assert_eq!(archive_kind(&tar_gz), "tar");
        let zip = dir.path().join("data.zip");
        assert_eq!(archive_stem(&zip), "data");
        assert_eq!(archive_kind(&zip), "zip");
        let sz = dir.path().join("data.7z");
        assert_eq!(archive_kind(&sz), "7z");
        let gz = dir.path().join("notes.txt.gz");
        assert_eq!(archive_stem(&gz), "notes.txt");
        assert_eq!(archive_kind(&gz), "gz");
    }

    #[test]
    fn zip_and_extract_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("a.txt"), "hello").unwrap();
        std::fs::write(src.join("sub").join("b.txt"), "world").unwrap();

        let archive = dir.path().join("out.zip");
        let result = zip_files(
            src.to_str().unwrap(),
            archive.to_str().unwrap(),
            Some(".*"),
            "",
            "",
            false,
        );
        assert!(result.starts_with("Zipped 2 files"), "got: {result}");
        assert!(archive.exists());

        // extract into a fresh destination folder
        let dest = dir.path().join("extracted");
        let result = extract_archive(archive.to_str().unwrap(), None, false);
        assert!(result.starts_with("Extracted 2 entries"), "got: {result}");
        let _ = dest;
        let extracted = dir.path().join("out");
        assert_eq!(
            std::fs::read_to_string(extracted.join("a.txt")).unwrap(),
            "hello"
        );
        assert_eq!(
            std::fs::read_to_string(extracted.join("sub").join("b.txt")).unwrap(),
            "world"
        );
    }

    #[test]
    fn gz_decompression() {
        let dir = tempfile::tempdir().unwrap();
        let gz_path = dir.path().join("notes.txt.gz");
        {
            let file = std::fs::File::create(&gz_path).unwrap();
            let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            std::io::Write::write_all(&mut encoder, b"gzip content").unwrap();
            encoder.finish().unwrap();
        }
        let result = extract_archive(gz_path.to_str().unwrap(), None, false);
        assert!(result.starts_with("Extracted 1 entries"), "got: {result}");
        let out = dir.path().join("notes.txt");
        assert_eq!(std::fs::read_to_string(out).unwrap(), "gzip content");
    }

    #[test]
    fn unsupported_kind() {
        let dir = tempfile::tempdir().unwrap();
        let rar = dir.path().join("data.rar");
        std::fs::write(&rar, b"junk").unwrap();
        let result = extract_archive(rar.to_str().unwrap(), None, false);
        assert!(result.starts_with("Error: unsupported archive type"));
    }
}
