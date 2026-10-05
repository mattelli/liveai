//! File tools (Python `src/tools/files.py`): read/write text, search and
//! replace, plus the shared filename helpers used by web/archives/finance.

use crate::log_call;
use crate::tools::registry::{ToolDef, ToolOutput};
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Parse a comma separated extension list into lowercase dotted suffixes.
/// `"py, .MD ,txt"` -> `[".py", ".md", ".txt"]`.
pub fn normalize_extensions(value: &str) -> Vec<String> {
    let mut result = Vec::new();
    for item in value.split(',') {
        let item = item.trim().to_lowercase();
        if item.is_empty() {
            continue;
        }
        result.push(if let Some(stripped) = item.strip_prefix('.') {
            format!(".{stripped}")
        } else {
            format!(".{item}")
        });
    }
    result
}

/// Turn `^GSPC` / `BRK/B` into a filesystem-safe filename stem.
pub fn safe_filename(symbol: &str) -> String {
    let re = regex::Regex::new(r"[^A-Za-z0-9._-]+").unwrap();
    let cleaned = re.replace_all(symbol, "_");
    let cleaned = cleaned.trim_matches(|c| c == '.' || c == '_');
    if cleaned.is_empty() {
        "symbol".to_string()
    } else {
        cleaned.to_string()
    }
}

fn strip_component_of(path: &Path, component: &str) -> bool {
    path.iter().any(|c| c == std::ffi::OsStr::new(component))
}

/// Read a text file and return its contents as a string.
/// Useful when user asks for content description.
pub fn read_text_file(file: &str) -> String {
    let file_path = PathBuf::from(file);
    let file_path = if file_path.is_absolute() {
        file_path
    } else {
        std::env::current_dir().unwrap_or_default().join(&file_path)
    };
    if file_path.exists() && file_path.is_file() {
        return match std::fs::read_to_string(&file_path) {
            Ok(text) => text,
            Err(e) => format!("Error: {e:?}"),
        };
    }
    format!("Unable to read '{file}'")
}

/// Write a text file with the given its path and string content.
/// Useful when user asks for content generation, e.g. code generation.
/// Existing files are only replaced when `overwrite` is True.
pub fn write_text_file(file: &str, content: &str, overwrite: bool) -> String {
    let dest_path = PathBuf::from(file);
    let dest_path = if dest_path.is_absolute() {
        dest_path
    } else {
        std::env::current_dir().unwrap_or_default().join(&dest_path)
    };
    if dest_path.exists() {
        if dest_path.is_dir() {
            return format!("Error: '{}' is a directory.", dest_path.display());
        }
        if !overwrite {
            return "File already exists.".into();
        }
    }
    if let Some(parent) = dest_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&dest_path, content) {
        Ok(_) => format!("File created at '{}'", dest_path.display()),
        Err(e) => format!("Error: {e:?}"),
    }
}

/// Search for a pattern in a specified file.
/// Returns the line numbers where pattern is found.
pub fn search_in_file(pattern: &str, file: &str) -> String {
    let file_path = PathBuf::from(file);
    let file_path = if file_path.is_absolute() {
        file_path
    } else {
        std::env::current_dir().unwrap_or_default().join(&file_path)
    };
    if !file_path.exists() {
        return format!("Error: '{}' does not exist!", file_path.display());
    }
    if file_path.is_dir() {
        return format!("Error: '{}' is a directory!", file_path.display());
    }
    match std::fs::read_to_string(&file_path) {
        Err(e) => format!("Error reading '{}': {e}", file_path.display()),
        Ok(content) => {
            let found: Vec<String> = content
                .lines()
                .enumerate()
                .filter(|(_, line)| line.contains(pattern))
                .map(|(i, _)| (i + 1).to_string())
                .collect();
            if found.is_empty() {
                format!(
                    "Did not find pattern '{pattern}' in file '{}' ",
                    file_path.display()
                )
            } else {
                format!(
                    "Found pattern '{pattern}' in file '{}' at lines: {}\n",
                    file_path.display(),
                    found.join(", ")
                )
            }
        }
    }
}

/// Search for a pattern in files within a directory recursively.
/// Returns a list of the file paths and the line numbers where pattern is found.
/// `include_extensions`/`exclude_extensions` accept `py` or `.py`.
/// `exclude_glob` (default `".*"`) skips hidden entries.
pub fn search_pattern_in_files(
    pattern: &str,
    directory: &str,
    include_extensions: &str,
    exclude_extensions: &str,
    exclude_glob: Option<&str>,
) -> String {
    let include = normalize_extensions(include_extensions);
    let exclude = normalize_extensions(exclude_extensions);
    let mut results: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    let search_dir = PathBuf::from(directory);
    let search_dir = if search_dir.is_absolute() {
        search_dir
    } else {
        std::env::current_dir()
            .unwrap_or_default()
            .join(&search_dir)
    };
    if !search_dir.exists() {
        return format!("'{}' does not exist!", search_dir.display());
    }
    if !search_dir.is_dir() {
        return format!("'{}' is not a directory!", search_dir.display());
    }
    let excluded_items: Vec<String> = match exclude_glob {
        Some(glob) if !glob.is_empty() => walkdir::WalkDir::new(&search_dir)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .skip(1)
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                glob_match(glob, &name)
            })
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect(),
        _ => Vec::new(),
    };
    for entry in walkdir::WalkDir::new(&search_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        if !include.is_empty() && !include.contains(&ext) {
            continue;
        }
        if !exclude.is_empty() && exclude.contains(&ext) {
            continue;
        }
        if excluded_items
            .iter()
            .any(|item| strip_component_of(path, item))
        {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let lines: Vec<usize> = content
            .lines()
            .enumerate()
            .filter(|(_, line)| line.contains(pattern))
            .map(|(i, _)| i + 1)
            .collect();
        if !lines.is_empty() {
            results
                .entry(path.to_string_lossy().to_string())
                .or_default()
                .extend(lines);
        }
    }
    if results.is_empty() {
        return format!(
            "Did not find file having pattern '{pattern}' in directory '{}' ",
            search_dir.display()
        );
    }
    let mut out = format!(
        "Found '{}' files having pattern '{pattern}' in directory '{}' :\n",
        results.len(),
        search_dir.display()
    );
    for (path, lines) in &results {
        out.push_str(&format!(
            "- in file '{path}' at lines: {}\n",
            lines
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out
}

/// Tiny glob matcher supporting `*` and `?` (enough for the exclude globs).
pub fn glob_match(glob: &str, name: &str) -> bool {
    let re = Regex::new(&format!(
        "^{}$",
        regex::escape(glob).replace("\\*", ".*").replace("\\?", ".")
    ))
    .expect("glob regex");
    re.is_match(name)
}

/// Replace all literal occurrences of old_string in a file with new_string and
/// count replacements.
pub fn replace_string_in_file(path: &str, old_string: &str, new_string: &str) -> String {
    let file_path = PathBuf::from(path);
    let file_path = if file_path.is_absolute() {
        file_path
    } else {
        std::env::current_dir().unwrap_or_default().join(&file_path)
    };
    if file_path.is_dir() {
        return format!("Error: '{}' is a directory.", file_path.display());
    }
    match std::fs::read_to_string(&file_path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            format!("Error: File '{}' not found.", file_path.display())
        }
        Err(e) => format!("Error: {e}"),
        Ok(content) => {
            let count = content.matches(old_string).count();
            let replaced = content.replace(old_string, new_string);
            match std::fs::write(&file_path, replaced) {
                Ok(_) => format!(
                    "'{count}' occurrences of '{old_string}' have been replaced by '{new_string}'"
                ),
                Err(e) => format!("Error: {e}"),
            }
        }
    }
}

// ------------------------------------------------------------- registry

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "search_in_file".into(),
            description: "Search for a pattern in a specified file.\nReturns the line numbers where pattern is found.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "file": {"type": "string"}},
                "required": ["pattern", "file"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let file = args.get("file").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                log_call!("search_in_file", "pattern" => pattern, "file" => file);
                Ok(ToolOutput::text(search_in_file(&pattern, &file)))
            })),
        },
        ToolDef {
            name: "search_pattern_in_files".into(),
            description: "Search for a pattern in files within a directory recursively.\nReturns a list of the file paths and the line numbers where pattern is found.\n`include_extensions`/`exclude_extensions` accept `py` or `.py`.\n`exclude_glob` (default \".*\") skips hidden entries.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"},
                    "include_extensions": {"type": "string", "default": ""},
                    "exclude_extensions": {"type": "string", "default": ""},
                    "exclude_glob": {"type": "string", "default": ".*"}
                },
                "required": ["pattern", "directory"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let directory = args.get("directory").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let include = args.get("include_extensions").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let exclude = args.get("exclude_extensions").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let exclude_glob = args.get("exclude_glob").and_then(|v| v.as_str()).unwrap_or(".*").to_string();
                log_call!("search_pattern_in_files", "pattern" => pattern, "directory" => directory);
                Ok(ToolOutput::text(search_pattern_in_files(&pattern, &directory, &include, &exclude, Some(&exclude_glob))))
            })),
        },
        ToolDef {
            name: "read_text_file".into(),
            description: "Read a text file and return its contents as a string.\nUseful when user asks for content description.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"file": {"type": "string"}},
                "required": ["file"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let file = args.get("file").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                log_call!("read_text_file", "file" => file);
                Ok(ToolOutput::text(read_text_file(&file)))
            })),
        },
        ToolDef {
            name: "replace_string_in_file".into(),
            description: "Replace all literal occurrences of old_string in a file with new_string and count replacements.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"}
                },
                "required": ["path", "old_string", "new_string"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let old_string = args.get("old_string").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let new_string = args.get("new_string").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                log_call!("replace_string_in_file", "path" => path, "old_string" => old_string, "new_string" => new_string);
                Ok(ToolOutput::text(replace_string_in_file(&path, &old_string, &new_string)))
            })),
        },
        ToolDef {
            name: "write_text_file".into(),
            description: "Write a text file with the given its path and string content.\nUseful when user asks for content generation, e.g. code generation.\nExisting files are only replaced when `overwrite` is True.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": false}
                },
                "required": ["file", "content"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let file = args.get("file").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let overwrite = args.get("overwrite").and_then(|v| v.as_bool()).unwrap_or(false);
                log_call!("write_text_file", "file" => file, "content" => content, "overwrite" => overwrite);
                Ok(ToolOutput::text(write_text_file(&file, &content, overwrite)))
            })),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_extensions_matches_python() {
        assert_eq!(
            normalize_extensions("py, .MD ,txt"),
            vec![".py", ".md", ".txt"]
        );
        assert_eq!(normalize_extensions(""), Vec::<String>::new());
    }

    #[test]
    fn safe_filename_matches_python() {
        assert_eq!(safe_filename("^GSPC"), "GSPC");
        assert_eq!(safe_filename("BRK/B"), "BRK_B");
        assert_eq!(safe_filename(""), "symbol");
    }

    #[test]
    fn read_write_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        let written = write_text_file(file.to_str().unwrap(), "hello", false);
        assert!(written.starts_with("File created at "));
        assert_eq!(read_text_file(file.to_str().unwrap()), "hello");
        // no overwrite
        assert_eq!(
            write_text_file(file.to_str().unwrap(), "x", false),
            "File already exists."
        );
        assert!(write_text_file(file.to_str().unwrap(), "x", true).starts_with("File created"));
        assert_eq!(read_text_file(file.to_str().unwrap()), "x");
    }

    #[test]
    fn search_in_file_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("b.txt");
        std::fs::write(&file, "alpha\nbeta alpha\ngamma\n").unwrap();
        let out = search_in_file("alpha", file.to_str().unwrap());
        assert!(out.contains("at lines: 1, 2"), "got: {out}");
        let none = search_in_file("zzz", file.to_str().unwrap());
        assert!(none.starts_with("Did not find pattern"));
    }

    #[test]
    fn search_pattern_in_files_filters() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), "findme\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "findme\n").unwrap();
        let out = search_pattern_in_files(
            "findme",
            dir.path().to_str().unwrap(),
            ".py",
            "",
            Some(".*"),
        );
        assert!(out.contains("a.py"), "got: {out}");
        assert!(!out.contains("b.txt"), "got: {out}");
    }

    #[test]
    fn replace_counts() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("c.txt");
        std::fs::write(&file, "a b a b a\n").unwrap();
        let out = replace_string_in_file(file.to_str().unwrap(), "a", "z");
        assert_eq!(out, "'3' occurrences of 'a' have been replaced by 'z'");
        let missing =
            replace_string_in_file(dir.path().join("nope.txt").to_str().unwrap(), "a", "z");
        assert!(missing.starts_with("Error: File"));
    }

    #[test]
    fn glob_basics() {
        assert!(glob_match(".*", ".hidden"));
        assert!(!glob_match(".*", "normal"));
        assert!(glob_match("*.log", "a.log"));
    }
}
