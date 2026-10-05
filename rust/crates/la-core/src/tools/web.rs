//! Web tools (Python `src/tools/web.py` port): DuckDuckGo search, page text
//! extraction and file downloads — via `aurora_websearch` (search client +
//! content extractor) and `reqwest` (downloads), replacing the Selenium stack.

use crate::log_call;
use crate::tools::files::safe_filename;
use crate::tools::registry::{ToolDef, ToolOutput};
use anyhow::Result;
use aurora_websearch::AuroraSearch;
use std::sync::Arc;

/// Return the stripped text of matching headings/paragraph lines longer than
/// `min_len` (Python `extract_text` helper over h1/h2/p tags).
///
/// With aurora_websearch the page is converted to markdown first; heading
/// lines (`#`, `##`, ...) and plain lines are then filtered by tag family.
fn lines_from_markdown(markdown: &str, kind: Heading) -> Vec<String> {
    markdown
        .lines()
        .filter_map(|line| {
            let stripped = line.trim();
            match kind {
                Heading::H1 => stripped.strip_prefix("# ").map(|s| s.trim().to_string()),
                Heading::H2 => stripped.strip_prefix("## ").map(|s| s.trim().to_string()),
                Heading::Paragraph => {
                    if stripped.starts_with('#')
                        || stripped.starts_with("```")
                        || stripped.starts_with('-')
                        || stripped.starts_with('*')
                    {
                        None
                    } else if !stripped.is_empty()
                        && !stripped.starts_with('[')
                        && !stripped.contains("](http")
                    {
                        Some(stripped.to_string())
                    } else {
                        None
                    }
                }
            }
        })
        .filter(|text| text.chars().count() > 0)
        .collect()
}

#[derive(Clone, Copy)]
enum Heading {
    H1,
    H2,
    /// Kept for parity with the Python paragraph extraction; content
    /// extraction is delegated to aurora's `extract` (kept unconstructed).
    #[allow(dead_code)]
    Paragraph,
}

/// Render search results as the numbered block returned to the model
/// (Python `format_search_results`).
pub fn format_search_results(results: &[String], n_results: usize) -> String {
    let mut lines = vec!["\nSearch Results:\n".to_string()];
    for (index, item) in results.iter().take(n_results).enumerate() {
        lines.push(format!("{}) {}", index + 1, item.replace(" › ", "/")));
        lines.push("=".repeat(50));
    }
    format!("{}\n", lines.join("\n"))
}

/// Build the shared AuroraSearch client (one per context; Clone is cheap).
pub fn aurora() -> Result<AuroraSearch> {
    AuroraSearch::new().map_err(|e| anyhow::anyhow!("AuroraSearch init: {e}"))
}

/// Automatic extraction of headlines from a URL. Returns headlines as string.
/// Useful when user is asking to summarize webpage headlines given its URL.
///
/// Python scrapes `h1`/`h2` texts; aurora's markdown conversion preserves the
/// heading structure, so `#`/`##` lines reproduce the same behavior.
pub async fn webpage_headlines(url: &str, relevant_text_length: usize, timeout_s: u64) -> String {
    log_call!("webpage_headlines", "url" => url, "relevant_text_length" => relevant_text_length, "timeout_s" => timeout_s);
    let aurora = match aurora() {
        Ok(a) => a,
        Err(e) => return format!("Error fetching URL: {e}"),
    };
    let _ = timeout_s; // aurora applies its own configured timeout
    match aurora.extract_markdown(url).await {
        Err(e) => format!("Error fetching URL: {e}"),
        Ok(markdown) => {
            let mut headlines: Vec<String> = lines_from_markdown(&markdown, Heading::H1);
            headlines.extend(lines_from_markdown(&markdown, Heading::H2));
            headlines.retain(|h| h.chars().count() > relevant_text_length);
            format!("\nFrom [{url}]\n{}", headlines.join("\n"))
        }
    }
}

/// Automatic extraction paragraph content from a URL as string.
/// Useful when user is asking to summarize the webpage content of a webpage
/// given its URL.
pub async fn webpage_content(url: &str, relevant_text_length: usize, timeout_s: u64) -> String {
    log_call!("webpage_content", "url" => url, "relevant_text_length" => relevant_text_length, "timeout_s" => timeout_s);
    let aurora = match aurora() {
        Ok(a) => a,
        Err(e) => return format!("Error fetching URL: {e}"),
    };
    let _ = timeout_s;
    match aurora.extract_content(url).await {
        Err(e) => format!("Error fetching URL: {e}"),
        Ok(content) => {
            // The extracted main content is a paragraph flow; keep paragraphs
            // longer than the threshold (markdown-safe split).
            let paragraphs: Vec<String> = content
                .content
                .split("\n\n")
                .map(|p| p.replace('\n', " ").trim().to_string())
                .filter(|p| p.chars().count() > relevant_text_length)
                .collect();
            format!("\nFrom [{url}]\n{}", paragraphs.join("\n"))
        }
    }
}

/// Search web pages based on the query.
/// Useful when user asks for search on a topic.
pub async fn web_search(query: &str, n_results: usize, _timeout_s: f64) -> String {
    log_call!("web_search", "query" => query, "n_results" => n_results);
    let aurora = match aurora() {
        Ok(a) => a,
        Err(e) => return format!("Error: {e:?}"),
    };
    match aurora.search_with_limit(query, n_results).await {
        Err(e) => format!("Error: {e:?}"),
        Ok(results) => {
            let items: Vec<String> = results
                .results
                .iter()
                .map(|r| {
                    let mut text = format!("{} › {}", r.title, r.url);
                    if !r.snippet.is_empty() {
                        text.push_str(&format!(" › {}", r.snippet));
                    }
                    text
                })
                .collect();
            format_search_results(&items, n_results)
        }
    }
}

// ------------------------------------------------------------- downloads

/// Content types that complete a download name missing its extension.
const CONTENT_TYPE_EXT: &[(&str, &str)] = &[
    ("image/jpeg", ".jpg"),
    ("image/png", ".png"),
    ("image/gif", ".gif"),
    ("image/webp", ".webp"),
    ("image/bmp", ".bmp"),
    ("image/tiff", ".tiff"),
    ("video/mp4", ".mp4"),
    ("video/webm", ".webm"),
    ("video/quicktime", ".mov"),
    ("video/x-msvideo", ".avi"),
    ("audio/mpeg", ".mp3"),
    ("audio/wav", ".wav"),
    ("audio/ogg", ".ogg"),
    ("application/pdf", ".pdf"),
    ("application/zip", ".zip"),
];

/// Decode percent-escapes in a URL path segment (Python `urllib.parse.unquote`:
/// invalid escapes pass through verbatim, `+` stays a literal plus).
fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 3 <= bytes.len() => {
                match (hex_digit(bytes[i + 1]), hex_digit(bytes[i + 2])) {
                    (Some(hi), Some(lo)) => {
                        out.push(hi * 16 + lo);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Download name from Content-Disposition, else the URL path basename
/// (Python `_suggested_name`; supports RFC 5987 `filename*=UTF-8''...`).
pub fn suggested_name(url: &str, headers: &reqwest::header::HeaderMap) -> String {
    let header = headers
        .get(reqwest::header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if let Some(value) = find_rfc5987_filename(header) {
        return value;
    }
    if let Some(value) = find_quoted_filename(header) {
        return value;
    }
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let name = path.rsplit('/').next().unwrap_or("");
    unquote(name)
}

fn find_rfc5987_filename(header: &str) -> Option<String> {
    let lower = header.to_lowercase();
    let idx = lower.find("filename*=")?;
    let value = &header[idx + "filename*=".len()..];
    let value = value
        .strip_prefix("utf-8''")
        .or_else(|| value.strip_prefix("UTF-8''"))
        .unwrap_or(value);
    let end = value.find(';').unwrap_or(value.len());
    Some(unquote(value[..end].trim()).trim().to_string())
}

fn find_quoted_filename(header: &str) -> Option<String> {
    let lower = header.to_lowercase();
    let idx = lower.find("filename=")?;
    let value = &header[idx + "filename=".len()..];
    let value = value.trim_start_matches('"');
    let end = value.find([';', '"']).unwrap_or(value.len());
    Some(value[..end].trim().to_string())
}

/// Complete `name` with the content-type extension when it has none.
pub fn with_extension(name: &str, content_type: &str) -> String {
    let has_ext = std::path::Path::new(name)
        .extension()
        .map(|e| !e.is_empty())
        .unwrap_or(false);
    if has_ext {
        return name.to_string();
    }
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    for (ty, ext) in CONTENT_TYPE_EXT {
        if *ty == media_type {
            return format!("{name}{ext}");
        }
    }
    name.to_string()
}

/// Download a file such as a picture, movie, audio or any other file from a URL
/// and save it on disk. The file is streamed in chunks to `out_dir` (default:
/// current working directory) under `filename` when given, else the name
/// suggested by the Content-Disposition header or the URL itself (missing
/// extensions are completed from the Content-Type). Existing files are only
/// replaced when `overwrite` is True. Returns the saved path and its size.
pub async fn download_file(
    http: &reqwest::Client,
    url: &str,
    out_dir: Option<&std::path::Path>,
    filename: Option<&str>,
    overwrite: bool,
    timeout_s: f64,
) -> String {
    let response = http
        .get(url)
        .header("User-Agent", "Mozilla/5.0 (compatible; live-assistant)")
        .timeout(std::time::Duration::from_secs_f64(timeout_s.max(0.1)))
        .send()
        .await;
    let response = match response {
        Ok(r) => r,
        Err(e) => return format!("Error fetching URL: {e}"),
    };
    let response = match response.error_for_status() {
        Ok(r) => r,
        Err(e) => return format!("Error fetching URL: {e}"),
    };
    let headers = response.headers().clone();
    let content_type = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let name = match filename {
        Some(name) => name.to_string(),
        None => suggested_name(url, &headers).to_string(),
    };
    let name = with_extension(&name, &content_type);
    let out_directory = match out_dir {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
    };
    let dest_path = out_directory.join(safe_filename(&name));
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
    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut file = match tokio::fs::File::create(&dest_path).await {
        Ok(f) => f,
        Err(e) => return format!("Error downloading: {e}"),
    };
    use tokio::io::AsyncWriteExt;
    let mut total = 0u64;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if let Err(e) = file.write_all(&bytes).await {
                    return format!("Error downloading: {e}");
                }
                total += bytes.len() as u64;
            }
            Err(e) => return format!("Error downloading: {e}"),
        }
    }
    let _ = file.flush().await;
    format!("Saved '{}' ({total} bytes).", dest_path.display())
}

// ------------------------------------------------------------- registry

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "web_search".into(),
            description: "Search web pages based on the query.\nUseful when user asks for search on a topic.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "n_results": {"type": "integer", "default": 5},
                    "timeout_s": {"type": "number", "default": 10.0}
                },
                "required": ["query"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let n_results = args.get("n_results").and_then(|v| v.as_i64()).map(|v| v as usize).unwrap_or(5);
                let timeout_s = args.get("timeout_s").and_then(|v| v.as_f64()).unwrap_or(10.0);
                Ok(ToolOutput::text(web_search(&query, n_results, timeout_s).await))
            })),
        },
        ToolDef {
            name: "webpage_content".into(),
            description: "Automatic extraction paragraph content from a URL as string.\nUseful when user is asking to summarize the webpage content of a webpage given its URL.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "relevant_text_length": {"type": "integer", "default": 20},
                    "timeout_s": {"type": "integer", "default": 10}
                },
                "required": ["url"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let url = args.get("url").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let len = args.get("relevant_text_length").and_then(|v| v.as_i64()).map(|v| v as usize).unwrap_or(20);
                let timeout = args.get("timeout_s").and_then(|v| v.as_i64()).map(|v| v as u64).unwrap_or(10);
                Ok(ToolOutput::text(webpage_content(&url, len, timeout).await))
            })),
        },
        ToolDef {
            name: "webpage_headlines".into(),
            description: "Automatic extraction of headlines from a URL. Returns headlines as string.\nUseful when user is asking to summarize webpage headlines given its URL.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "default": "https://www.ansa.it/english/"},
                    "relevant_text_length": {"type": "integer", "default": 20},
                    "timeout_s": {"type": "integer", "default": 10}
                },
                "required": ["url"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("https://www.ansa.it/english/").to_string();
                let len = args.get("relevant_text_length").and_then(|v| v.as_i64()).map(|v| v as usize).unwrap_or(20);
                let timeout = args.get("timeout_s").and_then(|v| v.as_i64()).map(|v| v as u64).unwrap_or(10);
                Ok(ToolOutput::text(webpage_headlines(&url, len, timeout).await))
            })),
        },
        ToolDef {
            name: "download_file".into(),
            description: "Download a file such as a picture, movie, audio or any other file from a URL\nand save it on disk. The file is streamed in chunks to `out_dir` (default:\ncurrent working directory) under `filename` when given, else the name\nsuggested by the Content-Disposition header or the URL itself (missing\nextensions are completed from the Content-Type). Existing files are only\nreplaced when `overwrite` is True. Returns the saved path and its size.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "out_dir": {"type": "string"},
                    "filename": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": false},
                    "timeout_s": {"type": "number", "default": 30.0}
                },
                "required": ["url"]
            }),
            handler: Arc::new(|ctx, args| Box::pin(async move {
                let url = args.get("url").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let out_dir = args.get("out_dir").and_then(|v| v.as_str()).map(std::path::PathBuf::from);
                let filename = args.get("filename").and_then(|v| v.as_str()).map(str::to_string);
                let overwrite = args.get("overwrite").and_then(|v| v.as_bool()).unwrap_or(false);
                let timeout_s = args.get("timeout_s").and_then(|v| v.as_f64()).unwrap_or(30.0);
                let http = ctx.view.http.clone();
                Ok(ToolOutput::text(
                    download_file(&http, &url, out_dir.as_deref(), filename.as_deref(), overwrite, timeout_s).await,
                ))
            })),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_search_results_matches_python() {
        let out = format_search_results(&["Title › url".to_string()], 5);
        assert!(out.contains("1) Title/url"));
        assert!(out.contains("=".repeat(50).as_str()));
        assert!(out.starts_with("\nSearch Results:\n"));
    }

    #[test]
    fn suggested_name_rfc5987_and_plain() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_DISPOSITION,
            reqwest::header::HeaderValue::from_static("attachment; filename*=UTF-8''caf%C3%A9.txt"),
        );
        assert_eq!(suggested_name("http://x/y", &headers), "café.txt");
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_DISPOSITION,
            reqwest::header::HeaderValue::from_static("attachment; filename=\"report.pdf\""),
        );
        assert_eq!(suggested_name("http://x/y", &headers), "report.pdf");
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(
            suggested_name("http://x/img/a%20b.jpg", &headers),
            "a b.jpg"
        );
        assert_eq!(
            suggested_name("http://x/img/photo.png", &headers),
            "photo.png"
        );
    }

    #[test]
    fn with_extension_completes_from_content_type() {
        assert_eq!(with_extension("photo", "image/png"), "photo.png");
        assert_eq!(with_extension("photo.jpg", "image/png"), "photo.jpg");
        assert_eq!(with_extension("doc", "application/pdf"), "doc.pdf");
        assert_eq!(with_extension("doc", "text/html; charset=utf-8"), "doc");
    }

    #[test]
    fn unquote_decodes_percent_escapes() {
        assert_eq!(unquote("caf%C3%A9.txt"), "café.txt");
        assert_eq!(unquote("a+b.txt"), "a+b.txt"); // `+` is literal (Python unquote)
        assert_eq!(unquote("%41%2"), "A%2"); // trailing invalid escape kept
    }
}
