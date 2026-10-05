//! Document loading for the semantic-search corpus
//! (`src/document_loader.py` port).
//!
//! Extracts plain text (and, where available, images) from the common document
//! types so `EmbeddingCorpus::store` can index them:
//! - text-like: `.txt` `.md` `.markdown` `.tex` `.rst` `.log` `.csv`
//! - markup:    `.html` `.htm`
//! - office:    `.pdf` `.docx` `.xlsx`
//! - images:    `.png` `.jpg` `.jpeg` `.webp` `.bmp` `.tif` `.tiff`
//!
//! Errors are captured per document (never raised), like the Python loader.

use std::path::{Path, PathBuf};

pub const TEXT_EXTENSIONS: [&str; 7] = [".txt", ".md", ".markdown", ".tex", ".rst", ".log", ".csv"];
pub const IMAGE_EXTENSIONS: [&str; 7] = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"];
pub const DOCUMENT_EXTENSIONS: [&str; 5] = [".pdf", ".docx", ".xlsx", ".html", ".htm"];
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    ".txt",
    ".md",
    ".markdown",
    ".tex",
    ".rst",
    ".log",
    ".csv",
    ".pdf",
    ".docx",
    ".xlsx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
];

/// Embedded PDF images smaller than this (either side) or flatter than this
/// (grayscale stddev) are icons, masks or blank fills — not figures/diagrams.
pub const MIN_IMAGE_DIM: u32 = 64;
pub const MIN_IMAGE_STD: f64 = 6.0;

/// One input file: its extracted text and/or image paths.
///
/// `sections` carries `(page_number, text)` pairs for paginated documents
/// (PDF). For single-section documents it is empty and the caller falls back
/// to `(pages, text)`.
#[derive(Debug, Clone, Default)]
pub struct Document {
    pub source: String,
    pub kind: String, // "text" | "image"
    pub text: String,
    pub images: Vec<String>,
    pub pages: Option<u64>,
    pub error: String,
    pub sections: Vec<(Option<u64>, String)>,
}

impl Document {
    pub fn error(source: &str, error: impl Into<String>) -> Self {
        Self {
            source: source.to_string(),
            kind: "text".into(),
            error: error.into(),
            ..Default::default()
        }
    }
}

// --------------------------------------------------------------- helpers

/// Normalize `value` ("txt, .md") into lowercase dotted suffixes.
pub fn normalize_extensions(value: &str) -> Vec<String> {
    crate::tools::files::normalize_extensions(value)
}

fn safe_name(value: &str) -> String {
    let re = regex::Regex::new(r"[^A-Za-z0-9._-]+").unwrap();
    let cleaned = re.replace_all(value, "_");
    let cleaned = cleaned.trim_matches(|c| c == '.' || c == '_');
    if cleaned.is_empty() {
        "file".to_string()
    } else {
        cleaned.to_string()
    }
}

fn relative_source(path: &Path, root: Option<&Path>) -> String {
    match root {
        None => path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        Some(root) => {
            let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
            let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            match path.strip_prefix(&root) {
                Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
                Err(_) => path.to_string_lossy().replace('\\', "/"),
            }
        }
    }
}

/// Collapse blank lines (Python `_collapse_blank_lines`).
fn collapse_blank_lines(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Remove LaTeX comments, math and command noise while keeping words.
pub fn strip_tex(text: &str) -> String {
    let text = strip_tex_comments(text);
    let command = regex::Regex::new(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?").unwrap();
    let math = regex::Regex::new(r"\$[^$]*\$").unwrap();
    let braces = regex::Regex::new(r"[{}]").unwrap();
    let spaces = regex::Regex::new(r"[ \t]+").unwrap();
    let text = math.replace_all(&text, " ");
    let text = command.replace_all(&text, " ");
    let text = braces.replace_all(&text, " ");
    spaces.replace_all(&text, " ").trim().to_string()
}

/// Strip `% ...` comments line by line (Python `_TEX_COMMENT`, a
/// multiline regex with a negative look-behind). The `regex` crate has no
/// look-around support, so the "`%` not preceded by a backslash" rule is
/// applied manually: on each line the text is cut before the first `%`
/// whose previous byte is not a backslash (an escaped `\\%` stays);
/// the rest of the line is dropped.
fn strip_tex_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (content, newline) = match line.strip_suffix('\n') {
            Some(content) => (content, "\n"),
            None => (line, ""),
        };
        let bytes = content.as_bytes();
        let cut = (0..bytes.len()).find(|&i| bytes[i] == b'%' && (i == 0 || bytes[i - 1] != b'\\'));
        match cut {
            Some(cut) => out.push_str(&content[..cut]),
            None => out.push_str(content),
        }
        out.push_str(newline);
    }
    out
}

/// Collapse table-of-contents dot leaders (`. . . .`) into single spaces.
pub fn strip_dot_leaders(text: &str) -> String {
    let leader = regex::Regex::new(r"(?:\.\s*){3,}").unwrap();
    leader.replace_all(text, " ").to_string()
}

// ------------------------------------------------------------ extractors

/// Plain text file; `.tex` is cleaned with [`strip_tex`].
pub fn extract_text(path: &Path) -> Document {
    match std::fs::read_to_string(path) {
        Err(e) => Document::error(
            &path.file_name().unwrap_or_default().to_string_lossy(),
            e.to_string(),
        ),
        Ok(raw) => {
            let text = if path
                .extension()
                .map(|e| e.eq_ignore_ascii_case("tex"))
                .unwrap_or(false)
            {
                strip_tex(&raw)
            } else {
                raw
            };
            Document {
                source: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                kind: "text".into(),
                text,
                ..Default::default()
            }
        }
    }
}

/// HTML file -> plain text (html2text via html5ever).
pub fn extract_html(path: &Path) -> Document {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    match std::fs::read_to_string(path) {
        Err(e) => Document::error(&name, e.to_string()),
        Ok(raw) => {
            let text = html2text::config::plain()
                .string_from_read(raw.as_bytes(), 240)
                .unwrap_or_default();
            Document {
                source: name,
                kind: "text".into(),
                text: collapse_blank_lines(&text),
                ..Default::default()
            }
        }
    }
}

/// True when `data` decodes to an image with visible detail (not blank).
fn image_is_useful(data: &[u8]) -> bool {
    let image = image::load_from_memory(data).ok();
    match image {
        None => false,
        Some(img) => {
            let gray = img.to_luma8();
            let (w, h) = (gray.width(), gray.height());
            if w < MIN_IMAGE_DIM || h < MIN_IMAGE_DIM {
                return false;
            }
            // grayscale stddev (Python ImageStat.Stat.stddev)
            let n = gray.as_raw().len() as f64;
            let mean = gray.as_raw().iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = gray
                .as_raw()
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / n;
            var.sqrt() >= MIN_IMAGE_STD
        }
    }
}

/// PDF extractor (pdfium-render): per-page text + embedded useful images.
pub fn extract_pdf(path: &Path, image_dir: Option<&Path>, source: &str) -> Document {
    use pdfium_render::prelude::*;
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let bindings = Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
        .or_else(|_| Pdfium::bind_to_system_library());
    let bindings = match bindings {
        Err(e) => {
            return Document::error(
                &name,
                format!(
                    "PdfiumError: {e} (bundle pdfium.dll next to the binary; see rust/README.md)"
                ),
            );
        }
        Ok(b) => b,
    };
    let pdfium = Pdfium::new(bindings);
    let document = match pdfium.load_pdf_from_file(path, None) {
        Err(e) => return Document::error(&name, format!("PdfiumError: {e}")),
        Ok(d) => d,
    };
    let pages = document.pages().len() as u64;
    let stem = if source.is_empty() {
        path.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    } else {
        PathBuf::from(source)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    };
    let mut sections: Vec<(Option<u64>, String)> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    for (page_index, page) in document.pages().iter().enumerate() {
        let page_text = page.text().map(|t| t.all()).unwrap_or_default();
        let page_text = strip_dot_leaders(&page_text);
        sections.push((Some(page_index as u64 + 1), page_text));
        if let Some(dir) = image_dir {
            images.extend(extract_pdf_images(&page, page_index, dir, &stem));
        }
    }
    let text = sections
        .iter()
        .map(|(_, t)| t.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    Document {
        source: name,
        kind: "text".into(),
        text,
        images,
        pages: Some(pages),
        sections,
        ..Default::default()
    }
}

/// Save useful embedded page images under `image_dir` and return their paths.
/// Skips soft-mask xrefs, duplicate xrefs, tiny icons and (near-)blank images.
fn extract_pdf_images(
    page: &pdfium_render::prelude::PdfPage,
    page_index: usize,
    image_dir: &Path,
    stem: &str,
) -> Vec<String> {
    use pdfium_render::prelude::*;
    let mut paths: Vec<String> = Vec::new();
    let objects = page.objects();
    for object in objects.iter() {
        let Some(image) = object.as_image_object() else {
            continue;
        };
        let Ok(width) = image.width() else { continue };
        let Ok(height) = image.height() else { continue };
        if (width as i64) < (MIN_IMAGE_DIM as i64) || (height as i64) < (MIN_IMAGE_DIM as i64) {
            continue;
        }
        let bitmap = match image.get_raw_bitmap() {
            Err(_) => continue,
            Ok(b) => b,
        };
        let rendered = match bitmap.as_image() {
            Err(_) => continue,
            Ok(img) => img,
        };
        if !image_is_useful(&render_bytes(&rendered)) {
            continue;
        }
        let _ = std::fs::create_dir_all(image_dir);
        let target = image_dir.join(format!(
            "{}_p{}_{}.png",
            safe_name(stem),
            page_index + 1,
            paths.len()
        ));
        if rendered
            .save_with_format(&target, image::ImageFormat::Png)
            .is_ok()
        {
            paths.push(
                std::fs::canonicalize(&target)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| target.to_string_lossy().to_string()),
            );
        }
    }
    paths
}

fn render_bytes(img: &image::DynamicImage) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    let _ = img.write_to(&mut buf, image::ImageFormat::Png);
    buf.into_inner()
}

/// DOCX file -> paragraphs + table rows (zip + quick-xml).
pub fn extract_docx(path: &Path) -> Document {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let result = (|| -> anyhow::Result<String> {
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let Some(entry) = archive.by_name("word/document.xml").ok() else {
            return Err(anyhow::anyhow!("missing word/document.xml"));
        };
        let mut text_xml = String::new();
        std::io::Read::read_to_string(&mut { entry }, &mut text_xml)?;
        Ok(docx_extract_text(&text_xml))
    })();
    match result {
        Err(e) => Document::error(&name, e.to_string()),
        Ok(text) => Document {
            source: name,
            kind: "text".into(),
            text,
            ..Default::default()
        },
    }
}

/// Pull paragraph and table-cell text out of `word/document.xml`.
pub fn docx_extract_text(xml: &str) -> String {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Err(_) | Ok(Event::Eof) => break,
            Ok(Event::Start(ref e)) => {
                let name = e.name().as_ref().to_string();
                if name == "w:p" {
                    current.clear();
                }
            }
            Ok(Event::Empty(ref e)) => {
                let name = e.name().as_ref().to_string();
                if name == "w:p" && !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                    current.clear();
                }
            }
            Ok(Event::Text(ref e)) => {
                let text = e
                    .xml_content(quick_xml::XmlVersion::Implicit1_0)
                    .to_string();
                current.push_str(&text);
            }
            Ok(Event::End(ref e)) => {
                let name = e.name().as_ref().to_string();
                if name == "w:p" {
                    if !current.trim().is_empty() {
                        parts.push(current.trim().to_string());
                    }
                    current.clear();
                } else if name == "w:tc" && !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                    current.clear();
                }
            }
            _ => {}
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts.join("\n")
}

/// XLSX file -> sheet rows (calamine).
pub fn extract_xlsx(path: &Path) -> Document {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let result = (|| -> anyhow::Result<String> {
        use calamine::Reader;
        let mut workbook = calamine::open_workbook_auto(path)?;
        let mut parts: Vec<String> = Vec::new();
        for sheet in workbook.sheet_names() {
            let range = match workbook.worksheet_range(&sheet) {
                Err(_) => continue,
                Ok(r) => r,
            };
            let mut rows: Vec<String> = Vec::new();
            for row in range.rows() {
                let values: Vec<String> = row
                    .iter()
                    .filter(|c| !matches!(c, calamine::Data::Empty))
                    .map(|c| match c {
                        calamine::Data::Float(f) => f.to_string(),
                        calamine::Data::Int(i) => i.to_string(),
                        calamine::Data::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect();
                if !values.is_empty() {
                    rows.push(values.join(" | "));
                }
            }
            if !rows.is_empty() {
                parts.push(format!("# Sheet: {sheet}\n{}", rows.join("\n")));
            }
        }
        Ok(parts.join("\n"))
    })();
    match result {
        Err(e) => Document::error(&name, e.to_string()),
        Ok(text) => Document {
            source: name,
            kind: "text".into(),
            text,
            ..Default::default()
        },
    }
}

/// Image file -> name + dimensions metadata (kind "image").
pub fn extract_image(path: &Path) -> Document {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let mut text = name.clone();
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if let Ok(size) = image::image_dimensions(path) {
        text = format!("{name} ({}x{})", size.0, size.1);
    }
    Document {
        source: name,
        kind: "image".into(),
        text,
        images: vec![resolved.to_string_lossy().to_string()],
        ..Default::default()
    }
}

/// Load one file into a [`Document`]; errors are captured, not raised.
pub fn load_document(path: &Path, root: Option<&Path>, image_dir: Option<&Path>) -> Document {
    let source = relative_source(path, root);
    let suffix = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    if !SUPPORTED_EXTENSIONS.contains(&suffix.as_str()) {
        return Document::error(&source, format!("Unsupported file type: '{suffix}'"));
    }
    let mut document = if IMAGE_EXTENSIONS.contains(&suffix.as_str()) {
        extract_image(path)
    } else if suffix == ".pdf" {
        extract_pdf(path, image_dir, &source)
    } else if suffix == ".docx" {
        extract_docx(path)
    } else if suffix == ".xlsx" {
        extract_xlsx(path)
    } else if suffix == ".html" || suffix == ".htm" {
        extract_html(path)
    } else {
        extract_text(path)
    };
    document.source = source;
    document
}

/// Load every supported file under `directory` (recursively, sorted).
pub fn iter_documents(
    directory: &Path,
    extensions: Option<&[String]>,
    image_dir: Option<&Path>,
) -> anyhow::Result<Vec<Document>> {
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
    let suffixes: Vec<String> = extensions
        .map(|list| list.to_vec())
        .unwrap_or_else(|| SUPPORTED_EXTENSIONS.iter().map(|s| s.to_string()).collect());
    let mut documents = Vec::new();
    for path in walkdir::WalkDir::new(directory)
        .sort_by_file_name()
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = path.path();
        if !path.is_file() {
            continue;
        }
        let suffix = path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        if !suffixes.is_empty() && !suffixes.contains(&suffix) {
            continue;
        }
        documents.push(load_document(path, Some(directory), image_dir));
    }
    Ok(documents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_tex_removes_commands_and_math() {
        let text = "\\documentclass{article} % comment\nThe math $x^2$ is here \\textbf{bold}.\n";
        let cleaned = strip_tex(text);
        assert!(!cleaned.contains("\\documentclass"));
        assert!(!cleaned.contains("x^2"));
        assert!(cleaned.contains("The math"));
        assert!(cleaned.contains("bold"));
    }

    #[test]
    fn strip_tex_comments_matches_python_lookbehind() {
        // Python `_TEX_COMMENT` = `(?m)(?<!\\\\)%.*$`.
        // A comment is removed from the first unescaped `%` to end of line.
        assert_eq!(strip_tex_comments("keep % drop\n"), "keep \n");
        // `\\%` is an escaped percent: not a comment start.
        assert_eq!(strip_tex_comments("100\\% pure\n"), "100\\% pure\n");
        // Comment at line start clears the line but keeps the newline.
        assert_eq!(strip_tex_comments("% only a comment\nnext\n"), "\nnext\n");
        // No trailing newline handling.
        assert_eq!(strip_tex_comments("a % b"), "a ");
        // Double backslash before `%` still reads as escaped (single-char look).
        assert_eq!(strip_tex_comments("x\\\\% y\n"), "x\\\\% y\n");
    }

    #[test]
    fn strip_dot_leaders_collapses() {
        // Python `re.sub(r"(?:\.\\s*){3,}", " ", ...)`: the leader run is
        // replaced by one space, so the space before it remains -> double space.
        assert_eq!(strip_dot_leaders("Title . . . . . 12"), "Title  12");
        assert_eq!(strip_dot_leaders("No leaders here."), "No leaders here.");
        // Python test parity: normal sentences and abbreviations survive.
        assert_eq!(
            strip_dot_leaders("End. Next sentence."),
            "End. Next sentence."
        );
        assert_eq!(
            strip_dot_leaders("U.S.A. and e.g. text"),
            "U.S.A. and e.g. text"
        );
    }

    #[test]
    fn text_and_html_extraction() {
        let dir = tempfile::tempdir().unwrap();
        let txt = dir.path().join("a.txt");
        std::fs::write(&txt, "hello\n\n\nworld\n").unwrap();
        let doc = extract_text(&txt);
        assert_eq!(doc.kind, "text");
        assert!(doc.text.contains("hello"));

        let html = dir.path().join("b.html");
        std::fs::write(
            &html,
            "<html><body><p>First</p><div><script>evil()</script><p>Second</p></div></body></html>",
        )
        .unwrap();
        let doc = extract_html(&html);
        assert!(doc.text.contains("First"));
        assert!(doc.text.contains("Second"));
        assert!(!doc.text.contains("evil"));
    }

    #[test]
    fn docx_extraction_from_zip() {
        // minimal docx: word/document.xml with two paragraphs
        let dir = tempfile::tempdir().unwrap();
        let docx = dir.path().join("t.docx");
        let file = std::fs::File::create(&docx).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file(
            "word/document.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        std::io::Write::write_all(
            &mut zip,
            br#"<w:document><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:r><w:t>World</w:t></w:r></w:p></w:body></w:document>"#,
        )
        .unwrap();
        zip.finish().unwrap();
        let doc = extract_docx(&docx);
        assert!(doc.text.contains("Hello"));
        assert!(doc.text.contains("World"));
    }

    #[test]
    fn xlsx_extraction() {
        let dir = tempfile::tempdir().unwrap();
        let xlsx = dir.path().join("t.xlsx");
        // build with calamine-compatible writer? use zip+sheet xml instead:
        // simplest: use the `rust_xlsxwriter`? Not available; instead craft a
        // minimal xlsx via zip.
        let file = std::fs::File::create(&xlsx).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        // content types + workbook + worksheet (minimal valid package)
        zip.start_file("[Content_Types].xml", options).unwrap();
        std::io::Write::write_all(&mut zip, br#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>"#).unwrap();
        zip.start_file("_rels/.rels", options).unwrap();
        std::io::Write::write_all(&mut zip, br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#).unwrap();
        zip.start_file("xl/workbook.xml", options).unwrap();
        std::io::Write::write_all(&mut zip, br#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></sheets></workbook>"#).unwrap();
        zip.start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        std::io::Write::write_all(&mut zip, br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#).unwrap();
        zip.start_file("xl/sharedStrings.xml", options).unwrap();
        std::io::Write::write_all(&mut zip, br#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2"><si><t>Alpha</t></si><si><t>Beta</t></si></sst>"#).unwrap();
        zip.start_file("xl/worksheets/sheet1.xml", options).unwrap();
        std::io::Write::write_all(&mut zip, br#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c t="s" r="A1"><v>0</v></c><c t="s" r="B1"><v>1</v></c></row><row r="2"><c r="A2"><v>1.5</v></c><c r="B2"><v>2.5</v></c></row></sheetData></worksheet>"#).unwrap();
        zip.finish().unwrap();

        let doc = extract_xlsx(&xlsx);
        assert!(doc.error.is_empty(), "error: {}", doc.error);
        assert!(doc.text.contains("# Sheet: Sheet1"), "got: {}", doc.text);
        assert!(doc.text.contains("Alpha"), "got: {}", doc.text);
    }

    #[test]
    fn image_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("img.png");
        image::DynamicImage::new_rgb8(16, 9).save(&png).unwrap();
        let doc = extract_image(&png);
        assert_eq!(doc.kind, "image");
        assert!(doc.text.contains("16x9"), "got: {}", doc.text);
        assert_eq!(doc.images.len(), 1);
    }

    #[test]
    fn unsupported_extension_captured() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("a.exe");
        std::fs::write(&exe, b"MZ").unwrap();
        let doc = load_document(&exe, None, None);
        assert!(doc.error.contains("Unsupported file type"));
    }

    #[test]
    fn iter_documents_sorted_recursive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("a.md"), "a").unwrap();
        std::fs::write(dir.path().join("c.xyz"), "skip").unwrap();
        let docs = iter_documents(dir.path(), None, None).unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().any(|d| d.source == "b.txt"));
        assert!(
            docs.iter()
                .any(|d| d.source == "sub/a.md" || d.source.ends_with("a.md"))
        );
    }
}
