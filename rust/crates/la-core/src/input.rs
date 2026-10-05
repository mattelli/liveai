//! Interactive input layer for chat mode (`src/input_reader.py` port).
//!
//! Built on `rustyline`: line editing, persistent file history, inline history
//! suggestions (`HistoryHinter`), completion of the session `/commands`, and
//! the `/paste` ... `.` fallback for terminals without bracketed paste.
//! When `prompt_toolkit` is disabled in the config the reader degrades to the
//! plain `prompt_input` builtin (Python parity).

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::hint::HistoryHinter;
use rustyline::history::FileHistory;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper, Result as RlResult};
use std::path::PathBuf;

/// Session commands for completion (Python `ConversationSession.COMMANDS`).
pub const COMMANDS: &[&str] = &["/help", "/quit", "/clear", "/voice", "/tools", "/info"];

/// Outcome of one read (Python: line / KeyboardInterrupt / EOFError).
#[derive(Debug, Clone, PartialEq)]
pub enum ReadOutcome {
    Line(String),
    /// Ctrl+C at the prompt (chat: re-prompt; voice: quit).
    Interrupted,
    /// EOF (Ctrl+D).
    Eof,
}

/// Simple completer for the session `/commands`.
#[derive(Default)]
struct CommandCompleter;

impl Completer for CommandCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        _pos: usize,
        _ctx: &Context<'_>,
    ) -> RlResult<(usize, Vec<Pair>)> {
        if !line.starts_with('/') || line.contains(' ') {
            return Ok((0, Vec::new()));
        }
        let start = 0;
        let candidates = COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(line) && **cmd != line)
            .map(|cmd| Pair {
                display: (*cmd).to_string(),
                replacement: (*cmd).to_string(),
            })
            .collect();
        Ok((start, candidates))
    }
}

/// Bundled helper: completions + history hints (auto-suggest).
#[derive(Default)]
struct CommandHelper {
    completer: CommandCompleter,
    hinter: HistoryHinter,
}

impl Completer for CommandHelper {
    type Candidate = Pair;
    fn complete(&self, line: &str, pos: usize, ctx: &Context<'_>) -> RlResult<(usize, Vec<Pair>)> {
        self.completer.complete(line, pos, ctx)
    }
}

impl Hinter for CommandHelper {
    type Hint = String;
    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<String> {
        if !line.starts_with('/') {
            self.hinter.hint(line, pos, ctx)
        } else {
            None
        }
    }
}

impl Highlighter for CommandHelper {}
impl Validator for CommandHelper {}
impl Helper for CommandHelper {}

/// Owns one interactive prompt and maps EOF / interrupt correctly
/// (Python `InputReader`).
pub struct InputReader {
    session: Option<Editor<CommandHelper, FileHistory>>,
    history_path: Option<PathBuf>,
    use_prompt_toolkit: bool,
    /// stored for parity with the Python config; hints come from the helper
    #[allow(dead_code)]
    auto_suggest: bool,
    multiline: bool,
    explicit_paste: bool,
    continuation: String,
    closed: bool,
}

impl InputReader {
    /// Build a reader. `history_path` enables persistent history (Up/Down
    /// recall across sessions); the parent directory is created if needed.
    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    pub fn new(
        history_path: Option<PathBuf>,
        auto_suggest: bool,
        completion: bool,
        multiline: bool,
        explicit_paste: bool,
        use_prompt_toolkit: bool,
        continuation: &str,
    ) -> Self {
        Self {
            session: None,
            history_path,
            use_prompt_toolkit,
            auto_suggest,
            multiline,
            explicit_paste,
            continuation: continuation.to_string(),
            closed: false,
        }
    }

    fn ensure_session(&mut self) -> anyhow::Result<&mut Editor<CommandHelper, FileHistory>> {
        if self.session.is_none() {
            let mut config = rustyline::Config::builder().auto_add_history(true);
            if self.multiline {
                // Esc+Enter accepts (advanced, Python multiline=true).
                config = config.edit_mode(rustyline::EditMode::Emacs);
            }
            let config = config.build();
            let history = match &self.history_path {
                Some(path) => {
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    FileHistory::with_config(&config)
                }
                None => FileHistory::new(),
            };
            let mut editor = Editor::with_history(config, history)
                .map_err(|e| anyhow::anyhow!("editor: {e}"))?;
            let helper = CommandHelper::default();
            editor.set_helper(Some(helper));
            if let Some(path) = &self.history_path {
                let _ = editor.load_history(path);
            }
            self.session = Some(editor);
        }
        Ok(self.session.as_mut().unwrap())
    }

    /// Return the entered text (may contain newlines), or EOF.
    pub fn read(&mut self, label: &str) -> ReadOutcome {
        if !self.use_prompt_toolkit {
            return match crate::common::prompt_input(label, "yellow") {
                Some(line) => ReadOutcome::Line(line),
                None => ReadOutcome::Eof,
            };
        }
        let editor = match self.ensure_session() {
            Ok(e) => e,
            Err(_) => return ReadOutcome::Eof,
        };
        match editor.readline(label) {
            Ok(line) => {
                self.save_history();
                if self.explicit_paste && line.trim() == "/paste" {
                    self.read_explicit_paste()
                } else {
                    ReadOutcome::Line(line)
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => ReadOutcome::Interrupted,
            Err(rustyline::error::ReadlineError::Eof) => ReadOutcome::Eof,
            Err(_) => ReadOutcome::Eof,
        }
    }

    /// Read a line; if it is exactly `/paste`, collect lines until `.`.
    fn read_explicit_paste(&mut self) -> ReadOutcome {
        let continuation = self.continuation.clone();
        let mut lines: Vec<String> = Vec::new();
        loop {
            let editor = match self.ensure_session() {
                Ok(e) => e,
                Err(_) => return ReadOutcome::Eof,
            };
            match editor.readline(continuation.as_str()) {
                Ok(line) => {
                    if line.trim() == "." {
                        break;
                    }
                    lines.push(line);
                }
                Err(rustyline::error::ReadlineError::Interrupted) => break,
                Err(_) => break,
            }
        }
        self.save_history();
        ReadOutcome::Line(lines.join("\n"))
    }

    fn save_history(&mut self) {
        if let (Some(path), Some(editor)) = (&self.history_path, self.session.as_mut()) {
            let _ = editor.save_history(path);
        }
    }

    /// Flush history / release resources. Safe to call twice.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.save_history();
        self.session = None;
    }
}

impl crate::session::PromptSource for InputReader {
    fn read(&mut self) -> crate::input::ReadOutcome {
        self.read("[You] (type / for hints): ")
    }
}

/// Pure logic of the `/paste` collector, extracted for unit testing:
/// given a line provider, accumulate lines until a bare `.`.
pub fn collect_paste(mut next_line: impl FnMut(&str) -> Option<String>) -> Option<String> {
    let mut lines = Vec::new();
    while let Some(line) = next_line("... ") {
        if line.trim() == "." {
            break;
        }
        lines.push(line);
    }
    Some(lines.join("\n"))
}

/// Convert a raw argument string to a typed value (Python `_convert_argument`).
pub fn convert_argument(value: &str, annotation: &str) -> serde_json::Value {
    match annotation {
        "int" => value
            .parse::<i64>()
            .map(serde_json::Value::from)
            .unwrap_or_else(|_| value.into()),
        "float" => value
            .parse::<f64>()
            .map(serde_json::Value::from)
            .unwrap_or_else(|_| value.into()),
        "bool" => serde_json::Value::from(matches!(
            value.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        )),
        _ => serde_json::Value::from(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_collector_accumulates_until_dot() {
        let inputs = vec!["line one", "line two", "."];
        let mut it = inputs.into_iter();
        let result = collect_paste(|_| it.next().map(|s| s.to_string()));
        assert_eq!(result, Some("line one\nline two".to_string()));
    }

    #[test]
    fn paste_collector_eof_stops() {
        let inputs = vec!["only line"];
        let mut it = inputs.into_iter();
        let result = collect_paste(|_| it.next().map(|s| s.to_string()));
        assert_eq!(result, Some("only line".to_string()));
    }

    #[test]
    fn paste_collector_dot_is_not_included() {
        let result = collect_paste(|prompt| {
            if prompt == "... " {
                Some(".".into())
            } else {
                None
            }
        });
        assert_eq!(result, Some("".to_string()));
    }

    #[test]
    fn argument_conversion() {
        assert_eq!(convert_argument("42", "int"), serde_json::json!(42));
        assert_eq!(convert_argument("abc", "int"), serde_json::json!("abc"));
        assert_eq!(convert_argument("1.5", "float"), serde_json::json!(1.5));
        assert_eq!(convert_argument("true", "bool"), serde_json::json!(true));
        assert_eq!(convert_argument("no", "bool"), serde_json::json!(false));
        assert_eq!(
            convert_argument("hello", "string"),
            serde_json::json!("hello")
        );
    }

    #[test]
    fn commands_list() {
        assert_eq!(
            COMMANDS,
            ["/help", "/quit", "/clear", "/voice", "/tools", "/info"]
        );
    }

    #[test]
    fn reader_degraded_mode_eof() {
        // prompt_toolkit disabled -> builtin input() path; stdin is not a tty
        // in tests, so read_line gets EOF -> None.
        let mut reader = InputReader::new(None, false, false, false, false, false, "");
        let outcome = reader.read("You: ");
        assert_eq!(outcome, ReadOutcome::Eof);
        reader.close();
    }
}
