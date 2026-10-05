//! `StreamPrinter` — `src/common.py` streaming-printer port.
//!
//! Prints streamed model output with syntax-highlighted fenced code blocks:
//! plain text is emitted as soon as it arrives (cyan); text between ``` fences
//! is buffered and rendered with syntect once the closing fence arrives, using
//! the language tag from the opening fence. A braille-dots activity spinner
//! animates while nothing visible is happening and is erased before any real
//! output (via the [`common::set_output_hook`] integration).

use crate::common::{Color, set_output_hook};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const SPIN_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPIN_INTERVAL: Duration = Duration::from_millis(120);
const SPIN_CLEAR: &str = "\r\x1b[K"; // CR + erase to end of line

fn ansi(color: Color, bold: bool) -> String {
    let code = match color {
        "black" => "30",
        "red" => "31",
        "green" => "32",
        "yellow" => "33",
        "blue" => "34",
        "magenta" => "35",
        "cyan" => "36",
        "grey" => "90",
        "white" => "37",
        _ => "0",
    };
    format!("\x1b[{}{}m", if bold { "1;" } else { "" }, code)
}

fn write_stdout(text: &str) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(text.as_bytes());
    let _ = lock.flush();
}

/// Shared spinner state (between the printer and its animation thread).
struct SpinState {
    active: bool,
    suppressed: bool,
    drawn: bool,
    note: String,
    t0: Instant,
    dirty_line: bool,
    /// Set under the mutex to end the animation thread. Keeping the stop
    /// request in the same locked state as everything else the thread reads
    /// removes the start/stop race (no separate atomic, no missed wakeup).
    stop: bool,
}

pub struct StreamPrinter {
    text_color: Color,
    line: String,
    in_code: bool,
    code_lang: String,
    code_lines: Vec<String>,
    plain_started: bool,

    spin: Arc<(Mutex<SpinState>, std::sync::Condvar)>,
    spin_thread: Option<JoinHandle<()>>,
}

impl Default for StreamPrinter {
    fn default() -> Self {
        Self::new("cyan")
    }
}

impl StreamPrinter {
    pub fn new(text_color: Color) -> Self {
        Self {
            text_color,
            line: String::new(),
            in_code: false,
            code_lang: "text".into(),
            code_lines: Vec::new(),
            plain_started: false,
            spin: Arc::new((
                Mutex::new(SpinState {
                    active: false,
                    suppressed: false,
                    drawn: false,
                    note: String::new(),
                    t0: Instant::now(),
                    dirty_line: false,
                    stop: false,
                }),
                std::sync::Condvar::new(),
            )),
            spin_thread: None,
        }
    }

    // ------------------------------------------------------------ streaming

    /// Push a streamed fragment, emitting plain text as soon as it is safe.
    pub fn feed(&mut self, content: &str) {
        let mut start = 0;
        loop {
            match content[start..].find('\n') {
                None => {
                    let tail = &content[start..];
                    if !tail.is_empty() {
                        self.feed_partial(tail);
                    }
                    return;
                }
                Some(rel) => {
                    let newline = start + rel;
                    self.feed_partial(&content[start..newline]);
                    self.end_line();
                    start = newline + 1;
                }
            }
        }
    }

    fn feed_partial(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.in_code {
            self.line.push_str(text);
        } else if self.plain_started {
            self.emit(text);
        } else {
            self.line.push_str(text);
            if !Self::could_be_fence(&self.line) {
                let line = std::mem::take(&mut self.line);
                self.emit(&line);
                self.plain_started = true;
            }
        }
    }

    /// True while `partial` could still become a ``` fence opening.
    fn could_be_fence(partial: &str) -> bool {
        let stripped = partial.trim_start();
        if stripped.is_empty() {
            return true; // only indentation so far
        }
        let backticks = stripped.chars().take_while(|&c| c == '`').count();
        if backticks >= 3 {
            return true; // confirmed fence; wait for the newline to classify it
        }
        backticks == stripped.chars().count() // only 1-2 backticks so far; may grow
    }

    fn end_line(&mut self) {
        let line = std::mem::take(&mut self.line);
        let started = std::mem::take(&mut self.plain_started);
        let stripped = line.trim();
        if self.in_code {
            if stripped.starts_with("```") {
                self.in_code = false;
                self.flush_code();
            } else {
                self.code_lines.push(format!("{line}\n"));
            }
            return;
        }
        if let Some(lang) = stripped.strip_prefix("```") {
            self.in_code = true;
            self.code_lang = lang.trim().to_string();
            if self.code_lang.is_empty() {
                self.code_lang = "text".into();
            }
            self.code_lines.clear();
            return;
        }
        let out = if started {
            "\n".to_string()
        } else {
            format!("{line}\n")
        };
        self.emit(&out);
    }

    fn emit(&mut self, text: &str) {
        self.erase_spinner_locked();
        let color = ansi(self.text_color, false);
        write_stdout(&format!("{color}{text}\x1b[0m"));
        self.mark_dirty(!text.ends_with('\n'));
    }

    fn flush_code(&mut self) {
        let code = self.code_lines.concat();
        self.code_lines.clear();
        let code = code.trim_end_matches('\n');
        if code.is_empty() {
            return;
        }
        self.erase_spinner_locked();
        write_stdout(&highlight_code(code, &self.code_lang));
        self.mark_dirty(false);
    }

    fn mark_dirty(&mut self, dirty: bool) {
        let (lock, _) = &*self.spin;
        lock.lock().unwrap().dirty_line = dirty;
    }

    /// Flush any buffered partial line / unterminated code block.
    pub fn flush(&mut self) {
        let line = std::mem::take(&mut self.line);
        self.plain_started = false;
        if self.in_code {
            if !line.trim().is_empty() && !line.trim().starts_with("```") {
                self.code_lines.push(line);
            }
            self.in_code = false;
            self.flush_code();
            return;
        }
        if !line.trim().is_empty() && !line.trim().starts_with("```") {
            self.emit(&line);
        }
    }

    // ---------------------------------------------------- activity spinner

    /// Animate the spinner until the next `stop_spinner`.
    pub fn start_spinner(&mut self, note: &str) {
        // All state transitions happen under the mutex; the animation thread
        // re-reads the shared state every cycle, so a start immediately
        // followed by a stop (or reuse of a still-running thread) is safe.
        let needs_thread = {
            let (lock, _) = &*self.spin;
            let mut st = lock.lock().unwrap();
            st.active = true;
            st.suppressed = false;
            st.drawn = false;
            st.stop = false;
            st.note = note.to_string();
            st.t0 = Instant::now();
            st.dirty_line = false;
            match &self.spin_thread {
                Some(h) => h.is_finished(),
                None => true,
            }
        };
        if needs_thread {
            let spin = self.spin.clone();
            self.spin_thread = Some(
                std::thread::Builder::new()
                    .name("stream-spinner".into())
                    .spawn(move || spin_loop(spin))
                    .expect("spawn spinner"),
            );
        }
        let spin = self.spin.clone();
        let suppress: Arc<dyn Fn() + Send + Sync> =
            Arc::new(move || Self::suppress_for_output(spin.clone()));
        set_output_hook(Some(suppress));
    }

    fn suppress_for_output(spin: Arc<(Mutex<SpinState>, std::sync::Condvar)>) {
        let (lock, _) = &*spin;
        let mut st = lock.lock().unwrap();
        if st.drawn {
            write_stdout(SPIN_CLEAR);
            st.drawn = false;
        }
        st.suppressed = true;
    }

    /// Erase the spinner (if drawn) and stop the animation thread.
    pub fn stop_spinner(&mut self) {
        {
            let (lock, cvar) = &*self.spin;
            let mut st = lock.lock().unwrap();
            st.active = false;
            st.suppressed = false;
            st.stop = true;
            if st.drawn {
                write_stdout(SPIN_CLEAR);
                st.drawn = false;
            }
            // Notify while the state change is still protected by the mutex
            // so a thread between its stop-check and its wait cannot miss it.
            cvar.notify_all();
        }
        set_output_hook(None);
        let thread = self.spin_thread.take();
        if let Some(h) = thread
            && h.thread().id() != std::thread::current().id()
        {
            let _ = h.join();
        }
    }

    /// Update the spinner caption (e.g. prompt-processing progress).
    pub fn note(&mut self, text: &str) {
        let (lock, _) = &*self.spin;
        let mut st = lock.lock().unwrap();
        st.note = text.to_string();
        st.suppressed = false;
    }

    /// Declare the terminal line empty so the spinner may draw again.
    pub fn mark_line_clean(&mut self) {
        let (lock, _) = &*self.spin;
        lock.lock().unwrap().dirty_line = false;
    }

    fn erase_spinner_locked(&mut self) {
        let (lock, _) = &*self.spin;
        let mut st = lock.lock().unwrap();
        if st.drawn {
            write_stdout(SPIN_CLEAR);
            st.drawn = false;
        }
    }
}

fn spin_loop(spin: Arc<(Mutex<SpinState>, std::sync::Condvar)>) {
    let (lock, cvar) = &*spin;
    let mut guard = lock.lock().unwrap();
    loop {
        if guard.stop {
            return;
        }
        let (next, _timeout) = cvar.wait_timeout(guard, SPIN_INTERVAL).unwrap();
        guard = next;
        if guard.stop {
            return;
        }
        if guard.active && !guard.suppressed && !guard.dirty_line {
            let elapsed = guard.t0.elapsed();
            let idx =
                (elapsed.as_secs_f64() / SPIN_INTERVAL.as_secs_f64()) as usize % SPIN_FRAMES.len();
            let frame = SPIN_FRAMES[idx];
            let out = format!("{SPIN_CLEAR}{frame} {}", guard.note);
            write_stdout(&out);
            guard.drawn = true;
        }
    }
}

/// Highlight a code block with syntect and return ANSI-escaped lines.
/// Falls back to plain text when no matching syntax is found.
/// (Python: Rich `Syntax(code, lang, theme="monokai")`; syntect's default set
/// has no monokai theme — `base16-eighties.dark` is the closest built-in.)
fn highlight_code(code: &str, lang: &str) -> String {
    use syntect::easy::HighlightLines;
    use syntect::highlighting::ThemeSet;
    use syntect::parsing::SyntaxSet;
    use syntect::util::as_24_bit_terminal_escaped;

    static SETS: std::sync::OnceLock<(SyntaxSet, ThemeSet)> = std::sync::OnceLock::new();
    let (syntax_set, theme_set) = SETS.get_or_init(|| {
        (
            SyntaxSet::load_defaults_newlines(),
            ThemeSet::load_defaults(),
        )
    });

    let syntax = syntax_set
        .find_syntax_by_token(lang)
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());
    let theme = theme_set
        .themes
        .get("base16-eighties.dark")
        .cloned()
        .unwrap_or_default();
    let mut hl = HighlightLines::new(syntax, &theme);
    let mut out = String::new();
    for line in syntect::util::LinesWithEndings::from(code) {
        if let Ok(ranges) = hl.highlight_line(line, syntax_set) {
            out.push_str(&as_24_bit_terminal_escaped(&ranges, false));
        }
    }
    out.push_str("\x1b[0m\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{cprint, get_output_hook};

    /// Collected output of the printer (tests capture via local writer).
    fn collect(printer: &mut StreamPrinter) -> String {
        let _ = printer;
        String::new()
    }

    #[test]
    fn could_be_fence_matches_python() {
        assert!(StreamPrinter::could_be_fence(""));
        assert!(StreamPrinter::could_be_fence("   "));
        assert!(StreamPrinter::could_be_fence("`"));
        assert!(StreamPrinter::could_be_fence("``"));
        assert!(StreamPrinter::could_be_fence("```"));
        assert!(!StreamPrinter::could_be_fence("hello"));
        assert!(!StreamPrinter::could_be_fence("``a"));
    }

    #[test]
    fn plain_text_streamed_line_by_line() {
        let mut p = StreamPrinter::new("cyan");
        p.feed("hello");
        p.feed(" world\n");
        p.flush();
        // nothing buffered, no fence classification state left
        assert!(p.line.is_empty());
        assert!(!p.in_code);
        assert!(p.code_lines.is_empty());
    }

    #[test]
    fn code_block_classification() {
        let mut p = StreamPrinter::new("cyan");
        p.feed("```python\nprint(1)\n```\nafter\n");
        p.flush();
        assert!(!p.in_code);
        assert!(
            p.code_lines.is_empty(),
            "code buffer flushed on closing fence"
        );
        assert!(p.line.is_empty());
    }

    #[test]
    fn fence_line_is_not_emitted_as_text() {
        let mut p = StreamPrinter::new("cyan");
        p.feed("``\n"); // 2 backticks then newline: NOT a fence -> plain text
        p.flush();
        assert!(!p.in_code);
    }

    #[test]
    fn unclosed_code_flushes() {
        let mut p = StreamPrinter::new("cyan");
        p.feed("```rust\nlet x = 1;\n");
        p.flush();
        assert!(!p.in_code, "flush closes an unterminated code block");
        assert!(p.code_lines.is_empty());
    }

    /// Serialize tests that touch the process-global `OUTPUT_HOOK` and spawn
    /// spinner threads. The hook is a global singleton (Python parity), so
    /// parallel tests would otherwise observe each other's hook state and
    /// leave leaked spinner threads behind.
    fn hook_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::common::output_hook_test_lock()
    }

    /// Panic-safe cleanup for a printer with a running spinner: on drop
    /// (including assertion failures unwinding) it stops the animation loop
    /// and clears the global hook, so a failing test cannot poison later ones.
    struct SpinnerGuard {
        spin: Arc<(Mutex<SpinState>, std::sync::Condvar)>,
    }
    impl Drop for SpinnerGuard {
        fn drop(&mut self) {
            let (lock, cvar) = &*self.spin;
            if let Ok(mut st) = lock.lock() {
                st.active = false;
                st.drawn = false;
                st.stop = true;
            }
            cvar.notify_all();
            crate::common::set_output_hook(None);
        }
    }

    #[test]
    fn spinner_state_transitions() {
        let _hook = hook_guard();
        let mut p = StreamPrinter::new("cyan");
        p.start_spinner("thinking");
        let _spin = SpinnerGuard {
            spin: p.spin.clone(),
        };
        {
            let (lock, _) = &*p.spin;
            let st = lock.lock().unwrap();
            assert!(st.active);
            assert_eq!(st.note, "thinking");
        }
        p.note("working");
        {
            let (lock, _) = &*p.spin;
            let st = lock.lock().unwrap();
            assert_eq!(st.note, "working");
        }
        p.stop_spinner();
        {
            let (lock, _) = &*p.spin;
            let st = lock.lock().unwrap();
            assert!(!st.active);
            assert!(!st.drawn);
        }
    }

    #[test]
    fn output_hook_registered_and_cleared() {
        let _hook = hook_guard();
        let mut p = StreamPrinter::new("cyan");
        assert!(get_output_hook().is_none());
        p.start_spinner("note");
        let _spin = SpinnerGuard {
            spin: p.spin.clone(),
        };
        assert!(
            get_output_hook().is_some(),
            "spinner installs the output hook"
        );
        p.stop_spinner();
        assert!(get_output_hook().is_none(), "stop clears the hook");
        let _ = collect(&mut p);
    }

    #[test]
    fn dirty_line_blocks_spinner_draw() {
        let _hook = hook_guard();
        let mut p = StreamPrinter::new("cyan");
        p.start_spinner("n");
        let _spin = SpinnerGuard {
            spin: p.spin.clone(),
        };
        p.mark_line_clean();
        {
            let (lock, _) = &*p.spin;
            let st = lock.lock().unwrap();
            assert!(!st.dirty_line);
        }
        p.emit("text");
        {
            let (lock, _) = &*p.spin;
            let st = lock.lock().unwrap();
            assert!(st.dirty_line, "emitted text dirties the line");
        }
        p.stop_spinner();
    }

    #[test]
    fn cprint_from_printer_path_uses_hook() {
        // cprint's hook integration must exist (parity with Python set_output_hook).
        // Also serialized: it installs a process-global hook while running.
        let _hook = hook_guard();
        let counter = Arc::new(Mutex::new(0));
        let c2 = counter.clone();
        crate::common::set_output_hook(Some(Arc::new(move || {
            *c2.lock().unwrap() += 1;
        })));
        cprint("x", "grey", false, "\n");
        crate::common::set_output_hook(None);
        assert_eq!(*counter.lock().unwrap(), 1);
    }
}
