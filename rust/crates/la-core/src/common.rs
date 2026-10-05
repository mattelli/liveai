//! Console helpers (`src/common.py` port): colored printing, prompt input,
//! the pre-output hook used to suppress the activity spinner, JSON file IO
//! and the `[CALL]` logging helper.

use std::io::Write;
use std::sync::{Arc, OnceLock, RwLock};

pub type Color = &'static str; // 'black','red','green','yellow','blue','magenta','cyan','white','grey','reset'

type OutputHook = Arc<dyn Fn() + Send + Sync>;
static OUTPUT_HOOK: OnceLock<RwLock<Option<OutputHook>>> = OnceLock::new();

fn output_hook() -> &'static RwLock<Option<OutputHook>> {
    OUTPUT_HOOK.get_or_init(|| RwLock::new(None))
}

/// Register/unregister a callback invoked before every `cprint`
/// (used to clear the activity spinner before raw prints).
pub fn set_output_hook(hook: Option<OutputHook>) {
    *output_hook().write().unwrap() = hook;
}

/// The registered output hook, if any (used by the StreamPrinter).
pub fn get_output_hook() -> Option<OutputHook> {
    output_hook().read().unwrap().clone()
}

/// Test-only serialization for the process-global hook: every test that
/// installs, clears or asserts on the output hook (in this module or in
/// `printer`) must hold this lock, otherwise parallel tests race on the
/// shared singleton (§14.4 printer-test isolation).
#[cfg(test)]
pub(crate) fn output_hook_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn color_code(color: &str) -> &'static str {
    match color {
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
    }
}

/// Print text in the terminal with ANSI colors.
///
/// Examples: `cprint("Hello", "green")`, `cprint("Error", "red", false, "", true)`
pub fn cprint(text: &str, color: Color, bold: bool, end: &str) {
    if let Some(hook) = get_output_hook() {
        hook();
    }
    let style = if bold { "1;" } else { "" };
    let line = format!("\x1b[{}{}m{}\x1b[0m", style, color_code(color), text);
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(line.as_bytes());
    let _ = lock.write_all(end.as_bytes());
    let _ = lock.flush();
}

/// Convenience overloads matching the common Python call shapes.
pub fn cprint_text(text: &str, color: Color) {
    cprint(text, color, false, "\n");
}

/// Write label (flushed, no newline) and read a line.
///
/// Returns the raw line (without trailing newline), or None on EOF.
pub fn prompt_input(label: &str, color: Color) -> Option<String> {
    cprint(label, color, false, "");
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => None, // EOF
        Ok(_) => Some(line.trim_end_matches(['\n', '\r']).to_string()),
        Err(_) => None,
    }
}

/// Read a JSON file into a serde value.
pub fn read_json(path: impl AsRef<std::path::Path>) -> anyhow::Result<serde_json::Value> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

/// Write a JSON file (pretty, indent 2).
pub fn write_json(
    path: impl AsRef<std::path::Path>,
    data: &serde_json::Value,
) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(data)?;
    std::fs::write(path, text)?;
    Ok(())
}

/// Truncate a string to `n` chars (like `str(value)[:50]` in Python).
pub fn trunc(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Log a function call with truncated parameter values
/// (Python `@dump_func_call` decorator equivalent; tools call this at entry).
pub fn log_call(name: &str, args: &[(&str, String)]) {
    let joined: Vec<String> = args
        .iter()
        .map(|(k, v)| format!("{}={:?}", k, trunc(v, 50)))
        .collect();
    cprint(
        &format!("[CALL] {}({})", name, joined.join(", ")),
        "grey",
        true,
        "\n",
    );
}

#[macro_export]
macro_rules! log_call {
    ($name:expr) => {
        $crate::common::log_call($name, &[])
    };
    ($name:expr $(, $k:expr => $v:expr)* $(,)?) => {
        $crate::common::log_call($name, &[$(($k, $v.to_string())),*])
    };
}

/// The grey `---` + context line printed between act() rounds (Python parity).
pub fn context_line(usage_pct: f64, context_window: u64) {
    cprint(
        &format!(
            "{}\ncontext: {:.2} % [{}k]\n",
            "-".repeat(50),
            usage_pct,
            context_window / 1000
        ),
        "grey",
        false,
        "\n",
    );
}

/// The "generation cancelled" placeholder padded into the history (Python parity).
pub const CANCELLED_PLACEHOLDER: &str = "(generation cancelled by user)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_codes() {
        assert_eq!(color_code("red"), "31");
        assert_eq!(color_code("cyan"), "36");
        assert_eq!(color_code("grey"), "90");
        assert_eq!(color_code("reset"), "0");
        assert_eq!(color_code("unknown"), "0");
    }

    #[test]
    fn trunc_works() {
        assert_eq!(trunc("abcdef", 4), "abcd");
        assert_eq!(trunc("", 4), "");
    }

    #[test]
    fn output_hook_fires() {
        let _guard = output_hook_test_lock();
        let counter = Arc::new(std::sync::Mutex::new(0));
        let c2 = counter.clone();
        set_output_hook(Some(Arc::new(move || {
            *c2.lock().unwrap() += 1;
        })));
        cprint("hook test", "red", false, "\n");
        set_output_hook(None);
        cprint("no hook", "red", false, "\n");
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn json_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.json");
        let value = serde_json::json!({"a": 1, "b": [1, 2, 3]});
        write_json(&path, &value).unwrap();
        let back = read_json(&path).unwrap();
        assert_eq!(back, value);
    }
}
