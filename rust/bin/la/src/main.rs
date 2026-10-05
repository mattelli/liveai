//! `la` — the Live Assistant main binary (`main.py` port).
//!
//! Modes: voice (default), chat (`--chat`), one-shot (`--prompt`),
//! with `--no-tools` and `--config <path>` switches. Ctrl+C cancels the
//! in-flight generation or quits from the voice prompt; on exit every worker
//! process (listen/speak/imggen) is stopped gracefully.

use anyhow::Result;
use clap::Parser;
use la_core::agent::Agent;
use la_core::brain::OpenAICore;
use la_core::common::cprint;
use la_core::config::load_config;
use la_core::input::{InputReader, ReadOutcome};
use la_core::session::{ConversationSession, PromptSource};
use la_core::workers::Workers;
use la_proto::ListenParams;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Live assistant (voice or chat mode)
#[derive(Parser, Debug)]
#[command(name = "la", about = "Live assistant (voice or chat mode)")]
struct Args {
    /// Run in text chat mode
    #[arg(long)]
    chat: bool,
    /// Disable tool usage
    #[arg(long)]
    no_tools: bool,
    /// Path to config.toml
    #[arg(long)]
    config: Option<PathBuf>,
    /// Run a single prompt and exit
    #[arg(long)]
    prompt: Option<String>,
}

/// Default config path: search upwards from the executable and the current
/// directory (mirrors `WORK_DIR / "config.toml"` of main.py).
fn default_config_file() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut base = exe.parent().map(Path::to_path_buf).unwrap_or_default();
        for _ in 0..4 {
            base = match base.parent() {
                Some(parent) => parent.to_path_buf(),
                None => break,
            };
            candidates.push(base.join("config.toml"));
        }
    }
    candidates.push(PathBuf::from("config.toml"));
    candidates
        .into_iter()
        .find(|c| c.exists())
        .unwrap_or_else(|| PathBuf::from("config.toml"))
}

/// A one-shot reader that yields exactly one prompt then EOF
/// (Python: `iter([args.prompt, None]).__next__`).
struct OneShot {
    prompt: Option<String>,
}

impl PromptSource for OneShot {
    fn read(&mut self) -> ReadOutcome {
        self.prompt
            .take()
            .map(ReadOutcome::Line)
            .unwrap_or(ReadOutcome::Eof)
    }
}

/// The voice prompt source: one listening round on the `la-listen` worker
/// per call (Python `SpeechDetector.listen_and_transcribe`).
struct VoiceReader {
    workers: Arc<Workers>,
    handle: tokio::runtime::Handle,
    params_base: ListenParams,
    silence_timeout_s: f64,
}

impl VoiceReader {
    fn new(workers: Arc<Workers>, config: &la_core::Config) -> Self {
        Self {
            workers,
            handle: tokio::runtime::Handle::current(),
            params_base: ListenParams {
                model_path: config
                    .rust
                    .speech_det_model_path
                    .clone()
                    .unwrap_or(config.speech_det.model_path.clone())
                    .to_string_lossy()
                    .to_string(),
                language: config.speech_det.language.clone(),
                beam_size: config.speech_det.beam_size,
                show_levels: config.speech_det.show_levels,
                calibration_chunks: config.speech_det.calibration_chunks,
                guard_factor: config.speech_det.guard_factor,
                samplerate: 16_000,
                silence_timeout_s: config.speech_det.silence_timeout_s,
            },
            silence_timeout_s: config.speech_det.silence_timeout_s,
        }
    }
}

impl PromptSource for VoiceReader {
    fn read(&mut self) -> ReadOutcome {
        let workers = self.workers.clone();
        let params = ListenParams {
            silence_timeout_s: self.silence_timeout_s,
            ..self.params_base.clone()
        };
        let show_levels = params.show_levels;
        let mut on_level = |rms: f64, threshold: f64| {
            let line = la_audio::format_level(rms, threshold, show_levels);
            if !line.is_empty() {
                cprint(&line, "reset", false, "");
            }
        };
        let result = self
            .handle
            .block_on(async { workers.listen(params, Some(&mut on_level)).await });
        match result {
            Ok(text) => ReadOutcome::Line(text),
            Err(e) => {
                cprint(&format!("❌ {e}"), "red", false, "\n");
                ReadOutcome::Eof
            }
        }
    }
}

/// Setup logging at the configured level (Python `logging.basicConfig`).
fn setup_logging(config: &la_core::Config) {
    let level = config.logging.level.to_uppercase();
    let filter = match level.as_str() {
        "DEBUG" => log::LevelFilter::Debug,
        "INFO" => log::LevelFilter::Info,
        "ERROR" => log::LevelFilter::Error,
        _ => log::LevelFilter::Warn,
    };
    env_logger::Builder::new()
        .filter_level(filter)
        .format_timestamp_secs()
        .try_init()
        .ok();
}

async fn main_inner(args: Args) -> Result<i32> {
    let config_path = args.config.clone().unwrap_or_else(default_config_file);
    let config = load_config(&config_path)?;
    // Workers read the same config file via LA_CONFIG (single-threaded startup).
    // SAFETY: called once from main before any other thread is spawned.
    unsafe {
        std::env::set_var("LA_CONFIG", config_path.to_string_lossy().to_string());
    }
    setup_logging(&config);

    // Brain core first (client + metadata), then the tool context with
    // worker handles and (optionally) the ML encoders.
    let mut core = OpenAICore::new(&config.brain)?;
    core.resolve_metadata().await;
    let workers = Arc::new(Workers::new());
    let context = la_core::tools::registry::ToolContext::with_parts(
        &config,
        core.view(),
        workers.clone(),
        encoders_for(&config),
    );
    let agent = Agent::with_parts(&config, core, context.clone()).await?;

    // Speech hook: agent responses are spoken through the speak worker
    // when speech is enabled.
    let speak_ctx = context.clone();
    agent.set_speak(Arc::new(move |text: String| {
        let ctx = speak_ctx.clone();
        tokio::spawn(async move {
            if let Err(e) = ctx.workers.speak_if_enabled(&text).await {
                log::debug!("speak failed: {e}");
            }
        });
    }));

    // Prompt source wiring (Python `_build_chat_reader` / SpeechDetector).
    let reader: Arc<std::sync::Mutex<dyn PromptSource>> = if args.prompt.is_some() {
        // Non-interactive: exactly one turn, no stdin reads.
        Arc::new(std::sync::Mutex::new(OneShot {
            prompt: args.prompt.clone(),
        }))
    } else if args.chat {
        let input_cfg = &config.input;
        let history = input_cfg
            .history_file
            .clone()
            .filter(|p| !p.as_os_str().is_empty() && p != &PathBuf::from("."));
        Arc::new(std::sync::Mutex::new(InputReader::new(
            history,
            input_cfg.auto_suggest,
            input_cfg.completion,
            input_cfg.multiline,
            input_cfg.explicit_paste,
            input_cfg.prompt_toolkit,
            "... ",
        )))
    } else {
        match la_audio::InputDeviceInfo::discover() {
            Err(e) => {
                cprint(&format!("❌ {e}"), "red", false, "\n");
                return Ok(1);
            }
            Ok(device) => {
                log::info!(
                    "ℹ️  Using audio input device: {} (index {})",
                    device.name,
                    device.index
                );
                Arc::new(std::sync::Mutex::new(VoiceReader::new(workers, &config)))
            }
        }
    };

    let mut session = ConversationSession::new(
        agent,
        reader,
        context.clone(),
        !args.no_tools,
        !args.chat, // voice / non-interactive: Ctrl+C quits
    );
    let code = session.run().await;
    context.workers.shutdown().await;
    Ok(code)
}

/// Wire the candle-based encoders (compiled only with `--features ml`).
#[cfg(feature = "ml")]
fn encoders_for(config: &la_core::Config) -> la_core::encoders::Encoders {
    let device = la_ml::resolve_device(&config.summarization.device);
    let text = la_ml::minilm::MiniLM::new(&config.summarization.model_path, device.clone())
        .map(|m| Arc::new(m) as Arc<dyn la_core::encoders::TextEmbedder>)
        .ok();
    let clip = la_ml::clip::ClipEmbedder::new(
        config
            .embedding
            .image_model_path
            .clone()
            .unwrap_or_else(|| config.embedding.model_path.clone()),
        device.clone(),
    )
    .map(|c| {
        let text: Arc<dyn la_core::encoders::TextEmbedder> = Arc::new(c.clone());
        let image: Arc<dyn la_core::encoders::ImageEmbedder> = Arc::new(c);
        (text, image)
    })
    .ok();
    let cross =
        la_ml::cross_encoder::TinyBertCrossEncoder::new(&config.document_query.model_path, device)
            .map(|c| Arc::new(c) as Arc<dyn la_core::encoders::CrossEncoder>)
            .ok();
    la_core::encoders::Encoders {
        text: text.clone(),
        clip_text: clip.as_ref().map(|(t, _)| t.clone()).or(text),
        image: clip.map(|(_, i)| i),
        cross,
    }
}

#[cfg(not(feature = "ml"))]
fn encoders_for(_config: &la_core::Config) -> la_core::encoders::Encoders {
    la_core::encoders::Encoders::default()
}

fn main() -> std::process::ExitCode {
    // Windows hardening (PLAN §3.3): put the main process into a Job object
    // with KILL_ON_JOB_CLOSE. Workers spawned later inherit the job, so they
    // cannot outlive a crashed main process. Best-effort: without it the
    // graceful stop path (workers.shutdown) still handles normal exits.
    #[cfg(windows)]
    {
        let mut info = win32job::ExtendedLimitInfo::new();
        info.limit_kill_on_job_close();
        match win32job::Job::create_with_limit_info(&info) {
            Ok(job) => {
                // Leak the handle: closing it would kill every process in the
                // job, including this one.
                if let Err(e) = job.assign_current_process() {
                    log::debug!("Job object assignment failed: {e:?}");
                } else {
                    std::mem::forget(job);
                }
            }
            Err(e) => log::debug!("Job object creation failed: {e:?}"),
        }
    }
    let args = Args::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let code = runtime.block_on(async move {
        match main_inner(args).await {
            Ok(code) => code,
            Err(e) => {
                cprint(&format!("❌ {e}"), "red", false, "\n");
                1
            }
        }
    });
    std::process::ExitCode::from(code as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Args {
        Args::parse_from(std::iter::once("la").chain(args.iter().copied()))
    }

    #[test]
    fn defaults_match_python_parse_args() {
        // main.py: --chat/--no-tools store_true (false), --prompt None,
        // --config defaults to CONFIG_FILE next to the binary.
        let args = parse(&[]);
        assert!(!args.chat);
        assert!(!args.no_tools);
        assert!(args.prompt.is_none());
        assert!(args.config.is_none());
    }

    #[test]
    fn switches_are_recognized() {
        let args = parse(&["--chat", "--no-tools"]);
        assert!(args.chat);
        assert!(args.no_tools);

        let args = parse(&["--prompt", "hello world"]);
        assert_eq!(args.prompt.as_deref(), Some("hello world"));
        assert!(!args.chat);

        let args = parse(&["--config", "some/other.toml"]);
        assert_eq!(
            args.config.as_deref(),
            Some(std::path::Path::new("some/other.toml"))
        );
    }

    #[test]
    fn prompt_and_chat_are_mutually_exclusive_by_usage() {
        // argparse allows combining --chat and --prompt; the session runs the
        // one-shot turn (Python: args.prompt takes precedence in main()).
        let args = parse(&["--chat", "--prompt", "x"]);
        assert!(args.chat && args.prompt.is_some());
    }

    #[test]
    fn unknown_flag_is_rejected() {
        assert!(Args::try_parse_from(["la", "--nope"]).is_err());
    }

    #[test]
    fn default_config_file_prefers_existing_candidates() {
        // Must always return *a* path (mirrors CONFIG_FILE fallback semantics).
        assert!(!default_config_file().as_os_str().is_empty());
    }
}
