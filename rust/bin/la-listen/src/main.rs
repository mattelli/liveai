//! `la-listen` — worker process: microphone capture + speech detection +
//! Whisper (whisper.cpp) transcription, one JSON-lines request per utterance.
//!
//! Protocol (stdin/stdout, see `la-proto`): `listen` requests answer with a
//! `transcript` (or `error`); live RMS levels stream as unsolicited `level`
//! events; `stop` / stdin EOF ends the process. The whisper model loads once
//! and the ambient-noise calibration runs once, both persisting across
//! requests (Python SpeechDetector behavior, one process per session).
//!
//! Build the real backend with `--features whisper` (whisper.cpp ggml models).

use anyhow::{Result, anyhow};
use la_audio::{
    AudioError, CaptureParams, ChunkedReader, capture_speech, compute_silence_threshold,
};
use la_proto::{Envelope, ListenParams, MainToWorker, WorkerToMain};
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};

#[cfg(feature = "whisper")]
mod backend {
    use super::*;
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    /// whisper.cpp STT: the ggml model is loaded once per process.
    pub struct Whisper {
        ctx: WhisperContext,
        language: String,
        beam_size: u32,
    }

    impl Whisper {
        pub fn new(model_path: &str, language: &str, beam_size: u32) -> Result<Self> {
            let file = resolve_model_file(model_path).ok_or_else(|| {
                anyhow!("Model file not found: {model_path} (whisper.cpp needs a ggml .bin model)")
            })?;
            log::info!("🚀 Loading Audio to Text model from {} ...", file.display());
            let ctx = WhisperContext::new_with_params(
                file.to_string_lossy().as_ref(),
                WhisperContextParameters::default(),
            )
            .map_err(|e| anyhow!("whisper load: {e}"))?;
            log::info!("✅ Model loaded.");
            Ok(Self {
                ctx,
                language: language.to_string(),
                beam_size,
            })
        }

        /// Transcribe int16 mono samples at 16 kHz.
        pub fn transcribe(&self, samples: &[i16]) -> Result<String> {
            let floats = whisper_rs::convert_integer_to_float_audio(samples)
                .map_err(|e| anyhow!("audio convert: {e}"))?;
            let mut state = self
                .ctx
                .create_state()
                .map_err(|e| anyhow!("whisper state: {e}"))?;
            let mut params = FullParams::new(SamplingStrategy::BeamSearch {
                beam_size: self.beam_size.clamp(1, 64) as i32,
                best_of: -1,
            });
            if !self.language.is_empty() {
                params.set_language(Some(&self.language));
            }
            params.set_print_progress(false);
            params.set_print_special(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            state
                .full(params, &floats)
                .map_err(|e| anyhow!("whisper inference: {e}"))?;
            let n = state
                .full_n_segments()
                .map_err(|e| anyhow!("whisper segments: {e}"))?;
            let mut text = String::new();
            for i in 0..n {
                let segment = state
                    .full_get_segment_text(i)
                    .map_err(|e| anyhow!("whisper segment text: {e}"))?;
                text.push_str(&segment);
            }
            Ok(text.trim().to_string())
        }
    }

    /// Resolve a ggml model file: the path itself, or a directory containing
    /// the first `*.bin` file (the Python config points at the faster-whisper
    /// CT2 folder; whisper.cpp needs ggml — see rust/README.md).
    pub fn resolve_model_file(model_path: &str) -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(model_path);
        if path.is_file() {
            return Some(path);
        }
        if path.is_dir() {
            let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(&path)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "bin").unwrap_or(false))
                .collect();
            entries.sort();
            return entries.into_iter().next();
        }
        None
    }
}

/// Session state of the worker (model + calibration threshold).
#[derive(Default)]
struct State {
    #[cfg(feature = "whisper")]
    whisper: Option<backend::Whisper>,
    threshold: Option<f64>,
}

fn send(w: &mut dyn Write, id: Option<u64>, body: &WorkerToMain) -> Result<()> {
    let envelope = match id {
        Some(id) => Envelope::with_id(id, body.clone()),
        None => Envelope::new(body.clone()),
    };
    la_proto::write_msg(w, &envelope)?;
    Ok(())
}

fn main() -> std::process::ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .try_init()
        .ok();
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // surface fatal setup errors to the parent before exiting
            let mut out = BufWriter::new(std::io::stdout().lock());
            let _ = send(
                &mut out,
                None,
                &WorkerToMain::Error {
                    message: e.to_string(),
                },
            );
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut writer = BufWriter::new(std::io::stdout().lock());
    let state: Arc<Mutex<State>> = Arc::new(Mutex::new(State::default()));

    loop {
        let Some(envelope) = la_proto::read_msg::<Envelope<MainToWorker>>(&mut reader)? else {
            return Ok(()); // parent closed stdin
        };
        let id = envelope.id;
        match envelope.body {
            MainToWorker::Listen(params) => {
                if let Err(e) = listen_round(&state, &params, &mut writer, id) {
                    send(
                        &mut writer,
                        id,
                        &WorkerToMain::Error {
                            message: e.to_string(),
                        },
                    )?;
                }
            }
            MainToWorker::Stop => return Ok(()),
            other => {
                send(
                    &mut writer,
                    id,
                    &WorkerToMain::Error {
                        message: format!("la-listen does not handle {other:?}"),
                    },
                )?;
            }
        }
    }
}

/// One capture + transcribe round.
fn listen_round(
    state: &Arc<Mutex<State>>,
    params: &ListenParams,
    writer: &mut dyn Write,
    request_id: Option<u64>,
) -> Result<()> {
    let samplerate = params.samplerate.max(8_000);

    // 1) ensure the whisper model is loaded (once per process)
    #[cfg(feature = "whisper")]
    {
        let mut guard = state.lock().unwrap();
        if guard.whisper.is_none() {
            guard.whisper = Some(backend::Whisper::new(
                &params.model_path,
                &params.language,
                params.beam_size,
            )?);
        }
    }

    // 2) calibrate the silence threshold once
    let threshold = {
        let mut guard = state.lock().unwrap();
        if guard.threshold.is_none() {
            let _ = send(
                writer,
                None,
                &WorkerToMain::Note {
                    text: "⚙️  Calibrating silence threshold...".into(),
                },
            );
            writer.flush()?;
            let mut factory = open_stream();
            let (threshold, mean, std) = compute_silence_threshold(
                &mut factory,
                samplerate,
                0.1,
                params.calibration_chunks.max(1),
                params.guard_factor,
            )?;
            log::info!(
                "✅ Silence threshold set to: {threshold:.3}, (Mean RMS: {mean:.3}, Std RMS: {std:.3})"
            );
            guard.threshold = Some(threshold);
        }
        guard.threshold.unwrap()
    };

    // 3) capture speech with level events
    let show_levels = params.show_levels;
    let mut on_level = |rms: f64| {
        let line = la_audio::format_level(rms, threshold, show_levels);
        if !line.is_empty() {
            let _ = send(writer, None, &WorkerToMain::Level { rms, threshold });
        }
    };
    let mut factory = open_stream();
    let audio = capture_speech(
        &mut factory,
        CaptureParams {
            threshold,
            samplerate,
            chunk_time_s: 0.1,
            silence_timeout_s: params.silence_timeout_s.max(0.1),
            clear_timeout_s: 30.0,
            max_record_chunks: None,
        },
        Some(&mut on_level),
        &|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0)
        },
    )?;

    // 4) transcribe
    let text = if audio.is_empty() {
        String::new()
    } else {
        #[cfg(feature = "whisper")]
        {
            let guard = state.lock().unwrap();
            match &guard.whisper {
                Some(whisper) => whisper.transcribe(&audio)?,
                None => return Err(anyhow!("whisper model not loaded")),
            }
        }
        #[cfg(not(feature = "whisper"))]
        {
            let _ = &audio;
            return Err(anyhow!(
                "la-listen was built without whisper support; rebuild with `cargo build -p la-listen --features whisper` \
                 and store a whisper.cpp ggml model (see rust/README.md)"
            ));
        }
    };
    log::debug!("Text Length: {}", text.len());
    send(writer, request_id, &WorkerToMain::Transcript { text })?;
    Ok(())
}

/// Adapter so the capture loop's `&mut dyn FnMut(u32) -> ...` factory seam can
/// open a fresh stream per calibration/capture call.
fn open_stream() -> impl FnMut(u32) -> Result<Box<dyn ChunkedReader>, AudioError> {
    move |_rate| {
        let reader = la_audio::CpalChunkedReader::open_default()?;
        Ok(Box::new(reader) as Box<dyn ChunkedReader>)
    }
}
