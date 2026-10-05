//! `la-speak` — worker process: Piper text-to-speech (sherpa-onnx VITS voices)
//! synthesis and playback, one JSON-lines request per utterance.
//!
//! Protocol: `enable` creates the generator from `[speech_gen]` in the config
//! (path via `LA_CONFIG` env var, set by the main process), `disable` releases
//! it; `speak` synthesizes + plays and answers with `spoken` (bytes written);
//! `speak` with `file` writes a WAV instead (Python `to_wav`); `unload` frees
//! the voice. `stop` / stdin EOF ends the process.
//!
//! Build the real backend with `--features tts` (sherpa-rs, prebuilt binaries).

use anyhow::{Result, anyhow};
use la_proto::{Envelope, MainToWorker, WorkerToMain};
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};

/// Remove formatting characters and emoji so they are not read aloud
/// (Python `SpeechGenerator.clean_for_speech`).
pub fn clean_for_speech(text: &str) -> String {
    // markdown formatting characters: **, __, *, _, #, `, >, ~, [ ] ( )
    let formatting = regex::Regex::new(r"[*_#`>~\[\]()]").unwrap();
    // emoji and other pictographic symbols (same ranges as the Python code)
    let emoji = regex::Regex::new(
        r"[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{1F1E6}-\u{1F1FF}\u{FE0F}\u{200D}\u{2190}-\u{21FF}\u{2B00}-\u{2BFF}]",
    )
    .unwrap();
    let whitespace = regex::Regex::new(r"\s+").unwrap();
    whitespace
        .replace_all(
            &emoji.replace_all(&formatting.replace_all(text, ""), ""),
            " ",
        )
        .trim()
        .to_string()
}

#[cfg(feature = "tts")]
mod backend {
    use super::*;
    use sherpa_rs::tts::{VitsTts, VitsTtsConfig};

    /// The Piper voice via sherpa-onnx VITS; loaded once per process.
    pub struct SpeechGenerator {
        tts: VitsTts,
        sample_rate: usize,
        volume: f64,
    }

    impl SpeechGenerator {
        pub fn new(config: &crate::SpeechGenCfg) -> Result<Self> {
            let model_path = std::path::PathBuf::from(&config.model_path);
            if !model_path.exists() {
                return Err(anyhow!("Voice model file not found: {}", config.model_path));
            }
            if !(0.0..=1.0).contains(&config.volume) {
                return Err(anyhow!("volume must be between 0.0 and 1.0"));
            }
            if config.speech_speed <= 0.0 {
                return Err(anyhow!("speech_speed must be greater than 0"));
            }
            let json_path = model_path.with_extension("onnx.json");
            log::info!(
                "🚀 Loading Audio Generator model from {} ...",
                config.model_path
            );
            let tts_config = VitsTtsConfig {
                model: model_path.clone(),
                tokens: Some(model_path.with_file_name("tokens.txt")),
                data_dir: Some(model_path.parent().unwrap_or_default().to_path_buf()),
                lexicon: None,
                dict_dir: None,
                sample_rate: config.samplerate as i32,
                noise_scale: config.noise_scale as f32,
                noise_scale_w: config.noise_w_scale as f32,
                length_scale: (1.0 / config.speech_speed) as f32,
                chunk_size: 1024,
                phonemize: true,
                use_rule_fst: false,
                espeak_voice: None,
            };
            let tts = VitsTts::new(tts_config).map_err(|e| anyhow!("voice load: {e}"))?;
            // samplerate check (Python _ensure_voice)
            let voice_rate = tts.sample_rate();
            if voice_rate != config.samplerate as usize {
                return Err(anyhow!(
                    "Output samplerate {} does not match voice samplerate {}; set [speech_gen] samplerate = {}",
                    config.samplerate,
                    voice_rate,
                    voice_rate
                ));
            }
            Ok(Self {
                tts,
                sample_rate: voice_rate,
                volume: config.volume,
            })
        }

        /// Synthesize to f32 samples (volume applied as digital gain).
        pub fn synthesize(&mut self, text: &str) -> Result<(usize, Vec<f32>)> {
            let audio = self
                .tts
                .synthesize(text, 0, 1.0, None)
                .map_err(|e| anyhow!("synthesize: {e}"))?;
            let samples: Vec<f32> = audio
                .samples
                .iter()
                .map(|&s| (s * self.volume as f32).clamp(-1.0, 1.0))
                .collect();
            Ok((audio.sample_rate, samples))
        }

        pub fn sample_rate(&self) -> usize {
            self.sample_rate
        }
    }
}

/// Worker state (Python module-level `_speech_generator`).
#[derive(Default)]
struct State {
    #[cfg(feature = "tts")]
    generator: Option<backend::SpeechGenerator>,
    enabled: bool,
}

fn send(w: &mut impl Write, id: Option<u64>, body: &WorkerToMain) -> Result<()> {
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
            MainToWorker::Enable => {
                if let Err(e) = enable(&state) {
                    send(
                        &mut writer,
                        id,
                        &WorkerToMain::Error {
                            message: e.to_string(),
                        },
                    )?;
                }
            }
            MainToWorker::Disable => {
                let mut guard = state.lock().unwrap();
                guard.enabled = false;
                #[cfg(feature = "tts")]
                {
                    guard.generator = None;
                }
            }
            MainToWorker::Speak { text, strip, file } => {
                if let Err(e) = speak(&state, &text, strip, file.as_deref(), &mut writer) {
                    send(
                        &mut writer,
                        id,
                        &WorkerToMain::Error {
                            message: e.to_string(),
                        },
                    )?;
                }
            }
            MainToWorker::Unload => {
                let mut guard = state.lock().unwrap();
                #[cfg(feature = "tts")]
                {
                    guard.generator = None;
                }
                guard.enabled = false;
                let _ = send(
                    &mut writer,
                    id,
                    &WorkerToMain::Done {
                        path: String::new(),
                        seed: None,
                        seconds: 0.0,
                    },
                );
            }
            MainToWorker::Stop => return Ok(()),
            other => {
                send(
                    &mut writer,
                    id,
                    &WorkerToMain::Error {
                        message: format!("la-speak does not handle {other:?}"),
                    },
                )?;
            }
        }
    }
}

/// Create the speech generator from `[speech_gen]` of the shared config.
fn enable(state: &Arc<Mutex<State>>) -> Result<()> {
    #[cfg(feature = "tts")]
    {
        let mut guard = state.lock().unwrap();
        if guard.generator.is_some() {
            return Ok(());
        }
        let config = config_speech_gen()?;
        guard.generator = Some(backend::SpeechGenerator::new(&config)?);
        guard.enabled = true;
        return Ok(());
    }
    #[cfg(not(feature = "tts"))]
    {
        let _state = state;
        Err(anyhow!(
            "la-speak was built without tts support; rebuild with `cargo build -p la-speak --features tts` \
             (sherpa-rs with prebuilt sherpa-onnx binaries)"
        ))
    }
}

#[cfg(feature = "tts")]
fn config_speech_gen() -> Result<crate::SpeechGenCfg> {
    let path = std::env::var("LA_CONFIG").unwrap_or_else(|_| "config.toml".into());
    Ok(la_core::config::load_config(&path)?.speech_gen)
}

/// Synthesize + play (or write a WAV); answers with `spoken`/`done`.
fn speak(
    _state: &Arc<Mutex<State>>,
    text: &str,
    strip: bool,
    file: Option<&str>,
    writer: &mut impl Write,
) -> Result<()> {
    let _started = std::time::Instant::now();
    let text = if strip {
        log::debug!("Removing formatted syntax...");
        clean_for_speech(text)
    } else {
        text.to_string()
    };
    if text.is_empty() {
        log::debug!("Nothing to speak after cleaning; skipping.");
        if file.is_some() {
            return Err(anyhow!("Cannot synthesize WAV: empty text"));
        }
        send(writer, None, &WorkerToMain::Spoken { bytes: 0 })?;
        return Ok(());
    }

    #[cfg(feature = "tts")]
    {
        let mut guard = state.lock().unwrap();
        if !guard.enabled || guard.generator.is_none() {
            // Parity: speak only when enabled.
            send(writer, None, &WorkerToMain::Spoken { bytes: 0 })?;
            return Ok(());
        }
        let generator = guard.generator.as_mut().unwrap();
        let (sample_rate, samples) = generator.synthesize(&text)?;
        log::info!("🔊 Streaming output audio ...");
        if let Some(path) = file {
            let i16_samples: Vec<i16> = samples.iter().map(|&s| (s * 32_767.0) as i16).collect();
            let path = la_audio::audio_to_wav_file(&i16_samples, sample_rate as u32, path)?;
            log::info!(
                "💾 Audio saved to {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            );
            send(
                writer,
                None,
                &WorkerToMain::Done {
                    path: path.to_string_lossy().to_string(),
                    seed: None,
                    seconds: started.elapsed().as_secs_f64(),
                },
            )?;
            return Ok(());
        }
        let bytes = play(&samples, sample_rate as u32)?;
        log::info!("🔊 Finished streaming audio.");
        send(
            writer,
            None,
            &WorkerToMain::Spoken {
                bytes: bytes as u64,
            },
        )?;
        return Ok(());
    }
    #[cfg(not(feature = "tts"))]
    Err(anyhow!(
        "la-speak was built without tts support; rebuild with `cargo build -p la-speak --features tts`"
    ))
}

/// Play f32 samples through the default output device; returns bytes written.
#[cfg(feature = "tts")]
fn play(samples: &[f32], _samplerate: u32) -> Result<usize> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::collections::VecDeque;
    use std::sync::Condvar;
    use std::sync::atomic::{AtomicBool, Ordering};

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("No audio output device available"))?;
    let supported = device
        .default_output_config()
        .map_err(|e| anyhow!("output config: {e}"))?;
    let mut config = supported.config();
    config.channels = 1;

    let queue: Arc<(Mutex<VecDeque<f32>>, Condvar)> =
        Arc::new((Mutex::new(VecDeque::from(samples.to_vec())), Condvar::new()));
    let done = Arc::new(AtomicBool::new(false));
    let q2 = queue.clone();
    let d2 = done.clone();
    let sample_rate_cfg = config.sample_rate;
    let err_fn = |e| log::debug!("cpal output error: {e}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            config,
            move |out: &mut [f32], _| drain_f32(&q2, out, &d2),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_output_stream(
            config,
            move |out: &mut [i16], _| drain_i16(&q2, out, &d2),
            err_fn,
            None,
        ),
        _ => return Err(anyhow!("unsupported output sample format")),
    }
    .map_err(|e| anyhow!("open output stream: {e}"))?;
    let _ = sample_rate_cfg;
    stream
        .play()
        .map_err(|e| anyhow!("start output stream: {e}"))?;

    // Wait until the queue is drained (Python writes are synchronous).
    let (lock, cvar) = &*queue;
    let mut guard = lock.lock().unwrap();
    while !guard.is_empty() {
        guard = cvar
            .wait_timeout(guard, std::time::Duration::from_millis(50))
            .unwrap()
            .0;
    }
    drop(guard);
    // let the device play the tail
    std::thread::sleep(std::time::Duration::from_millis(120));
    done.store(true, Ordering::SeqCst);
    let _ = stream.pause();
    drop(stream);
    Ok(samples.len() * 2) // int16 bytes
}

#[cfg(feature = "tts")]
fn drain_f32(
    queue: &(Mutex<VecDeque<f32>>, std::sync::Condvar),
    out: &mut [f32],
    done: &std::sync::atomic::AtomicBool,
) {
    use std::sync::atomic::Ordering;
    let (lock, cvar) = queue;
    let mut q = lock.lock().unwrap();
    for slot in out.iter_mut() {
        *slot = q.pop_front().unwrap_or(0.0);
    }
    if q.is_empty() || done.load(Ordering::SeqCst) {
        cvar.notify_all();
    }
}

#[cfg(feature = "tts")]
fn drain_i16(
    queue: &(Mutex<VecDeque<f32>>, std::sync::Condvar),
    out: &mut [i16],
    done: &std::sync::atomic::AtomicBool,
) {
    use std::sync::atomic::Ordering;
    let (lock, cvar) = queue;
    let mut q = lock.lock().unwrap();
    for slot in out.iter_mut() {
        let sample = q.pop_front().unwrap_or(0.0);
        *slot = (sample * 32_767.0).clamp(-32768.0, 32767.0) as i16;
    }
    if q.is_empty() || done.load(Ordering::SeqCst) {
        cvar.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_for_speech_matches_python() {
        assert_eq!(
            clean_for_speech("**Hello** `world` [link](url) #tag"),
            "Hello world linkurl tag"
        );
        assert_eq!(clean_for_speech("Hello 👋 world!"), "Hello world!");
        assert_eq!(clean_for_speech("multi\n\nspace\ttext"), "multi space text");
        assert_eq!(clean_for_speech(""), "");
        assert_eq!(clean_for_speech("***"), "");
    }
}
