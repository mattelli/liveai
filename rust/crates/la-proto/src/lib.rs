//! JSON-lines IPC protocol for the Live Assistant worker processes
//! (`la-listen`, `la-speak`, `la-imggen`).
//!
//! Every message is one JSON object per UTF-8 line. Requests from the main
//! process carry a monotonic `id`; the worker answers with the same `id`.
//! Unolicited progress events (audio levels, notes) carry no `id`.

use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

pub const PROTOCOL_VERSION: u64 = 1;

/// A framed message: protocol version, optional request id, typed body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub v: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    #[serde(flatten)]
    pub body: T,
}

impl<T> Envelope<T> {
    pub fn new(body: T) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: None,
            body,
        }
    }
    pub fn with_id(id: u64, body: T) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: Some(id),
            body,
        }
    }
}

/// Parameters for one `listen` request (`la-listen`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenParams {
    /// whisper.cpp ggml model path (file or directory containing a .bin)
    pub model_path: String,
    /// ISO 639-1 language code (e.g. "en")
    pub language: String,
    pub beam_size: u32,
    pub show_levels: bool,
    /// Ambient-noise calibration chunk count (first request only)
    pub calibration_chunks: usize,
    pub guard_factor: f64,
    pub samplerate: u32,
    /// Per-request silence timeout in seconds
    pub silence_timeout_s: f64,
}

/// Text-to-image generation request (`la-imggen`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T2iParams {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    /// negative -> pick a random seed
    pub seed: i64,
    pub steps: u32,
    pub out_file: String,
}

/// Image-to-image / inpainting request (`la-imggen`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct I2iParams {
    pub image: String,
    /// required for inpaint mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    pub denoise: f64,
    pub guidance: f64,
    pub out_file: String,
    pub inpaint: bool,
}

/// Body of a message sent by the main process to a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MainToWorker {
    Listen(ListenParams),
    /// Synthesize `text`; when `strip` is true the formatting/emoji cleaner runs.
    /// When `file` is set, write a WAV file instead of playing.
    Speak {
        text: String,
        strip: bool,
        file: Option<String>,
    },
    /// Enable (create) or disable (release) the speech generator.
    Enable,
    Disable,
    T2i(T2iParams),
    I2i(I2iParams),
    /// Free the loaded diffusion pipeline (imggen) / speech generator (speak).
    Unload,
    /// Graceful stop; the worker exits its loop.
    Stop,
}

/// Body of a message sent by a worker to the main process.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerToMain {
    /// Live microphone level (listen worker, only when show_levels is on).
    Level { rms: f64, threshold: f64 },
    /// Transcription of the captured utterance ("" when no speech detected).
    Transcript { text: String },
    /// Audio playback finished; number of bytes written.
    Spoken { bytes: u64 },
    /// A WAV file was written / the image was saved / unload finished.
    Done {
        path: String,
        seed: Option<i64>,
        seconds: f64,
    },
    /// Unrecoverable error for the pending request (or a background failure).
    Error { message: String },
    /// Spinner/progress note (e.g. calibration status).
    Note { text: String },
}

/// Write one message as a JSON line.
pub fn write_msg<T: Serialize>(w: &mut dyn Write, msg: &T) -> io::Result<()> {
    let line =
        serde_json::to_string(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    w.write_all(line.as_bytes())?;
    w.write_all(b"\n")?;
    w.flush()
}

/// Read one JSON line; `Ok(None)` on EOF.
pub fn read_msg<T: for<'de> Deserialize<'de>>(r: &mut impl BufRead) -> io::Result<Option<T>> {
    let mut line = String::new();
    let n = r.read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
    let msg = serde_json::from_str(line.trim())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_listen() {
        let env = Envelope::with_id(
            7,
            MainToWorker::Listen(ListenParams {
                model_path: "m.bin".into(),
                language: "en".into(),
                beam_size: 5,
                show_levels: true,
                calibration_chunks: 20,
                guard_factor: 1.0,
                samplerate: 16000,
                silence_timeout_s: 2.0,
            }),
        );
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("\"type\":\"listen\""));
        let back: Envelope<MainToWorker> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, Some(7));
        match back.body {
            MainToWorker::Listen(p) => assert_eq!(p.beam_size, 5),
            _ => panic!("wrong body"),
        }
    }

    #[test]
    fn round_trip_all_bodies() {
        let bodies = vec![
            MainToWorker::Speak {
                text: "hi".into(),
                strip: true,
                file: None,
            },
            MainToWorker::Enable,
            MainToWorker::Disable,
            MainToWorker::Unload,
            MainToWorker::Stop,
            MainToWorker::T2i(T2iParams {
                prompt: "p".into(),
                negative_prompt: None,
                width: 512,
                height: 512,
                seed: 42,
                steps: 50,
                out_file: "o.png".into(),
            }),
            MainToWorker::I2i(I2iParams {
                image: "i.png".into(),
                mask: Some("m.png".into()),
                prompt: "p".into(),
                negative_prompt: None,
                denoise: 0.5,
                guidance: 5.0,
                out_file: "o.png".into(),
                inpaint: true,
            }),
        ];
        for b in bodies {
            let json = serde_json::to_string(&b).unwrap();
            let back: MainToWorker = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&back).unwrap(), json);
        }
    }

    #[test]
    fn round_trip_worker_to_main() {
        let bodies = vec![
            WorkerToMain::Level {
                rms: 0.1,
                threshold: 0.2,
            },
            WorkerToMain::Transcript {
                text: "hello".into(),
            },
            WorkerToMain::Spoken { bytes: 1024 },
            WorkerToMain::Done {
                path: "a.png".into(),
                seed: Some(1),
                seconds: 1.5,
            },
            WorkerToMain::Error {
                message: "boom".into(),
            },
            WorkerToMain::Note {
                text: "calibrating".into(),
            },
        ];
        for b in bodies {
            let json = serde_json::to_string(&b).unwrap();
            let back: WorkerToMain = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&back).unwrap(), json);
        }
    }

    #[test]
    fn framing() {
        let mut buf = Vec::new();
        write_msg(&mut buf, &Envelope::with_id(1, MainToWorker::Stop)).unwrap();
        let mut cursor = io::BufReader::new(&buf[..]);
        let msg: Option<Envelope<MainToWorker>> = read_msg(&mut cursor).unwrap();
        assert!(msg.is_some());
        let msg: Option<Envelope<MainToWorker>> = read_msg(&mut cursor).unwrap();
        assert!(msg.is_none(), "EOF must read as None");
    }
}
