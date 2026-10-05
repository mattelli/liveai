//! Shared audio primitives for the Live Assistant.
//!
//! Rust port of the capture half of `src/audio_wrapper.py`:
//! input-device discovery, RMS levels, WAV buffers/files, ambient-noise
//! calibration and the chunked speech-capture loop with silence timeout.
//!
//! The blocking "read a chunk" seam (`ChunkedReader`) keeps the capture loop
//! unit-testable without hardware, mirroring the Python tests' fake
//! `InputStream`.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavReader, WavWriter};
use std::collections::VecDeque;
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const DEFAULT_SAMPLERATE: u32 = 16_000;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// No usable audio input device is available.
    #[error("No audio input device available. Check microphone settings.")]
    NoInputDevice,
    /// No audio input devices found.
    #[error("No audio input devices found")]
    NoDevices,
    #[error("audio stream error: {0}")]
    Stream(String),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("cannot write WAV: no audio samples")]
    EmptyAudio,
}

/// Info about the input device used for capture (Python `InputAudioDevice`).
#[derive(Debug, Clone)]
pub struct InputDeviceInfo {
    pub name: String,
    pub index: usize,
    pub sample_rate: u32,
}

impl InputDeviceInfo {
    /// The default input device of the default audio host.
    pub fn discover() -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevices)?;
        let name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "Unknown Device".into());
        let default_config = device
            .default_input_config()
            .map_err(|e| AudioError::Stream(format!("input config: {e}")));
        let sample_rate = default_config
            .as_ref()
            .map(|c| c.sample_rate())
            .unwrap_or(DEFAULT_SAMPLERATE);
        Ok(Self {
            name,
            index: 0,
            sample_rate,
        })
    }
}

/// Blocking "read exactly `size` int16 samples" seam (Python `stream.read`).
pub trait ChunkedReader: Send {
    /// Read exactly `size` samples; may return fewer when the reader is closed.
    fn read_chunk(&mut self, size: usize) -> io::Result<Vec<i16>>;
    /// Stop the underlying device (safe to call more than once).
    fn close(&mut self) {}
}

/// A `cpal`-backed reader: a callback thread pushes samples into a ring
/// buffer; [`ChunkedReader::read_chunk`] pops a chunk, blocking until the
/// requested size is available (Python `InputStream.read(chunk_size)`).
pub struct CpalChunkedReader {
    shared: Arc<(Mutex<VecDeque<i16>>, Condvar)>,
    abort: Arc<AtomicBool>,
    stream: Mutex<Option<cpal::Stream>>,
    device_name: String,
}

impl CpalChunkedReader {
    /// Open the default input device as a chunked reader.
    pub fn open_default() -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevices)?;
        let name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "Unknown Device".into());
        let default_config = device
            .default_input_config()
            .map_err(|e| AudioError::Stream(format!("input config: {e}")))?;
        let sample_format = default_config.sample_format();
        // Force mono (channels=1) like the Python capture path.
        let mut config = default_config.config();
        config.channels = 1;
        let shared: Arc<(Mutex<VecDeque<i16>>, Condvar)> =
            Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let abort = Arc::new(AtomicBool::new(false));
        let push = shared.clone();
        let err_fn = move |e| log::debug!("cpal input error: {e}");
        let stream = match sample_format {
            cpal::SampleFormat::I16 => device.build_input_stream(
                config,
                move |d: &[i16], _| push_all(&push, d),
                err_fn,
                None,
            ),
            cpal::SampleFormat::F32 => device.build_input_stream(
                config,
                move |d: &[f32], _| push_f32(&push, d),
                err_fn,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                config,
                move |d: &[u16], _| push_u16(&push, d),
                err_fn,
                None,
            ),
            _ => return Err(AudioError::Stream("unsupported sample format".into())),
        }
        .map_err(|e| AudioError::Stream(format!("open input stream: {e}")))?;
        stream
            .play()
            .map_err(|e| AudioError::Stream(format!("start input stream: {e}")))?;
        Ok(Self {
            shared,
            abort,
            stream: Mutex::new(Some(stream)),
            device_name: name,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

fn push_all(shared: &(Mutex<VecDeque<i16>>, Condvar), data: &[i16]) {
    if shared.0.lock().unwrap().len() > 1 << 22 {
        return; // drop samples when the consumer is stuck (protect memory)
    }
    shared.0.lock().unwrap().extend(data.iter().copied());
    shared.1.notify_all();
}

fn push_f32(shared: &(Mutex<VecDeque<i16>>, Condvar), data: &[f32]) {
    let scaled: Vec<i16> = data
        .iter()
        .map(|&s| (s * 32_767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();
    push_all(shared, &scaled);
}

fn push_u16(shared: &(Mutex<VecDeque<i16>>, Condvar), data: &[u16]) {
    let scaled: Vec<i16> = data.iter().map(|&s| (s as i32 - 32_768) as i16).collect();
    push_all(shared, &scaled);
}

impl ChunkedReader for CpalChunkedReader {
    fn read_chunk(&mut self, size: usize) -> io::Result<Vec<i16>> {
        let (lock, cvar) = &*self.shared;
        let mut queue = lock.lock().unwrap();
        let mut deadline = std::time::Instant::now() + Duration::from_secs(3600);
        loop {
            if queue.len() >= size {
                let out: Vec<i16> = queue.drain(..size).collect();
                return Ok(out);
            }
            if self.abort.load(Ordering::Relaxed) {
                let out: Vec<i16> = queue.drain(..).collect();
                if out.is_empty() {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "stream closed"));
                }
                return Ok(out);
            }
            let (guard, _timeout) = cvar
                .wait_timeout(queue, Duration::from_millis(100))
                .unwrap();
            queue = guard;
            if std::time::Instant::now() >= deadline {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "read timed out"));
            }
            // Reset the (generous) deadline while data keeps flowing.
            if !queue.is_empty() {
                deadline = std::time::Instant::now() + Duration::from_secs(3600);
            }
        }
    }

    fn close(&mut self) {
        self.abort.store(true, Ordering::Relaxed);
        if let Some(stream) = self.stream.lock().unwrap().take() {
            let _ = stream.pause();
            drop(stream);
        }
        self.shared.1.notify_all();
    }
}

impl Drop for CpalChunkedReader {
    fn drop(&mut self) {
        self.close();
    }
}

/// Root-mean-square level of int16 samples. 0.0 for empty input.
pub fn rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_sq: f64 = samples
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / samples.len() as f64;
    mean_sq.sqrt()
}

/// Generate an int16 sine tone. The peak sample is `amplitude * 32767`.
pub fn get_audio_tone(duration: f64, samplerate: u32, frequency: f64, amplitude: f64) -> Vec<i16> {
    let n = (samplerate as f64 * duration) as usize;
    let two_pi_f = 2.0 * std::f64::consts::PI * frequency / samplerate as f64;
    (0..n)
        .map(|i| (amplitude * (two_pi_f * i as f64).sin() * 32_767.0) as i16)
        .collect()
}

/// Build the live RMS level line. Empty string when show is False.
pub fn format_level(level: f64, threshold: f64, show: bool) -> String {
    if !show {
        return String::new();
    }
    format!("\rIN-Audio RMS: {level:.3} | Silence threshold [{threshold:.3}] ")
}

/// Encode int16 mono samples as an in-memory WAV file.
pub fn audio_to_wav_buffer(samples: &[i16], samplerate: u32) -> Result<Vec<u8>, AudioError> {
    if samples.is_empty() {
        return Err(AudioError::EmptyAudio);
    }
    let mut buf = Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: samplerate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    {
        let mut writer =
            WavWriter::new(&mut buf, spec).map_err(|e| AudioError::Io(io::Error::other(e)))?;
        for &s in samples {
            writer
                .write_sample(s)
                .map_err(|e| AudioError::Io(io::Error::other(e)))?;
        }
        writer
            .finalize()
            .map_err(|e| AudioError::Io(io::Error::other(e)))?;
    }
    Ok(buf.into_inner())
}

/// Write audio to a WAV file and return its path.
pub fn audio_to_wav_file(
    samples: &[i16],
    samplerate: u32,
    out_file: impl AsRef<Path>,
) -> Result<PathBuf, AudioError> {
    let data = audio_to_wav_buffer(samples, samplerate)?;
    let path = out_file.as_ref().to_path_buf();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, data)?;
    Ok(path)
}

/// Read int16 mono samples back from a WAV file (tests / debugging).
pub fn wav_to_audio(path: impl AsRef<Path>) -> Result<Vec<i16>, AudioError> {
    let mut reader = WavReader::open(path).map_err(|e| AudioError::Io(io::Error::other(e)))?;
    let mut out = Vec::new();
    for sample in reader.samples::<i16>() {
        out.push(sample.map_err(|e| AudioError::Io(io::Error::other(e)))?);
    }
    Ok(out)
}

/// A blocking factory that opens one capture stream (test seam).
pub type StreamFactory<'a> = &'a mut dyn FnMut(u32) -> Result<Box<dyn ChunkedReader>, AudioError>;

/// Return `(threshold, mean_rms, std_rms)` sampled from ambient noise.
///
/// Python: `compute_silence_threshold(device, ...)`; the stream is opened,
/// `chunks` RMS values are collected, then the stream is closed.
pub fn compute_silence_threshold(
    factory: StreamFactory<'_>,
    samplerate: u32,
    chunk_time_s: f64,
    chunks: usize,
    guard_factor: f64,
) -> Result<(f64, f64, f64), AudioError> {
    let chunk_size = (chunk_time_s * samplerate as f64) as usize;
    let mut stream = factory(samplerate)?;
    let mut values = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        let chunk = stream.read_chunk(chunk_size)?;
        values.push(rms(&chunk));
    }
    stream.close();
    if values.is_empty() {
        values.push(0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
    let std = var.sqrt();
    Ok((guard_factor * (mean + 2.0 * std), mean, std))
}

/// Parameters of one speech capture (Python `capture_speech` defaults).
#[derive(Debug, Clone)]
pub struct CaptureParams {
    pub threshold: f64,
    pub samplerate: u32,
    pub chunk_time_s: f64,
    pub silence_timeout_s: f64,
    pub clear_timeout_s: f64,
    /// Bound the total number of chunks read (None = unbounded).
    pub max_record_chunks: Option<usize>,
}

impl Default for CaptureParams {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            samplerate: DEFAULT_SAMPLERATE,
            chunk_time_s: 0.1,
            silence_timeout_s: 1.5,
            clear_timeout_s: 30.0,
            max_record_chunks: None,
        }
    }
}

/// Record until `silence_timeout_s` of consecutive sub-threshold chunks.
///
/// Returns the trimmed speech samples, or an empty vector when no speech was
/// detected. Direct port of `src.audio_wrapper.capture_speech`.
pub fn capture_speech(
    factory: StreamFactory<'_>,
    params: CaptureParams,
    mut on_level: Option<&mut dyn FnMut(f64)>,
    time_now: &dyn Fn() -> f64,
) -> Result<Vec<i16>, AudioError> {
    let chunk_size = (params.chunk_time_s * params.samplerate as f64) as usize;
    let timeout_chunks = (params.silence_timeout_s / params.chunk_time_s).max(1.0) as usize;
    let mut chunks: Vec<Vec<i16>> = Vec::new();
    let mut speech_started = false;
    let mut silent_chunks: usize = 0;
    let mut start_chunk: usize = 0;
    let mut start_time = time_now();
    let mut read_count: usize = 0;

    let mut stream = factory(params.samplerate)?;
    let result = loop {
        let chunk = match stream.read_chunk(chunk_size) {
            Ok(c) => c,
            Err(e) => break Err(AudioError::Stream(format!("read failed: {e}"))),
        };
        read_count += 1;
        let level = rms(&chunk);
        if let Some(cb) = on_level.as_deref_mut() {
            cb(level);
        }
        chunks.push(chunk);
        if level > params.threshold {
            if !speech_started {
                speech_started = true;
                start_chunk = chunks.len() - 1;
            }
            silent_chunks = 0;
        } else if speech_started {
            silent_chunks += 1;
            if silent_chunks >= timeout_chunks {
                break Ok(());
            }
        }
        if !speech_started && (time_now() - start_time) > params.clear_timeout_s {
            chunks.clear();
            silent_chunks = 0;
            start_time = time_now();
        }
        if let Some(max) = params.max_record_chunks
            && read_count >= max
        {
            break Ok(());
        }
    };
    stream.close();
    result?;

    let empty: Vec<i16> = Vec::new();
    if !speech_started || chunks.is_empty() {
        return Ok(empty);
    }
    let tolerance = timeout_chunks / 2;
    let begin = start_chunk.saturating_sub(tolerance);
    let end = chunks.len().saturating_sub(tolerance).max(begin);
    let audio: Vec<i16> = if begin >= end {
        chunks.concat()
    } else {
        chunks[begin..end].concat()
    };
    if rms(&audio) < params.threshold {
        return Ok(empty);
    }
    Ok(audio)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed-sequence fake stream (mirrors tests/conftest.py FakeStream).
    struct FakeStream {
        chunks: std::vec::IntoIter<Vec<i16>>,
    }
    impl FakeStream {
        fn new(chunks: Vec<Vec<i16>>) -> Box<Self> {
            Box::new(Self {
                chunks: chunks.into_iter(),
            })
        }
    }
    impl ChunkedReader for FakeStream {
        fn read_chunk(&mut self, size: usize) -> io::Result<Vec<i16>> {
            match self.chunks.next() {
                Some(c) => {
                    assert_eq!(c.len(), size, "fake chunks must match chunk_size");
                    Ok(c)
                }
                None => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "exhausted")),
            }
        }
    }

    fn factory(
        chunks: Vec<Vec<i16>>,
    ) -> impl FnMut(u32) -> Result<Box<dyn ChunkedReader>, AudioError> {
        move |_| Ok(FakeStream::new(chunks.clone()))
    }

    #[test]
    fn rms_of_silence_and_tone() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[0, 0, 0]), 0.0);
        let tone = get_audio_tone(1.0, 8000, 440.0, 1.0);
        assert!((rms(&tone) - 32767.0 / 2f64.sqrt()).abs() < 500.0);
    }

    #[test]
    fn tone_peak() {
        let tone = get_audio_tone(0.1, 8000, 440.0, 0.5);
        assert!(tone.iter().any(|&s| s.abs() > 15_000));
    }

    #[test]
    fn silence_threshold_from_values() {
        let chunks: Vec<Vec<i16>> = (0..20).map(|i| vec![(i * 100) as i16; 1600]).collect();
        let (threshold, mean, std) =
            compute_silence_threshold(&mut factory(chunks), 16000, 0.1, 20, 1.0).unwrap();
        assert!(threshold > mean + std);
        assert_eq!(threshold, mean + 2.0 * std);
    }

    #[test]
    fn capture_detects_speech_and_trims() {
        // 3 quiet chunks, 3 loud, 3 quiet (timeout=2 chunks -> stop after 2 quiet)
        let quiet = vec![0i16; 1600];
        let loud = vec![30_000i16; 1600];
        let chunks = vec![
            quiet.clone(),
            quiet.clone(),
            quiet.clone(),
            loud.clone(),
            loud.clone(),
            loud.clone(),
            quiet.clone(),
            quiet.clone(),
        ];
        let levels = std::cell::RefCell::new(Vec::new());
        let mut cb = |l: f64| levels.borrow_mut().push(l);
        let audio = capture_speech(
            &mut factory(chunks),
            CaptureParams {
                threshold: 1000.0,
                silence_timeout_s: 0.2,
                chunk_time_s: 0.1,
                ..Default::default()
            },
            Some(&mut cb),
            &(|| 0.0),
        )
        .unwrap();
        assert!(!audio.is_empty());
        assert!(audio.contains(&30_000));
        assert!(levels.borrow().len() >= 8);
    }

    #[test]
    fn capture_no_speech_returns_empty() {
        let quiet = vec![0i16; 1600];
        let chunks: Vec<Vec<i16>> = (0..5).map(|_| quiet.clone()).collect();
        let audio = capture_speech(
            &mut factory(chunks),
            CaptureParams {
                threshold: 1000.0,
                max_record_chunks: Some(5),
                ..Default::default()
            },
            None,
            &(|| 0.0),
        )
        .unwrap();
        assert!(audio.is_empty());
    }

    #[test]
    fn wav_round_trip() {
        let samples = get_audio_tone(0.05, 8000, 440.0, 0.8);
        let bytes = audio_to_wav_buffer(&samples, 8000).unwrap();
        assert!(!bytes.is_empty());
        let path = std::env::temp_dir().join(format!("la_audio_test_{}.wav", std::process::id()));
        let written = audio_to_wav_file(&samples, 8000, &path).unwrap();
        assert_eq!(written, path);
        let back = wav_to_audio(&path).unwrap();
        assert_eq!(back.len(), samples.len());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn format_level_matches_python() {
        assert_eq!(
            format_level(0.5, 0.3, true),
            "\rIN-Audio RMS: 0.500 | Silence threshold [0.300] "
        );
        assert_eq!(format_level(0.5, 0.3, false), "");
    }
}
