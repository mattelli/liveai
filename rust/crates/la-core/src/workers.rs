//! Supervisor of the worker processes (`la-listen`, `la-speak`, `la-imggen`).
//!
//! The main process owns no audio/ML backends: listening, speech and image
//! generation each run in a separate process that speaks the JSON-lines
//! protocol from `la-proto`. Workers are spawned on demand, restarted when
//! they died, and all of them are stopped (graceful, then killed) by the
//! Ctrl+C shutdown path. Workers also exit on stdin EOF (parent-death
//! watchdog); on Windows the main binary additionally attaches them to a Job
//! object with KILL_ON_JOB_CLOSE so orphans cannot survive a crashed parent.
//!
//! Workers locate the shared `config.toml` through the `LA_CONFIG` env var
//! (inherited from the main process at spawn).

use anyhow::{Context, Result, anyhow};
use la_proto::{Envelope, ListenParams, MainToWorker, T2iParams, WorkerToMain};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{broadcast, oneshot};

/// The worker-facing part of a spawned process (cloneable: shared via Arcs).
#[derive(Clone)]
struct WorkerHandle {
    stdin: Arc<tokio::sync::Mutex<tokio::process::ChildStdin>>,
    next_id: Arc<AtomicU64>,
    pending: Arc<std::sync::Mutex<HashMap<u64, oneshot::Sender<WorkerToMain>>>>,
    events: broadcast::Sender<WorkerToMain>,
    child: Arc<tokio::sync::Mutex<tokio::process::Child>>,
    alive: Arc<AtomicBool>,
}

/// Spawn one worker process and start its response/event reader task.
fn spawn_worker(exe_dir: &Path, name: &str) -> Result<WorkerHandle> {
    #[cfg(windows)]
    let exe_name = format!("{name}.exe");
    #[cfg(not(windows))]
    let exe_name = name.to_string();
    let path = exe_dir.join(exe_name);
    if !path.exists() {
        return Err(anyhow!(
            "worker binary not found: '{}' (build the workspace first)",
            path.display()
        ));
    }
    let mut command = tokio::process::Command::new(&path);
    command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let mut child = command.spawn().with_context(|| format!("spawn {name}"))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("{name}: no stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("{name}: no stdout"))?;
    let pending: Arc<std::sync::Mutex<HashMap<u64, oneshot::Sender<WorkerToMain>>>> =
        Arc::new(std::sync::Mutex::new(HashMap::new()));
    let (events, _) = broadcast::channel::<WorkerToMain>(256);
    let alive = Arc::new(AtomicBool::new(true));

    let pending_task = pending.clone();
    let events_task = events.clone();
    let alive_task = alive.clone();
    let name_task = name.to_string();
    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let envelope: Envelope<WorkerToMain> = match serde_json::from_str(&line) {
                Ok(e) => e,
                Err(e) => {
                    log::debug!("{name_task}: unparsable line: {e}");
                    continue;
                }
            };
            if let Some(id) = envelope.id {
                let sender = pending_task.lock().unwrap().remove(&id);
                if let Some(sender) = sender {
                    let _ = sender.send(envelope.body);
                    continue;
                }
            }
            let _ = events_task.send(envelope.body);
        }
        alive_task.store(false, Ordering::SeqCst);
    });

    Ok(WorkerHandle {
        stdin: Arc::new(tokio::sync::Mutex::new(stdin)),
        next_id: Arc::new(AtomicU64::new(1)),
        pending,
        events,
        child: Arc::new(tokio::sync::Mutex::new(child)),
        alive,
    })
}

/// Send a helper for the mpsc-free design: unused marker type.
pub type WorkerEvent = WorkerToMain;

pub struct Workers {
    exe_dir: PathBuf,
    listen: std::sync::Mutex<Option<WorkerHandle>>,
    speak: std::sync::Mutex<Option<WorkerHandle>>,
    imggen: std::sync::Mutex<Option<WorkerHandle>>,
    speech_enabled: AtomicBool,
}

impl Workers {
    pub fn new() -> Self {
        Self {
            exe_dir: std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from(".")),
            listen: std::sync::Mutex::new(None),
            speak: std::sync::Mutex::new(None),
            imggen: std::sync::Mutex::new(None),
            speech_enabled: AtomicBool::new(false),
        }
    }

    fn ensure(
        &self,
        slot: &std::sync::Mutex<Option<WorkerHandle>>,
        name: &str,
    ) -> Result<WorkerHandle> {
        let mut guard = slot.lock().unwrap();
        let needs_spawn = match guard.as_ref() {
            Some(handle) => !handle.alive.load(Ordering::SeqCst),
            None => true,
        };
        if needs_spawn {
            *guard = Some(spawn_worker(&self.exe_dir, name)?);
        }
        Ok(guard.as_ref().unwrap().clone())
    }

    /// Send `body` and await the matching response.
    async fn call(
        slot: &std::sync::Mutex<Option<WorkerHandle>>,
        exe_dir: &Path,
        name: &str,
        body: MainToWorker,
    ) -> Result<WorkerToMain> {
        let handle = Self::ensure_slot(slot, exe_dir, name)?;
        let id = handle.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        handle.pending.lock().unwrap().insert(id, tx);
        let envelope = Envelope::with_id(id, body);
        let mut line = serde_json::to_string(&envelope)?;
        line.push('\n');
        {
            let mut stdin = handle.stdin.lock().await;
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await?;
        }
        match rx.await {
            Ok(response) => Ok(response),
            Err(_) => Err(anyhow!("{name} worker died before responding")),
        }
    }

    fn ensure_slot(
        slot: &std::sync::Mutex<Option<WorkerHandle>>,
        exe_dir: &Path,
        name: &str,
    ) -> Result<WorkerHandle> {
        let mut guard = slot.lock().unwrap();
        let needs_spawn = match guard.as_ref() {
            Some(handle) => !handle.alive.load(Ordering::SeqCst),
            None => true,
        };
        if needs_spawn {
            *guard = Some(spawn_worker(exe_dir, name)?);
        }
        Ok(guard.as_ref().unwrap().clone())
    }

    /// Stop one worker gracefully (Stop, wait ≤2s, kill).
    async fn stop_one(slot: &std::sync::Mutex<Option<WorkerHandle>>) {
        let handle = slot.lock().unwrap().take();
        let Some(handle) = handle else { return };
        let stop = Envelope::new(MainToWorker::Stop);
        if let Ok(line) = serde_json::to_string(&stop) {
            let mut stdin = handle.stdin.lock().await;
            let _ = stdin.write_all(line.as_bytes()).await;
            let _ = stdin.write_all(b"\n").await;
            let _ = stdin.flush().await;
            drop(stdin);
        }
        // wait for exit up to 2 s, then kill
        let waited = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            let mut guard = handle.child.lock().await;
            let _ = guard.wait().await;
        })
        .await;
        if waited.is_err() {
            let mut guard = handle.child.lock().await;
            let _ = guard.kill().await;
        }
    }

    // ------------------------------------------------------------ listening

    /// Run one listening round on `la-listen`; level events are surfaced via
    /// the callback while the transcript is awaited.
    pub async fn listen(
        &self,
        params: ListenParams,
        mut on_level: Option<&mut (dyn FnMut(f64, f64) + Send)>,
    ) -> Result<String> {
        let handle = self.ensure(&self.listen, "la-listen")?;
        let mut events = handle.events.subscribe();
        let id = handle.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, mut rx) = oneshot::channel();
        handle.pending.lock().unwrap().insert(id, tx);
        let envelope = Envelope::with_id(id, MainToWorker::Listen(params));
        let mut line = serde_json::to_string(&envelope)?;
        line.push('\n');
        {
            let mut stdin = handle.stdin.lock().await;
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await?;
        }

        // Await the transcript while surfacing level events.
        loop {
            tokio::select! {
                response = &mut rx => {
                    return match response {
                        Ok(WorkerToMain::Transcript { text }) => Ok(text),
                        Ok(WorkerToMain::Error { message }) => Err(anyhow!("{message}")),
                        Ok(other) => Err(anyhow!("unexpected listen response: {other:?}")),
                        Err(_) => Err(anyhow!("la-listen worker died before responding")),
                    };
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                    while let Ok(event) = events.try_recv() {
                        if let WorkerToMain::Level { rms, threshold } = event
                            && let Some(cb) = on_level.as_deref_mut()
                        {
                            cb(rms, threshold);
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------- speech

    /// Enable the speech generator (spawn `la-speak` + Enable).
    pub async fn speak_enable(&self) -> Result<()> {
        Self::call(&self.speak, &self.exe_dir, "la-speak", MainToWorker::Enable)
            .await
            .map(|_| ())?;
        self.speech_enabled.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Disable the speech generator.
    pub async fn speak_disable(&self) {
        let _ = Self::call(
            &self.speak,
            &self.exe_dir,
            "la-speak",
            MainToWorker::Disable,
        )
        .await;
        self.speech_enabled.store(false, Ordering::SeqCst);
    }

    /// Speak `text` when speech is enabled (Python `speak_if_enabled`).
    pub async fn speak_if_enabled(&self, text: &str) -> Result<()> {
        if !self.speech_enabled.load(Ordering::SeqCst) {
            return Ok(());
        }
        match Self::call(
            &self.speak,
            &self.exe_dir,
            "la-speak",
            MainToWorker::Speak {
                text: text.to_string(),
                strip: true,
                file: None,
            },
        )
        .await
        {
            Ok(WorkerToMain::Spoken { .. }) | Ok(WorkerToMain::Done { .. }) => Ok(()),
            Ok(WorkerToMain::Error { message }) => Err(anyhow!("{message}")),
            Ok(other) => Err(anyhow!("unexpected speak response: {other:?}")),
            Err(e) => {
                self.speech_enabled.store(false, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    // ---------------------------------------------------------- image gen

    /// Generate one image on `la-imggen`; returns (path, seed used).
    pub async fn generate_image(&self, params: &T2iParams) -> Result<(PathBuf, Option<i64>)> {
        match Self::call(
            &self.imggen,
            &self.exe_dir,
            "la-imggen",
            MainToWorker::T2i(params.clone()),
        )
        .await?
        {
            WorkerToMain::Done { path, seed, .. } => Ok((PathBuf::from(path), seed)),
            WorkerToMain::Error { message } => Err(anyhow!("{message}")),
            other => Err(anyhow!("unexpected imggen response: {other:?}")),
        }
    }

    /// Free the loaded diffusion pipeline.
    pub async fn unload_image_generator(&self) -> String {
        match Self::call(
            &self.imggen,
            &self.exe_dir,
            "la-imggen",
            MainToWorker::Unload,
        )
        .await
        {
            Ok(WorkerToMain::Done { .. }) => "Image generation model unloaded, memory freed".into(),
            Ok(WorkerToMain::Error { message })
                if message.contains("No image generation model is loaded") =>
            {
                "No image generation model is loaded".into()
            }
            Ok(_) => "Image generation model unloaded, memory freed".into(),
            Err(_) => "No image generation model is loaded".into(),
        }
    }

    // ------------------------------------------------------------ shutdown

    /// Stop every worker (graceful, then kill). Ctrl+C shutdown path.
    pub async fn shutdown(&self) {
        let (a, b, c) = tokio::join!(
            Self::stop_one(&self.listen),
            Self::stop_one(&self.speak),
            Self::stop_one(&self.imggen),
        );
        let _ = (a, b, c);
    }
}

impl Default for Workers {
    fn default() -> Self {
        Self::new()
    }
}
