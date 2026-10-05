//! `la-imggen` — worker process: local Stable-Diffusion image generation
//! (candle, txt2img / img2img / inpaint), one JSON-lines request per image.
//!
//! Protocol: `t2i` / `i2i` generate + save and answer with `done`
//! (path, seed used, seconds); `unload` frees the pipeline (Python
//! `unload_t2i_pipe` / `unload_image_generator`); `stop` / stdin EOF ends the
//! process. The pipeline is loaded once and survives across requests
//! (Python module-level cache), keeping the main process free of heavy ML
//! dependencies.
//!
//! Build the real backend with `--features sd` (candle + tokenizers). Models
//! come from a local folder or the HuggingFace cache (see rust/setup_rust_res.ps1).

use anyhow::{Result, anyhow};
use la_proto::{Envelope, I2iParams, MainToWorker, T2iParams, WorkerToMain};
use std::io::{BufWriter, Write};

/// Worker state: the loaded pipeline (Python `_generator` module-level cache).
/// The worker is single-threaded, so a `static mut` behind a small accessor is
/// used instead of a Mutex (the loop processes one request at a time).
#[cfg(feature = "sd")]
mod backend {
    use anyhow::{Result, anyhow};
    use candle_core::{D, DType, Device, IndexOp, Tensor};
    use candle_nn::Module;
    use candle_transformers::models::stable_diffusion as sd;
    use sd::StableDiffusionConfig;
    use tokenizers::Tokenizer;

    /// One loaded SD model (tokenizer + CLIP + UNet + VAE).
    pub struct LoadedModel {
        tokenizer: Tokenizer,
        clip: sd::clip::ClipTextTransformer,
        unet: sd::unet_2d::UNet2DConditionModel,
        vae: sd::vae::AutoEncoderKL,
        config: StableDiffusionConfig,
        device: Device,
        dtype: DType,
        use_guide_scale: bool,
        guidance_scale: f64,
    }

    /// Resolve the torch dtype for the device (Python `_torch_dtype`).
    pub fn torch_dtype(device: &Device) -> DType {
        if device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        }
    }

    fn resolve_model_dir(model: &str) -> Result<std::path::PathBuf> {
        let path = std::path::PathBuf::from(model);
        if path.exists() && path.is_dir() {
            return Ok(path);
        }
        // HuggingFace id: use the local HF cache (downloaded by setup_rust_res.ps1)
        let cache = std::env::var("HF_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("USERPROFILE")
                    .or_else(|_| std::env::var("HOME"))
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|_| std::path::PathBuf::from("."))
                    .join(".cache/huggingface")
            });
        let hub = cache.join("hub");
        let safe = model.replace('/', "--");
        let dir = hub.join(format!("models--{safe}"));
        if !dir.exists() {
            return Err(anyhow!(
                "model folder '{model}' not found locally; run rust/setup_rust_res.ps1 or point [image_gen] at a local folder"
            ));
        }
        for entry in dir.join("snapshots").read_dir()?.flatten() {
            if entry.path().join("unet").is_dir() {
                return Ok(entry.path());
            }
        }
        Err(anyhow!(
            "model '{model}' found in the HF cache but no usable snapshot"
        ))
    }

    /// The inpaint variant needs the inpainting repo layout.
    pub const INPAINT_MODEL: &str = "stable-diffusion-v1-5/stable-diffusion-inpainting";

    impl LoadedModel {
        /// Load a full text-to-image model (Python `load_t2i_pipe`).
        pub fn load_t2i(model: &str, device: Device, dtype: DType) -> Result<Self> {
            let dir = resolve_model_dir(model)?;
            println!(
                "Loading text2image model '{model}' on {} ...",
                if device.is_cuda() { "cuda" } else { "cpu" }
            );
            let clip_weights = dir.join("text_encoder/model.safetensors");
            let tokenizer_file = dir.join("tokenizer/tokenizer.json");
            let unet_weights = dir.join("unet/diffusion_pytorch_model.safetensors");
            let vae_weights = dir.join("vae/diffusion_pytorch_model.safetensors");
            let config = StableDiffusionConfig::v1_5(false, None, None);
            let tokenizer =
                Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow!("tokenizer: {e}"))?;
            let clip =
                sd::build_clip_transformer(&config.clip, &clip_weights, &device, DType::F32)?;
            let vae = config.build_vae(&vae_weights, &device, dtype)?;
            let unet = config.build_unet(&unet_weights, &device, 4, false, dtype)?;
            println!("Model loaded.");
            Ok(Self {
                tokenizer,
                clip,
                unet,
                vae,
                config,
                device,
                dtype,
                use_guide_scale: true,
                guidance_scale: 7.5,
            })
        }

        /// Load the image-to-image (or inpainting) model (Python
        /// `load_i2i_pipe`); the two variants are mutually exclusive, so this
        /// replaces any loaded t2i model.
        pub fn load_i2i(
            model: &str,
            device: Device,
            dtype: DType,
            in_paint: bool,
            guidance: f64,
        ) -> Result<Self> {
            let dir = resolve_model_dir(model)?;
            let mode = if in_paint { "inpaint" } else { "img2img" };
            println!(
                "Loading {mode} model '{model}' on {} ...",
                if device.is_cuda() { "cuda" } else { "cpu" }
            );
            let clip_weights = dir.join("text_encoder/model.safetensors");
            let tokenizer_file = dir.join("tokenizer/tokenizer.json");
            let unet_weights = dir.join("unet/diffusion_pytorch_model.safetensors");
            let vae_weights = dir.join("vae/diffusion_pytorch_model.safetensors");
            let config = StableDiffusionConfig::v1_5(false, None, None);
            let tokenizer =
                Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow!("tokenizer: {e}"))?;
            let clip =
                sd::build_clip_transformer(&config.clip, &clip_weights, &device, DType::F32)?;
            let vae = config.build_vae(&vae_weights, &device, dtype)?;
            let in_channels = if in_paint { 9 } else { 4 };
            let unet = config.build_unet(&unet_weights, &device, in_channels, false, dtype)?;
            println!("Model loaded.");
            Ok(Self {
                tokenizer,
                clip,
                unet,
                vae,
                config,
                device,
                dtype,
                use_guide_scale: guidance > 1.0,
                guidance_scale: guidance,
            })
        }

        fn text_embeddings(&self, prompt: &str, uncond_prompt: &str) -> Result<Tensor> {
            let pad_id = match &self.config.clip.pad_with {
                Some(padding) => *self
                    .tokenizer
                    .get_vocab(true)
                    .get(padding.as_str())
                    .unwrap(),
                None => *self.tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
            };
            let mut tokens = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            if tokens.len() > self.config.clip.max_position_embeddings {
                anyhow::bail!(
                    "the prompt is too long, {} > max-tokens ({})",
                    tokens.len(),
                    self.config.clip.max_position_embeddings
                );
            }
            while tokens.len() < self.config.clip.max_position_embeddings {
                tokens.push(pad_id);
            }
            let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let text_embeddings = self.clip.forward(&tokens)?;

            if self.use_guide_scale {
                let mut uncond_tokens = self
                    .tokenizer
                    .encode(uncond_prompt, true)
                    .map_err(|e| anyhow!("encode: {e}"))?
                    .get_ids()
                    .to_vec();
                if uncond_tokens.len() > self.config.clip.max_position_embeddings {
                    anyhow::bail!(
                        "the negative prompt is too long, {} > max-tokens ({})",
                        uncond_tokens.len(),
                        self.config.clip.max_position_embeddings
                    );
                }
                while uncond_tokens.len() < self.config.clip.max_position_embeddings {
                    uncond_tokens.push(pad_id);
                }
                let uncond_tokens =
                    Tensor::new(uncond_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let uncond_embeddings = self.clip.forward(&uncond_tokens)?;
                Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(self.dtype)
            } else {
                text_embeddings.to_dtype(self.dtype)
            }
        }

        /// Generate an image from a text prompt (Python `text2image`).
        pub fn text2image(
            &self,
            prompt: &str,
            negative_prompt: &str,
            width: u32,
            height: u32,
            seed: u64,
            steps: usize,
        ) -> Result<image::DynamicImage> {
            self.device.set_seed(seed)?;
            let mut scheduler = self.config.build_scheduler(steps)?;
            let text_embeddings = self.text_embeddings(prompt, negative_prompt)?;
            let latents = Tensor::randn(
                0f32,
                1f32,
                (1usize, 4, (height / 8) as usize, (width / 8) as usize),
                &self.device,
            )?;
            let mut latents = (latents * scheduler.init_noise_sigma())?.to_dtype(self.dtype)?;
            let use_guide_scale = self.use_guide_scale;

            for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };
                let latent_model_input =
                    scheduler.scale_model_input(latent_model_input, timestep)?;
                let noise_pred =
                    self.unet
                        .forward(&latent_model_input, timestep as f64, &text_embeddings)?;
                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
                    (noise_pred_uncond
                        + ((noise_pred_text - noise_pred_uncond)? * self.guidance_scale)?)?
                } else {
                    noise_pred
                };
                latents = scheduler.step(&noise_pred, timestep, &latents)?;
            }
            self.decode_image(&latents)
        }

        /// Transform an input image following the prompt (Python
        /// `image2image`); when the model was loaded with `in_paint`, edit
        /// only the masked region.
        pub fn image2image(
            &self,
            image_path: &str,
            mask_path: Option<&str>,
            prompt: &str,
            negative_prompt: &str,
            denoise: f64,
            use_inpaint: bool,
        ) -> Result<image::DynamicImage> {
            let steps = 50usize;
            let mut scheduler = self.config.build_scheduler(steps)?;
            let text_embeddings = self.text_embeddings(prompt, negative_prompt)?;
            let init_image = image_preprocess(image_path, &self.device, self.dtype)?;
            let init_latent_dist = self.vae.encode(&init_image)?;

            let vae_scale = 0.18215;
            let latents_0 = (init_latent_dist.sample()? * vae_scale)?.to_device(&self.device)?;
            let t_start = steps - ((steps as f64 * denoise.clamp(0.0, 1.0)) as usize);
            let timesteps = scheduler.timesteps().to_vec();
            let mut latents = if t_start < timesteps.len() {
                let noise = latents_0.randn_like(0f64, 1f64)?;
                scheduler.add_noise(&latents_0, noise, timesteps[t_start])?
            } else {
                latents_0
            }
            .to_dtype(self.dtype)?;

            // Inpainting tensors (mask latents + scaled mask + mask_4).
            let (mask_latents, mask, mask_4) = if use_inpaint {
                let mask_path =
                    mask_path.ok_or_else(|| anyhow!("Image-mask is required for inpainting"))?;
                let mask_tensor = mask_preprocess(mask_path)?
                    .to_device(&self.device)?
                    .to_dtype(self.dtype)?;
                let xmask = mask_tensor
                    .le(0.5)?
                    .repeat(&[1, 3, 1, 1])?
                    .to_dtype(self.dtype)?;
                let masked_img = (init_image * xmask)?;
                let shape = masked_img.shape();
                let (w, h) = (shape.dims()[3] / 8, shape.dims()[2] / 8);
                let mask_small = mask_tensor.interpolate2d(w, h)?;
                let mask_latents = (self.vae.encode(&masked_img)?.sample()? * vae_scale)?
                    .to_device(&self.device)?;
                let mask_4 = mask_small.repeat(&[1, 4, 1, 1])?;
                let (mask_latents, mask_small) = if self.use_guide_scale {
                    (
                        Tensor::cat(&[&mask_latents, &mask_latents], 0)?,
                        Tensor::cat(&[&mask_small, &mask_small], 0)?,
                    )
                } else {
                    (mask_latents, mask_small)
                };
                (Some(mask_latents), Some(mask_small), Some(mask_4))
            } else {
                (None, None, None)
            };

            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let latent_model_input = if self.use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };
                let mut latent_model_input =
                    scheduler.scale_model_input(latent_model_input, timestep)?;
                if use_inpaint {
                    latent_model_input = Tensor::cat(
                        &[
                            &latent_model_input,
                            mask.as_ref().unwrap(),
                            mask_latents.as_ref().unwrap(),
                        ],
                        1,
                    )?;
                }
                latent_model_input = latent_model_input.to_device(&self.device)?;
                let noise_pred =
                    self.unet
                        .forward(&latent_model_input, timestep as f64, &text_embeddings)?;
                let noise_pred = if self.use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
                    (noise_pred_uncond
                        + ((noise_pred_text - noise_pred_uncond)? * self.guidance_scale)?)?
                } else {
                    noise_pred
                };
                latents = scheduler.step(&noise_pred, timestep, &latents)?;
            }
            self.decode_image(&latents)
        }

        /// VAE decode → RGBA8 image (Python `save_image` internals).
        fn decode_image(&self, latents: &Tensor) -> Result<image::DynamicImage> {
            let vae_scale = 0.18215;
            let images = self.vae.decode(&(latents / vae_scale)?)?;
            let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
            let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
            let image = images.i(0)?;
            // (c, h, w) -> hwc
            let (c, h, w) = (image.dims()[0], image.dims()[1], image.dims()[2]);
            let image = image.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
            image::RgbImage::from_raw(w as u32, h as u32, image)
                .map(image::DynamicImage::ImageRgb8)
                .ok_or_else(|| anyhow!("image decode failed ({c} channels)"))
        }
    }

    /// Preprocess an init image into a [1, 3, h, w] tensor in [-1, 1].
    pub fn image_preprocess(path: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let img = image::ImageReader::open(path)?.decode()?;
        let (height, width) = (img.height() as usize, img.width() as usize);
        let height = height - height % 32;
        let width = width - width % 32;
        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let img = img.to_rgb8().into_raw();
        Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?
            .unsqueeze(0)?
            .to_device(device)?
            .to_dtype(dtype)
            .map_err(|e| e.into())
    }

    /// Convert the mask image to a single channel tensor, multiple of 32.
    pub fn mask_preprocess(path: &str) -> Result<Tensor> {
        let img = image::open(path)?.to_luma8();
        let (new_width, new_height) = {
            let (width, height) = img.dimensions();
            (width - width % 32, height - height % 32)
        };
        let img = image::imageops::resize(
            &img,
            new_width,
            new_height,
            image::imageops::FilterType::CatmullRom,
        )
        .into_raw();
        Ok(
            Tensor::from_vec(img, (new_height as usize, new_width as usize), &Device::Cpu)?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?
                .div(255.0)?
                .unsqueeze(0)?,
        )
    }

    impl super::SdPipeline {
        pub fn new(device_str: &str) -> Result<Self> {
            let device = match device_str {
                "cuda" => Device::new_cuda(0).map_err(|e| anyhow!("cuda: {e}"))?,
                _ => Device::Cpu,
            };
            Ok(Self {
                device,
                t2i: None,
                i2i: None,
                i2i_was_inpaint: false,
            })
        }

        pub fn dtype(&self) -> DType {
            torch_dtype(&self.device)
        }
    }
}

/// The loaded pipelines (mutually exclusive like Python `_t2i_pipe`/`_i2i_pipe`).
#[cfg(feature = "sd")]
pub struct SdPipeline {
    pub device: candle_core::Device,
    pub t2i: Option<backend::LoadedModel>,
    pub i2i: Option<backend::LoadedModel>,
    pub i2i_was_inpaint: bool,
}

#[cfg(feature = "sd")]
static PIPELINE: std::sync::Mutex<Option<SdPipeline>> = std::sync::Mutex::new(None);

#[cfg(feature = "sd")]
fn state_pipeline() -> std::sync::MutexGuard<'static, Option<SdPipeline>> {
    PIPELINE.lock().unwrap()
}

#[cfg(feature = "sd")]
fn device_string() -> String {
    // [image_gen] device from the shared config ("auto" -> cuda when available)
    let device = config_image_gen().device;
    if device == "auto" {
        if candle_core::Device::cuda_if_available(0).is_ok() {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        }
    } else {
        device
    }
}

#[cfg(feature = "sd")]
fn config_image_gen() -> la_core::config::ImageGenCfg {
    let path = std::env::var("LA_CONFIG").unwrap_or_else(|_| "config.toml".into());
    la_core::config::load_config(&path)
        .map(|c| c.image_gen)
        .unwrap_or_default()
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

    loop {
        let Some(envelope) = la_proto::read_msg::<Envelope<MainToWorker>>(&mut reader)? else {
            return Ok(()); // parent closed stdin
        };
        let id = envelope.id;
        match envelope.body {
            MainToWorker::T2i(params) => {
                if let Err(e) = text2image(&params, &mut writer, id) {
                    send(
                        &mut writer,
                        id,
                        &WorkerToMain::Error {
                            message: e.to_string(),
                        },
                    )?;
                }
            }
            MainToWorker::I2i(params) => {
                if let Err(e) = image2image(&params, &mut writer, id) {
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
                #[cfg(feature = "sd")]
                let mut state = state_pipeline();
                #[cfg(not(feature = "sd"))]
                let _state: Option<()> = None;
                #[cfg(feature = "sd")]
                {
                    if state.is_some() {
                        *state = None;
                        send(
                            &mut writer,
                            id,
                            &WorkerToMain::Done {
                                path: String::new(),
                                seed: None,
                                seconds: 0.0,
                            },
                        )?;
                    } else {
                        send(
                            &mut writer,
                            id,
                            &WorkerToMain::Error {
                                message: "No image generation model is loaded".into(),
                            },
                        )?;
                    }
                }
                #[cfg(not(feature = "sd"))]
                {
                    send(
                        &mut writer,
                        id,
                        &WorkerToMain::Error {
                            message: "No image generation model is loaded".into(),
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
                        message: format!("la-imggen does not handle {other:?}"),
                    },
                )?;
            }
        }
    }
}

/// Text2image round: generate + save (Python `generate_image` body).
fn text2image(params: &T2iParams, writer: &mut impl Write, id: Option<u64>) -> Result<()> {
    let start = std::time::Instant::now();
    #[cfg(feature = "sd")]
    {
        {
            let mut state = state_pipeline();
            if state.is_none() {
                *state = Some(SdPipeline::new(&device_string())?);
            }
            let pipeline = state.as_ref().unwrap();
            if pipeline.t2i.is_none() {
                let loaded = backend::LoadedModel::load_t2i(
                    &config_image_gen().t2i_model,
                    pipeline.device.clone(),
                    backend::torch_dtype(&pipeline.device),
                )?;
                let state = state_pipeline();
                state.as_mut().unwrap().t2i = Some(loaded);
            }
        }
        let seed = if params.seed < 0 {
            rand_seed()
        } else {
            params.seed as u64
        };
        let image = {
            let state = state_pipeline();
            let pipeline = state.as_ref().unwrap();
            pipeline.t2i.as_ref().unwrap().text2image(
                &params.prompt,
                params.negative_prompt.as_deref().unwrap_or_default(),
                params.width,
                params.height,
                seed,
                params.steps.max(1) as usize,
            )?
        };
        save_image(&image, &params.out_file)?;
        println!(
            "Generation Time: {:.2} seconds",
            start.elapsed().as_secs_f64()
        );
        send(
            writer,
            id,
            &WorkerToMain::Done {
                path: params.out_file.clone(),
                seed: Some(seed as i64),
                seconds: start.elapsed().as_secs_f64(),
            },
        )?;
        return Ok(());
    }
    #[cfg(not(feature = "sd"))]
    {
        let _ = (params, writer, id, start);
        Err(anyhow!(
            "image generation needs the diffusers stack; rebuild with `cargo build -p la-imggen --features sd` \
             (candle + tokenizers; models via rust/setup_rust_res.ps1)"
        ))
    }
}

/// Image2image / inpaint round (Python `image2image`).
fn image2image(params: &I2iParams, writer: &mut impl Write, id: Option<u64>) -> Result<()> {
    let start = std::time::Instant::now();
    #[cfg(feature = "sd")]
    {
        let config = config_image_gen();
        {
            let mut state = state_pipeline();
            if state.is_none() {
                *state = Some(SdPipeline::new(&config.device)?);
            }
            let pipeline = state.as_ref().unwrap();
            if pipeline.i2i.is_none() || params.inpaint != pipeline.i2i_was_inpaint {
                let loaded = backend::LoadedModel::load_i2i(
                    &config.i2i_model,
                    pipeline.device.clone(),
                    backend::torch_dtype(&pipeline.device),
                    params.inpaint,
                    params.guidance,
                )?;
                let state = state_pipeline();
                let pipeline = state.as_mut().unwrap();
                pipeline.t2i = None; // the two variants are mutually exclusive
                pipeline.i2i = Some(loaded);
                pipeline.i2i_was_inpaint = params.inpaint;
            }
        }
        let image = {
            let state = state_pipeline();
            let pipeline = state.as_ref().unwrap();
            pipeline.i2i.as_ref().unwrap().image2image(
                &params.image,
                params.mask.as_deref(),
                &params.prompt,
                params.negative_prompt.as_deref().unwrap_or_default(),
                params.denoise,
                params.inpaint,
            )?
        };
        save_image(&image, &params.out_file)?;
        println!(
            "Generation Time: {:.2} seconds",
            start.elapsed().as_secs_f64()
        );
        send(
            writer,
            id,
            &WorkerToMain::Done {
                path: params.out_file.clone(),
                seed: None,
                seconds: start.elapsed().as_secs_f64(),
            },
        )?;
        return Ok(());
    }
    #[cfg(not(feature = "sd"))]
    {
        let _ = (params, writer, id, start);
        Err(anyhow!(
            "image generation needs the diffusers stack; rebuild with `cargo build -p la-imggen --features sd`"
        ))
    }
}

/// A random 32-bit seed for `seed < 0` requests (Python `random.randint`).
#[cfg_attr(not(feature = "sd"), allow(dead_code))]
fn rand_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs() << 32))
        .unwrap_or(42);
    nanos
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        & 0xFFFF_FFFF
}

/// Save an image, creating parent directories (Python `ImageGenerator.save`).
#[cfg_attr(not(feature = "sd"), allow(dead_code))]
fn save_image(image: &image::DynamicImage, path: &str) -> Result<()> {
    let path = std::path::PathBuf::from(path);
    if path.is_dir() {
        return Err(anyhow!("'{}' should not be a directory", path.display()));
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    image.save_with_format(&path, image::ImageFormat::Png)?;
    println!("Image '{}' stored", path.display());
    Ok(())
}
