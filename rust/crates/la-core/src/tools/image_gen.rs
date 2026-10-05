//! Image-generation tool (Python `src/tools/image_gen.py` port): text-to-image
//! via the local Stable-Diffusion pipeline, routed to the `la-imggen` worker
//! process (the pipeline is loaded once and survives across tool calls).

use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use anyhow::Result;
use chrono::Local;
use std::path::PathBuf;
use std::sync::Arc;

pub const GENERATED_IMAGE: &str = "generated_image.png";

/// Validate a generation dimension: Stable Diffusion needs positive multiples
/// of 8 (Python `check_size`).
pub fn check_size(value: i64, name: &str) -> Result<u32> {
    if value <= 0 || value % 8 != 0 {
        return Err(anyhow::anyhow!(
            "{name} must be a positive multiple of 8 (got {value})"
        ));
    }
    Ok(value as u32)
}

/// Fill a `T2iParams` from the `[image_gen]` config defaults (Python:
/// `ImageGenerator.generate_image` uses `self.defaults` when a parameter is
/// None). Split out of `generate_image` so the defaulting rules are testable
/// without a worker.
#[allow(clippy::too_many_arguments)] // mirrors the tool signature (Python parity)
fn t2i_params_from_config(
    cfg: &crate::config::ImageGenCfg,
    prompt: &str,
    negative_prompt: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
    seed: Option<i64>,
    steps: Option<u32>,
    out_path: PathBuf,
) -> la_proto::T2iParams {
    la_proto::T2iParams {
        prompt: prompt.to_string(),
        negative_prompt: negative_prompt.or_else(|| Some(cfg.negative_prompt.clone())),
        width: width.map(|v| v as u32).unwrap_or(cfg.width),
        height: height.map(|v| v as u32).unwrap_or(cfg.height),
        seed: seed.unwrap_or(cfg.seed),
        steps: steps.unwrap_or(cfg.steps),
        out_file: out_path.to_string_lossy().to_string(),
    }
}

/// Timestamped default output path under the configured `out_dir`
/// (Python `_default_out_path`).
pub fn default_out_path(ctx: &ToolContext) -> PathBuf {
    let out_dir = if ctx.config.image_gen.out_dir.as_os_str().is_empty() {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        ctx.config.image_gen.out_dir.clone()
    };
    let stem = PathBuf::from(GENERATED_IMAGE)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let stamp = Local::now().format("%Y%m%d-%H%M%S");
    out_dir.join(format!("{stem}_{stamp}.png"))
}

/// Generate an image from a text description and save it as an image file.
/// Useful when the user asks to create, draw, generate or imagine a
/// picture/image/artwork/illustration from a prompt. The first call loads the
/// diffusion model, which can take a while; later calls reuse it. Parameters
/// left empty use the [image_gen] defaults from config.toml (512x512, 50
/// steps). `negative_prompt` describes what to avoid, `width`/`height`
/// must be multiples of 8, `steps` trades quality for speed, and a `seed`
/// < 0 picks a random seed. Returns the absolute path of the saved image with
/// the seed used, so the same image can be reproduced or varied with a new seed.
#[allow(clippy::too_many_arguments)] // Python parity: mirrors the tool signature
pub async fn generate_image(
    ctx: Arc<ToolContext>,
    prompt: &str,
    negative_prompt: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
    seed: Option<i64>,
    steps: Option<u32>,
    out_file: Option<String>,
) -> String {
    log_call!(
        "generate_image",
        "prompt" => prompt,
        "negative_prompt" => negative_prompt.clone().unwrap_or_default(),
        "width" => width.map(|v| v.to_string()).unwrap_or_default(),
        "height" => height.map(|v| v.to_string()).unwrap_or_default(),
        "seed" => seed.map(|v| v.to_string()).unwrap_or_default(),
        "steps" => steps.map(|v| v.to_string()).unwrap_or_default(),
        "out_file" => out_file.clone().unwrap_or_default()
    );
    if prompt.trim().is_empty() {
        return "Error: a prompt is required.".into();
    }
    for (name, value) in [("width", width), ("height", height)] {
        if let Some(value) = value
            && let Err(e) = check_size(value, name)
        {
            return format!("Error: {e}");
        }
    }
    let out_path = match out_file {
        Some(file) => {
            let mut path = PathBuf::from(file);
            if path.extension().map(|e| e.is_empty()).unwrap_or(true) {
                path.set_extension("png");
            }
            if path.is_dir() {
                return format!(
                    "Error: '{}' is a directory, not a file name",
                    path.display()
                );
            }
            path
        }
        None => default_out_path(&ctx),
    };

    // Route to the imggen worker (pipeline cached in the worker process).
    let params = t2i_params_from_config(
        &ctx.config.image_gen,
        prompt,
        negative_prompt,
        width,
        height,
        seed,
        steps,
        out_path,
    );
    match ctx.workers.generate_image(&params).await {
        Ok((path, seed_used)) => {
            let abs = std::fs::canonicalize(&path).unwrap_or(path);
            format!(
                "{} (seed={})",
                abs.display(),
                seed_used
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "None".into())
            )
        }
        Err(e) => format!("Error: {e}"),
    }
}

/// Free the memory (RAM/VRAM) used by the text-to-image model loaded by
/// generate_image. Useful when the user asks to free memory after generating
/// images, or before running heavy work. Returns a short status message.
pub async fn unload_image_generator(ctx: Arc<ToolContext>) -> String {
    log_call!("unload_image_generator");
    ctx.workers.unload_image_generator().await
}

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "generate_image".into(),
            description: "Generate an image from a text description and save it as an image file.\nUseful when the user asks to create, draw, generate or imagine a\npicture/image/artwork/illustration from a prompt. The first call loads the\ndiffusion model, which can take a while; later calls reuse it. Parameters\nleft empty use the [image_gen] defaults from config.toml (512x512, 50\nsteps). `negative_prompt` describes what to avoid, `width`/`height`\nmust be multiples of 8, `steps` trades quality for speed, and a `seed`\n< 0 picks a random seed. Returns the absolute path of the saved image with\nthe seed used, so the same image can be reproduced or varied with a new seed.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "seed": {"type": "integer"},
                    "steps": {"type": "integer"},
                    "out_file": {"type": "string"}
                },
                "required": ["prompt"]
            }),
            handler: Arc::new(|ctx, args| Box::pin(async move {
                let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let negative = args.get("negative_prompt").and_then(|v| v.as_str()).map(str::to_string);
                let width = args.get("width").and_then(|v| v.as_i64());
                let height = args.get("height").and_then(|v| v.as_i64());
                let seed = args.get("seed").and_then(|v| v.as_i64());
                let steps = args.get("steps").and_then(|v| v.as_i64()).map(|v| v as u32);
                let out_file = args.get("out_file").and_then(|v| v.as_str()).map(str::to_string);
                Ok(ToolOutput::text(generate_image(ctx, &prompt, negative, width, height, seed, steps, out_file).await))
            })),
        },
        ToolDef {
            name: "unload_image_generator".into(),
            description: "Free the memory (RAM/VRAM) used by the text-to-image model loaded by\ngenerate_image. Useful when the user asks to free memory after generating\nimages, or before running heavy work. Returns a short status message.".into(),
            schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            handler: Arc::new(|ctx, _args| Box::pin(async move {
                Ok(ToolOutput::text(unload_image_generator(ctx).await))
            })),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn check_size_rules() {
        assert_eq!(check_size(512, "width").unwrap(), 512);
        assert!(check_size(500, "width").is_err());
        assert!(check_size(0, "width").is_err());
        assert!(check_size(-8, "width").is_err());
        assert_eq!(check_size(8, "width").unwrap(), 8);
    }

    #[test]
    fn default_out_path_stamped() {
        let config = Config::default();
        let view = crate::brain::AgentView {
            client: async_openai::Client::with_config(async_openai::config::OpenAIConfig::new()),
            http: reqwest::Client::new(),
            base_url: "http://localhost".into(),
            model_name: "test".into(),
            temperature: 0.6,
            max_tokens: 100,
            context_window: 1000,
            context_window_source: "config",
            vision: false,
            vision_source: "config",
            messages: Arc::new(std::sync::Mutex::new(vec![])),
            model_meta: Arc::new(std::sync::RwLock::new(None)),
        };
        let ctx = ToolContext::new(&config, view);
        let path = default_out_path(&ctx);
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("generated_image_"), "got: {name}");
        assert!(name.ends_with(".png"));
    }

    #[test]
    fn t2i_defaults_come_from_image_gen_config() {
        let mut cfg = Config::default();
        cfg.image_gen.width = 640;
        cfg.image_gen.height = 384;
        cfg.image_gen.seed = 42;
        cfg.image_gen.steps = 30;
        cfg.image_gen.negative_prompt = "blurry".into();
        let params = t2i_params_from_config(
            &cfg.image_gen,
            "a cat",
            None,
            None,
            None,
            None,
            None,
            PathBuf::from("out.png"),
        );
        assert_eq!(params.prompt, "a cat");
        assert_eq!(params.negative_prompt.as_deref(), Some("blurry"));
        assert_eq!(params.width, 640);
        assert_eq!(params.height, 384);
        assert_eq!(params.seed, 42);
        assert_eq!(params.steps, 30);
        assert_eq!(params.out_file, "out.png");

        // Explicit arguments override the defaults.
        let params = t2i_params_from_config(
            &cfg.image_gen,
            "a cat",
            Some("soft".into()),
            Some(512),
            Some(512),
            Some(-1),
            Some(10),
            PathBuf::from("other.jpg"),
        );
        assert_eq!(params.negative_prompt.as_deref(), Some("soft"));
        assert_eq!(params.width, 512);
        assert_eq!(params.height, 512);
        assert_eq!(params.seed, -1);
        assert_eq!(params.steps, 10);
        assert_eq!(params.out_file, "other.jpg");
    }
}
