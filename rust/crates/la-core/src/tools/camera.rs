//! Camera tool (Python `src/tools/camera.py` port): capture a frame from a
//! webcam and save it as an image file. Uses `nokhwa` (Media Foundation on
//! Windows) behind the `camera` feature; without it the tool answers with a
//! clear rebuild hint.

use crate::tools::registry::{ToolDef, ToolOutput};
use std::sync::Arc;

pub const CAPTURED_IMAGE: &str = "captured_image.jpg";

/// Captures a frame from a camera and saves it as an image file.
/// Useful when user asks for taking a picture.
#[cfg(feature = "camera")]
pub fn capture_image(out_file: &str, camera_index: u32) -> String {
    log_call!("capture_image", "out_file" => out_file, "camera_index" => camera_index);
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

    let index = CameraIndex::index(camera_index);
    let requested =
        RequestedFormat::<RgbFormat>::new(RequestedFormatType::AbsoluteHighestResolution);
    let mut camera = match nokhwa::Camera::new(index, requested) {
        Ok(c) => c,
        Err(_) => return format!("Webcam '{camera_index}' not available"),
    };
    if camera.open_stream().is_err() {
        return format!("Webcam '{camera_index}' not available");
    }
    let frame = match camera.frame() {
        Ok(f) => f,
        Err(_) => {
            let _ = camera.stop_stream();
            return "Unable to capture frame".into();
        }
    };
    let _ = camera.stop_stream();
    let image = match frame.decode_image::<RgbFormat>() {
        Ok(img) => img,
        Err(_) => return "Unable to capture frame".into(),
    };
    let path = std::path::PathBuf::from(out_file);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match image.save(&path) {
        Err(_) => format!("Unable to save image '{out_file}'"),
        Ok(_) => std::fs::canonicalize(&path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| out_file.to_string()),
    }
}

#[cfg(not(feature = "camera"))]
pub fn capture_image(_out_file: &str, _camera_index: u32) -> String {
    "Error: camera support is not compiled in; rebuild with `--features camera` \
     (cargo build -p la --features camera)"
        .into()
}

pub fn tool_def() -> ToolDef {
    ToolDef {
        name: "capture_image".into(),
        description: "Captures a frame from a camera and saves it as an image file.\nUseful when user asks for taking a picture.".into(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "out_file": {"type": "string", "default": CAPTURED_IMAGE},
                "camera_index": {"type": "integer", "default": 0}
            },
            "required": []
        }),
        handler: Arc::new(|_ctx, args| Box::pin(async move {
            let out_file = args.get("out_file").and_then(|v| v.as_str()).unwrap_or(CAPTURED_IMAGE).to_string();
            let camera_index = args.get("camera_index").and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let out_file2 = out_file.clone();
            let result = tokio::task::spawn_blocking(move || capture_image(&out_file2, camera_index))
                .await
                .unwrap_or_else(|e| format!("Error: {e}"));
            Ok(ToolOutput::text(result))
        })),
    }
}
