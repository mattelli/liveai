//! Vision tool (Python `src/tools/vision.py` port): describe an image through
//! the multimodal chat model. Unlike the other tool modules there is no config
//! section here: everything it needs (client, model, temperature, vision
//! capability, live history) belongs to the running chat session.

use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
};
use std::path::Path;
use std::sync::Arc;

/// Guess the mime type of an image from its extension.
fn guess_mime(path: &str) -> String {
    let ext = Path::new(path)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        _ => "image/png",
    }
    .to_string()
}

/// Provide text response form an input image.
/// Useful when user asks for getting a description of the input image content.
pub async fn image_response(ctx: Arc<ToolContext>, image: &str, prompt: &str) -> String {
    log_call!("image_response", "image" => image, "prompt" => prompt);
    if !ctx.view.vision {
        if ctx.view.vision_source == "server" {
            return "Cannot handle images: the server reports the model has no 'vision' capability"
                .into();
        }
        return "Cannot handle images: no 'vision' capability reported by the server and no [brain] vision = true override set".into();
    }
    let image_url = if image.starts_with("http://") || image.starts_with("https://") {
        image.to_string()
    } else {
        let data = match std::fs::read(image) {
            Ok(d) => d,
            Err(e) => return format!("Error: {e}"),
        };
        let mime = guess_mime(image);
        format!("data:{mime};base64,{}", base64_encode(&data))
    };
    let message = ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Array(vec![
            ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText {
                    text: prompt.to_string(),
                    prompt_cache_breakpoint: None,
                },
            ),
            ChatCompletionRequestUserMessageContentPart::ImageUrl(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: async_openai::types::chat::ImageUrl {
                        url: image_url,
                        detail: None,
                    },
                    prompt_cache_breakpoint: None,
                },
            ),
        ]),
        name: None,
    });
    let mut messages: Vec<ChatCompletionRequestMessage> = ctx.view.messages.lock().unwrap().clone();
    messages.push(message);
    let response = ctx
        .view
        .client
        .chat()
        .create(request_for(&ctx, messages))
        .await;
    match response {
        Err(e) => {
            // drop the image turn: provider rejected it
            ctx.view.messages.lock().unwrap().pop();
            format!("Error: {e}")
        }
        Ok(resp) => {
            let answer = resp
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();
            ctx.view
                .messages
                .lock()
                .unwrap()
                .push(ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessage {
                        content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                            answer.clone(),
                        )),
                        ..Default::default()
                    },
                ));
            answer
        }
    }
}

fn request_for(
    ctx: &Arc<ToolContext>,
    messages: Vec<ChatCompletionRequestMessage>,
) -> async_openai::types::chat::CreateChatCompletionRequest {
    let mut req = async_openai::types::chat::CreateChatCompletionRequestArgs::default();
    req.model(ctx.view.model_name.clone())
        .messages(messages)
        .temperature(ctx.view.temperature);
    req.build().expect("valid vision request")
}

/// Minimal base64 encoder (standard alphabet, padded).
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(triple >> 18) as usize & 63] as char);
        out.push(ALPHABET[(triple >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(triple >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[triple as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub fn tool_def() -> ToolDef {
    ToolDef {
        name: "image_response".into(),
        description: "Provide text response form an input image.\nUseful when user asks for getting a description of the input image content.".into(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "prompt": {"type": "string", "default": "describe the image"}
            },
            "required": ["image"]
        }),
        handler: Arc::new(|ctx, args| Box::pin(async move {
            let image = args.get("image").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("describe the image").to_string();
            Ok(ToolOutput::text(image_response(ctx, &image, &prompt).await))
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn mime_guess() {
        assert_eq!(guess_mime("a.png"), "image/png");
        assert_eq!(guess_mime("a.JPG"), "image/jpeg");
        assert_eq!(guess_mime("a.webp"), "image/webp");
        assert_eq!(guess_mime("a.unknown"), "image/png");
    }
}
