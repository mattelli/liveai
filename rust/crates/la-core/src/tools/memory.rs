//! Personal-memory and conversation-history tools
//! (Python `src/tools/memory.py` port).
//!
//! `store_personal` appends personal notes from the conversation to a history
//! file, `recall_personal` reads them back, and `clear_chat` resets the live
//! conversation (re-seeded with the recalled personal data).

use crate::log_call;
use crate::tools::registry::{ToolContext, ToolDef, ToolOutput};
use anyhow::Result;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent,
};
use std::path::PathBuf;
use std::sync::Arc;

fn history_file(ctx: &ToolContext) -> PathBuf {
    ctx.config.brain.history_path.clone()
}

fn bootstrap_prompt(ctx: &ToolContext) -> Option<String> {
    let prompt = ctx.config.agent.bootstrap_prompt.clone();
    if prompt.is_empty() {
        None
    } else {
        Some(prompt)
    }
}

/// User messages of the live history (text content of user turns).
fn shaped_user_messages(ctx: &ToolContext) -> Vec<String> {
    let messages = ctx.view.messages.lock().unwrap();
    messages
        .iter()
        .filter(|m| matches!(m, ChatCompletionRequestMessage::User(_)))
        .filter_map(|m| match m {
            ChatCompletionRequestMessage::User(user) => match &user.content {
                ChatCompletionRequestUserMessageContent::Text(text) => Some(text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Read users personal data from file.
/// Useful when the users ask to recall their personal data.
pub fn recall_personal(ctx: &ToolContext) -> String {
    log_call!("recall_personal");
    let history = history_file(ctx);
    if !history.exists() {
        return "No personal data found!".into();
    }
    std::fs::read_to_string(&history).unwrap_or_else(|e| format!("Error: {e}"))
}

/// Store the user personal content to a text file for later recall with
/// recall_personal. Useful when the users explicitly mention that the
/// prompted text is "personal" or ask to save/remember something about them.
/// Pass the personal text in `content`; when omitted, the last user message
/// of the conversation is stored, preferring messages that mention
/// "personal". Returns the file path written.
pub fn store_personal(ctx: &ToolContext, content: &str) -> String {
    log_call!("store_personal", "content" => content);
    let result: Result<String> = (|| {
        let mut text = content.trim().to_string();
        if text.is_empty() {
            let mut user_msgs = shaped_user_messages(ctx);
            if let Some(bootstrap) = bootstrap_prompt(ctx)
                && user_msgs.first().map(|m| m == &bootstrap).unwrap_or(false)
            {
                user_msgs = user_msgs[1..].to_vec();
            }
            let selected: Vec<String> = user_msgs
                .iter()
                .filter(|m| m.to_lowercase().contains("personal"))
                .cloned()
                .collect();
            let selected = if selected.is_empty() && !user_msgs.is_empty() {
                vec![user_msgs.last().unwrap().clone()]
            } else {
                selected
            };
            if selected.is_empty() {
                return Ok("Error: no user message found to store.".into());
            }
            text = selected.last().unwrap().clone();
        }
        let history = history_file(ctx);
        let now = chrono::Local::now();
        let timestamp = now.format("%Y-%m-%d %H:%M:%S");
        let user_prompt = format!("At {timestamp} user wrote: {text} \n");
        if !history.exists() {
            if let Some(parent) = history.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(&history, user_prompt)?;
        } else {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().append(true).open(&history)?;
            file.write_all(user_prompt.as_bytes())?;
        }
        Ok(format!("Conversation written to {}", history.display()))
    })();
    match result {
        Ok(text) => text,
        Err(e) => format!("Error: {e}"),
    }
}

/// Useful when user ask for cleaning conversation history.
pub fn clear_chat(ctx: &ToolContext) -> String {
    log_call!("clear_chat");
    let result: Result<String> = {
        let mut messages = ctx.view.messages.lock().unwrap();
        let system_prompt = ctx.config.brain.system_prompt.clone();
        let system = messages
            .iter()
            .find_map(|m| match m {
                ChatCompletionRequestMessage::System(sys) => match &sys.content {
                    ChatCompletionRequestSystemMessageContent::Text(t) => Some(t.clone()),
                    _ => None,
                },
                _ => None,
            })
            .unwrap_or(system_prompt);
        let user_msgs: Vec<String> = messages
            .iter()
            .filter_map(|m| match m {
                ChatCompletionRequestMessage::User(user) => match &user.content {
                    ChatCompletionRequestUserMessageContent::Text(t) => Some(t.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        messages.clear(); // in place: the session keeps the same list object
        messages.push(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(system),
                name: None,
            },
        ));
        messages.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(recall_personal(ctx)),
                name: None,
            },
        ));
        if let Some(last) = user_msgs.last() {
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(last.clone()),
                    name: None,
                },
            ));
        }
        Ok("Chat cleared successfully".into())
    };
    match result {
        Ok(text) => text,
        Err(e) => format!("Error: {e}"),
    }
}

// ------------------------------------------------------------- registry

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "recall_personal".into(),
            description: "Read users personal data from file.\nUseful when the users ask to recall their personal data.".into(),
            schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            handler: Arc::new(|ctx, _args| Box::pin(async move {
                Ok(ToolOutput::text(recall_personal(&ctx)))
            })),
        },
        ToolDef {
            name: "store_personal".into(),
            description: "Store the user personal content to a text file for later recall with\nrecall_personal. Useful when the users explicitly mention that the\nprompted text is \"personal\" or ask to save/remember something about them.\nPass the personal text in `content`; when omitted, the last user message\nof the conversation is stored, preferring messages that mention\n\"personal\". Returns the file path written.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"content": {"type": "string", "default": ""}},
                "required": []
            }),
            handler: Arc::new(|ctx, args| Box::pin(async move {
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                Ok(ToolOutput::text(store_personal(&ctx, &content)))
            })),
        },
        ToolDef {
            name: "clear_chat".into(),
            description: "Useful when user ask for cleaning conversation history.".into(),
            schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            handler: Arc::new(|ctx, _args| Box::pin(async move {
                Ok(ToolOutput::text(clear_chat(&ctx)))
            })),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::AgentView;
    use crate::config::Config;

    fn test_ctx() -> (tempfile::TempDir, Arc<ToolContext>) {
        let dir = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.brain.history_path = dir.path().join("personal.log");
        config.brain.system_prompt = "You are a helpful assistant".into();
        config.agent.bootstrap_prompt = "Recall user personal content".into();
        let view = test_view();
        (dir, ToolContext::new(&config, view))
    }

    fn test_view() -> AgentView {
        crate::brain::AgentView {
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
        }
    }

    #[test]
    fn store_and_recall_personal() {
        let (_dir, ctx) = test_ctx();
        let out = store_personal(&ctx, "My name is Max");
        assert!(out.starts_with("Conversation written to"), "got: {out}");
        let recalled = recall_personal(&ctx);
        assert!(recalled.contains("My name is Max"));
    }

    #[test]
    fn recall_missing_file() {
        let (_dir, ctx) = test_ctx();
        assert_eq!(recall_personal(&ctx), "No personal data found!");
    }

    #[test]
    fn store_from_history_prefers_personal() {
        let (_dir, ctx) = test_ctx();
        {
            let mut messages = ctx.view.messages.lock().unwrap();
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(
                        "Recall user personal content".into(),
                    ),
                    name: None,
                },
            ));
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(
                        "my personal password is 1234".into(),
                    ),
                    name: None,
                },
            ));
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text("tell me a joke".into()),
                    name: None,
                },
            ));
        }
        let out = store_personal(&ctx, "");
        assert!(out.starts_with("Conversation written to"));
        let recalled = recall_personal(&ctx);
        assert!(
            recalled.contains("password"),
            "should pick the personal message: {recalled}"
        );
    }

    #[test]
    fn clear_chat_reseeds_with_recall() {
        let (_dir, ctx) = test_ctx();
        store_personal(&ctx, "personal note");
        {
            let mut messages = ctx.view.messages.lock().unwrap();
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text("hello".into()),
                    name: None,
                },
            ));
        }
        let out = clear_chat(&ctx);
        assert_eq!(out, "Chat cleared successfully");
        let messages = ctx.view.messages.lock().unwrap();
        assert_eq!(messages.len(), 3, "system + recall + last user");
        assert!(matches!(
            &messages[0],
            ChatCompletionRequestMessage::System(_)
        ));
        let recall_text = match &messages[1] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
                _ => String::new(),
            },
            _ => String::new(),
        };
        assert!(recall_text.contains("personal note"));
    }

    #[test]
    fn store_without_content_and_no_history() {
        let (_dir, ctx) = test_ctx();
        let out = store_personal(&ctx, "");
        assert_eq!(out, "Error: no user message found to store.");
    }
}
