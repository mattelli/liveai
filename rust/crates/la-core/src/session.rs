//! `ConversationSession` — shared interaction loop for text and voice modes
//! (`src/session.py` port).
//!
//! Owns prompt acquisition, slash commands, per-turn error recovery and
//! shutdown. All collaborators are injected: the agent ([`AgentLike`]) and the
//! prompt source ([`PromptSource`]), so heavy modules are not imported.

use crate::agent::AgentLike;
use crate::common::cprint;
use crate::input::ReadOutcome;
use crate::tools::registry::ToolContext;
use std::sync::{Arc, Mutex};

/// Blocking prompt source (rustyline reader, voice listener or one-shot).
pub trait PromptSource: Send {
    fn read(&mut self) -> ReadOutcome;
}

/// One scripted prompt source for tests (mirrors the Python test seams).
pub struct ScriptedPrompts {
    lines: std::collections::VecDeque<ReadOutcome>,
}

impl ScriptedPrompts {
    pub fn new(lines: impl IntoIterator<Item = ReadOutcome>) -> Self {
        Self {
            lines: lines.into_iter().collect(),
        }
    }
}

impl PromptSource for ScriptedPrompts {
    fn read(&mut self) -> ReadOutcome {
        self.lines.pop_front().unwrap_or(ReadOutcome::Eof)
    }
}

pub struct ConversationSession {
    agent: Arc<dyn AgentLike>,
    reader: Arc<Mutex<dyn PromptSource>>,
    context: Arc<ToolContext>,
    use_tools: bool,
    quit_on_prompt_interrupt: bool,
    should_quit: bool,
    speech_enabled: bool,
    shutdown_done: bool,
}

impl ConversationSession {
    pub fn new(
        agent: Arc<dyn AgentLike>,
        reader: Arc<Mutex<dyn PromptSource>>,
        context: Arc<ToolContext>,
        use_tools: bool,
        quit_on_prompt_interrupt: bool,
    ) -> Self {
        Self {
            agent,
            reader,
            context,
            use_tools,
            quit_on_prompt_interrupt,
            should_quit: false,
            speech_enabled: false,
            shutdown_done: false,
        }
    }

    // ------------------------------------------------------------------ loop

    pub async fn run(&mut self) -> i32 {
        self.banner();
        if let Some(bootstrap) = self.agent.bootstrap() {
            self.generation_step(&bootstrap).await;
        }

        while !self.should_quit {
            // Blocking prompt reads run on a blocking thread.
            let reader = self.reader.clone();
            let outcome = tokio::task::spawn_blocking(move || reader.lock().unwrap().read())
                .await
                .unwrap_or(ReadOutcome::Eof);
            let prompt = match outcome {
                ReadOutcome::Interrupted => {
                    // Ctrl+C while waiting for input: quit for voice /
                    // non-interactive prompt sources, otherwise re-prompt.
                    if self.quit_on_prompt_interrupt {
                        break;
                    }
                    continue;
                }
                ReadOutcome::Eof => None, // EOF
                ReadOutcome::Line(line) => Some(line),
            };
            let Some(prompt) = prompt else { break };
            let text = prompt.trim().to_string();
            if text.is_empty() {
                continue;
            }
            // A multi-line paste goes to the agent even if its first line looks
            // like a command; only a single-line string may be a command.
            if !text.contains('\n') && self.handle_command(&text).await {
                continue;
            }
            self.generation_step(&text).await;
            cprint("", "reset", false, "\n"); // blank separator line after each turn
        }
        self.shutdown().await
    }

    /// Run one agent generation (bootstrap or user turn) with Ctrl+C support.
    ///
    /// Ctrl+C cancels the current generation; the session keeps running. The
    /// agent cancel (server-side stream close) happens first, then the turn
    /// unwinds — parity of the Python SIGINT guard.
    async fn generation_step(&mut self, prompt: &str) {
        let agent = self.agent.clone();
        let step = agent.run_turn(prompt, self.use_tools);
        tokio::pin!(step);
        let mut interrupted = false;
        loop {
            tokio::select! {
                res = &mut step => { let _ = res; break; }
                _ = tokio::signal::ctrl_c() => {
                    agent.cancel();
                    interrupted = true;
                    // give the generation loop a moment to observe the cancel
                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                }
            }
        }
        self.agent.flush();
        if interrupted || self.agent.is_cancelled() {
            cprint(" ⏹  generation cancelled", "yellow", false, "\n");
        }
    }

    // -------------------------------------------------------------- commands

    /// True if the text was consumed as a command.
    async fn handle_command(&mut self, text: &str) -> bool {
        if !text.starts_with('/') {
            if text.eq_ignore_ascii_case("exit") {
                self.should_quit = true;
                return true;
            }
            return false;
        }

        let (cmd, arg) = match text.split_once(' ') {
            Some((cmd, arg)) => (cmd.to_lowercase(), arg),
            None => (text.to_lowercase(), ""),
        };

        match cmd.as_str() {
            "/quit" => {
                self.should_quit = true;
                true
            }
            "/help" => {
                self.cmd_help();
                true
            }
            "/clear" => {
                self.cmd_clear();
                true
            }
            "/voice" => {
                self.cmd_voice(arg).await;
                true
            }
            "/tools" => {
                self.cmd_tools(arg).await;
                true
            }
            "/info" => {
                self.cmd_info();
                true
            }
            _ => {
                cprint(
                    &format!("Unknown command: {text}. Type /help."),
                    "red",
                    false,
                    "\n",
                );
                true
            }
        }
    }

    fn cmd_help(&self) {
        cprint("Available commands:", "cyan", false, "\n");
        cprint("  /help              - show this help", "cyan", false, "\n");
        cprint(
            "  /quit              - exit the session (alias: exit)",
            "cyan",
            false,
            "\n",
        );
        cprint(
            "  /clear             - clear the conversation history",
            "cyan",
            false,
            "\n",
        );
        cprint(
            "  /voice on|off      - toggle speech output",
            "cyan",
            false,
            "\n",
        );
        cprint(
            "  /tools             - list and run an available tool",
            "cyan",
            false,
            "\n",
        );
        cprint(
            "  /info              - show model/context info",
            "cyan",
            false,
            "\n",
        );
        cprint(
            "  Ctrl+C             - stop the response being generated",
            "cyan",
            false,
            "\n",
        );
        cprint(
            "  Ctrl+D             - exit the session",
            "cyan",
            false,
            "\n",
        );
    }

    fn cmd_clear(&mut self) {
        let out = crate::tools::memory::clear_chat(&self.context);
        cprint(
            &out,
            if out.starts_with("Chat cleared") {
                "green"
            } else {
                "red"
            },
            false,
            "\n",
        );
    }

    async fn cmd_voice(&mut self, arg: &str) {
        let arg = arg.trim().to_lowercase();
        match arg.as_str() {
            "on" => {
                self.speech_enabled = true;
                let out = crate::tools::speech::allow_speech(self.context.clone(), true).await;
                cprint(
                    &out,
                    if out.starts_with("Speech") {
                        "green"
                    } else {
                        "red"
                    },
                    false,
                    "
",
                );
            }
            "off" => {
                self.speech_enabled = false;
                crate::tools::speech::allow_speech(self.context.clone(), false).await;
                cprint(
                    "Voice disabled",
                    "green",
                    false,
                    "
",
                );
            }
            _ => cprint(
                "Usage: /voice on|off",
                "yellow",
                false,
                "
",
            ),
        }
    }

    async fn cmd_tools(&mut self, arg: &str) {
        if !arg.trim().is_empty() {
            cprint("Usage: /tools", "yellow", false, "\n");
            return;
        }
        self.launch_tool().await;
    }

    /// List available tools, let the user pick one and run it.
    ///
    /// Required parameters (those without a default value) are prompted for;
    /// a tool that takes no parameters runs immediately after selection.
    async fn launch_tool(&mut self) {
        let tools = self.agent.tools();
        if tools.is_empty() {
            cprint("No tools available.", "grey", false, "\n");
            return;
        }

        cprint("Available tools:", "cyan", false, "\n");
        for (index, tool) in tools.iter().enumerate() {
            let summary = tool.summary();
            let line = if summary.is_empty() {
                format!("  {:>2}. {}", index + 1, tool.name)
            } else {
                format!("  {:>2}. {} - {}", index + 1, tool.name, summary)
            };
            cprint(&line, "cyan", false, "\n");
        }

        let Some(selection) = self
            .read_tool_input("Select a tool (number or name, blank to cancel): ")
            .await
        else {
            return;
        };
        let selection = selection.trim().to_string();
        if selection.is_empty() {
            cprint("Cancelled.", "grey", false, "\n");
            return;
        }

        let tool = if let Ok(index) = selection.parse::<usize>() {
            tools.get(index - 1)
        } else {
            tools
                .iter()
                .find(|t| t.name.eq_ignore_ascii_case(&selection))
        };
        let Some(tool) = tool else {
            cprint(&format!("Unknown tool: {selection}"), "red", false, "\n");
            return;
        };

        let mut arguments = serde_json::Map::new();
        for (name, type_hint) in tool.required_params() {
            let Some(raw) = self
                .read_tool_input(&format!("{name} ({type_hint}): "))
                .await
            else {
                return;
            };
            let value = crate::input::convert_argument(&raw, &type_hint);
            arguments.insert(name, value);
        }

        cprint(&format!("▶ {}", tool.name), "green", false, "\n");
        let result =
            (tool.handler)(self.context.clone(), serde_json::Value::Object(arguments)).await;
        match result {
            Ok(output) => cprint(&output.text, "green", false, "\n"),
            Err(e) => cprint(&format!("❌ {e}"), "red", false, "\n"),
        }
    }

    /// Prompt for a tool interaction, handling EOF and Ctrl-C.
    async fn read_tool_input(&mut self, hint: &str) -> Option<String> {
        cprint(
            hint, "yellow", false, "
",
        );
        let reader = self.reader.clone();
        let outcome = tokio::task::spawn_blocking(move || reader.lock().unwrap().read())
            .await
            .unwrap_or(ReadOutcome::Eof);
        match outcome {
            ReadOutcome::Line(line) => Some(line),
            ReadOutcome::Interrupted => {
                self.should_quit = true;
                None
            }
            ReadOutcome::Eof => {
                self.should_quit = true;
                None
            }
        }
    }

    fn cmd_info(&mut self) {
        match self.agent.model_info() {
            Some(info) => cprint(&info, "cyan", false, "\n"),
            None => cprint("Info unavailable", "grey", false, "\n"),
        }
    }

    // -------------------------------------------------------------- shutdown

    async fn shutdown(&mut self) -> i32 {
        self.agent.flush();
        if self.speech_enabled {
            self.speech_enabled = false;
            let ctx = self.context.clone();
            crate::tools::speech::allow_speech(ctx, false).await;
        }
        if !self.shutdown_done {
            cprint("👋 Bye!", "green", false, "\n");
            self.shutdown_done = true;
        }
        0
    }

    // ---------------------------------------------------------------- banner

    fn banner(&self) {
        let hint = if self.quit_on_prompt_interrupt {
            ""
        } else {
            " Ctrl+C stops a response; Ctrl+D quits."
        };
        cprint(
            &format!(
                "Agent ready. Tools are {}.{hint}",
                if self.use_tools {
                    "enabled"
                } else {
                    "disabled"
                }
            ),
            "green",
            false,
            "\n",
        );
    }
}
