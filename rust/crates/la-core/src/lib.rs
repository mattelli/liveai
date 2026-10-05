//! `la-core` — the agent core library of the Live Assistant Rust port.
//!
//! Mirrors the Python `src` package: configuration, console printing
//! (with streaming printer + spinner), interactive input, the OpenAI-compatible
//! Brain/Agent, the conversation session, the tool registry and the
//! document/embedding/summarization toolkits.

pub mod agent;
pub mod brain;
pub mod common;
pub mod config;
pub mod document_loader;
pub mod document_query;
pub mod embedding_search;
pub mod encoders;
pub mod image_search;
pub mod input;
pub mod lex_rank;
pub mod printer;
pub mod session;
pub mod summarizer;
pub mod tools;
pub mod workers;

pub use config::Config;
