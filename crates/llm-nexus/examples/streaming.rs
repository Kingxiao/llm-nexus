//! Streaming example — prints tokens as they arrive.
//!
//! Usage: cargo run --example streaming -p llm-nexus --features full
//!
//! Set the appropriate API key environment variable before running.
//! Model can be overridden via NEXUS_MODEL env var.

use futures::StreamExt;
use llm_nexus::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_dir = std::path::Path::new("config");
    let client = NexusClient::from_config_dir(config_dir)?;

    let model = std::env::var("NEXUS_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".into());

    let mut stream = client
        .chat_stream(&ChatRequest {
            model,
            messages: vec![Message::user("Write a haiku about Rust programming.")],
            ..Default::default()
        })
        .await?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(c) => {
                if let Some(text) = c.delta_content {
                    print!("{text}");
                }
            }
            Err(e) => eprintln!("\nStream error: {e}"),
        }
    }
    println!();

    Ok(())
}
