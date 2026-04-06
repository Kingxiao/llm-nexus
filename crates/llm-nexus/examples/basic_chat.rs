//! Basic chat example — sends a single message and prints the response.
//!
//! Usage: cargo run --example basic_chat -p llm-nexus --features full
//!
//! Set the appropriate API key environment variable before running.
//! Model can be overridden via NEXUS_MODEL env var (default: reads from config).

use llm_nexus::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_dir = std::path::Path::new("config");
    let client = NexusClient::from_config_dir(config_dir)?;

    let model = std::env::var("NEXUS_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".into());

    let response = client
        .chat(&ChatRequest {
            model,
            messages: vec![Message::user(
                "Explain Rust's ownership model in one paragraph.",
            )],
            ..Default::default()
        })
        .await?;

    println!("Model: {}", response.model);
    println!("Response: {}", response.content);
    println!(
        "Tokens: {} prompt + {} completion = {} total",
        response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens
    );

    // Show cost from metrics
    let stats = client.stats(&StatsFilter::default()).await?;
    println!("Estimated cost: ${:.6}", stats.total_cost_usd);

    Ok(())
}
