//! Cost routing example — automatically selects the cheapest model.
//!
//! Usage: cargo run --example cost_routing -p llm-nexus --features full
//!
//! Set at least one provider's API key environment variable before running.

use llm_nexus::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_dir = std::path::Path::new("config");
    let client = NexusClient::from_config_dir(config_dir)?;

    // Route to cheapest model with Chat capability
    let request = ChatRequest {
        messages: vec![Message::user("What is the meaning of life?")],
        ..Default::default()
    };
    let response = client
        .chat_with_routing(
            &request,
            &RouteContext {
                required_capabilities: vec![Capability::Chat],
                ..Default::default()
            },
        )
        .await?;

    println!("Routed to: {}", response.model);
    println!("Response: {}", response.content);

    Ok(())
}
