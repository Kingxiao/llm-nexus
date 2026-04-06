<p align="center">
  <h1 align="center">LLM Nexus</h1>
  <p align="center"><strong>The Rust-native LLM gateway with zero Python in the dependency tree.</strong></p>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.85+-orange?logo=rust" alt="Rust 1.85+"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="License"></a>
  <a href="https://github.com/Kingxiao/llm-nexus/actions"><img src="https://img.shields.io/badge/tests-300%2B%20passing-brightgreen" alt="Tests"></a>
  <a href="https://crates.io/crates/llm-nexus"><img src="https://img.shields.io/crates/v/llm-nexus.svg" alt="crates.io"></a>
</p>

---

Unified interface to every major LLM provider. One `ChatRequest`, one middleware pipeline, any model. Ship to production without worrying about your adapter library getting [supply-chain attacked](https://www.bleepingcomputer.com/news/security/litellm-risks-data-breach-with-high-severity-vulnerability/).

## Why Nexus?

| | LLM Nexus | LiteLLM | Portkey | BricksLLM |
|---|---|---|---|---|
| **Language** | Rust (native binary) | Python | TypeScript (SaaS) | Go |
| **Supply chain** | 0 Python deps, `cargo audit` | [CVE-2025-0319](https://nvd.nist.gov/vuln/detail/CVE-2025-0319) SSRF | Closed-source SaaS | Open-source |
| **Middleware** | Composable trait pipeline | Hook-based | Plugin system | N/A |
| **Routing** | Cost / latency / weighted + cooldown | Fallback list | Load balance | Round-robin |
| **Self-hosted proxy** | Single binary, ~10MB | Python process | N/A (SaaS) | Docker |
| **Streaming** | Native `Stream<Item>` + SSE | SSE | SSE | SSE |
| **Cost tracking** | Built-in + Prometheus | Via callbacks | Dashboard | Dashboard |
| **License** | MIT / Apache-2.0 | MIT | Proprietary | MIT |

## Features

- **7 native providers** -- OpenAI, Anthropic, Gemini, DeepSeek, Azure OpenAI, AWS Bedrock, Vertex AI
- **Infinite extensibility** -- any OpenAI-compatible API works out of the box (OpenRouter, MiniMax, 302.AI, Ollama, vLLM, LM Studio, etc.)
- **Middleware pipeline** -- `ChatMiddleware` trait, onion-model execution. Plug in cache, guardrails, budget control, retry, timeout, logging, tracing -- or write your own in 20 lines
- **Intelligent routing** -- cost-optimized, latency-optimized, weighted composite strategies with 429-aware cooldown and A/B experiment support
- **Proxy server** -- one command to start an OpenAI-compatible HTTP proxy with virtual keys and budget enforcement
- **Cost tracking** -- in-memory or SQLite metrics store, Prometheus `/metrics` endpoint
- **Structured output** -- JSON Schema `response_format` across OpenAI, Gemini, Anthropic
- **Multimodal** -- text, image, audio, document content types
- **Streaming** -- SSE with tool_calls delta support across all providers
- **MCP Gateway** -- JSON-RPC client for MCP tool discovery and execution
- **17 crates, ~300 tests, ~16K lines of Rust**

## Quick Start

### Install

```toml
# Cargo.toml
[dependencies]
llm-nexus = { version = "0.1", features = ["openai", "anthropic"] }
tokio = { version = "1", features = ["full"] }
```

Feature flags: `openai`, `anthropic`, `gemini`, `deepseek`, `azure`, `bedrock`, `vertex`, `full` (all providers).

### Basic Chat

```rust
use llm_nexus::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = NexusClient::from_config_dir(Path::new("config"))?;

    let request = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message::user("Explain supply chain attacks in 3 sentences.")],
        ..Default::default()
    };

    let response = client.chat(&request).await?;
    println!("{}", response.content);
    // Token usage tracked automatically: response.usage
    Ok(())
}
```

### Middleware Pipeline

Stack middlewares like tower layers. Each one can short-circuit (e.g. cache hit) or transform requests/responses.

```rust
use llm_nexus::prelude::*;
use llm_nexus_cache::ResponseCache;
use llm_nexus_guardrail::KeywordFilter;

let client = NexusClient::builder()
    .config_dir(Path::new("config"))?
    .auto_register_providers()
    // Onion model: cache runs first, guardrail runs second
    .middleware(Arc::new(ResponseCache::new(Duration::from_secs(300))))
    .middleware(Arc::new(KeywordFilter::deny(vec!["password".into()])))
    .build()?;

// Cached responses skip the provider entirely
let response = client.chat(&request).await?;
```

### Proxy Server (One Command)

```bash
# Set your keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Start the proxy
cargo run -p llm-nexus-proxy --features full
```

Now point any OpenAI-compatible client at `http://localhost:8080`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $NEXUS_PROXY_AUTH_TOKEN" \
  -d '{"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, `/metrics` (Prometheus).

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              NexusClient                    │
                          │  .chat() / .chat_stream() / .chat_batch()  │
                          └──────────────────┬──────────────────────────┘
                                             │
                          ┌──────────────────▼──────────────────────────┐
                          │          Middleware Pipeline                │
                          │  Cache → Guardrail → Budget → Logging → …  │
                          │         (onion model, composable)          │
                          └──────────────────┬──────────────────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │                   Router                        │
                    │  cost / latency / weighted / cooldown / A|B     │
                    └────────────────────────┬────────────────────────┘
                                             │
        ┌────────┬────────┬────────┬────────┼────────┬────────┬────────┐
        ▼        ▼        ▼        ▼        ▼        ▼        ▼        ▼
    ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
    │ OpenAI ││Anthropic││ Gemini ││DeepSeek││ Azure  ││Bedrock ││ Vertex │
    └────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘
        ▲                                                          ▲
        │              Any OpenAI-compatible API                   │
        └──── OpenRouter, Ollama, vLLM, LM Studio, 302.AI, … ─────┘
```

### Crate Map

| Crate | Role |
|---|---|
| `llm-nexus` | Facade -- the only crate you import |
| `llm-nexus-core` | Traits (`ChatProvider`, `EmbeddingProvider`, `ChatMiddleware`), types, error |
| `llm-nexus-registry` | Model metadata knowledge base (TOML + OpenRouter remote sync) |
| `llm-nexus-router` | Routing strategies: cost, latency, composite, cooldown, experiment |
| `llm-nexus-metrics` | In-memory / SQLite / Prometheus metrics collection |
| `llm-nexus-cache` | Response caching middleware |
| `llm-nexus-guardrail` | Content moderation middleware |
| `llm-nexus-budget` | Spend limit middleware |
| `llm-nexus-mcp` | MCP Gateway (tool discovery + execution) |
| `llm-nexus-proxy` | Axum HTTP proxy with virtual keys |
| `llm-nexus-provider-*` | 7 provider implementations |

## Supported Providers

| Provider | Auth | Streaming | Tool Calls | Structured Output |
|---|---|---|---|---|
| OpenAI | Bearer token | Yes | Yes | Yes |
| Anthropic | `x-api-key` header | Yes | Yes | Yes |
| Google Gemini | Query param `?key=` | Yes | Yes | Yes |
| DeepSeek | Bearer token | Yes | Yes | -- |
| Azure OpenAI | API key / AAD | Yes | Yes | Yes |
| AWS Bedrock | SigV4 | Yes | Yes | -- |
| Vertex AI | OAuth2 | Yes | Yes | Yes |
| **Any OpenAI-compatible** | Bearer token | Yes | Yes | Varies |

## Configuration

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS credentials for Bedrock |
| `NEXUS_HOST` | Proxy bind address (default: `0.0.0.0`) |
| `NEXUS_PORT` | Proxy port (default: `8080`) |
| `NEXUS_PROXY_AUTH_TOKEN` | Bearer token for proxy auth |
| `NEXUS_MAX_CONCURRENT_REQUESTS` | Concurrency limit (default: `100`) |
| `NEXUS_CONFIG_DIR` | Config directory path (default: `config`) |

### Config Files

```
config/
  providers.toml   # Provider endpoints, auth, base URLs
  models.toml      # Model metadata, pricing, capabilities
```

All provider URLs, model IDs, and pricing data live in TOML config -- not hardcoded in Rust source.

## Development

```bash
cargo build --workspace --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-features -- -D warnings
cargo fmt --all -- --check
```

### Integration Tests

Real API tests are gated behind `NEXUS_INTEGRATION=1`:

```bash
cp .env.example .env   # fill in real keys
source .env
NEXUS_INTEGRATION=1 cargo test -p llm-nexus --test chat_providers --features full
```

## Contributing

1. Fork & create a feature branch
2. Write tests (we use `wiremock` for HTTP mocking -- no real API calls in CI)
3. `cargo test --workspace --all-features && cargo clippy --workspace --all-features -- -D warnings`
4. Open a PR

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
