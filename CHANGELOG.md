# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-06

Initial release of LLM Nexus — a Rust-native unified LLM adapter library.

### Added

#### Core
- Unified type system with `ChatRequest`, `ChatResponse`, and streaming types across all providers
- Trait-based provider abstraction (`ChatProvider`, `EmbeddingProvider`) for compile-time dispatch
- Middleware pipeline with `RequestContext` and `KeyValueStore` for extensible request processing
- Retry middleware with configurable backoff and attempt limits
- Rate limiter middleware for per-provider throughput control
- Auth middleware with configurable scheme (Bearer default)
- Cache middleware with deterministic hash keying
- Guardrail middleware for content filtering
- Budget middleware with per-key spend tracking and TOCTOU-safe enforcement
- Logging backend integration
- Multimodal content types: audio and document support alongside text and images
- Batch API support for bulk request processing
- `thiserror`-based error hierarchy

#### Providers
- **OpenAI-compatible**: Chat completion, streaming with SSE, and model listing
- **Anthropic**: Full chat support with tool_calls conversion in both request and response
- **Gemini**: Chat via `generateContent` API, image URL to InlineData/FileData conversion, embedding API with single and batch modes
- **DeepSeek**: Provider delegating to the OpenAI-compatible adapter with correct base URL
- **Azure**: Support for both v1 (`/openai/deployments/`) and classic deployment URL formats
- **Bedrock**: AWS Converse API integration with credential chain auth, ConverseStream, and tool_use conversion
- **Vertex AI**: OAuth2 token auto-refresh via `gcloud`, RwLock-based token management, config-driven setup with graceful degradation

#### Routing
- Cost-based routing with model pricing awareness
- Fallback chain for automatic provider failover
- Composite weighted scorer combining multiple routing signals
- Latency-aware router with factory construction
- Cooldown routing to temporarily exclude unhealthy providers
- A/B experiment routing for traffic splitting
- Structured output routing with JSON Schema support

#### Metrics
- In-memory metrics backend with cost calculation and aggregation
- SQLite persistent backend with auto-migration and SQL-based aggregation
- Prometheus exporter with `/metrics` endpoint on the proxy
- Streaming metrics recording for real-time usage tracking
- Capacity-based eviction to bound memory usage

#### Registry
- TOML-based config loading for static model metadata
- Remote sync with three-level merge (local, remote, cache) and cache fallback
- Programmatic model registration via `NexusClientBuilder::with_model()`

#### Proxy
- Axum-based HTTP proxy server with SSE streaming relay
- Model listing endpoint
- Bearer token auth middleware
- Concurrency-based rate limiting
- Configurable CORS via `NEXUS_CORS_ORIGINS` environment variable
- Graceful shutdown on SIGTERM/Ctrl+C

#### Middleware
- Virtual keys with per-key model restrictions
- Tracing abstraction layer
- MCP gateway client integration

#### Streaming
- Tool calls delta parsing across all providers
- Budget spend recording during streaming responses

#### Testing
- Integration test framework gated behind `NEXUS_INTEGRATION` env variable
- End-to-end pipeline integration tests (7 scenarios)
- Live integration tests for OpenRouter, 302.ai, Minimax, and cache
- Wiremock-based HTTP mocking for unit tests

#### Build & CI
- GitHub Actions CI workflow
- crates.io publish metadata for all workspace crates
- Feature flags for conditional provider compilation (`deepseek` implicitly enables `openai`)

### Changed

- `chat_with_routing` accepts a full request parameter instead of individual fields
- Extracted provider factory into dedicated module, split Azure provider files
- Removed `cfg` gating from `AppState` for cleaner proxy architecture
- Enforced retry/timeout as innermost middleware ordering for correct behavior

### Fixed

- Default `auth_scheme` to Bearer when not explicitly configured
- Minimax base URL corrected to `api.minimaxi.com` with proper model name mapping
- Bedrock `ConverseStream` response parsing and tool_use conversion
- Gemini image URL conversion to `InlineData` and `FileData` parts
- Budget middleware TOCTOU race condition
- MCP client safety improvements
- Routing pipeline bypass edge case
- Mutex poison handling made consistent across all middleware
- Constant-time token comparison in auth middleware to prevent timing attacks
- Idle budget `key_locks` sweep to prevent unbounded memory growth
- Cache hash determinism across requests
- Regex compilation warnings suppressed
- Tracing dead code eliminated
- Virtual key model restriction enforcement
- Streaming budget spend now correctly recorded

### Security

- Constant-time token comparison in proxy auth to mitigate timing side-channels
- Configurable CORS origins (deny-by-default) via environment variable
- Virtual key model restrictions enforced at request time
- Budget TOCTOU fix prevents over-spend race conditions

[0.1.0]: https://github.com/nichochar/llm-adapter/releases/tag/v0.1.0
