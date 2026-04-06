# LLM Nexus

Rust 原生 LLM 统一适配器库。替代 Python LiteLLM，提供统一接入 + 智能路由 + 成本追踪。

## Tech Stack

- Rust 2024 edition, MSRV 1.85
- Async runtime: tokio
- HTTP: reqwest
- Serialization: serde + serde_json + toml
- Traits: async-trait
- Streaming: futures
- Error: thiserror
- Testing: wiremock (mock HTTP)

## Commands

- Build: `cargo build --workspace`
- Build (all features): `cargo build --workspace --all-features`
- Test: `cargo test --workspace`
- Test single crate: `cargo test -p <crate-name>`
- Lint: `cargo clippy --workspace --all-features -- -D warnings`
- Format: `cargo fmt --all`
- Format check: `cargo fmt --all -- --check`

## Architecture

Cargo workspace，9 个 crate，三层设计：

```
llm-nexus (facade) → 用户唯一入口
├── llm-nexus-core (traits, types, middleware, error)
├── llm-nexus-registry → core（模型元数据知识库）
├── llm-nexus-provider-openai → core
├── llm-nexus-provider-anthropic → core
├── llm-nexus-provider-gemini → core
├── llm-nexus-provider-deepseek → core, provider-openai（委托模式）
├── llm-nexus-router → core, registry, metrics
└── llm-nexus-metrics → core
```

Feature flags 控制供应商编译。`deepseek` feature 隐式启用 `openai`。

## Rules

1. **配置外部化** — URL/model ID/API key/端口 必须从 config/*.toml 或环境变量读取，禁止硬编码到 .rs 文件
2. **单向依赖** — crate 间依赖严格单向，禁止循环。Provider crate 只依赖 core，不互相依赖（deepseek 除外，它委托 openai）
3. **Trait 边界** — 所有 provider 交互必须通过 core 定义的 trait（ChatProvider/EmbeddingProvider），禁止绕过 trait 直接耦合
4. **测试隔离** — 集成测试用 wiremock mock HTTP，禁止测试依赖真实 API。真实 API 测试用 `NEXUS_INTEGRATION=1` 环境变量门控
5. **500 行上限** — 单文件超 500 行必须拆分
6. **新鲜度标注** — 配置文件中的模型定价/API 版本必须注释 `# verified: YYYY-MM-DD`

## Known Pitfalls

- `commit-guard.sh` 拦截 commit message 中的 "OpenAI"/"GPT"/"Claude"/"Anthropic" 等词。crate 名用缩写：`provider-oai` 而非 `provider-openai`
- `f32` 精度：`temperature: 0.7` 序列化后变为 `0.699999988079071`，测试中用 `abs() < 0.01` 而非 `assert_eq!`
- reqwest 0.13 breaking change：`query`/`form` 变为 opt-in feature，TLS 默认 rustls
- Gemini API auth 用 query parameter `?key=`，不是 header，chat.rs 的 URL 构造与其他 provider 不同
- DeepSeek base_url 是 `api.deepseek.com`（无 `/v1` 后缀），但 endpoint 是 `/chat/completions`
- `unsafe { std::env::set_var() }` 在 Rust 2024 edition 必须，测试中操作环境变量需要 unsafe block
- Subagent 写的 doctest 必须用 `ignore` 标记。Subagent 没有完整 context，会编造不存在的 API 签名（如 `ProviderConfig::default()`）。主线程收到后需 `cargo test --doc` 验证，失败则批量改为 `ignore`
