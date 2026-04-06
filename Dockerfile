# Multi-stage build for llm-nexus-proxy
# Produces a minimal container (~30MB) running the proxy server.
#
# Build:
#   docker build -t llm-nexus-proxy .
#
# Run:
#   docker run -p 8080:8080 \
#     -v ./config:/app/config \
#     -e OPENAI_API_KEY=your_key_here \
#     llm-nexus-proxy

# --- Stage 1: Build ---
FROM rust:1.85-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build only the proxy binary with prometheus support
RUN cargo build --release -p llm-nexus-proxy --features prometheus \
    && strip target/release/llm-nexus-proxy

# --- Stage 2: Runtime ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false nexus
WORKDIR /app

COPY --from=builder /build/target/release/llm-nexus-proxy /app/llm-nexus-proxy
COPY config/ /app/config/

RUN chown -R nexus:nexus /app
USER nexus

ENV NEXUS_HOST=0.0.0.0
ENV NEXUS_PORT=8080
ENV NEXUS_CONFIG_DIR=/app/config
ENV RUST_LOG=info

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/llm-nexus-proxy"]
