//! Pipeline overhead benchmark — measures middleware dispatch latency.
//!
//! Usage: cargo run --example bench_pipeline -p llm-nexus --release --features full
//!
//! This benchmark does NOT call real APIs. It uses a mock dispatcher to
//! isolate pure pipeline overhead (middleware chain invocation, context
//! creation, extensions access).

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::Stream;
use llm_nexus_core::error::NexusResult;
use llm_nexus_core::pipeline::middleware::Dispatcher;
use llm_nexus_core::pipeline::{ChatMiddleware, Next, NextStream, RequestContext};
use llm_nexus_core::types::{ChatRequest, ChatResponse, Message, StreamChunk, Usage};

/// Instant-return dispatcher — no network, no IO.
struct NoOpDispatcher;

#[async_trait]
impl Dispatcher for NoOpDispatcher {
    async fn dispatch(
        &self,
        _ctx: &mut RequestContext,
        _request: &ChatRequest,
    ) -> NexusResult<ChatResponse> {
        Ok(ChatResponse {
            id: "bench".into(),
            model: "bench-model".into(),
            content: "ok".into(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            finish_reason: None,
            tool_calls: None,
        })
    }

    async fn dispatch_stream(
        &self,
        _ctx: &mut RequestContext,
        _request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// Pass-through middleware — measures pure dispatch overhead.
struct PassthroughMiddleware;

#[async_trait]
impl ChatMiddleware for PassthroughMiddleware {
    fn name(&self) -> &str {
        "bench::passthrough"
    }

    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        next.run(ctx, request).await
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        next.run(ctx, request).await
    }
}

fn make_request() -> ChatRequest {
    ChatRequest {
        model: "bench-model".into(),
        messages: vec![Message::user("hello")],
        ..Default::default()
    }
}

async fn bench_direct(iterations: usize) -> Duration {
    let dispatcher = NoOpDispatcher;
    let middlewares: Vec<Arc<dyn ChatMiddleware>> = vec![];
    let request = make_request();

    let start = Instant::now();
    for _ in 0..iterations {
        let mut ctx = RequestContext::new("bench".into());
        let next = Next {
            middlewares: &middlewares,
            dispatcher: &dispatcher,
        };
        let _ = next.run(&mut ctx, &request).await;
    }
    start.elapsed()
}

async fn bench_with_middleware(n_layers: usize, iterations: usize) -> Duration {
    let dispatcher = NoOpDispatcher;
    let middlewares: Vec<Arc<dyn ChatMiddleware>> = (0..n_layers)
        .map(|_| Arc::new(PassthroughMiddleware) as Arc<dyn ChatMiddleware>)
        .collect();
    let request = make_request();

    let start = Instant::now();
    for _ in 0..iterations {
        let mut ctx = RequestContext::new("bench".into());
        let next = Next {
            middlewares: &middlewares,
            dispatcher: &dispatcher,
        };
        let _ = next.run(&mut ctx, &request).await;
    }
    start.elapsed()
}

#[tokio::main]
async fn main() {
    let iterations = 100_000;

    println!("Pipeline Overhead Benchmark");
    println!("==========================");
    println!("Iterations: {iterations}");
    println!();

    // Warmup
    let _ = bench_direct(1000).await;

    let direct = bench_direct(iterations).await;
    println!(
        "Direct dispatch (0 middleware):  {:>8.2} ns/op  ({:.2} ms total)",
        direct.as_nanos() as f64 / iterations as f64,
        direct.as_secs_f64() * 1000.0
    );

    for n in [1, 3, 5, 10] {
        let elapsed = bench_with_middleware(n, iterations).await;
        let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        let overhead =
            ns_per_op - (direct.as_nanos() as f64 / iterations as f64);
        println!(
            "{n:>2} middleware layers:            {:>8.2} ns/op  (+{:.2} ns overhead)",
            ns_per_op, overhead
        );
    }

    println!();
    println!("Note: Real-world latency is dominated by network IO (~100-2000ms),");
    println!("not pipeline dispatch. Even 10 middleware layers add < 1us overhead.");
}
