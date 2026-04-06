//! ChatMiddleware trait and Next continuation type.
//!
//! The middleware pipeline intercepts every `chat()` and `chat_stream()` call,
//! enabling caching, guardrails, budget, logging, retry, etc. without modifying
//! the provider or client code.
//!
//! ## Design
//!
//! ```text
//! client.chat(request)
//!   → middleware[0].process(ctx, req, next)
//!     → middleware[1].process(ctx, req, next)
//!       → ... → provider.chat(req)
//! ```
//!
//! Each middleware calls `next.run()` to pass to the next layer, or returns
//! directly to short-circuit (e.g. cache hit).

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;

use crate::error::NexusResult;
use crate::types::{ChatRequest, ChatResponse, StreamChunk};

use super::context::RequestContext;

/// Middleware that intercepts chat requests in the pipeline.
///
/// Implement this trait to add cross-cutting concerns (caching, logging,
/// retry, guardrails, budget, etc.) to the request path.
#[async_trait::async_trait]
pub trait ChatMiddleware: Send + Sync + 'static {
    /// Middleware name for identification (used for pipeline ordering).
    /// Default: type name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Process a non-streaming chat request.
    ///
    /// Call `next.run(ctx, request)` to forward to the next middleware/provider.
    /// Return directly to short-circuit the pipeline (e.g. cache hit).
    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse>;

    /// Process a streaming chat request.
    ///
    /// Default: delegates to `process_stream_inner` which calls `next.run_stream`.
    /// Override for middleware that needs to intercept streaming (e.g. logging).
    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        next.run(ctx, request).await
    }
}

/// Continuation handle for non-streaming requests.
///
/// Calling `run()` advances to the next middleware in the chain, or to the
/// provider dispatch if this is the last middleware.
pub struct Next<'a> {
    #[doc(hidden)]
    pub middlewares: &'a [Arc<dyn ChatMiddleware>],
    #[doc(hidden)]
    pub dispatcher: &'a dyn Dispatcher,
}

impl<'a> Next<'a> {
    /// Forward the request to the next middleware or provider.
    pub async fn run(
        self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
    ) -> NexusResult<ChatResponse> {
        if let Some((current, rest)) = self.middlewares.split_first() {
            let next = Next {
                middlewares: rest,
                dispatcher: self.dispatcher,
            };
            current.process(ctx, request, next).await
        } else {
            self.dispatcher.dispatch(ctx, request).await
        }
    }
}

/// Continuation handle for streaming requests.
pub struct NextStream<'a> {
    #[doc(hidden)]
    pub middlewares: &'a [Arc<dyn ChatMiddleware>],
    #[doc(hidden)]
    pub dispatcher: &'a dyn Dispatcher,
}

impl<'a> NextStream<'a> {
    /// Forward the request to the next middleware or provider (streaming).
    pub async fn run(
        self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        if let Some((current, rest)) = self.middlewares.split_first() {
            let next = NextStream {
                middlewares: rest,
                dispatcher: self.dispatcher,
            };
            current.process_stream(ctx, request, next).await
        } else {
            self.dispatcher.dispatch_stream(ctx, request).await
        }
    }
}

/// Internal trait for the pipeline terminal — dispatches to the actual provider.
///
/// This is an implementation detail. Use [`NexusClient`] instead of implementing
/// this trait directly.
#[doc(hidden)]
#[async_trait::async_trait]
pub trait Dispatcher: Send + Sync {
    async fn dispatch(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
    ) -> NexusResult<ChatResponse>;

    async fn dispatch_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>>;
}
