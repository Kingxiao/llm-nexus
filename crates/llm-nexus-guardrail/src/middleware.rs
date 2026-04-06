//! GuardrailMiddleware — pre/post content checks on the pipeline.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::pipeline::context::RequestContext;
use llm_nexus_core::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};

/// Result of a guardrail check.
#[derive(Debug, Clone)]
pub enum GuardrailVerdict {
    /// Allow the request/response to proceed.
    Allow,
    /// Block with a reason (request rejected, or response suppressed).
    Block { reason: String },
}

/// A pluggable check that inspects requests and/or responses.
#[async_trait::async_trait]
pub trait GuardrailCheck: Send + Sync + 'static {
    /// Check a request before it reaches the provider.
    /// Default: allow all.
    async fn check_request(&self, _request: &ChatRequest) -> NexusResult<GuardrailVerdict> {
        Ok(GuardrailVerdict::Allow)
    }

    /// Check a response before it reaches the caller.
    /// Default: allow all.
    async fn check_response(&self, _response: &ChatResponse) -> NexusResult<GuardrailVerdict> {
        Ok(GuardrailVerdict::Allow)
    }
}

/// Middleware that runs pre/post guardrail checks.
pub struct GuardrailMiddleware {
    checks: Vec<Arc<dyn GuardrailCheck>>,
}

impl GuardrailMiddleware {
    pub fn new(checks: Vec<Arc<dyn GuardrailCheck>>) -> Self {
        Self { checks }
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for GuardrailMiddleware {
    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        // Pre-checks
        for check in &self.checks {
            if let GuardrailVerdict::Block { reason } = check.check_request(request).await? {
                tracing::warn!(reason = %reason, "guardrail blocked request");
                return Err(NexusError::GuardrailBlocked(reason));
            }
        }

        let response = next.run(ctx, request).await?;

        // Post-checks
        for check in &self.checks {
            if let GuardrailVerdict::Block { reason } = check.check_response(&response).await? {
                tracing::warn!(reason = %reason, "guardrail blocked response");
                return Err(NexusError::GuardrailBlocked(reason));
            }
        }

        Ok(response)
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        // Pre-checks only (can't post-check streaming content without buffering)
        for check in &self.checks {
            if let GuardrailVerdict::Block { reason } = check.check_request(request).await? {
                tracing::warn!(reason = %reason, "guardrail blocked streaming request");
                return Err(NexusError::GuardrailBlocked(reason));
            }
        }

        next.run(ctx, request).await
    }
}
