//! Tracing abstraction for observability integration.
//!
//! Defines a minimal [`SpanExporter`] trait that can be implemented with
//! OpenTelemetry, Datadog, or any other tracing system. No external
//! dependencies — the user brings their own tracing backend.
//!
//! ## GenAI Semantic Conventions
//!
//! When implementing, use these attribute keys (OTel GenAI conventions):
//! - `gen_ai.system` — provider name (e.g. "openai", "anthropic")
//! - `gen_ai.request.model` — model ID
//! - `gen_ai.usage.input_tokens` — prompt tokens consumed
//! - `gen_ai.usage.output_tokens` — completion tokens generated
//! - `gen_ai.request.temperature` — temperature parameter
//! - `gen_ai.request.max_tokens` — max_tokens parameter
//! - `gen_ai.response.finish_reason` — stop, length, tool_calls, etc.

use std::collections::HashMap;

/// A completed span with LLM-specific attributes.
#[derive(Debug, Clone)]
pub struct LlmSpan {
    /// Span name (e.g. "chat gpt-5.4").
    pub name: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Whether the request succeeded.
    pub success: bool,
    /// Key-value attributes following GenAI semantic conventions.
    pub attributes: HashMap<String, SpanValue>,
}

/// Attribute value types.
#[derive(Debug, Clone)]
pub enum SpanValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// Trait for exporting LLM request spans.
///
/// Implement this with your preferred tracing backend:
///
/// ```ignore
/// use opentelemetry::global;
/// struct OTelExporter { tracer: global::BoxedTracer }
///
/// impl SpanExporter for OTelExporter {
///     fn export(&self, span: LlmSpan) {
///         // Convert LlmSpan attributes to OTel span attributes
///     }
/// }
/// ```
pub trait SpanExporter: Send + Sync + 'static {
    /// Export a completed LLM span.
    fn export(&self, span: LlmSpan);
}
