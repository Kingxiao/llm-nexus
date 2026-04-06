//! Pipeline terminal — dispatches requests to the actual provider.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::pipeline::context::RequestContext;
use llm_nexus_core::pipeline::middleware::Dispatcher;
use llm_nexus_core::traits::registry::ModelRegistry;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};
use llm_nexus_registry::StaticRegistry;

use crate::provider_map::ProviderMap;

/// Dispatches to the actual [`ChatProvider`] at the end of the middleware chain.
///
/// Handles model resolution (registry lookup) and provider selection.
pub(crate) struct ProviderDispatcher {
    pub(crate) providers: ProviderMap,
    pub(crate) registry: Arc<StaticRegistry>,
}

impl ProviderDispatcher {
    async fn resolve_and_dispatch(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
    ) -> NexusResult<(
        Arc<dyn llm_nexus_core::traits::chat::ChatProvider>,
        llm_nexus_core::types::model::ModelMetadata,
    )> {
        let model_meta = self
            .registry
            .get_model(&request.model)
            .await?
            .ok_or_else(|| NexusError::ModelNotFound(request.model.clone()))?;

        // Store metadata in context for downstream use (metrics, logging)
        ctx.model_meta = Some(model_meta.clone());
        ctx.provider_id = Some(model_meta.provider.clone());

        let provider = self
            .providers
            .resolve_provider_for_model(&request.model, &model_meta.provider)?;

        Ok((provider, model_meta))
    }
}

#[async_trait::async_trait]
impl Dispatcher for ProviderDispatcher {
    async fn dispatch(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
    ) -> NexusResult<ChatResponse> {
        let (provider, _) = self.resolve_and_dispatch(ctx, request).await?;
        provider.chat(request).await
    }

    async fn dispatch_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let (provider, _) = self.resolve_and_dispatch(ctx, request).await?;
        provider.chat_stream(request).await
    }
}
