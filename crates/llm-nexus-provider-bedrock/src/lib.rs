//! AWS Bedrock provider adapter for llm-nexus.
//!
//! Implements [`ChatProvider`](llm_nexus_core::traits::chat::ChatProvider) via the
//! Bedrock Converse API. Uses the official AWS SDK for Rust and supports the full
//! credential chain (env vars, IAM roles, instance profiles, SSO).
//!
//! Model IDs follow the Bedrock format: `anthropic.claude-sonnet-4-6-20250514-v1:0`
//!
//! # Examples
//!
//! ```rust,ignore
//! use llm_nexus_provider_bedrock::BedrockProvider;
//!
//! # async fn run() -> llm_nexus_core::error::NexusResult<()> {
//! let provider = BedrockProvider::from_env().await?;
//! # Ok(())
//! # }
//! ```

pub mod convert;

use std::pin::Pin;

use async_trait::async_trait;
use aws_sdk_bedrockruntime::Client as BedrockClient;
use futures::Stream;
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};

use convert::{from_converse_output, to_converse_input};

/// AWS Bedrock provider using the Converse API.
pub struct BedrockProvider {
    client: BedrockClient,
    /// AWS region (e.g. "us-east-1"). Used for cross-region inference routing.
    #[allow(dead_code)]
    region: String,
}

impl BedrockProvider {
    /// Create from the default AWS credential chain.
    ///
    /// Reads credentials from:
    /// - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` env vars
    /// - IAM instance profile
    /// - SSO / credential process
    ///
    /// Region from `AWS_REGION` or `AWS_DEFAULT_REGION` env var.
    pub async fn from_env() -> NexusResult<Self> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let region = config
            .region()
            .map(|r| r.to_string())
            .unwrap_or_else(|| "us-east-1".into());
        let client = BedrockClient::new(&config);

        Ok(Self { client, region })
    }

    /// Create with explicit region override.
    pub async fn with_region(region: impl Into<String>) -> NexusResult<Self> {
        let region_str: String = region.into();
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_sdk_bedrockruntime::config::Region::new(
                region_str.clone(),
            ))
            .load()
            .await;
        let client = BedrockClient::new(&config);

        Ok(Self {
            client,
            region: region_str,
        })
    }

    /// Create with an existing SDK client (for testing).
    pub fn from_client(client: BedrockClient, region: impl Into<String>) -> Self {
        Self {
            client,
            region: region.into(),
        }
    }
}

#[async_trait]
impl ChatProvider for BedrockProvider {
    fn provider_id(&self) -> &str {
        "aws_bedrock"
    }

    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        let (messages, system, inference_config) = to_converse_input(request)?;

        let mut req = self
            .client
            .converse()
            .model_id(&request.model)
            .set_messages(Some(messages));

        if let Some(sys) = system {
            req = req.system(sys);
        }
        if let Some(config) = inference_config {
            req = req.inference_config(config);
        }

        let output = req.send().await.map_err(|e| NexusError::ProviderError {
            provider: "aws_bedrock".into(),
            message: format!("Bedrock Converse error: {e}"),
            status_code: None,
        })?;

        from_converse_output(output, &request.model)
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        use aws_sdk_bedrockruntime::types::ConverseStreamOutput as CSOEvent;
        use llm_nexus_core::types::response::{FinishReason, Usage};

        let (messages, system, inference_config) = to_converse_input(request)?;

        let mut req = self
            .client
            .converse_stream()
            .model_id(&request.model)
            .set_messages(Some(messages));

        if let Some(sys) = system {
            req = req.system(sys);
        }
        if let Some(config) = inference_config {
            req = req.inference_config(config);
        }

        let output = req.send().await.map_err(|e| NexusError::ProviderError {
            provider: "aws_bedrock".into(),
            message: format!("Bedrock ConverseStream error: {e}"),
            status_code: None,
        })?;

        // Convert EventReceiver into a channel-based Stream
        let (tx, rx) = tokio::sync::mpsc::channel::<NexusResult<StreamChunk>>(32);
        let mut event_receiver = output.stream;

        tokio::spawn(async move {
            loop {
                match event_receiver.recv().await {
                    Ok(Some(event)) => {
                        let chunk = match event {
                            CSOEvent::ContentBlockDelta(delta_event) => match delta_event.delta() {
                                Some(aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(
                                    text,
                                )) => Some(StreamChunk {
                                    delta_content: Some(text.clone()),
                                    delta_tool_call: None,
                                    finish_reason: None,
                                    usage: None,
                                }),
                                _ => None,
                            },
                            CSOEvent::MessageStop(stop_event) => {
                                let reason = match stop_event.stop_reason() {
                                    aws_sdk_bedrockruntime::types::StopReason::EndTurn => {
                                        FinishReason::Stop
                                    }
                                    aws_sdk_bedrockruntime::types::StopReason::MaxTokens => {
                                        FinishReason::Length
                                    }
                                    aws_sdk_bedrockruntime::types::StopReason::ToolUse => {
                                        FinishReason::ToolCalls
                                    }
                                    _ => FinishReason::Stop,
                                };
                                Some(StreamChunk {
                                    delta_content: None,
                                    delta_tool_call: None,
                                    finish_reason: Some(reason),
                                    usage: None,
                                })
                            }
                            CSOEvent::Metadata(meta) => {
                                let usage = meta.usage().map(|u| Usage {
                                    prompt_tokens: u.input_tokens() as u32,
                                    completion_tokens: u.output_tokens() as u32,
                                    total_tokens: (u.input_tokens() + u.output_tokens()) as u32,
                                });
                                Some(StreamChunk {
                                    delta_content: None,
                                    delta_tool_call: None,
                                    finish_reason: None,
                                    usage,
                                })
                            }
                            _ => None, // ContentBlockStart, ContentBlockStop, MessageStart
                        };
                        if let Some(chunk) = chunk
                            && tx.send(Ok(chunk)).await.is_err()
                        {
                            break;
                        }
                    }
                    Ok(None) => break, // stream ended
                    Err(e) => {
                        let _ = tx
                            .send(Err(NexusError::StreamError(format!(
                                "Bedrock stream error: {e}"
                            ))))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn list_models(&self) -> NexusResult<Vec<String>> {
        // Bedrock model listing requires a different API (ListFoundationModels)
        Ok(vec![])
    }
}
