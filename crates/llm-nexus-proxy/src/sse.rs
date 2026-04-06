//! SSE streaming support for OpenAI-compatible streaming responses.

use std::pin::Pin;

use axum::response::sse::{Event, Sse};
use futures::{Stream, StreamExt};
use llm_nexus::NexusResult;
use llm_nexus::types::response::StreamChunk;

use crate::types::OaiStreamChunk;

/// Convert a NexusClient stream into an axum SSE response.
pub fn stream_to_sse(
    stream: Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>,
    request_id: String,
    model: String,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let event_stream = stream
        .map(move |result| {
            let event = match result {
                Ok(chunk) => {
                    let oai_chunk = OaiStreamChunk::from_nexus(&chunk, &request_id, &model);
                    match serde_json::to_string(&oai_chunk) {
                        Ok(json) => Event::default().data(json),
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize stream chunk");
                            Event::default().data(
                                r#"{"error":{"message":"serialization error","type":"server_error"}}"#,
                            )
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "stream error from provider");
                    Event::default().data(format!(
                        r#"{{"error":{{"message":"{}","type":"upstream_error"}}}}"#,
                        e.to_string().replace('"', "\\\"")
                    ))
                }
            };
            Ok::<_, std::convert::Infallible>(event)
        })
        .chain(futures::stream::once(async {
            Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
        }));

    Sse::new(event_stream)
}
