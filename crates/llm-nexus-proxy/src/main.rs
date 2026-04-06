//! llm-nexus-proxy binary — starts the OpenAI-compatible proxy server.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use llm_nexus::NexusClient;
use llm_nexus_proxy::{build_router_with_state, AppState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug".into()),
        )
        .init();

    let config_dir = std::env::var("NEXUS_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("config"));

    let host = std::env::var("NEXUS_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = std::env::var("NEXUS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    tracing::info!(config_dir = %config_dir.display(), "loading configuration");
    let client = NexusClient::from_config_dir(&config_dir)?;
    let client = Arc::new(client);

    tracing::info!(
        providers = ?client.provider_ids(),
        "registered providers"
    );

    let metrics_gatherer: Option<Arc<llm_nexus_proxy::MetricsGatherer>> = {
        #[cfg(feature = "prometheus")]
        {
            use llm_nexus_metrics::PrometheusExporter;
            // Wrap the client's own metrics backend so Prometheus counters
            // reflect the same data as client.stats() queries.
            let exporter = Arc::new(PrometheusExporter::new(client.metrics().clone()));
            let exporter_ref = exporter.clone();
            tracing::info!("prometheus metrics enabled at /metrics");
            Some(Arc::new(move || exporter_ref.gather()))
        }
        #[cfg(not(feature = "prometheus"))]
        { None }
    };

    let state = AppState {
        client,
        metrics_gatherer,
        virtual_keys: None, // Set to Some(Arc::new(VirtualKeyStore)) for multi-tenant mode
    };
    let app = build_router_with_state(state);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!(%addr, "starting proxy server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("proxy server shut down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received Ctrl+C, shutting down"),
        () = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}
