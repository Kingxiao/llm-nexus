//! Vertex AI OAuth2 token management.
//!
//! Token sources (tried in order):
//! 1. `VERTEX_ACCESS_TOKEN` env var (static, user manages refresh)
//! 2. `gcloud auth print-access-token` CLI command (auto-refresh)
//!
//! Tokens are cached and refreshed 5 minutes before expiry.
//! Uses RwLock with double-check to prevent thundering herd on refresh.

use std::time::{Duration, Instant};

use tokio::sync::RwLock;

/// Cached OAuth2 access token with expiry tracking.
pub struct TokenCache {
    inner: RwLock<CachedToken>,
}

struct CachedToken {
    token: String,
    obtained_at: Instant,
    /// Conservative TTL: refresh 5 min before 1h expiry.
    ttl: Duration,
    /// Full TTL: use as fallback if refresh fails.
    full_ttl: Duration,
    /// Whether the token came from a static env var (don't refresh).
    is_static: bool,
}

impl TokenCache {
    /// Create a new token cache. Resolves the initial token immediately.
    pub fn new() -> Result<Self, String> {
        let (token, is_static) = resolve_initial_token()?;
        Ok(Self {
            inner: RwLock::new(CachedToken {
                token,
                obtained_at: Instant::now(),
                ttl: Duration::from_secs(55 * 60), // refresh at 55 min
                full_ttl: Duration::from_secs(60 * 60), // hard expiry at 60 min
                is_static,
            }),
        })
    }

    /// Get a valid access token, refreshing if needed.
    ///
    /// Uses RwLock double-check to prevent thundering herd:
    /// multiple readers can get the cached token simultaneously,
    /// but only one writer refreshes when expired.
    pub async fn get_token(&self) -> Result<String, String> {
        // Fast path: read lock
        {
            let cached = self.inner.read().await;
            if cached.is_static || cached.obtained_at.elapsed() < cached.ttl {
                return Ok(cached.token.clone());
            }
        }

        // Slow path: write lock (only one caller refreshes)
        let mut cached = self.inner.write().await;

        // Double-check: another caller may have refreshed while we waited
        if cached.obtained_at.elapsed() < cached.ttl {
            return Ok(cached.token.clone());
        }

        // Refresh via gcloud
        match refresh_token_gcloud().await {
            Ok(new_token) => {
                cached.token = new_token.clone();
                cached.obtained_at = Instant::now();
                tracing::debug!("vertex AI token refreshed");
                Ok(new_token)
            }
            Err(e) => {
                // Graceful degradation: if old token is still within full TTL,
                // return it with a warning instead of failing the request.
                if cached.obtained_at.elapsed() < cached.full_ttl {
                    tracing::warn!(
                        error = %e,
                        "vertex AI token refresh failed, using existing token (still within 1h TTL)"
                    );
                    Ok(cached.token.clone())
                } else {
                    Err(format!("vertex AI token expired and refresh failed: {e}"))
                }
            }
        }
    }
}

/// Resolve initial token from env or gcloud.
fn resolve_initial_token() -> Result<(String, bool), String> {
    // 1. Static token from env var
    if let Ok(token) = std::env::var("VERTEX_ACCESS_TOKEN") {
        if token.is_empty() {
            tracing::debug!("VERTEX_ACCESS_TOKEN is empty, trying gcloud");
        } else {
            tracing::info!("vertex AI: using static VERTEX_ACCESS_TOKEN");
            return Ok((token, true));
        }
    }

    // 2. Try gcloud CLI
    match std::process::Command::new("gcloud")
        .args(["auth", "print-access-token"])
        .output()
    {
        Ok(output) if output.status.success() => {
            let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if token.is_empty() {
                return Err("gcloud returned empty token".into());
            }
            tracing::info!("vertex AI: using gcloud auth token");
            Ok((token, false))
        }
        Ok(output) => Err(format!(
            "gcloud auth failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(e) => Err(format!(
            "gcloud not found: {e}. Set VERTEX_ACCESS_TOKEN or install gcloud CLI"
        )),
    }
}

/// Refresh token via gcloud CLI (async).
async fn refresh_token_gcloud() -> Result<String, String> {
    let output = tokio::process::Command::new("gcloud")
        .args(["auth", "print-access-token"])
        .output()
        .await
        .map_err(|e| format!("gcloud exec failed: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "gcloud auth refresh failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err("gcloud returned empty token on refresh".into());
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_token_from_env() {
        unsafe { std::env::set_var("VERTEX_ACCESS_TOKEN", "test-token-123") };
        let cache = TokenCache::new().unwrap();
        let token = cache.get_token().await.unwrap();
        assert_eq!(token, "test-token-123");

        // Multiple calls should return the same static token without refresh
        for _ in 0..10 {
            assert_eq!(cache.get_token().await.unwrap(), "test-token-123");
        }
        unsafe { std::env::remove_var("VERTEX_ACCESS_TOKEN") };
    }
}
