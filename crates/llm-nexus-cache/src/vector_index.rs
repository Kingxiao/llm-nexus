//! In-memory vector index for semantic similarity search.
//!
//! Uses brute-force cosine similarity — sufficient for caches up to ~10K entries.
//! For larger deployments, swap with an ANN backend (HNSW, etc.).

use std::sync::RwLock;
use std::time::{Duration, Instant};

/// A cached embedding entry with its associated response bytes.
struct Entry {
    embedding: Vec<f32>,
    response: Vec<u8>,
    inserted_at: Instant,
    ttl: Duration,
}

/// Thread-safe brute-force vector index with TTL eviction.
pub struct VectorIndex {
    entries: RwLock<Vec<Entry>>,
    capacity: usize,
}

impl VectorIndex {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(capacity)),
            capacity,
        }
    }

    /// Search for the most similar entry above `threshold`.
    /// Returns the cached response bytes if found.
    pub fn search(&self, query: &[f32], threshold: f32) -> Option<Vec<u8>> {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_response = None;

        for entry in entries.iter() {
            if now.duration_since(entry.inserted_at) > entry.ttl {
                continue; // expired
            }
            let score = cosine_similarity(query, &entry.embedding);
            if score >= threshold && score > best_score {
                best_score = score;
                best_response = Some(entry.response.clone());
            }
        }

        best_response
    }

    /// Insert a new entry. Evicts expired entries and oldest if at capacity.
    pub fn insert(&self, embedding: Vec<f32>, response: Vec<u8>, ttl: Duration) {
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        // Evict expired entries
        entries.retain(|e| now.duration_since(e.inserted_at) <= e.ttl);

        // Evict oldest if at capacity
        if entries.len() >= self.capacity {
            entries.remove(0);
        }

        entries.push(Entry {
            embedding,
            response,
            inserted_at: now,
            ttl,
        });
    }

    /// Number of non-expired entries.
    pub fn len(&self) -> usize {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        entries
            .iter()
            .filter(|e| now.duration_since(e.inserted_at) <= e.ttl)
            .count()
    }
}

/// Cosine similarity between two vectors. Returns 0.0 on zero-norm vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_index_insert_and_search() {
        let index = VectorIndex::new(100);
        let emb = vec![1.0, 0.0, 0.0];
        let data = b"cached-response".to_vec();

        index.insert(emb.clone(), data.clone(), Duration::from_secs(60));

        let result = index.search(&emb, 0.95);
        assert_eq!(result, Some(data));
    }

    #[test]
    fn test_index_search_below_threshold() {
        let index = VectorIndex::new(100);
        index.insert(
            vec![1.0, 0.0, 0.0],
            b"resp".to_vec(),
            Duration::from_secs(60),
        );

        // Orthogonal query
        let result = index.search(&[0.0, 1.0, 0.0], 0.8);
        assert!(result.is_none());
    }

    #[test]
    fn test_index_ttl_expiry() {
        let index = VectorIndex::new(100);
        index.insert(
            vec![1.0, 0.0],
            b"expired".to_vec(),
            Duration::from_millis(1),
        );

        std::thread::sleep(Duration::from_millis(10));

        let result = index.search(&[1.0, 0.0], 0.9);
        assert!(result.is_none());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_index_capacity_eviction() {
        let index = VectorIndex::new(2);
        index.insert(vec![1.0, 0.0], b"first".to_vec(), Duration::from_secs(60));
        index.insert(vec![0.0, 1.0], b"second".to_vec(), Duration::from_secs(60));
        index.insert(vec![0.5, 0.5], b"third".to_vec(), Duration::from_secs(60));

        assert_eq!(index.len(), 2);
        // "first" should be evicted
        let result = index.search(&[1.0, 0.0], 0.99);
        assert!(result.is_none());
    }

    #[test]
    fn test_index_finds_best_match() {
        let index = VectorIndex::new(100);
        index.insert(
            vec![1.0, 0.0, 0.0],
            b"exact".to_vec(),
            Duration::from_secs(60),
        );
        index.insert(
            vec![0.9, 0.1, 0.0],
            b"close".to_vec(),
            Duration::from_secs(60),
        );

        // Query exactly matches first entry
        let result = index.search(&[1.0, 0.0, 0.0], 0.8);
        assert_eq!(result, Some(b"exact".to_vec()));
    }
}
