use llm_nexus_core::traits::metrics::{AggregatedStats, CallRecord, StatsFilter};

/// Aggregate a slice of call records according to the given filter.
pub fn aggregate(records: &[CallRecord], filter: &StatsFilter) -> AggregatedStats {
    let filtered: Vec<&CallRecord> = records
        .iter()
        .filter(|r| matches_filter(r, filter))
        .collect();

    if filtered.is_empty() {
        return AggregatedStats::default();
    }

    let total_calls = filtered.len() as u64;
    let successful = filtered.iter().filter(|r| r.success).count() as u64;
    let failed = total_calls - successful;
    let total_prompt_tokens: u64 = filtered.iter().map(|r| u64::from(r.prompt_tokens)).sum();
    let total_completion_tokens: u64 = filtered
        .iter()
        .map(|r| u64::from(r.completion_tokens))
        .sum();
    let total_cost: f64 = filtered.iter().map(|r| r.estimated_cost_usd).sum();
    let avg_latency: f64 =
        filtered.iter().map(|r| r.latency_ms as f64).sum::<f64>() / total_calls as f64;

    // P99 latency
    let mut latencies: Vec<u64> = filtered.iter().map(|r| r.latency_ms).collect();
    latencies.sort_unstable();
    let p99_idx = ((latencies.len() as f64) * 0.99).ceil() as usize;
    let p99 = latencies.get(p99_idx.saturating_sub(1)).copied();

    AggregatedStats {
        total_calls,
        successful_calls: successful,
        failed_calls: failed,
        total_prompt_tokens,
        total_completion_tokens,
        total_cost_usd: total_cost,
        avg_latency_ms: avg_latency,
        p99_latency_ms: p99,
    }
}

fn matches_filter(record: &CallRecord, filter: &StatsFilter) -> bool {
    if let Some(ref provider) = filter.provider_id
        && record.provider_id != *provider
    {
        return false;
    }
    if let Some(ref model) = filter.model_id
        && record.model_id != *model
    {
        return false;
    }
    if let Some(since) = filter.since
        && record.timestamp < since
    {
        return false;
    }
    if let Some(until) = filter.until
        && record.timestamp > until
    {
        return false;
    }
    true
}
