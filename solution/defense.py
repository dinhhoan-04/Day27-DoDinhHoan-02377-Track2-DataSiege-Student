"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
import math
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    """
    Checks data_batch for volume_spike, volume_drop, null_spike,
    freshness_lag, and distribution_shift.
    Uses single-dimension baseline 3-sigma thresholds and a composite
    multi-dimensional Euclidean distance (norm2) for correlated shifts.
    """
    batch_id = payload.get("batch_id")
    prof = ctx.tools.batch_profile(batch_id)
    if not prof or "error" in prof:
        return Verdict(alert=False, pillar="checks")

    row_count = prof.get("row_count", 0)
    null_rate = prof.get("null_rate", {}).get("customer_id", 0.0)
    mean_amount = prof.get("mean_amount", 0.0)
    staleness_min = prof.get("staleness_min", 0.0)

    # 1. Single-dimensional checks against published 3-sigma baselines
    b = ctx.baseline
    if row_count < b["row_count_min"]:
        return Verdict(alert=True, pillar="checks", reason=f"Volume drop: row_count={row_count} < {b['row_count_min']}")
    if row_count > b["row_count_max"]:
        return Verdict(alert=True, pillar="checks", reason=f"Volume spike: row_count={row_count} > {b['row_count_max']}")
    if null_rate > b["null_rate_max"]:
        return Verdict(alert=True, pillar="checks", reason=f"Null spike: null_rate={null_rate} > {b['null_rate_max']}")
    if mean_amount < b["mean_amount_min"]:
        return Verdict(alert=True, pillar="checks", reason=f"Distribution shift: mean_amount={mean_amount} < {b['mean_amount_min']}")
    if mean_amount > b["mean_amount_max"]:
        return Verdict(alert=True, pillar="checks", reason=f"Distribution shift: mean_amount={mean_amount} > {b['mean_amount_max']}")
    if staleness_min > b["staleness_min_max"]:
        return Verdict(alert=True, pillar="checks", reason=f"Freshness lag: staleness_min={staleness_min} > {b['staleness_min_max']}")

    # 2. Multi-signal Composite Distance:
    # Normalized Euclidean distance across dimensions to catch joint subtle shifts
    mu_rows = (b["row_count_min"] + b["row_count_max"]) / 2.0
    sigma_rows = (b["row_count_max"] - b["row_count_min"]) / 6.0
    mu_mean = (b["mean_amount_min"] + b["mean_amount_max"]) / 2.0
    sigma_mean = (b["mean_amount_max"] - b["mean_amount_min"]) / 6.0

    z_rows = (row_count - mu_rows) / sigma_rows
    z_mean = (mean_amount - mu_mean) / sigma_mean
    norm_null = null_rate / b["null_rate_max"]
    norm_stale = staleness_min / b["staleness_min_max"]

    norm2 = math.sqrt(z_rows ** 2 + z_mean ** 2 + norm_null ** 2 + norm_stale ** 2)
    if norm2 > 3.2:
        return Verdict(alert=True, pillar="checks", reason=f"Multi-signal anomaly: composite norm2={norm2:.2f} > 3.2")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    """
    Checks contract_checkpoint for ODCS contract violations:
    schema_hash_mismatch, type_violation, and SLA freshness violations.
    """
    contract_id = payload.get("contract_id")
    checkpoint_batch_id = payload.get("checkpoint_batch_id")
    diff = ctx.tools.contract_diff(contract_id, checkpoint_batch_id)
    if not diff or "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    # 1. Direct contract violations (schema hash, amount type)
    violations = diff.get("violations", [])
    if violations:
        return Verdict(alert=True, pillar="contracts", reason=f"Contract violations: {', '.join(violations)}")

    # 2. SLA freshness delay vs declared SLA / baseline
    freshness_delay_min = diff.get("freshness_delay_min", 0.0)
    declared_sla = payload.get("declared_sla", {}).get("freshness_min", 15)
    if freshness_delay_min > declared_sla or freshness_delay_min > ctx.baseline.get("freshness_delay_max_min", 11.1141):
        return Verdict(alert=True, pillar="contracts", reason=f"SLA violation: delay={freshness_delay_min} min exceeds limit")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    """
    Checks lineage_run for DAG transform integrity:
    orphan_output, missing_upstream, and runtime_anomaly.
    """
    run_id = payload.get("run_id")
    slice_data = ctx.tools.lineage_graph_slice(run_id)
    if not slice_data or "error" in slice_data:
        return Verdict(alert=False, pillar="lineage")

    # 1. Orphan Output: Table produced with 0 downstream consumers
    downstream_count = slice_data.get("actual_downstream_count", 1)
    if downstream_count == 0:
        return Verdict(alert=True, pillar="lineage", reason="Orphan output: actual_downstream_count is 0")

    # 2. Missing Upstream: dbt transform missing required input model
    actual_upstream = set(slice_data.get("actual_upstream", []))
    required_upstream = {"raw.orders", "raw.customers"}
    if not required_upstream.issubset(actual_upstream):
        missing = required_upstream - actual_upstream
        return Verdict(alert=True, pillar="lineage", reason=f"Missing upstream model: {missing}")

    # 3. Runtime Anomaly: Duration exceeding calibrated baseline max
    duration_ms = slice_data.get("duration_ms", 0.0)
    if duration_ms > ctx.baseline.get("lineage_duration_ms_max", 5134.9804):
        return Verdict(alert=True, pillar="lineage", reason=f"Runtime anomaly: duration_ms={duration_ms:.1f} > max")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    """
    Checks feature_materialization for Training-Serving skew and Feature Drift.
    Uses calibrated threshold separating clean variance from drift.
    """
    feature_view = payload.get("feature_view")
    batch_id = payload.get("batch_id")
    drift = ctx.tools.feature_drift(feature_view, batch_id)
    if not drift or "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    mean_shift_sigma = drift.get("mean_shift_sigma", 0.0)
    # Calibrated separation threshold: clean data max is ~0.472, smallest skew is ~0.970
    if mean_shift_sigma > 0.65:
        return Verdict(alert=True, pillar="ai_infra", reason=f"Feature drift: mean_shift_sigma={mean_shift_sigma} > 0.65")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    """
    Checks embedding_batch for vector embedding drift and corpus staleness.
    Separates natural clean variance from semantic drift / stale documents.
    """
    corpus = payload.get("corpus")
    chunk_batch_id = payload.get("chunk_batch_id")
    drift = ctx.tools.embedding_drift(corpus, chunk_batch_id)
    if not drift or "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    centroid_shift = drift.get("centroid_shift", 0.0)
    avg_doc_age_days = drift.get("avg_doc_age_days", 0.0)

    # 1. Embedding drift check: clean centroid shift <= 0.0387; drift >= 0.0400
    if centroid_shift > 0.0395:
        return Verdict(alert=True, pillar="ai_infra", reason=f"Embedding drift: centroid_shift={centroid_shift:.4f} > 0.0395")

    # 2. Corpus staleness check: clean doc age <= 37.5 days; stale corpus >= 48.0 days
    if avg_doc_age_days > 41.0:
        return Verdict(alert=True, pillar="ai_infra", reason=f"Corpus staleness: avg_doc_age_days={avg_doc_age_days:.1f} > 41.0")

    return Verdict(alert=False, pillar="ai_infra")
