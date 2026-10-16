# Reflection (≤1 page)

## 1. Which fault types were hardest to catch, and why?

Across the 5 event streams, the hardest fault types to detect were **subtle distribution shifts** in `data_batch`, **subtle vector embedding drift / corpus staleness** in `embedding_batch`, and **subtle runtime anomalies** in `lineage_run`:

- **Subtle distributional and vector shifts:** Unlike obvious faults (where single metrics shift beyond 5σ to 20σ), subtle faults in the private phase exhibit parameters that overlap significantly with the normal variance of clean data (e.g., centroid shifts around 0.028–0.031 while clean variance reaches 0.0387; document ages around 31–37 days while clean document ages span up to 37.5 days). 
- **The FPR penalty asymmetry:** The scoring objective heavily penalizes false alarms:
  $$\text{Score} = 100 \times (0.5 \cdot \text{TPR} - 0.3 \cdot \text{FPR} - 0.2 \cdot \min(\text{cost\_overage}, 1.0))$$
  With an FPR weight of $-30$, even a single false alarm on clean events ($N_{\text{clean}} = 146$ in private) costs approximately $0.21$ points, and a few false alarms quickly wipe out any marginal gains in TPR.
- **Why composite distance helped:** For `data_batch`, individual metrics like `row_count` or `mean_amount` often stayed just within individual $3\sigma$ bounds during subtle shifts. Implementing a normalized Euclidean composite distance ($norm_2 = \sqrt{z_{\text{rows}}^2 + z_{\text{mean}}^2 + (\text{null}/\text{null}_{\max})^2 + (\text{stale}/\text{stale}_{\max})^2}$) with a calibrated threshold of $3.2$ successfully captured multi-signal shifts in the public phase while guaranteeing $0\%$ false positive rate across all phases.

## 2. What would you change about your cost/coverage tradeoff, if you had another pass?

- **Current strategy:** Every event is profiled by calling exactly one dedicated metered tool (`batch_profile`, `contract_diff`, `lineage_graph_slice`, `feature_drift`, `embedding_drift`), resulting in:
  - **Practice:** $180.0 / 220.0$ credits (Cost overage: $0.0$, TPR: $1.0$, Score: $50.00$)
  - **Public:** $240.0 / 220.0$ credits (Cost overage: $0.0909$, TPR: $1.0$, Score: $48.18$)
  - **Private:** $300.0 / 320.0$ credits (Cost overage: $0.0$, TPR: $0.6481$, Score: $32.41$)
- **Trade-off evaluation:** In the public phase, spending $240.0$ credits incurred a small $1.82$ point overage penalty, but secured $100\%$ TPR ($+50$ points) across all 4 pillars, which was far superior to skipping checks to save budget.
- **Future refinement:** If given another pass, I would introduce a **free metadata heuristic pre-filter** using `payload` attributes and state history before invoking high-cost tools (such as $2.0$-credit `feature_drift` and `embedding_drift`). For example, tracking temporal regularity or skip-sampling for models with high historical stability could conserve credits and eliminate the $0.0909$ overage in the public phase.

## 3. Improvements to `ctx.state` statistical mechanisms

With additional time, I would leverage the zero-cost in-memory `ctx.state` dictionary to build:
1. **Online Exponential Moving Average (EMA) & Welford's Algorithm:** Continuously update running estimates of $\mu$ and $\sigma$ per table/job across consecutive batches. This would detect local step-changes and sudden variance shifts rather than relying strictly on static global baseline thresholds.
2. **CUSUM (Cumulative Sum Control Chart):** Accumulate small deviations over time to reliably catch subtle drift without triggering false positives on single-event fluctuations.
3. **Producer Trust Scores:** Maintain error frequency counters for contract producers (`checkout-service`) to dynamically adapt inspection depths based on historical reliability.
