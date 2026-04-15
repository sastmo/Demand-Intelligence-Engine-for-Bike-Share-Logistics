# Station-Level Analysis Memo
## Metro Bike Program: Why the Station-Level Forecasting Path Should Be Built Around a Global Station-Day Model With Explicit Slices

## Purpose

This memo consolidates the key insights from the station-level diagnosis and planning work completed so far. Its purpose is to document, in a clear and technically grounded way, **what the station network actually looks like**, **why a simple average-station view is misleading**, and **why the forecasting path should start from one global station-day workflow rather than fragmented station-specific or cluster-specific models**.

---

## Executive Summary

The station network is **heterogeneous, uneven in maturity, and operationally mixed**.

The analysis covers **381 observed stations** between **2019-01-01 and 2024-12-31**. Within that station universe:

- **304 stations are mature**
- **77 stations are newborn or young**
- **158 stations are not recently active**
- **6 stations are nearly always zero**
- The observed station count is **41 higher than the expected 340**, likely because the inventory still contains temporary, retired, or nearly empty stations.

The most important conclusion is that the system is **not composed of one “typical” station**. It contains several distinct station populations:

- a productive and reliable core,
- a broad middle of mixed-use stations,
- a meaningful weekend-oriented segment,
- a sparse and weak-signal tail,
- a very small anomaly-heavy group,
- and a non-trivial short-history population that should not be interpreted as a true behavior type.

This matters because the forecasting problem is not just “many station time series.” It is an **unbalanced panel of stations with different start dates, different operational lifetimes, different levels of signal quality, and different behavioral patterns**.

The station-level evidence supports a clear modeling direction:

1. forecast at the **station-day** level,
2. start with **one global model** as the default benchmark,
3. evaluate performance through **meaningful slices** rather than starting from many separate first-stage models,
4. treat **short-history** and **sparse** stations as special cases in evaluation and later policy.

That path is not a compromise. It is the most defensible way to balance statistical learning, operational simplicity, and model governance.

---

## 1. What Was Diagnosed

The station-level work used a station-day view and summarized each station using several behavior and quality signals, including:

- average demand,
- zero-rate,
- coefficient of variation,
- weekly persistence,
- weekday/weekend effect strength,
- weekend ratio,
- outlier rate,
- history length and maturity,
- recent activity,
- relationship with system-level movement.

These signals were then used to build:

1. **maturity groups**  
2. **activity and inactivity flags**  
3. **human-readable behavioral categories**  
4. **data-driven mature-station clusters**

This created a layered view of the station universe:

- **maturity** tells us how much history a station has,
- **activity** tells us whether it is currently meaningfully alive,
- **categories** tell us how a station behaves in practical terms,
- **clusters** tell us which mature stations are statistically similar across multiple features.

---

## 2. The Station Universe Is Broader and Messier Than the Operational Headcount

### Station universe validation

The station diagnosis identified:

| Measure | Value |
|---|---:|
| Expected stations | 340 |
| Observed unique stations | 381 |
| Gap | +41 |
| Mature stations | 304 |
| Newborn stations | 46 |
| Young stations | 31 |
| Short-history stations | 77 |
| Not recently active | 158 |
| Nearly always-zero | 6 |

### What this means

The observed universe is **wider than the expected operational station count**. The most plausible explanation is not a counting error. It is that the data still includes:

- temporary stations,
- retired stations,
- stations with very short operating windows,
- or stations that exist in inventory but contribute almost no usable recent demand.

This is a critical station-level insight because it changes the interpretation of the forecasting problem.

The challenge is not only demand prediction. It is also **defining which stations belong to which forecasting population**.

In other words, the station portfolio is not just behaviorally heterogeneous. It is also **operationally uneven**.

---

## 3. Why the “Average Station” Is the Wrong Mental Model

Across all 381 stations:

- mean station average demand = **7.04**
- median station average demand = **2.46**
- 75th percentile = **4.89**
- 90th percentile = **12.48**
- 95th percentile = **27.20**

### Interpretation

The mean is almost **3x the median**. That is a classic signal that the station universe is **highly skewed**.

This tells us that the network is not mostly made of medium-demand stations. Instead, it is made of:

- a smaller number of strong stations,
- a broad group of modest stations,
- and a long tail of low-demand stations.

That matters strategically because averages hide structure. In a portfolio like this, one “representative station” does not really exist.

The practical implication is that station-level forecasting should not be framed as 381 copies of the same problem. It is **one family of related problems with shared structure and unequal signal quality**.

---

## 4. The Four Interpretation Layers

## 4.1 Maturity groups

The station maturity split is:

| Maturity group | Count |
|---|---:|
| newborn | 46 |
| young | 31 |
| mature | 304 |

### Meaning

- **newborn** and **young** stations have limited historical evidence,
- **mature** stations have enough history for stable behavior estimation.

This distinction is foundational. A short-history station should not be interpreted in the same way as a mature station, even if a few summary metrics look superficially similar.

Short history is not just a smaller sample size. It affects:

- lag quality,
- stability of averages,
- reliability of seasonality estimates,
- and fairness of model evaluation.

---

## 4.2 Activity lens

The station activity view identifies:

| Activity lens | Count |
|---|---:|
| not recently active | 158 |
| nearly always-zero | 6 |

This means a large share of the observed station universe is not currently behaving like a strong live forecasting asset.

This matters because there is a major difference between:

1. a station that is in service and receives true zero demand on some days,
2. a station that is structurally sparse,
3. a station that is technically present in the data but not recently active,
4. and a station that may no longer be operationally relevant.

For forecasting, those cases should not be mentally collapsed into a single “low demand” bucket.

---

## 4.3 Behavioral categories

The station analysis produced **7 categories**:

| Category | Count |
|---|---:|
| mixed_profile | 123 |
| short_history | 77 |
| weekend_leisure | 74 |
| busy_stable | 50 |
| sparse_intermittent | 44 |
| anomaly_heavy | 7 |
| seasonal_commuter | 6 |

### Why categories matter

Categories are the most useful **human interpretation layer**. They answer:

> “What kind of station does this look like?”

They are useful because they convert many raw metrics into simple station stories.

---

## 4.4 Mature-station clusters

The station analysis also produced **4 meaningful clusters** for mature stations, plus a separate non-cluster bucket for short-history stations.

| Cluster label | Count |
|---|---:|
| cluster_1 | 32 |
| cluster_2 | 139 |
| cluster_3 | 43 |
| cluster_4 | 90 |
| not_clustered_short_history | 77 |

The selected solution used **k = 4** with a silhouette score of **0.226**, chosen because it gave usable separation without tiny unstable clusters.

### Why clusters matter

Clusters answer a different question from categories.

Categories answer:

> “What behavior label best describes this station?”

Clusters answer:

> “Which mature stations are numerically similar across many behavior signals?”

So categories are primarily interpretive. Clusters are primarily structural.

---

## 5. Category-Level Insights

## 5.1 busy_stable

| Measure | Value |
|---|---:|
| station count | 50 |
| mean avg_demand | 8.31 |
| mean zero_rate | 0.041 |
| mean outlier_rate | 0.011 |
| mature share | 1.00 |

### Interpretation

This is the **productive and reliable core** of the network.

These stations:
- are active on most days,
- have relatively low zero-rates,
- carry strong usable signal,
- and are the clearest candidates for stable pattern learning.

This group is not necessarily small in business importance just because it is only 50 stations. It likely contains a disproportionate share of the forecastable signal.

### Strategic reading

This group tells us the network does contain strong recurring structure. The station portfolio is not chaotic. It has a meaningful stable core.

---

## 5.2 mixed_profile

| Measure | Value |
|---|---:|
| station count | 123 |
| mean avg_demand | 2.63 |
| mean zero_rate | 0.239 |
| mean outlier_rate | 0.022 |
| mature share | 1.00 |

### Interpretation

This is the **broad middle** of the network.

These stations are neither strongly sparse nor strongly specialized. They are the most typical representation of a moderate, usable but not dominant station.

### Strategic reading

This group is important because it makes up the largest single category. Any forecasting approach that only works on top stations will miss the center of the network.

---

## 5.3 weekend_leisure

| Measure | Value |
|---|---:|
| station count | 74 |
| mean avg_demand | 3.70 |
| mean zero_rate | 0.296 |
| mean outlier_rate | 0.019 |
| mature share | 1.00 |

### Interpretation

This is the clearest non-commuter behavioral segment.

These stations show a stronger weekend signature and indicate that the network has a meaningful leisure and discretionary usage component.

### Strategic reading

This is important because it prevents a false assumption that metro bike demand is mostly a weekday commuter problem. It is not. A material part of the network behaves differently.

---

## 5.4 sparse_intermittent

| Measure | Value |
|---|---:|
| station count | 44 |
| mean avg_demand | 0.39 |
| mean zero_rate | 0.761 |
| mean outlier_rate | 0.012 |
| mature share | 1.00 |

### Interpretation

This is the **weak-signal tail** of the network.

These stations:
- are quiet most of the time,
- have many zero days,
- and generate unstable or less trustworthy station-level metrics.

The diagnosis correctly warns that sparse stations should not dominate interpretation because their ratios and autocorrelation measures become less reliable.

### Strategic reading

This group matters more as a **modeling and governance problem** than as a demand-volume driver.

If this slice is not treated explicitly, it can distort model comparison and hide useful performance on healthier stations.

---

## 5.5 anomaly_heavy

| Measure | Value |
|---|---:|
| station count | 7 |
| mean avg_demand | 6.39 |
| mean zero_rate | 0.226 |
| mean outlier_rate | 0.114 |
| mature share | 1.00 |

### Interpretation

This is a very small but important monitoring group.

These stations show a much higher concentration of unusual spikes or behavior shifts. They may reflect:
- event-driven demand,
- operational interventions,
- disruptions,
- or special local context.

### Strategic reading

This slice should not define the default modeling path, but it should be tracked explicitly so forecast instability is not hidden inside aggregate metrics.

---

## 5.6 seasonal_commuter

| Measure | Value |
|---|---:|
| station count | 6 |
| mean avg_demand | 4.43 |
| mean zero_rate | 0.225 |
| mean outlier_rate | 0.036 |
| mature share | 1.00 |

### Interpretation

This is the cleanest weekday-oriented segment, but it is very small.

### Strategic reading

This is one of the most useful negative findings from the diagnosis: **the network is not dominated by classic commuter-only stations**.

A pure commute framing would overstate one pattern and understate the actual diversity in the station portfolio.

---

## 5.7 short_history

| Measure | Value |
|---|---:|
| station count | 77 |
| mean avg_demand | 20.55 |
| median avg_demand | 2.47 |
| mean zero_rate | 0.267 |
| mean outlier_rate | 0.033 |
| mature share | 0.00 |

### Interpretation

This is not a true behavior segment in the same sense as the other categories.

It is a **data maturity bucket**.

The mean average demand appears very high, but that is misleading. It is inflated by stations with very short observation windows, including one-day records with large ride counts.

The better interpretation is:

- short-history stations are too new or too incomplete to classify confidently,
- their raw averages can look extreme,
- and they should not be treated as evidence of a genuinely high-demand mature station class.

### Strategic reading

This is one of the most important insights in the whole station analysis.

If short-history is interpreted as a normal behavioral category, the network story becomes distorted. Short-history is fundamentally a **confidence and maturity issue**, not a stable operating identity.

---

## 6. Cluster-Level Insights

The mature-station clustering adds another layer of structure.

## 6.1 cluster_1: strong core

| Measure | Value |
|---|---:|
| station count | 32 |
| mean avg_demand | 12.40 |
| mean zero_rate | 0.096 |
| mean lag7 autocorr | 0.656 |
| mean correlation with system | 0.534 |

### Interpretation

This is the strongest mature cluster.

It contains many of the better-performing, more system-aligned stations with stronger recurring pattern structure.

### Strategic reading

This cluster is evidence that the network contains a dense, learnable signal core.

---

## 6.2 cluster_2: mainstream operating base

| Measure | Value |
|---|---:|
| station count | 139 |
| mean avg_demand | 3.70 |
| mean zero_rate | 0.153 |
| mean lag7 autocorr | 0.362 |
| mean correlation with system | 0.414 |

### Interpretation

This is the main operating middle of the mature station universe.

It contains the largest number of mature stations and looks like the broad, usable center of the network.

### Strategic reading

This cluster is likely where a practical global model earns most of its business value, because it is large enough to matter and structured enough to learn from.

---

## 6.3 cluster_3: weekend-sensitive and more volatile

| Measure | Value |
|---|---:|
| station count | 43 |
| mean avg_demand | 2.52 |
| mean zero_rate | 0.331 |
| mean weekend_ratio | 1.536 |
| mean cv | 1.300 |

### Interpretation

This cluster is the clearest weekend-sensitive mature group.

It is more variable, more weekend-oriented, and less regular than the strong core.

### Strategic reading

This cluster shows that heterogeneity is not random. It has recognizable internal structure.

---

## 6.4 cluster_4: weak-signal mature tail

| Measure | Value |
|---|---:|
| station count | 90 |
| mean avg_demand | 0.90 |
| mean zero_rate | 0.569 |
| mean lag7 autocorr | 0.143 |
| mean correlation with system | 0.189 |

### Interpretation

This is the weakest mature cluster.

It contains low-demand mature stations, with high inactivity and weaker recurring structure.

### Strategic reading

This is the mature analogue of the sparse tail. It reinforces the need for explicit sparse treatment later, without making sparse-specific modeling the first design choice.

---

## 7. Important Edge Cases and Watchlists

The diagnosis also surfaced several watchlists.

## Raw busiest stations by average demand

The top raw list includes stations such as:
- 4407,
- 4670,
- 4633,
- 4646,
- 4402.

But these are short-history one-day stations in the raw ranking.

### Interpretation

This is a perfect example of why station maturity matters.

A station can appear “busiest” in a raw average table simply because it has one observed day with a large value. That does not make it a stable high-demand station.

### Strategic reading

Raw station rankings should never be read without maturity context.

---

## Sparsest stations

The sparsest stations include:
- 4629,
- 4625,
- 4403,
- 4634,
- 4395.

These are genuine weak-signal stations and should be treated as part of the sparse tail, not as minor variations of healthy active stations.

---

## Most volatile stations

The most volatile stations include:
- 4331,
- 4135,
- 4457,
- 4134,
- 4602.

High volatility in low-demand series should be interpreted carefully. Sometimes it reflects true instability; sometimes it reflects the mathematics of low denominators and intermittent activity.

---

## Most anomaly-heavy stations

The most anomaly-heavy stations include:
- 4539,
- 3030,
- 3014,
- 4669,
- 4614.

These stations should be tracked because they show where event sensitivity or irregular behavior may matter later.

---

## 8. What This Means for Forecasting Design

This memo is not the full forecasting implementation plan, but it should explain why the modeling path points in one direction rather than another.

## 8.1 Why not start with 381 separate station models?

Because the station universe contains:

- many stations with limited history,
- many stations with low signal,
- many stations with shared calendar structure,
- and several small behavior groups that are too small to justify independent first-stage pipelines.

A station-by-station modeling strategy would:
- fragment the data,
- reduce the ability to borrow signal across stations,
- and create avoidable maintenance complexity.

---

## 8.2 Why not start with category-specific or cluster-specific models?

Because heterogeneity does not automatically mean the first solution should be segmentation-first.

The diagnosis supports that:
- clusters are meaningful,
- categories are useful,
- but the default first step should still be a **single global workflow** with **slice-based evaluation**.

That is the right compromise between learning power and operational simplicity.

---

## 8.3 Why a global station-day model is the right default

A global station-day workflow is the strongest default because the series still share:

- calendar structure,
- weekly rhythm,
- broad system dynamics,
- and common demand drivers.

A global model can borrow strength across stations while still respecting differences through station identity, lags, history, and other features.

The station-level evidence supports the following logic:

- train globally,
- evaluate locally and by slice,
- refine only after error analysis shows where refinement is truly needed.

This is better than assuming fragmentation before proving it is beneficial.

---

## 8.4 Why slices matter even if the model is global

A single model should not be judged by one aggregate metric alone.

The diagnosis shows at least four distinct evaluation populations:

1. **mature core stations**  
2. **short-history stations**  
3. **sparse/intermittent stations**  
4. **behavioral and cluster slices**

That means the right question is not:

> “Does one number summarize the whole system?”

It is:

> “Does the model perform acceptably on the core, and how does it degrade across more difficult station groups?”

That is the correct professional framing for this station portfolio.

---

## 9. Recommended Forecasting Decision Logic

The station analysis justifies the following decision logic.

### A. Forecasting unit
Use **station-day** as the primary forecasting unit.

### B. Core benchmark population
Use the **mature active non-sparse core** as the main signal slice for early benchmark judgment.

### C. Keep all stations in reporting
Do not throw short-history or sparse stations away entirely. Keep them in the evaluation universe, but score them separately.

### D. Treat short-history as maturity, not behavior
Do not interpret `short_history` as a normal behavior class.

### E. Treat sparse stations as a special policy issue
Keep them visible, but do not let them dominate first-stage model selection.

### F. Use clusters as a later lens
Clusters are useful for later refinement, not the first split.

This logic is the main reason the station-level path points toward:

- seasonal naive and naive baselines,
- one pooled station-day tree model,
- then a global probabilistic model such as DeepAR,
- with cluster refinement considered only later if the residuals demand it.

---

## 10. Final Conclusion

The station-level diagnosis does not show chaos. It shows **structured heterogeneity**.

That is the most important conclusion.

The metro bike station universe contains:

- a reliable productive core,
- a broad mixed middle,
- a clear weekend-oriented segment,
- a sparse tail,
- a small anomaly-heavy group,
- and a large enough short-history population that maturity must be treated explicitly.

Because of that structure, the most credible forecasting direction is not many disconnected station models and not immediate cluster-specific pipelines. It is a **single global station-day forecasting workflow with explicit evaluation slices and disciplined treatment of maturity and sparsity**.

That path is:

- statistically stronger,
- operationally simpler,
- easier to explain,
- easier to govern,
- and directly supported by the station-level evidence.
