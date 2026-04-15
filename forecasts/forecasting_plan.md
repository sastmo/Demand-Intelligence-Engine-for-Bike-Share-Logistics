# System-Level Forecasting Next Steps

## Purpose

This document translates the system-level analysis into a practical next-step plan. It is written for a customer-facing audience and focuses on what should happen next, why it matters, and how progress should be judged.

---

## Current Position

The system-level forecasting workflow is now functional end to end, with clear separation between diagnosis and forecasting, rolling backtesting for the validated horizons, and a practical multi-model comparison framework.

This means the project is no longer in a setup phase. It is in a refinement phase.

The key question is no longer whether the system can produce forecasts. It can.

The key question is how to improve forecast quality and confidence in the horizons that matter most for planning and operations.

---

## What We Have Learned So Far

The current analysis supports five clear conclusions:

1. **The aggregate demand signal is forecastable.** The system-level series has enough persistence and recurring structure to support practical forecasting.
2. **Short-horizon forecasting is the strongest current use case.** ETS remains a healthy benchmark and short-horizon behavior is more stable than longer-horizon behavior.
3. **Medium-horizon forecasting is promising but still needs work.** Fourier-based dynamic regression is currently strong, but the 30-day window is not yet as mature as it should be.
4. **SARIMAX is improving, but should still be treated carefully.** It is a monitored candidate, not yet a default choice.
5. **Long-horizon outputs should remain conservative.** Ninety-day forecasts are useful for sensitivity review, but not yet ready to be positioned as validated production-grade outputs.

---

## Strategic Objective for the Next Phase

The next phase should aim to move the system from **technically working** to **operationally trusted**.

That means:

- improving medium-horizon quality
- preserving rigorous validation standards
- communicating forecast confidence appropriately by horizon
- resisting the temptation to overcomplicate the solution too early

This is a quality and trust phase, not a feature accumulation phase.

---

## Recommended Workstreams

## Workstream 1: Improve 30-Day Forecast Quality

### Why this matters

The 30-day horizon is the most important gap between current usability and stronger customer value. It is already workable, but not yet strong enough to be described as fully mature.

### What to do next

- continue rolling-origin backtests focused specifically on 30-day performance
- compare all serious candidates against the same baseline and evaluation windows
- concentrate on forecast quality, not dashboard or presentation enhancements
- isolate what improves accuracy versus what only increases complexity

### What success looks like

- lower forecast error at the 30-day horizon
- more stable performance across rolling windows
- clearer model-selection evidence rather than occasional isolated wins

---

## Workstream 2: Keep Model Selection Disciplined

### Why this matters

Model credibility is lost when a flexible model is promoted before it is stable. The current results show that some classical approaches are still highly competitive, which is a strength, not a weakness.

### What to do next

- keep ETS as a strong short-horizon reference point
- continue using Fourier-based dynamic regression as a leading medium-horizon challenger
- keep SARIMAX in active evaluation, but not as an automatic production default
- treat nonlinear challenger models as evidence-driven options, not assumptions

### What success looks like

- each deployed model has a clear horizon-specific role
- model selection is justified by repeatable backtest evidence
- the production recommendation is based on reliability, not complexity

---

## Workstream 3: Protect Evaluation Integrity

### Why this matters

A forecasting system is only as trustworthy as its evaluation design. The current validation philosophy is strong and should remain intact.

### What to do next

- keep walk-forward validation as the standard method
- continue reporting MAE, RMSE, MASE, and bias
- avoid random train/test splits
- keep anomalies in the evaluation windows
- report results separately for 7-day and 30-day horizons

### What success looks like

- evaluation results reflect real deployment conditions
- model gains are believable and reproducible
- stakeholder confidence improves because the methodology is consistent

---

## Workstream 4: Manage 90-Day Outputs Carefully

### Why this matters

Long-horizon numbers are often attractive to stakeholders, but they can create false confidence if they are presented too strongly before validation is mature.

### What to do next

- keep 90-day outputs available for directional planning and sensitivity review
- use conservative wording in customer-facing reporting
- clearly distinguish between validated operating horizons and exploratory long-horizon outputs

### What success looks like

- long-horizon forecasts are used appropriately
- customer expectations stay aligned with current evidence
- trust is protected while long-horizon capability continues to improve

---

## Workstream 5: Expand Uncertainty Reporting in the Right Order

### Why this matters

Uncertainty estimates are valuable, but interval reporting should not distract from the more urgent task of improving point forecast quality.

### What to do next

- keep the current residual-based interval approach as a practical foundation
- improve point forecasts first, especially at 30 days
- then evaluate interval coverage and width by horizon

### What success looks like

- uncertainty bands are informative rather than cosmetic
- intervals reflect real forecast behavior
- coverage and sharpness can be explained with confidence

---

## Practical Recommendation for Customer Messaging

The strongest customer-facing message is not that the system is complete. It is that the system is **credible, useful, and improving in a disciplined way**.

A good external message would be:

- the pipeline is operational end to end
- the short-horizon forecasting capability is already on solid ground
- the medium-horizon capability is promising and is the main focus of improvement
- longer-horizon outputs are currently best used for directional planning rather than firm commitments

This message is strong because it is realistic. It builds confidence without overclaiming.

---

## Recommended Immediate Priorities

### Priority 1
Improve 30-day model quality and stability.

### Priority 2
Keep model selection evidence-based, especially for SARIMAX versus ETS and Fourier-based regression.

### Priority 3
Maintain strict horizon-based evaluation and preserve anomaly-inclusive testing.

### Priority 4
Keep 90-day communication conservative until validation strengthens.

### Priority 5
Expand interval evaluation only after point forecast performance improves further.

---

## Final Takeaway

The next step is clear.

Do not widen the scope unnecessarily. Do not over-promote the most flexible model. Do not dilute the evaluation design.

Instead, focus on improving the medium-horizon forecasting quality, preserving transparent validation, and communicating capability by horizon with discipline.

That approach will turn the current system from a technically functioning forecasting pipeline into a more trusted and customer-ready forecasting solution.
