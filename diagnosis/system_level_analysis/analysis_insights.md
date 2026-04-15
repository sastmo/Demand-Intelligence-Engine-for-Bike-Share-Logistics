# System-Level Forecasting Insights Report

## Purpose

This report summarizes what we learned from the system-level analysis of aggregated daily demand, what is already working well in the forecasting pipeline, where the current limitations are, and what we recommend as the next step.

The goal of this document is to provide a practical, customer-facing view of the analysis. It is not a raw technical log. It is a decision-oriented summary of what the data is telling us and how that should guide the next phase of work.

---

## Executive Summary

The system-level forecasting pipeline is now operating end to end and has reached a stage where it can support meaningful analysis and model comparison. At the aggregate level, the demand signal shows repeatable structure, persistent behavior, seasonality, and sensitivity to anomalies. This means the problem is forecastable, but only if evaluation is done carefully and model claims remain tied to horizon-specific evidence.

Several important lessons are already clear.

First, strong baseline and classical models remain highly competitive for this system-level problem. In particular, ETS continues to perform well for short-horizon forecasting, while Fourier-based dynamic regression is currently one of the strongest options for medium-horizon forecasting.

Second, more complex models should not be promoted simply because they are more flexible. SARIMAX has improved and remains promising, but it is not yet consistently strong enough to be treated as the default production choice without continued monitoring.

Third, forecast quality changes materially by horizon. Short-horizon results are in a healthier state than longer-horizon results. Seven-day forecasting is the most stable decision window. Thirty-day forecasting is usable, but still needs improvement before it should be positioned as a fully mature planning tool. Ninety-day output can still provide directional value, but it should be framed as a sensitivity view rather than as a production-grade forecast.

Overall, the analysis supports a measured conclusion: the system-level pipeline is viable, the modeling direction is sound, and the next phase should focus on strengthening forecast quality and operational trust, not on adding unnecessary model complexity or presentation features too early.

---

## What We Analyzed

The current work focuses only on **system-level forecasting**, meaning the aggregated daily demand across the full network.

This is important because system-level forecasting behaves differently from station-level forecasting:

- the signal is smoother at the aggregate level
- recurring demand structure is easier to detect
- practical baselines can be very strong
- interpretability matters more than model novelty at this stage

The pipeline and repository structure now separate **diagnosis** from **forecasting**, which is a meaningful improvement. This makes it easier to understand whether a result comes from data understanding, feature and model design, or final forecasting outputs.

---

## What the Data Tells Us

### 1. The system-level signal is forecastable

The aggregated daily total is not random. It shows persistence, seasonality, and recurring temporal structure. That is why baseline and classical time-series methods remain strong. The data has enough pattern to support forecasting, but not in a way that justifies skipping careful validation.

### 2. Horizon matters more than model branding

The analysis shows that a model that performs well at one horizon may not be the best model at another. This is one of the most important findings for stakeholders.

In practical terms:

- the best short-horizon model is not automatically the best monthly-horizon model
- performance claims must always be tied to the target forecast window
- deployment decisions should be horizon-specific, not model-name-specific

### 3. Anomalies are part of the real business problem

The evaluation approach correctly keeps anomalies in the test windows. This is the right decision. Removing difficult periods may make a model look better in presentation, but it would reduce real-world credibility. The system should be judged on how it behaves under realistic conditions, including abnormal periods, demand spikes, and operational irregularities.

### 4. Long-horizon outputs currently need cautious interpretation

The current evidence does not support positioning 90-day forecasts as fully validated production outputs. They can still be useful for directional planning, scenario review, and sensitivity discussion, but the wording around them should remain conservative.

---

## What Is Working Well

### End-to-end pipeline availability

The system-level pipeline now runs from input preparation through backtesting, model comparison, and forecast generation. This is a major foundation milestone because it means future work can focus on improving quality rather than building basic plumbing.

### Clear separation of responsibilities

The repository now distinguishes diagnosis from forecasting more clearly. This reduces confusion during iteration and supports more reliable future development.

### Strong baseline discipline

The evaluation philosophy starts with a strong seasonal baseline rather than assuming a complex model must be better. This is a healthy practice and improves trust in the final recommendations.

### Rolling backtesting for validated horizons

The use of rolling-origin or walk-forward evaluation is appropriate for a forecasting problem of this kind. It better reflects how the model will behave in practice than random train/test splitting.

### Probabilistic interval calibration from out-of-sample residuals

The pipeline already supports interval construction using out-of-sample residual calibration. This is a practical and grounded starting point for uncertainty estimation.

### Standardized system-level output structure

Outputs are now organized under a dedicated `forecasts/system_level/` area, making the workflow easier to maintain, audit, and explain.

---

## Model-Level Insights

### Seasonal naive remains an important benchmark

This is not just a trivial baseline. At the system level, weekly repeat structure is visible enough that a seasonal naive forecast is a meaningful reference point. Any proposed model has to beat this benchmark fairly.

### ETS remains a healthy short-horizon benchmark

ETS continues to be one of the most reliable short-horizon choices in the current setup. This is a strong result because ETS is interpretable, practical, and well matched to aggregate signals that show stable level, trend, and seasonality.

### Fourier dynamic regression is a strong medium-horizon candidate

In the current setup, Fourier-based dynamic regression is emerging as one of the strongest medium-horizon options. This suggests that smoother seasonal structure is being captured well in a regression-style framework.

### SARIMAX has improved, but it is still not a model to over-promote

SARIMAX remains useful and has shown improvement, especially as a candidate that can incorporate persistence and structured external signals. However, the current evidence suggests it still needs stabilization. It should stay in the monitored candidate set rather than being presented as the unquestioned production leader.

### More ambitious model families should remain challengers, not assumptions

The model selection strategy is appropriately practical. Structural state-space methods and nonlinear lag-feature models are sensible challengers. Deep models are not the first priority for a single aggregated daily series, and that is the right choice at this stage.

---

## What Needs Improvement

### 30-day quality is usable, but not yet strong enough

The medium-horizon forecast quality is workable, but it is not yet where it should be if the objective is a confident customer-facing planning product. This is the most important near-term model-quality issue.

### SARIMAX still needs stabilization

Although SARIMAX has improved, its medium-horizon behavior is not yet consistently better than the leading benchmark models. Promoting it too aggressively would create unnecessary operational risk.

### 90-day forecasting is not yet production-grade

Long-horizon outputs should continue to be treated as directional rather than fully validated. Stakeholders should not interpret those numbers with the same level of confidence as the shorter validated windows.

### Communication discipline matters

A technically sound pipeline can still lose trust if horizon limitations are not communicated clearly. The current state supports confident messaging for shorter horizons, measured messaging for 30 days, and conservative messaging for 90 days.

---

## What This Means for Stakeholders

From a customer-facing perspective, the findings support several practical conclusions.

### The system is already useful

The work has moved beyond experimentation. There is now a functioning system-level forecasting process with meaningful validation and comparable model outputs.

### The right message is maturity with discipline

The best external message is not that the system solves every forecasting horizon equally well. The better message is that the system has a validated short-horizon forecasting capability, a promising medium-horizon capability that is still being improved, and a long-horizon view that should currently be used for sensitivity and planning discussion rather than committed operational decisions.

### Confidence should be tied to use case

A customer-facing forecast should reflect the decision being supported:

- near-term operational planning can be supported with higher confidence
- medium-horizon planning can be supported with more caution and continued refinement
- long-horizon planning should remain scenario-based until validation improves

---

## Recommended Next Steps

The next phase should focus on **quality hardening**, not feature inflation.

### 1. Strengthen the 30-day forecasting window

This is the most important next step. The current state is usable, but the strongest opportunity for customer value is to improve medium-horizon stability and accuracy.

Priority actions:

- continue model comparison using rolling-origin backtests
- focus on quality gains rather than adding interface features
- test whether better feature design and calibration improve medium-horizon behavior
- keep results separated by forecast horizon so improvements are measurable

### 2. Keep SARIMAX in monitored evaluation, not default deployment

SARIMAX should remain in the candidate stack, but it should not be over-positioned until it shows stable superiority or a clear complementary role.

Priority actions:

- monitor consistency across rolling windows
- compare directly against ETS and Fourier-based regression
- avoid presenting isolated wins as a full model-selection conclusion

### 3. Maintain conservative language for 90-day outputs

The 90-day forecast should remain a sensitivity and directional review tool until stronger evidence supports production-grade use.

Priority actions:

- label 90-day outputs clearly
- prevent over-interpretation in reporting
- prioritize validation quality before expanding long-horizon claims

### 4. Continue baseline-first validation discipline

The evaluation design is one of the strongest parts of the current work and should not be relaxed.

Priority actions:

- keep walk-forward validation as the standard
- continue reporting MAE, RMSE, MASE, and bias
- avoid random train/test evaluation
- keep anomalies in the test windows

### 5. Expand uncertainty evaluation only after point forecasts are stronger

Intervals are important, but the first priority should remain point forecast quality, especially at the 30-day horizon.

Priority actions:

- improve point forecasts first
- then evaluate empirical coverage and interval width
- compare interval behavior separately by horizon

---

## Final Takeaway

The system-level analysis shows a forecasting pipeline that is operational, credible, and directionally strong, but still in a quality-hardening phase rather than a final optimization phase.

The current evidence supports confidence in the pipeline foundation and in the short-horizon forecasting capability. It also provides a clear roadmap for what should happen next: improve medium-horizon quality, keep model claims disciplined, preserve rigorous evaluation standards, and communicate long-horizon outputs conservatively.

That is a strong customer-facing story because it is both positive and honest. It shows real progress, practical value, and a clear path to higher confidence in the next iteration.
