# Model Bias

Post-training model bias analysis. Import from `fair_perf_ml.model_bias`.

Supported metrics: Difference in Positive Predicted Labels (DPPL), Disparate Impact (DI),
Accuracy Difference, Recall Difference, Difference in Conditional Acceptance (DCA),
Difference in Acceptance Rate (DAR), Speciality Difference, Difference in Conditional
Rejection (DCR), Difference in Rejection Rate (DRR), Treatment Equity (TE),
Conditional Demographic Disparity in Predicted Labels (CDDPL), Generalized Entropy (GE).

---

## Batch analysis

::: fair_perf_ml.model_bias.core.model_bias_perform_analysis

::: fair_perf_ml.model_bias.core.model_bias_perform_analysis_explicit_segmentation

---

## Runtime comparison

::: fair_perf_ml.model_bias.core.model_bias_runtime_comparison

::: fair_perf_ml.model_bias.core.model_bias_partial_runtime_comparison

---

## Streaming

::: fair_perf_ml.model_bias.streaming.ModelBiasStreaming
