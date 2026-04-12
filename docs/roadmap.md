
# Fair-Perf-ML — Real-Time Observability Roadmap

## Overview

**Fair-Perf-ML** is a lightweight, composable observability toolkit for machine-learning systems.  
It provides a unified API for computing **performance**, **fairness**, and soon **drift** metrics in both
batch and streaming settings.

The next major phase of the project focuses on **real-time drift detection** and **online monitoring** —
building the bridge between offline evaluation and live production observability.

---

## ✨ Vision

Enable engineers to detect model degradation, bias, or data drift **before** ground truth arrives.  
Fair-Perf-ML will support real-time metric computation over streaming or micro-batch data, with
plug-and-play integration for existing ML pipelines (SageMaker, Airflow, Kafka, Kinesis, etc.).

---

## 🧩 Core Goals

1. **Unified metric API** — same interface for performance, fairness, and drift metrics.
2. **Real-time drift monitoring** — lightweight statistical proxies for early detection.
3. **Streaming-friendly design** — incremental metrics, windowed evaluation, low memory.
4. **Fairness-aware drift** — group-level monitoring for ethical robustness.
5. **Interoperability** — easy export to OLAP, Prometheus, CloudWatch, or S3 JSON logs.

---

## Roadmap

### **Phase 1 – Drift Metrics **

**Goal:** introduce a first-class `fair_perf_ml.drift` module with standard metrics.

#### Features
- [X] `PopulationStabilityIndex` (PSI) — categorical + numeric.
- [X] `KSTestDrift` — two-sample Kolmogorov–Smirnov test.
- [X] `JensenShannonDrift` — symmetric divergence [0, 1].
- [ ] `WassersteinDrift` — Earth-Mover’s distance for continuous features.
- [X] Unified base class `BaseDriftMetric` with `compute(baseline, current)`.

#### Deliverables
- [X] Comprehensive unit tests for numeric / categorical inputs.
- [X] Serialization for baseline histograms (`.json`).

---

### **Phase 2 – Real-Time Monitoring Engine **

**Goal:** make drift and fairness metrics stream-aware and incremental.

#### Features
- [X] `DriftMonitor` class for rolling window updates.
- [ ] Online statistics via Welford’s algorithm / T-Digest summaries.
- [ ] Config-driven thresholds and per-feature policies.
- [ ] Aggregation & alert logic (e.g., drift detected in ≥ K % of features).
- [ ] Async / background execution hooks.

#### Deliverables
- [ ] Example micro-batch monitor (5-min window) for Kafka/Kinesis.
- [ ] CLI & YAML config interface.
- [ ] Integration guide: “Using Fair-Perf-ML in a streaming service”.

---

### **Phase 3 – Fairness Drift & Group Analytics **

**Goal:** extend drift detection to subgroup and fairness metrics.

#### Features
- [ ] Group-level PSI / JSD (e.g., `PSI(ŷ | gender)`).
- [ ] Fairness delta tracking (Δ FPR, Δ TPR, Δ PPV over time).
- [ ] Group-specific drift alerts.
- [ ] Fairness-drift visualization templates.

#### Deliverables
- [ ] Example notebook: “Monitoring fairness drift in real time”.
- [ ] API design doc for `fair_perf_ml.fairness.monitor`.

---

### **Phase 4 – Observability Integrations (Q3 2026)**

**Goal:** make Fair-Perf-ML metrics observable and interoperable with production systems.

#### Features
- [ ] Exporters for:
  - Prometheus Pushgateway
  - AWS CloudWatch
  - OpenTelemetry Metrics API
  - S3 / DynamoDB JSON logs
- [ ] `MetricPublisher` abstraction layer.
- [ ] Prebuilt Grafana dashboards for PSI, drift scores, fairness deltas.

#### Deliverables
- [ ] Example Docker sidecar for model endpoints.
- [ ] `fair_perf_ml.client` module with async publishers.
- [ ] Integration tests with SageMaker & local streaming setup.

---

### **Phase 5 – Continuous Evaluation & Ground-Truth Integration **

**Goal:** combine real-time proxy metrics with delayed ground-truth metrics.

#### Features
- [ ] Join inference logs with delayed labels for backfilled evaluation.
- [ ] Compare leading (proxy) vs lagging (ground-truth) metrics.
- [ ] Automatic model performance dashboards (F1, AUC, Recall drift).
- [ ] Correlation tracking between proxy drift and true performance drop.

---

## 🧠 Architecture Summary

