use super::PreTraining;
use crate::data_handler::BiasSegmentationCriteria;
use crate::errors::{ModelPerfResult, ModelPerformanceError};
use crate::metrics::DataBiasMetric;
use crate::reporting::{DataBiasAnalysisReport, DataBiasDriftSnapshot, DriftReport};
use crate::runtime::DataBiasRuntime;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::StreamingDataBias;
    use crate::data_handler::{
        py_types_handler::{report_to_py_dict, PyDictResult},
        BiasSegmentationCriteria, BiasSegmentationType,
    };
    use crate::metrics::DataBiasMetricVec;
    use pyo3::prelude::*;
    use pyo3::types::IntoPyDict;

    // for the bias streaming, the python api here will accept already labeled data to limit the
    // type complexity here. So the labeling will happen in the python api

    #[pyclass]
    pub(crate) struct PyDataBiasStreaming {
        inner: StreamingDataBias<i8, i8>,
    }

    // segmentation logic lives in Python wrapper type, thus no reset baseline method here

    #[pymethods]
    impl PyDataBiasStreaming {
        #[new]
        fn new(feature_data: Vec<i8>, gt_data: Vec<i8>) -> PyResult<PyDataBiasStreaming> {
            let inner = StreamingDataBias::new(
                &feature_data,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
                &gt_data,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
            )?;
            Ok(PyDataBiasStreaming { inner })
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn push(&mut self, f: i8, g: i8) {
            self.inner.push(&f, &g)
        }

        fn push_batch(&mut self, f: Vec<i8>, g: Vec<i8>) -> PyResult<()> {
            self.inner.push_batch(&f, &g)?;
            Ok(())
        }

        fn reset_baseline(&mut self, f: Vec<i8>, g: Vec<i8>) -> PyResult<()> {
            self.inner.reset_baseline(&f, &g)?;
            Ok(())
        }

        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.drift_snapshot()?;
            Ok(report_to_py_dict(py, report))
        }

        fn drift_report<'py>(&self, py: Python<'py>, drift_threshold: f32) -> PyDictResult<'py> {
            let report = self.inner.drift_report(Some(drift_threshold))?;
            report.into_py_dict(py)
        }

        fn drift_report_partial_metrics<'py>(
            &self,
            py: Python<'py>,
            metrics: Vec<String>,
            drift_threshold: f32,
        ) -> PyDictResult<'py> {
            let m_vec = DataBiasMetricVec::try_from(metrics.as_ref())?;
            let report = self
                .inner
                .drift_report_partial_metrics(m_vec.as_ref(), Some(drift_threshold))?;
            report.into_py_dict(py)
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.performance_snapshot()?;
            Ok(report_to_py_dict(py, report))
        }
    }
}

/// A streaming data bias manager. At construction, baseline data is required, then through the
/// lifetime of the type instance, runtime data can be accumulated and a snapshot drift report can
/// be generated to give a point in time snapshot into the current data bias drift. This type
/// stores only the data needed to compute the supported bias metrics, and does so compactly.
/// Reseting the baseline is also supported, which may be appropriate at some time cadence, or
/// after a model refresh. Feature class and ground truth class labeling will be done with the
/// provided 'data_handler::BiasSegmentationCriteria<T>'.
#[derive(Debug)]
pub struct StreamingDataBias<G, F>
where
    G: PartialOrd,
    F: PartialOrd,
{
    feature_seg_criteria: BiasSegmentationCriteria<F>,
    gt_seg_criteria: BiasSegmentationCriteria<G>,
    baseline_report: DataBiasRuntime,
    rt: PreTraining, // stores the runtime/accumulated distributions for both facets
}

impl<G, F> StreamingDataBias<G, F>
where
    F: PartialOrd,
    G: PartialOrd,
{
    /// Construct a new streaming instance with baseline features and ground truth dataset. Feature and
    /// ground truth label segmentation criteria is required to segment features and labels into
    /// advantaged and disadvantaged classes.
    pub fn new(
        feature_data: &[F],
        feature_seg_criteria: BiasSegmentationCriteria<F>,
        gt_data: &[G],
        gt_seg_criteria: BiasSegmentationCriteria<G>,
    ) -> ModelPerfResult<StreamingDataBias<G, F>> {
        let bl = PreTraining::new_from_segmentation(
            feature_data,
            &feature_seg_criteria,
            gt_data,
            &gt_seg_criteria,
        )?;
        let rt = PreTraining::default();

        let baseline_report = DataBiasRuntime::new_from_pre_training(&bl)?;

        Ok(StreamingDataBias {
            feature_seg_criteria,
            gt_seg_criteria,
            baseline_report,
            rt,
        })
    }

    /// Push a single runtime example ad hoc into the stream.
    pub fn push(&mut self, f: &F, g: &G) {
        self.rt
            .accumulate_runtime_single(f, &self.feature_seg_criteria, g, &self.gt_seg_criteria)
    }

    /// Method to batch push new runtime examples to the stream and update stream state. The segmentation logic
    /// provided at type construction will do all feature class and outcome class labeling.
    pub fn push_batch(&mut self, feature: &[F], gt: &[G]) -> ModelPerfResult<()> {
        if let Err(e) = self.rt.accumulate_runtime_batch(
            feature,
            &self.feature_seg_criteria,
            gt,
            &self.gt_seg_criteria,
        ) {
            return Err(ModelPerformanceError::BiasError(e));
        }

        Ok(())
    }

    /// Flush the runtime state accumulated in the stream. Resets the stream to be empty.
    pub fn flush(&mut self) {
        self.rt.clear()
    }

    /// Reset the baseline data. This may be used to refresh the data on a retraining, or to move
    /// forward in time from a given baseline set if data drifts. This method will clear all
    /// runtime data. This method will use the same segmentation criteria established at
    /// construction, to update the reset the baseline state and update the segmentation criteria,
    /// use the `reset_baseline_and_segmentation`. This will also reset the runtime stream state to
    /// be empty.
    pub fn reset_baseline(&mut self, feature_data: &[F], gt_data: &[G]) -> ModelPerfResult<()> {
        let new_bl = PreTraining::new_from_segmentation(
            feature_data,
            &self.feature_seg_criteria,
            gt_data,
            &self.gt_seg_criteria,
        )?;

        self.baseline_report = DataBiasRuntime::new_from_pre_training(&new_bl)?;
        self.flush();
        Ok(())
    }

    /// Reset the baseline state and update the class segmentation criteria. This is the only
    /// method that allows for updating the segmentation criteria, which requires a new baseline
    /// state and emptying the runtime stream for consistency across runtime analysis.
    pub fn reset_baseline_and_segmentation(
        &mut self,
        feature_data: &[F],
        feat_seg: BiasSegmentationCriteria<F>,
        gt_data: &[G],
        gt_seg: BiasSegmentationCriteria<G>,
    ) -> ModelPerfResult<()> {
        self.feature_seg_criteria = feat_seg;
        self.gt_seg_criteria = gt_seg;
        self.reset_baseline(feature_data, gt_data)?;
        Ok(())
    }

    /// Generate a point in time drift snapshot. This method will report out the current
    /// drift across all metrics in 'metrics::DataBiasMetric'. Returns an error when there the
    /// runtime bins are empty.
    pub fn drift_snapshot(&self) -> ModelPerfResult<DataBiasDriftSnapshot> {
        if self.rt.size() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let curr_rt = DataBiasRuntime::new_from_pre_training(&self.rt)?;
        let rt_report = curr_rt.runtime_drift_report(&self.baseline_report);
        Ok(rt_report)
    }

    /// Generates a point in time drift report on the full metric suite. Any metric drift
    /// observation that exceeds the provided drift threhold (default 0.10_f32) will result
    /// in a failed drift report.
    pub fn drift_report(
        &self,
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<DataBiasMetric>> {
        if self.rt.size() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let curr_rt = DataBiasRuntime::new_from_pre_training(&self.rt)?;
        let rt_report = curr_rt.runtime_drift_report(&self.baseline_report);
        let drift_threshold =
            drift_threshold_opt.unwrap_or(crate::reporting::DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            rt_report
                .into_iter()
                .filter(|(_, v)| *v >= drift_threshold)
                .collect(),
        ))
    }

    /// Same as ['StreamingDataBias::drift_report'] but only on the provided subset of metrics.
    pub fn drift_report_partial_metrics(
        &self,
        metrics: &[DataBiasMetric],
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<DataBiasMetric>> {
        if self.rt.size() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let curr_rt = DataBiasRuntime::new_from_pre_training(&self.rt)?;
        let rt_report = curr_rt.runtime_drift_report(&self.baseline_report);
        let drift_threshold =
            drift_threshold_opt.unwrap_or(crate::reporting::DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            rt_report
                .into_iter()
                .filter(|(m, v)| *v >= drift_threshold && metrics.contains(m))
                .collect(),
        ))
    }

    /// Generate a point in time performance snapshot irrespective of the baseline data. Returns an
    /// error when the runtime bins are empty.
    pub fn performance_snapshot(&self) -> ModelPerfResult<DataBiasAnalysisReport> {
        if self.rt.size() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let curr_rt = DataBiasRuntime::new_from_pre_training(&self.rt)?;
        Ok(curr_rt.generate_report())
    }
}

#[cfg(test)]
mod data_bias_streaming_tests {
    use super::*;
    use crate::data_bias::statistics as stats;
    use crate::data_handler::{BiasSegmentationCriteria, BiasSegmentationType};
    use crate::metrics::DataBiasMetric as M;
    use std::collections::HashMap;

    fn test_data() -> (Vec<i32>, Vec<i32>) {
        (
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
            vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        )
    }

    fn seg() -> BiasSegmentationCriteria<i32> {
        BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label)
    }

    fn baseline_runtime(feat: &[i32], gt: &[i32]) -> DataBiasRuntime {
        DataBiasRuntime {
            ci: stats::class_imbalance(feat, seg(), gt, seg()).unwrap(),
            dpl: stats::diff_in_proportion_of_labels(feat, seg(), gt, seg()).unwrap(),
            kl: stats::kl_divergence(feat, seg(), gt, seg()).unwrap(),
            js: stats::jensen_shannon(feat, seg(), gt, seg()).unwrap(),
            lpnorm: stats::lp_norm(feat, seg(), gt, seg()).unwrap(),
            tvd: stats::total_variation_distance(feat, seg(), gt, seg()).unwrap(),
            ks: stats::kolmogorov_smirnov(feat, seg(), gt, seg()).unwrap(),
        }
    }

    fn make_streaming(feat: &[i32], gt: &[i32]) -> StreamingDataBias<i32, i32> {
        StreamingDataBias::new(feat, seg(), gt, seg()).unwrap()
    }

    #[test]
    fn test_bl_construction() {
        let (feat, gt) = test_data();
        assert_eq!(
            baseline_runtime(&feat, &gt),
            make_streaming(&feat, &gt).baseline_report
        );
    }

    #[test]
    fn test_bl_reset() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.reset_baseline(&feat, &gt).unwrap();
        assert_eq!(baseline_runtime(&feat, &gt), streaming.baseline_report);
    }

    #[test]
    fn reset_baseline_and_seg() {
        let (feat, gt) = test_data();
        let feat2: Vec<i32> = feat.iter().map(|v| v + 1).collect();
        let gt2: Vec<i32> = gt.iter().map(|v| v + 1).collect();
        let mut streaming = make_streaming(&feat, &gt);
        streaming
            .reset_baseline_and_segmentation(
                &feat2,
                BiasSegmentationCriteria::new(2_i32, BiasSegmentationType::Label),
                &gt2,
                BiasSegmentationCriteria::new(2_i32, BiasSegmentationType::Label),
            )
            .unwrap();
        assert_eq!(baseline_runtime(&feat, &gt), streaming.baseline_report);
    }

    #[test]
    fn perf_report() {
        let (feat, gt) = test_data();
        let base = baseline_runtime(&feat, &gt);
        let expected: HashMap<M, f32> = [
            (M::ClassImbalance, base.ci),
            (M::DifferenceInProportionOfLabels, base.dpl),
            (M::KlDivergence, base.kl),
            (M::JsDivergence, base.js),
            (M::LpNorm, base.lpnorm),
            (M::TotalVariationDistance, base.tvd),
            (M::KolmogorovSmirnov, base.ks),
        ]
        .into_iter()
        .collect();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&feat, &gt).unwrap();
        assert_eq!(expected, streaming.performance_snapshot().unwrap());
    }

    #[test]
    fn drift_report() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&feat, &gt).unwrap();
        let expected: HashMap<M, f32> = [
            (M::ClassImbalance, 0_f32),
            (M::DifferenceInProportionOfLabels, 0_f32),
            (M::KlDivergence, 0_f32),
            (M::JsDivergence, 0_f32),
            (M::LpNorm, 0_f32),
            (M::TotalVariationDistance, 0_f32),
            (M::KolmogorovSmirnov, 0_f32),
        ]
        .into_iter()
        .collect();
        assert_eq!(expected, streaming.drift_snapshot().unwrap());
    }

    // Runtime data where both facets have 50% acceptance, vs baseline 70%/20%.
    // DPL drift = |0.0 - 0.5| = 0.5, KL drift ≈ 0.58 — both well above any reasonable threshold.
    // CI remains 0 for both (balanced classes), giving a metric with zero drift to test against.
    fn drifted_data() -> (Vec<i32>, Vec<i32>) {
        (
            vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            vec![1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        )
    }

    // --- push ---

    #[test]
    fn push_single_example_populates_stream() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push(&1_i32, &1_i32);
        // Should error here as 1 examples results in only 1 class having data.
        assert!(streaming.drift_snapshot().is_err());
    }

    // --- flush ---

    #[test]
    fn flush_clears_runtime_and_subsequent_operations_return_empty_error() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&feat, &gt).unwrap();
        streaming.flush();
        assert!(streaming.drift_snapshot().is_err());
        assert!(streaming.drift_report(None).is_err());
        assert!(streaming.performance_snapshot().is_err());
    }

    // --- empty stream errors ---

    #[test]
    fn drift_snapshot_on_empty_stream_returns_error() {
        let (feat, gt) = test_data();
        assert!(make_streaming(&feat, &gt).drift_snapshot().is_err());
    }

    #[test]
    fn drift_report_on_empty_stream_returns_error() {
        let (feat, gt) = test_data();
        assert!(make_streaming(&feat, &gt).drift_report(None).is_err());
    }

    #[test]
    fn performance_snapshot_on_empty_stream_returns_error() {
        let (feat, gt) = test_data();
        assert!(make_streaming(&feat, &gt).performance_snapshot().is_err());
    }

    // --- drift_snapshot with actual non-zero drift ---

    #[test]
    fn drift_snapshot_nonzero_when_runtime_differs_from_baseline() {
        let (feat, gt) = test_data();
        let (rt_feat, rt_gt) = drifted_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&rt_feat, &rt_gt).unwrap();
        let snapshot = streaming.drift_snapshot().unwrap();
        assert!(snapshot.values().any(|v| *v > 0.0));
    }

    // --- drift_report threshold ---

    #[test]
    fn drift_report_passes_when_runtime_matches_baseline() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&feat, &gt).unwrap();
        let report = streaming.drift_report(Some(0.10)).unwrap();
        assert!(report.passed);
        assert!(report.failed_report.is_none());
    }

    #[test]
    fn drift_report_fails_when_drift_exceeds_threshold() {
        let (feat, gt) = test_data();
        let (rt_feat, rt_gt) = drifted_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&rt_feat, &rt_gt).unwrap();
        let report = streaming.drift_report(Some(0.10)).unwrap();
        assert!(!report.passed);
        assert!(report.failed_report.is_some());
    }

    // --- drift_report_partial_metrics ---

    #[test]
    fn drift_report_partial_metrics_filters_to_requested_subset() {
        let (feat, gt) = test_data();
        let (rt_feat, rt_gt) = drifted_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&rt_feat, &rt_gt).unwrap();

        // CI has zero drift (both baseline and runtime have balanced classes), so
        // restricting to CI only should produce a passing report even though full
        // drift is large.
        let ci_only = streaming
            .drift_report_partial_metrics(&[M::ClassImbalance], Some(0.10))
            .unwrap();
        assert!(ci_only.passed);

        // DPL drifts by ~0.5, so restricting to DPL should produce a failing report.
        let dpl_only = streaming
            .drift_report_partial_metrics(&[M::DifferenceInProportionOfLabels], Some(0.10))
            .unwrap();
        assert!(!dpl_only.passed);
    }

    // --- push_batch error cases ---

    #[test]
    fn push_batch_mismatched_lengths_returns_error() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        assert!(streaming.push_batch(&[1_i32, 0, 1], &[0_i32, 1]).is_err());
    }

    #[test]
    fn push_batch_empty_returns_error() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        assert!(streaming.push_batch(&[], &[]).is_err());
    }

    // --- reset_baseline clears accumulated runtime ---

    #[test]
    fn reset_baseline_clears_accumulated_runtime() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        streaming.push_batch(&feat, &gt).unwrap();
        streaming.reset_baseline(&feat, &gt).unwrap();
        assert!(streaming.drift_snapshot().is_err());
    }

    // --- reset_baseline error case ---

    #[test]
    fn reset_baseline_with_empty_data_returns_error() {
        let (feat, gt) = test_data();
        let mut streaming = make_streaming(&feat, &gt);
        assert!(streaming.reset_baseline(&[], &[]).is_err());
    }
}
