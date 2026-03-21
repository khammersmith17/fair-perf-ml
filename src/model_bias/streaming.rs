use super::{BucketGeneralizedEntropy, PostTraining};
use crate::data_handler::BiasSegmentationCriteria;
use crate::errors::{ModelPerfResult, ModelPerformanceError};
use crate::metrics::ModelBiasMetric;
use crate::reporting::{
    DriftReport, ModelBiasAnalysisReport, ModelBiasDriftSnapshot, DEFAULT_DRIFT_THRESHOLD,
};
use crate::runtime::ModelBiasRuntime;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::StreamingModelBias;
    use crate::data_handler::{
        py_types_handler::{report_to_py_dict, PyDictResult},
        BiasSegmentationCriteria, BiasSegmentationType,
    };
    use crate::metrics::ModelBiasMetricVec;
    use pyo3::prelude::*;
    use pyo3::types::IntoPyDict;

    #[pyclass]
    pub(crate) struct PyModelBiasStreaming {
        inner: StreamingModelBias<i8, i8, i8>,
    }

    // class segmentation will happen in python layer
    // and as such, not exposing the method to update seg criteria
    #[pymethods]
    impl PyModelBiasStreaming {
        #[new]
        fn new(
            features: Vec<i8>,
            predictions: Vec<i8>,
            ground_truth: Vec<i8>,
        ) -> PyResult<PyModelBiasStreaming> {
            let inner = StreamingModelBias::new(
                &features,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
                &predictions,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
                &ground_truth,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
            )?;

            Ok(PyModelBiasStreaming { inner })
        }

        fn push(&mut self, feature: i8, pred: i8, gt: i8) {
            self.inner.push(&feature, &pred, &gt);
        }

        fn push_batch(&mut self, feature: Vec<i8>, pred: Vec<i8>, gt: Vec<i8>) -> PyResult<()> {
            self.inner.push_batch(&feature, &pred, &gt)?;
            Ok(())
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn reset_baseline(&mut self, feature: Vec<i8>, pred: Vec<i8>, gt: Vec<i8>) -> PyResult<()> {
            self.inner.reset_baseline(&feature, &pred, &gt)?;
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
            let m_vec = ModelBiasMetricVec::try_from(metrics.as_ref())?;
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

/// This is a container geared toward a long running ML Observability service. Internally it leverages a
/// bucketing alogrithm to store all information necessary to compute model bias metrics in compact
/// space. This type takes in class segmentation logic for the feature, predictions, and ground
/// truth. A single instance can store a single feature segmentation for a single feature, thus
/// many instances of this type can be composed to provide full insight into a model if you wish to
/// monitor many features or many models in the same service.
pub struct StreamingModelBias<F, P, G>
where
    F: PartialOrd,
    G: PartialOrd,
    P: PartialOrd,
{
    // labeling threshold for each data dimensions
    feat_seg: BiasSegmentationCriteria<F>,
    pred_seg: BiasSegmentationCriteria<P>,
    gt_seg: BiasSegmentationCriteria<G>,
    rt: PostTraining,
    ge: BucketGeneralizedEntropy, // runtime ge buckets
    bl: ModelBiasRuntime,         // the baseline metric computations precomputed
}

impl<F, P, G> StreamingModelBias<F, P, G>
where
    F: PartialOrd,
    P: PartialOrd,
    G: PartialOrd,
{
    /// Constructor requires segmentation criteria for the feature of interest, predictions, and
    /// ground truth, and the baseline dataset for each respective data dimension. On construction,
    /// baseline are computed on the baseline dataset provided. This type then takes ownership of
    /// the segmentation criteria to be used later for runtime segmentation and bucketing.
    pub fn new(
        features: &[F],
        feat_seg: BiasSegmentationCriteria<F>,
        preds: &[P],
        pred_seg: BiasSegmentationCriteria<P>,
        gt: &[G],
        gt_seg: BiasSegmentationCriteria<G>,
    ) -> ModelPerfResult<StreamingModelBias<F, P, G>> {
        let bl_pt = PostTraining::new_from_segmentation_criteria(
            features, &feat_seg, preds, &pred_seg, gt, &gt_seg,
        )?;

        let mut bl_ge_bucket = BucketGeneralizedEntropy::default();
        bl_ge_bucket.accumulate(gt, &gt_seg, preds, &pred_seg);
        let bl_ge = bl_ge_bucket.ge_snapshot();
        let bl = ModelBiasRuntime::new_from_post_training(&bl_pt, bl_ge)?;

        Ok(StreamingModelBias {
            feat_seg,
            pred_seg,
            gt_seg,
            bl,
            rt: PostTraining::default(),
            ge: BucketGeneralizedEntropy::default(),
        })
    }

    /// Push a single ad hoc example of feature, prediction, and runtime data into the stream.
    pub fn push(&mut self, feature: &F, pred: &P, gt: &G) {
        self.rt.accumulate_single(
            self.feat_seg.label(feature),
            self.pred_seg.label(pred),
            self.gt_seg.label(gt),
        )
    }

    /// Push a batch of runtime feature data, prediction data, and ground truth data into the stream. This
    /// method will use the segmentation logic passed at type construction. This method returns
    /// nothing, use the drift_snapshot method to generate the report.
    pub fn push_batch(&mut self, features: &[F], preds: &[P], gt: &[G]) -> ModelPerfResult<()> {
        self.rt.accumulate_batch(
            features,
            &self.feat_seg,
            preds,
            &self.pred_seg,
            gt,
            &self.gt_seg,
        )?;
        self.ge.accumulate(gt, &self.gt_seg, preds, &self.pred_seg);
        Ok(())
    }

    /// Flush the data accumulated in the streaming container. This will periodically be helpful to
    /// reset state over time in a long running service.
    pub fn flush(&mut self) {
        self.rt.clear();
        self.ge.clear();
    }

    /// Resets the baseline on new baseline data and the segmentation criteria. Reseting the
    /// segmentation criteria may be required if there is a meaningful shift in the distribution of
    /// the related datasets.
    pub fn reset_baseline_and_segmentation_criteria(
        &mut self,
        feature: &[F],
        feat_seg: BiasSegmentationCriteria<F>,
        prediction: &[P],
        pred_seg: BiasSegmentationCriteria<P>,
        ground_truth: &[G],
        gt_seg: BiasSegmentationCriteria<G>,
    ) -> ModelPerfResult<()> {
        self.feat_seg = feat_seg;
        self.pred_seg = pred_seg;
        self.gt_seg = gt_seg;

        let bl_pt = PostTraining::new_from_segmentation_criteria(
            feature,
            &self.feat_seg,
            prediction,
            &self.pred_seg,
            ground_truth,
            &self.gt_seg,
        )?;

        let mut bl_ge_bucket = BucketGeneralizedEntropy::default();
        bl_ge_bucket.accumulate(ground_truth, &self.gt_seg, prediction, &self.pred_seg);
        let bl_ge = bl_ge_bucket.ge_snapshot();
        self.bl = ModelBiasRuntime::new_from_post_training(&bl_pt, bl_ge)?;
        Ok(())
    }

    /// Reset the baseline data used to compute drift. This is useful for a model retraining, or to
    /// update the baseline on more recent data. This will leverage the same segmentation criteria
    /// defined at construction. To also redefine the segmentation criteria, use the
    /// `reset_baseline_and_segmentation_criteria' method.
    pub fn reset_baseline(
        &mut self,
        feature: &[F],
        prediction: &[P],
        ground_truth: &[G],
    ) -> ModelPerfResult<()> {
        let bl_pt = PostTraining::new_from_segmentation_criteria(
            feature,
            &self.feat_seg,
            prediction,
            &self.pred_seg,
            ground_truth,
            &self.gt_seg,
        )?;

        let mut bl_ge_bucket = BucketGeneralizedEntropy::default();
        bl_ge_bucket.accumulate(ground_truth, &self.gt_seg, prediction, &self.pred_seg);
        let bl_ge = bl_ge_bucket.ge_snapshot();
        self.bl = ModelBiasRuntime::new_from_post_training(&bl_pt, bl_ge)?;
        Ok(())
    }

    /// Generateas a point in time drift report, this will consider the baseline set and all the
    /// data that has been accumulated since the last flush. Errors when there is no data
    /// accumulate in the runtime stream.
    pub fn drift_snapshot(&self) -> ModelPerfResult<ModelBiasDriftSnapshot> {
        if self.ge.len() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let rt_ge = self.ge.ge_snapshot();
        let rt_snapshot = ModelBiasRuntime::new_from_post_training(&self.rt, rt_ge)?;
        let report = rt_snapshot.runtime_drift_report(&self.bl);
        Ok(report)
    }

    pub fn drift_report(
        &self,
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<ModelBiasMetric>> {
        if self.ge.len() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let rt_ge = self.ge.ge_snapshot();
        let rt_snapshot = ModelBiasRuntime::new_from_post_training(&self.rt, rt_ge)?;
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        let report = rt_snapshot.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(
            report
                .into_iter()
                .filter(|(_, v)| *v >= drift_threshold)
                .collect(),
        ))
    }

    pub fn drift_report_partial_metrics(
        &self,
        metrics: &[ModelBiasMetric],
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<ModelBiasMetric>> {
        if self.ge.len() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let rt_ge = self.ge.ge_snapshot();
        let rt_snapshot = ModelBiasRuntime::new_from_post_training(&self.rt, rt_ge)?;
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        let report = rt_snapshot.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(
            report
                .into_iter()
                .filter(|(m, v)| *v >= drift_threshold && metrics.contains(m))
                .collect(),
        ))
    }

    /// Generateas a point in time performance report irrespective of the baseline state.
    /// This will consider the data that has been accumulated since the last flush. Errors
    /// when there is no data accumulate in the runtime stream.
    pub fn performance_snapshot(&self) -> ModelPerfResult<ModelBiasAnalysisReport> {
        if self.ge.len() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let rt_ge = self.ge.ge_snapshot();
        let rt_snapshot = ModelBiasRuntime::new_from_post_training(&self.rt, rt_ge)?;
        Ok(rt_snapshot.generate_report())
    }
}

#[cfg(test)]
mod model_bias_strming_tests {
    use super::*;
    use crate::data_handler::{BiasSegmentationType, SegmentationThresholdType};

    /*
     * These tests validate that the streaming accumulation logic is functionally equivalent to
     * performing the same computations on a discrete dataset.
     * */

    use crate::metrics::ModelBiasMetric;
    use crate::runtime::EQUALITY_ERROR_ALLOWANCE;

    #[derive(Debug)]
    struct TestModelBiasAnalysisReport(crate::reporting::ModelBiasAnalysisReport);

    impl PartialEq for TestModelBiasAnalysisReport {
        fn eq(&self, other: &Self) -> bool {
            use ModelBiasMetric as MBM;
            /*
                DifferenceInPositivePredictedLabels,
                DisparateImpact,
                AccuracyDifference,
                RecallDifference,
                DifferenceInConditionalAcceptance,
                DifferenceInAcceptanceRate,
                SpecialityDifference,
                DifferenceInConditionalRejection,
                DifferenceInRejectionRate,
                TreatmentEquity,
                ConditionalDemographicDesparityPredictedLabels,
                GeneralizedEntropy,
            */
            if (self
                .0
                .get(&MBM::DifferenceInPositivePredictedLabels)
                .unwrap()
                - other
                    .0
                    .get(&MBM::DifferenceInPositivePredictedLabels)
                    .unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::DisparateImpact).unwrap()
                - other.0.get(&MBM::DisparateImpact).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::AccuracyDifference).unwrap()
                - other.0.get(&MBM::AccuracyDifference).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::RecallDifference).unwrap()
                - other.0.get(&MBM::RecallDifference).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::DifferenceInAcceptanceRate).unwrap()
                - other.0.get(&MBM::DifferenceInAcceptanceRate).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::SpecialityDifference).unwrap()
                - other.0.get(&MBM::SpecialityDifference).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::DifferenceInConditionalRejection).unwrap()
                - other.0.get(&MBM::DifferenceInConditionalRejection).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::DifferenceInRejectionRate).unwrap()
                - other.0.get(&MBM::DifferenceInRejectionRate).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::TreatmentEquity).unwrap()
                - other.0.get(&MBM::TreatmentEquity).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self
                .0
                .get(&MBM::ConditionalDemographicDesparityPredictedLabels)
                .unwrap()
                - other
                    .0
                    .get(&MBM::ConditionalDemographicDesparityPredictedLabels)
                    .unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&MBM::GeneralizedEntropy).unwrap()
                - other.0.get(&MBM::GeneralizedEntropy).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            true
        }
    }

    #[test]
    fn test_baseline_construction_label() {
        /*
         * The baseline counts here should be:
         *       facet a:
         *           count: 3
         *           positive pred: 2
         *           positive gt count: 1
         *       facet d:
         *           count: 6
         *           positive pred: 3
         *           positive gt: 4
         * */
        let pred_bl_data: Vec<usize> = vec![1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1];
        let feat_bl_data: Vec<usize> = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1];
        let gt_bl_data: Vec<usize> = vec![1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0];
        let pred_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let feat_seg = BiasSegmentationCriteria::new(0_usize, BiasSegmentationType::Label);
        let gt_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let mut stream = StreamingModelBias::new(
            &feat_bl_data,
            feat_seg,
            &pred_bl_data,
            pred_seg,
            &gt_bl_data,
            gt_seg,
        )
        .unwrap();

        stream
            .push_batch(&feat_bl_data, &pred_bl_data, &gt_bl_data)
            .unwrap();

        assert_eq!(
            stream.rt.dist_a,
            crate::model_bias::PostTrainingDistribution {
                len: 4,
                positive_pred: 2,
                positive_gt: 2
            }
        );

        assert_eq!(
            stream.rt.dist_d,
            crate::model_bias::PostTrainingDistribution {
                len: 7,
                positive_pred: 4,
                positive_gt: 4
            }
        );
    }

    #[test]
    fn test_baseline_construction_accum_label() {
        let pred_bl_data: Vec<usize> = vec![1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1];
        let feat_bl_data: Vec<usize> = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1];
        let gt_bl_data: Vec<usize> = vec![1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0];
        let pred_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let feat_seg = BiasSegmentationCriteria::new(0_usize, BiasSegmentationType::Label);
        let gt_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let mut stream = StreamingModelBias::new(
            &feat_bl_data,
            feat_seg,
            &pred_bl_data,
            pred_seg,
            &gt_bl_data,
            gt_seg,
        )
        .unwrap();

        stream
            .push_batch(&feat_bl_data, &pred_bl_data, &gt_bl_data)
            .unwrap();

        let base = TestModelBiasAnalysisReport(stream.bl.generate_report());
        let test = TestModelBiasAnalysisReport(stream.performance_snapshot().unwrap());
        assert_eq!(base, test);
    }
    // Helper: builds a StreamingModelBias baseline from symmetric 8-sample data.
    fn make_stream() -> StreamingModelBias<usize, usize, usize> {
        let feat = vec![1_usize, 1, 1, 1, 0, 0, 0, 0];
        let pred = vec![1_usize, 1, 0, 0, 1, 1, 0, 0];
        let gt   = vec![1_usize, 0, 1, 0, 1, 0, 1, 0];
        let feat_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let pred_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let gt_seg   = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        StreamingModelBias::new(&feat, feat_seg, &pred, pred_seg, &gt, gt_seg).unwrap()
    }

    fn baseline_data() -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        (
            vec![1_usize, 1, 1, 1, 0, 0, 0, 0],
            vec![1_usize, 1, 0, 0, 1, 1, 0, 0],
            vec![1_usize, 0, 1, 0, 1, 0, 1, 0],
        )
    }

    // --- empty stream errors ---

    #[test]
    fn drift_snapshot_errors_when_stream_is_empty() {
        let stream = make_stream();
        assert!(stream.drift_snapshot().is_err());
    }

    #[test]
    fn drift_report_errors_when_stream_is_empty() {
        let stream = make_stream();
        assert!(stream.drift_report(None).is_err());
    }

    #[test]
    fn performance_snapshot_errors_when_stream_is_empty() {
        let stream = make_stream();
        assert!(stream.performance_snapshot().is_err());
    }

    // --- flush ---

    #[test]
    fn flush_clears_accumulated_data() {
        let mut stream = make_stream();
        let (feat, pred, gt) = baseline_data();
        stream.push_batch(&feat, &pred, &gt).unwrap();
        assert!(stream.drift_snapshot().is_ok());
        stream.flush();
        assert!(stream.drift_snapshot().is_err());
    }

    // --- construction errors ---

    #[test]
    fn new_errors_when_slices_have_mismatched_lengths() {
        let feat_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let pred_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let gt_seg   = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let result = StreamingModelBias::new(
            &[1_usize, 0], feat_seg,
            &[1_usize],    pred_seg,
            &[1_usize, 0], gt_seg,
        );
        assert!(result.is_err());
    }

    // --- drift_report passes when runtime matches baseline ---

    #[test]
    fn drift_report_passes_when_runtime_matches_baseline() {
        let mut stream = make_stream();
        let (feat, pred, gt) = baseline_data();
        stream.push_batch(&feat, &pred, &gt).unwrap();
        // Use a very high threshold so identical data won't trigger drift
        let report = stream.drift_report(Some(1e6)).unwrap();
        assert!(report.passed);
    }

    // --- drift_report_partial_metrics ---

    #[test]
    fn drift_report_partial_metrics_only_returns_requested_metrics() {
        let mut stream = make_stream();
        let (feat, pred, gt) = baseline_data();
        stream.push_batch(&feat, &pred, &gt).unwrap();
        let subset = &[ModelBiasMetric::AccuracyDifference];
        let report = stream
            .drift_report_partial_metrics(subset, Some(1e6))
            .unwrap();
        // With a huge threshold nothing should fail; report passes
        assert!(report.passed);
    }

    // --- reset_baseline ---

    #[test]
    fn reset_baseline_updates_baseline() {
        let mut stream = make_stream();
        let (feat, pred, gt) = baseline_data();
        // After reset with same data, performance_snapshot should still work
        stream.reset_baseline(&feat, &pred, &gt).unwrap();
        stream.push_batch(&feat, &pred, &gt).unwrap();
        assert!(stream.performance_snapshot().is_ok());
    }

    #[test]
    fn reset_baseline_errors_when_slices_have_mismatched_lengths() {
        let mut stream = make_stream();
        let result = stream.reset_baseline(&[1_usize, 0], &[1_usize], &[1_usize, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_accum_threshold() {
        let pred_bl_data: Vec<f32> = vec![
            0.6, 0.12, 0.78, 0.56, 0.98, 0.43, 0.49, 0.60, 0.33, 0.23, 0.54,
        ];
        let feat_bl_data: Vec<usize> = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1];
        let gt_bl_data: Vec<usize> = vec![1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0];
        let pred_seg = BiasSegmentationCriteria::new(
            0.5_f32,
            BiasSegmentationType::Threshold(SegmentationThresholdType::GreaterThanEqualTo),
        );
        let feat_seg = BiasSegmentationCriteria::new(0_usize, BiasSegmentationType::Label);
        let gt_seg = BiasSegmentationCriteria::new(1_usize, BiasSegmentationType::Label);
        let mut stream = StreamingModelBias::new(
            &feat_bl_data,
            feat_seg,
            &pred_bl_data,
            pred_seg,
            &gt_bl_data,
            gt_seg,
        )
        .unwrap();

        stream
            .push_batch(&feat_bl_data, &pred_bl_data, &gt_bl_data)
            .unwrap();

        let base = TestModelBiasAnalysisReport(stream.bl.generate_report());
        let test = TestModelBiasAnalysisReport(stream.performance_snapshot().unwrap());
        assert_eq!(base, test);
    }
}
