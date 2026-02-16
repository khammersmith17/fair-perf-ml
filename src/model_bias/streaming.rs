use super::{BucketGeneralizedEntropy, PostTraining};
use crate::data_handler::BiasSegmentationCriteria;
use crate::errors::{ModelPerfResult, ModelPerformanceError};
use crate::metrics::ModelBiasMetric;
use crate::reporting::{DriftReport, ModelBiasAnalysisReport};
use crate::runtime::ModelBiasRuntime;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::StreamingModelBias;
    use crate::data_handler::{
        py_types_handler::{report_to_py_dict, PyDictResult},
        BiasSegmentationCriteria, BiasSegmentationType,
    };
    use crate::errors::ModelPerformanceError;
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
            )
            .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;

            Ok(PyModelBiasStreaming { inner })
        }

        fn push(&mut self, feature: i8, pred: i8, gt: i8) {
            // TODO: passing reference more expensive then passing data here
            self.inner.push(&feature, &pred, &gt);
        }

        fn push_batch(&mut self, feature: Vec<i8>, pred: Vec<i8>, gt: Vec<i8>) -> PyResult<()> {
            self.inner
                .push_batch(&feature, &pred, &gt)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn reset_baseline(&mut self, feature: Vec<i8>, pred: Vec<i8>, gt: Vec<i8>) -> PyResult<()> {
            self.inner
                .reset_baseline(&feature, &pred, &gt)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .drift_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(report.into_py_dict(py)?)
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .performance_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;

            Ok(report_to_py_dict(py, report))
        }
    }
}

/// This is a type geared toward a long running ML Observability service. Internally it leverages a
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
        )
        .map_err(|e| ModelPerformanceError::BiasError(e))?;
        dbg!(&bl_pt);

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
        self.rt
            .accumulate_batch(
                features,
                &self.feat_seg,
                preds,
                &self.pred_seg,
                gt,
                &self.gt_seg,
            )
            .map_err(|e| ModelPerformanceError::BiasError(e))?;
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
        )
        .map_err(|e| ModelPerformanceError::BiasError(e))?;

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
        )
        .map_err(|e| ModelPerformanceError::BiasError(e))?;

        let mut bl_ge_bucket = BucketGeneralizedEntropy::default();
        bl_ge_bucket.accumulate(ground_truth, &self.gt_seg, prediction, &self.pred_seg);
        let bl_ge = bl_ge_bucket.ge_snapshot();
        self.bl = ModelBiasRuntime::new_from_post_training(&bl_pt, bl_ge)?;
        Ok(())
    }

    /// Generateas a point in time drift report, this will consider the baseline set and all the
    /// data that has been accumulated since the last flush. Errors when there is no data
    /// accumulate in the runtime stream.
    pub fn drift_snapshot(&self) -> ModelPerfResult<DriftReport<ModelBiasMetric>> {
        if self.ge.len() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let rt_ge = self.ge.ge_snapshot();
        let rt_snapshot = ModelBiasRuntime::new_from_post_training(&self.rt, rt_ge)?;
        let report = rt_snapshot.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(report))
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
    #[test]
    fn test_baseline_construction_accum_threshold() {
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
