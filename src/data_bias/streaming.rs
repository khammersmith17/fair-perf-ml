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
    use crate::errors::ModelPerformanceError;
    use pyo3::prelude::*;

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
            )
            .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(PyDataBiasStreaming { inner })
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn push(&mut self, f: i8, g: i8) {
            self.inner.push(&f, &g)
        }

        fn push_batch(&mut self, f: Vec<i8>, g: Vec<i8>) -> PyResult<()> {
            self.inner
                .push_batch(&f, &g)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn reset_baseline(&mut self, f: Vec<i8>, g: Vec<i8>) -> PyResult<()> {
            self.inner
                .reset_baseline(&f, &g)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .drift_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;

            Ok(report_to_py_dict(py, report))
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

/// A streaming data bias manager. At construction, baseline data is required, then through the
/// lifetime of the type instance, runtime data can be accumulated and a snapshot drift report can
/// be generated to give a point in time snapshot into the current data bias drift. This type
/// stores only the data needed to compute the supported bias metrics, and does so compactly.
/// Reseting the baseline is also supported, which may be appropriate at some time cadence, or
/// after a model refresh. Feature class and ground truth class labeling will be done with the
/// provided 'data_handler::BiasSegmentationCriteria<T>'.
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
    pub fn drift_report_partial_metric(
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
    use crate::data_handler::{BiasSegmentationCriteria, BiasSegmentationType};

    #[test]
    fn test_bl_construction() {
        use crate::data_bias::statistics as stats;
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let ci = stats::class_imbalance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let dpl = stats::diff_in_proportion_of_labels(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let kl = stats::kl_divergence(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let js = stats::jensen_shannon(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let lpnorm = stats::lp_norm(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let tvd = stats::total_variation_distance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let ks = stats::kolmogorov_smirnov(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let base = DataBiasRuntime {
            ci,
            dpl,
            js,
            kl,
            ks,
            lpnorm,
            tvd,
        };

        let streaming = StreamingDataBias::new(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        assert_eq!(base, streaming.baseline_report)
    }

    #[test]
    fn test_bl_reset() {
        use crate::data_bias::statistics as stats;
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let ci = stats::class_imbalance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let dpl = stats::diff_in_proportion_of_labels(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let kl = stats::kl_divergence(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let js = stats::jensen_shannon(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let lpnorm = stats::lp_norm(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let tvd = stats::total_variation_distance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let ks = stats::kolmogorov_smirnov(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let base = DataBiasRuntime {
            ci,
            dpl,
            js,
            kl,
            ks,
            lpnorm,
            tvd,
        };

        let mut streaming = StreamingDataBias::new(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        streaming.reset_baseline(&feature_data, &gt_data).unwrap();
        assert_eq!(base, streaming.baseline_report)
    }

    #[test]
    fn reset_baseline_and_seg() {
        use crate::data_bias::statistics as stats;
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let ci = stats::class_imbalance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let dpl = stats::diff_in_proportion_of_labels(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let kl = stats::kl_divergence(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let js = stats::jensen_shannon(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let lpnorm = stats::lp_norm(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let tvd = stats::total_variation_distance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let ks = stats::kolmogorov_smirnov(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let base = DataBiasRuntime {
            ci,
            dpl,
            js,
            kl,
            ks,
            lpnorm,
            tvd,
        };

        let mut streaming = StreamingDataBias::new(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let feature_data2: Vec<i32> = feature_data.iter().map(|v| v + 1).collect();
        let gt_data2: Vec<i32> = gt_data.iter().map(|v| v + 1).collect();

        streaming
            .reset_baseline_and_segmentation(
                &feature_data2,
                BiasSegmentationCriteria::new(2_i32, BiasSegmentationType::Label),
                &gt_data2,
                BiasSegmentationCriteria::new(2_i32, BiasSegmentationType::Label),
            )
            .unwrap();
        assert_eq!(base, streaming.baseline_report)
    }

    #[test]
    fn perf_report() {
        use crate::data_bias::statistics as stats;
        use crate::metrics::DataBiasMetric as M;
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let ci = stats::class_imbalance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let dpl = stats::diff_in_proportion_of_labels(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let kl = stats::kl_divergence(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let js = stats::jensen_shannon(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let lpnorm = stats::lp_norm(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let tvd = stats::total_variation_distance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let ks = stats::kolmogorov_smirnov(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let mut streaming = StreamingDataBias::new(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let mut result = std::collections::HashMap::new();
        result.insert(M::ClassImbalance, ci);
        result.insert(M::DifferenceInProportionOfLabels, dpl);
        result.insert(M::KlDivergence, kl);
        result.insert(M::JsDivergence, js);
        result.insert(M::LpNorm, lpnorm);
        result.insert(M::TotalVariationDistance, tvd);
        result.insert(M::KolmorogvSmirnov, ks);

        streaming.push_batch(&feature_data, &gt_data).unwrap();

        assert_eq!(result, streaming.performance_snapshot().unwrap());
    }

    #[test]
    fn drift_report() {
        use crate::metrics::DataBiasMetric as M;
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let mut streaming = StreamingDataBias::new(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        streaming.push_batch(&feature_data, &gt_data).unwrap();

        let mut clean_rt = std::collections::HashMap::new();
        clean_rt.insert(M::ClassImbalance, 0_f32);
        clean_rt.insert(M::DifferenceInProportionOfLabels, 0_f32);
        clean_rt.insert(M::KlDivergence, 0_f32);
        clean_rt.insert(M::JsDivergence, 0_f32);
        clean_rt.insert(M::LpNorm, 0_f32);
        clean_rt.insert(M::TotalVariationDistance, 0_f32);
        clean_rt.insert(M::KolmorogvSmirnov, 0_f32);
        assert_eq!(clean_rt, streaming.drift_snapshot().unwrap());
    }
}
