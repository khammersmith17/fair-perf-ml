use super::PreTraining;
use crate::data_handler::BiasSegmentationCriteria;
use crate::errors::ModelPerformanceError;
use crate::metrics::DataBiasMetric;
use crate::reporting::DriftReport;
use crate::runtime::DataBiasRuntime;

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
    bl: PreTraining, // stores the baseline accumulated distributions for both facets
}

impl<G, F> StreamingDataBias<G, F>
where
    F: PartialOrd,
    G: PartialOrd,
{
    /// Construct a new streaming instance with baseline features and ground truth. Feature and
    /// ground truth label segmentation criteria is required to segment features and labels into
    /// advantaged and disadvantaged classes.
    pub fn new_with_baseline(
        feature_seg_criteria: BiasSegmentationCriteria<F>,
        gt_seg_criteria: BiasSegmentationCriteria<G>,
        feature_data: &[F],
        gt_data: &[G],
    ) -> StreamingDataBias<G, F> {
        let bl = PreTraining::new_from_segmentation(
            feature_data,
            &feature_seg_criteria,
            gt_data,
            &gt_seg_criteria,
        );
        let rt = PreTraining::default();

        let baseline_report = DataBiasRuntime::new_from_pre_training(&bl);

        StreamingDataBias {
            feature_seg_criteria,
            gt_seg_criteria,
            baseline_report,
            bl,
            rt,
        }
    }

    /// Method to push new data to the stream and update stream state. The segmentation logic
    /// provided at type construction will do all feature class and outcome class labeling.
    pub fn push_data(&mut self, feature: &[F], gt: &[G]) {
        self.rt.accumulate_runtime(
            feature,
            &self.feature_seg_criteria,
            gt,
            &self.gt_seg_criteria,
        );
    }

    pub fn flush(&mut self) {
        let _ = std::mem::take(&mut self.rt);
    }

    /// Reset the baseline data. This may be used to refresh the data on a retraining, or to move
    /// forward in time from a given baseline set if data drifts. This method will clear all
    /// runtime data.
    pub fn reset_baseline(&mut self, feature_data: &[F], gt_data: &[G]) {
        let new_bl = PreTraining::new_from_segmentation(
            feature_data,
            &self.feature_seg_criteria,
            gt_data,
            &self.gt_seg_criteria,
        );

        self.bl = new_bl;
        self.rt.clear();
    }

    /// Method to generate a point in time drift snapshot. This method will report out the current
    /// drift across all metrics in 'metrics::DataBiasMetric'. Returns an error when there the
    /// runtime bins are empty.
    pub fn generate_drift_snapshot(
        &self,
    ) -> Result<DriftReport<DataBiasMetric>, ModelPerformanceError> {
        if self.rt.size() == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let curr_rt = DataBiasRuntime::new_from_pre_training(&self.rt);
        let rt_report = curr_rt.runtime_drift_report(&self.baseline_report);
        Ok(DriftReport::from_runtime(rt_report))
    }
}

// the idea here is to register many different features
// so it will probably look some thing like a table of <feature name>: <StreamingDataBias>
// figure out how out how to make it store arbitrary types
// I guess they each streaming agent will then need to be boxed
pub struct StreamingDataBiasManager {}
