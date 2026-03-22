pub(crate) mod core;
#[cfg(feature = "python")]
pub(crate) mod python_impl;
pub mod statistics;
pub mod streaming;

use crate::data_handler::{BiasDataPayload, BiasSegmentationCriteria};
use crate::errors::{BiasError, DataBiasRuntimeError, ModelPerfResult, ModelPerformanceError};
use crate::metrics::{DataBiasMetric, DataBiasMetricVec, FULL_DATA_BIAS_METRICS};
use crate::reporting::{DataBiasAnalysisReport, DriftReport, DEFAULT_DRIFT_THRESHOLD};
use crate::runtime::DataBiasRuntime;
use crate::zip_iters;
use statistics::AdHocSegmentation;

/// Function to perform analysis on a dataset pretrainingm, or without regards to a model
/// prediction. This is to be used to indicate a bias between groups in real world outcomes in
/// dataset. The features and ground truth are passed in a BiasDataPayload type, where the
/// segmentaion criteria and segmentation type are provided. This is in turn used to segmented the
/// data into the two facets for bias analysis. This is best used for point in time analysis.
pub fn data_bias_analyzer<'a, F, G>(
    feature: BiasDataPayload<'a, F>,
    ground_truth: BiasDataPayload<'a, G>,
) -> Result<DataBiasAnalysisReport, BiasError>
where
    G: PartialOrd,
    F: PartialOrd,
{
    let pre_training = PreTraining::new_from_bias_payload(feature, ground_truth)?;
    Ok(core::pre_training_bias(pre_training)?)
}

/// Function to perform runtime check across all available DataBias metrics, see
/// [`crate::metrics::DataBiasMetric`] for the full list. The threshold determines whether the metric is
/// within the bounds of a "passing" score and represents the absolute percent drift from the
/// baseline metric score. This is optional and defaults to 0.1.
pub fn data_bias_runtime_check(
    baseline_report: DataBiasAnalysisReport,
    current_report: DataBiasAnalysisReport,
    threshold_opt: Option<f32>,
) -> Result<DriftReport<DataBiasMetric>, DataBiasRuntimeError> {
    let threshold = threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
    let baseline = DataBiasRuntime::try_from(baseline_report)?;
    let current = DataBiasRuntime::try_from(current_report)?;
    let check_res = current.runtime_check(baseline, threshold, &FULL_DATA_BIAS_METRICS);

    Ok(DriftReport::from_runtime(check_res))
}
/// Function to perform runtime check across a subset of available DataBias metrics, see
/// [`crate::metrics::DataBiasMetric`] for the full list. The method accepts a ['crate::metrics::DataBiasMetricVec']
/// which implements 'From<Vec<DataBiasMetric>>' and 'From<&[T]>' where T is string like.
/// The threshold determines whether the metric is within the bounds of a "passing" score and
/// represents the absolute percent drift from the baseline metric score. This is optional and defaults to 0.1.
pub fn data_bias_partial_check<V>(
    baseline_report: DataBiasAnalysisReport,
    current_report: DataBiasAnalysisReport,
    metrics: V,
    threshold_opt: Option<f32>,
) -> Result<DriftReport<DataBiasMetric>, DataBiasRuntimeError>
where
    V: Into<DataBiasMetricVec>,
{
    let threshold = threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
    let baseline = DataBiasRuntime::try_from(baseline_report)?;
    let current = DataBiasRuntime::try_from(current_report)?;
    let check_res = current.runtime_check(baseline, threshold, metrics.into().as_ref());

    Ok(DriftReport::from_runtime(check_res))
}

#[derive(Default, Debug, PartialEq)]
pub(crate) struct PreTrainingDistribution {
    pub positive: u64,
    pub len: u64,
}

impl PreTrainingDistribution {
    #[inline]
    pub(crate) fn acceptance(&self) -> ModelPerfResult<f32> {
        if self.len == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        Ok(self.positive as f32 / self.len as f32)
    }

    fn clear(&mut self) {
        self.positive = 0_u64;
        self.len = 0_u64;
    }
}

#[derive(Default, Debug, PartialEq)]
pub(crate) struct PreTraining {
    facet_a: PreTrainingDistribution,
    facet_d: PreTrainingDistribution,
}

impl From<AdHocSegmentation> for PreTraining {
    fn from(seg: AdHocSegmentation) -> Self {
        let AdHocSegmentation { facet_a, facet_d } = seg;
        Self { facet_a, facet_d }
    }
}

impl PreTraining {
    pub(crate) fn new_from_labeled(
        feature_data: &[i16],
        gt_data: &[i16],
    ) -> Result<PreTraining, BiasError> {
        let mut facet_a = PreTrainingDistribution::default();
        let mut facet_d = PreTrainingDistribution::default();

        for (f, gt) in zip_iters!(feature_data, gt_data) {
            if f.eq(&1_i16) {
                facet_a.positive += *gt as u64;
                facet_a.len += 1
            } else {
                facet_d.positive += *gt as u64;
                facet_d.len += 1
            }
        }

        Self::check_facet_deviation(&facet_a, &facet_d)?;

        Ok(PreTraining { facet_a, facet_d })
    }

    // Utility to validate there exists samples in both demographic classes.
    fn check_facet_deviation(
        facet_a: &PreTrainingDistribution,
        facet_d: &PreTrainingDistribution,
    ) -> Result<(), BiasError> {
        if facet_a.len == 0 || facet_d.len == 0 {
            return Err(BiasError::NoFacetDeviation);
        }
        Ok(())
    }

    pub(crate) fn new_from_bias_payload<'a, F: PartialOrd, G: PartialOrd>(
        feat: BiasDataPayload<'a, F>,
        gt: BiasDataPayload<'a, G>,
    ) -> Result<PreTraining, BiasError> {
        if feat.len() != gt.len() || feat.len() == 0 {
            return Err(BiasError::DataLengthError);
        }
        let mut facet_a = PreTrainingDistribution::default();
        let mut facet_d = PreTrainingDistribution::default();

        for (f, g) in zip_iters!(feat.data, gt.data) {
            let grp = feat.segmentation_criteria.label(f);
            let is_p = gt.segmentation_criteria.label(g);

            facet_a.len += grp as u64;
            facet_a.positive += (grp && is_p) as u64;

            facet_d.len += !grp as u64;
            facet_d.positive += (!grp && is_p) as u64;
        }

        Self::check_facet_deviation(&facet_a, &facet_d)?;

        Ok(PreTraining { facet_a, facet_d })
    }

    pub(crate) fn size(&self) -> u64 {
        self.facet_a.len + self.facet_d.len
    }

    pub(crate) fn clear(&mut self) {
        self.facet_a.clear();
        self.facet_d.clear();
    }

    pub(crate) fn new_from_segmentation<F, G>(
        feature: &[F],
        feat_seg: &BiasSegmentationCriteria<F>,
        gt: &[G],
        gt_seg: &BiasSegmentationCriteria<G>,
    ) -> Result<PreTraining, BiasError>
    where
        F: PartialOrd,
        G: PartialOrd,
    {
        if feature.len() != gt.len() || gt.is_empty() {
            return Err(BiasError::DataLengthError);
        }

        let mut facet_a = PreTrainingDistribution::default();
        let mut facet_d = PreTrainingDistribution::default();

        for (f, g) in zip_iters!(feature, gt) {
            let is_favored = feat_seg.label(f);
            let is_positive = gt_seg.label(g);

            facet_a.len += is_favored as u64;
            facet_a.positive += (is_favored && is_positive) as u64;
            facet_d.len += !is_favored as u64;
            facet_d.positive += (!is_favored && is_positive) as u64;
        }

        Self::check_facet_deviation(&facet_a, &facet_d)?;

        Ok(PreTraining { facet_a, facet_d })
    }

    #[inline]
    fn accumulate_runtime_single<F, G>(
        &mut self,
        f: &F,
        feat_seg: &BiasSegmentationCriteria<F>,
        g: &G,
        gt_seg: &BiasSegmentationCriteria<G>,
    ) where
        F: PartialOrd,
        G: PartialOrd,
    {
        let is_a = feat_seg.label(f);
        let is_positive = gt_seg.label(g);

        self.facet_a.len += is_a as u64;
        self.facet_d.len += !is_a as u64;
        self.facet_a.positive += (is_a && is_positive) as u64;
        self.facet_d.positive += (!is_a && is_positive) as u64;
    }

    fn accumulate_runtime_batch<F, G>(
        &mut self,
        features: &[F],
        feat_seg: &BiasSegmentationCriteria<F>,
        gt: &[G],
        gt_seg: &BiasSegmentationCriteria<G>,
    ) -> Result<(), BiasError>
    where
        F: PartialOrd,
        G: PartialOrd,
    {
        if features.len() != gt.len() || gt.is_empty() {
            return Err(BiasError::DataLengthError);
        }

        for (f, g) in zip_iters!(features, gt) {
            self.accumulate_runtime_single(f, feat_seg, g, gt_seg);
        }

        Ok(())
    }
}

#[cfg(test)]
mod data_bias_containers {
    use super::*;
    use crate::data_handler::{BiasSegmentationCriteria, BiasSegmentationType};
    use crate::metrics::DataBiasMetric as M;
    use std::collections::HashMap;

    #[test]
    fn data_bias_from_slice() {
        let gt = vec![1, 0, 1, 0, 1, 1, 1, 0];
        let feat = vec![1, 1, 0, 0, 1, 0, 1, 0];
        let gt_seg = BiasSegmentationCriteria::new(1, BiasSegmentationType::Label);
        let feat_seg = BiasSegmentationCriteria::new(0, BiasSegmentationType::Label);

        let pre_training =
            PreTraining::new_from_segmentation(&feat, &feat_seg, &gt, &gt_seg).unwrap();
        assert_eq!(
            pre_training.facet_a,
            PreTrainingDistribution {
                len: 4,
                positive: 2
            }
        );
        assert_eq!(
            pre_training.facet_d,
            PreTrainingDistribution {
                len: 4,
                positive: 3
            }
        );
    }

    #[test]
    fn pre_training_flush() {
        let gt = vec![1, 0, 1, 0, 1, 1, 1, 0];
        let feat = vec![1, 1, 0, 0, 1, 0, 1, 0];
        let gt_seg = BiasSegmentationCriteria::new(1, BiasSegmentationType::Label);
        let feat_seg = BiasSegmentationCriteria::new(0, BiasSegmentationType::Label);

        let mut pre_training =
            PreTraining::new_from_segmentation(&feat, &feat_seg, &gt, &gt_seg).unwrap();

        pre_training.clear();
        assert_eq!(pre_training, PreTraining::default());
    }

    #[test]
    fn test_data_bias_analyzer() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let ci = statistics::class_imbalance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let dpl = statistics::diff_in_proportion_of_labels(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let kl = statistics::kl_divergence(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let js = statistics::jensen_shannon(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let lp_norm = statistics::lp_norm(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let tvd = statistics::total_variation_distance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let ks = statistics::kolmogorov_smirnov(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let mut result: HashMap<M, f32> = HashMap::with_capacity(7);
        result.insert(M::ClassImbalance, ci);
        result.insert(M::DifferenceInProportionOfLabels, dpl);
        result.insert(M::KlDivergence, kl);
        result.insert(M::JsDivergence, js);
        result.insert(M::LpNorm, lp_norm);
        result.insert(M::TotalVariationDistance, tvd);
        result.insert(M::KolmogorovSmirnov, ks);

        let test = data_bias_analyzer(
            BiasDataPayload::new_from_criteria(
                &feature_data,
                BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            ),
            BiasDataPayload::new_from_criteria(
                &gt_data,
                BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            ),
        )
        .unwrap();

        assert_eq!(test, result)
    }

    #[test]
    fn pre_training_from_label() {
        let feature_data: Vec<i16> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i16> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let pre_training = PreTraining::new_from_labeled(&feature_data, &gt_data).unwrap();
        assert_eq!(pre_training.facet_a.len, 10);
        assert_eq!(pre_training.facet_d.len, 10);
        assert_eq!(pre_training.facet_a.positive, 7);
        assert_eq!(pre_training.facet_d.positive, 2);
    }

    #[test]
    fn pre_training_accum() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let feat_seg_criteria = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        let gt_seg_criteria = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        let mut pre_training = PreTraining::default();
        pre_training
            .accumulate_runtime_batch(
                &feature_data,
                &feat_seg_criteria,
                &gt_data,
                &gt_seg_criteria,
            )
            .unwrap();
        assert_eq!(pre_training.facet_a.len, 10);
        assert_eq!(pre_training.facet_d.len, 10);
        assert_eq!(pre_training.facet_a.positive, 7);
        assert_eq!(pre_training.facet_d.positive, 2);
    }

    #[test]
    fn no_facet_deviation() {
        let feature_data: Vec<i16> =
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let gt_data: Vec<i16> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let pre_training = PreTraining::new_from_labeled(&feature_data, &gt_data);
        assert!(pre_training.is_err());
        assert_eq!(pre_training.err().unwrap(), BiasError::NoFacetDeviation);
    }
}
