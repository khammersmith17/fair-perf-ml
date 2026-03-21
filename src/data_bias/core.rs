use super::{DataBiasAnalysisReport, PreTraining};
use crate::errors::ModelPerfResult;
use std::collections::HashMap;

pub fn pre_training_bias(data: PreTraining) -> ModelPerfResult<DataBiasAnalysisReport> {
    use super::statistics::inner as stats;
    use crate::metrics::DataBiasMetric as M;
    let mut result: HashMap<M, f32> = HashMap::with_capacity(7);
    result.insert(M::ClassImbalance, stats::class_imbalance(&data));
    result.insert(
        M::DifferenceInProportionOfLabels,
        stats::diff_in_proportion_of_labels(&data)?,
    );
    result.insert(M::KlDivergence, stats::kl_divergence(&data)?);
    result.insert(M::JsDivergence, stats::jensen_shannon(&data)?);
    result.insert(M::LpNorm, stats::lp_norm(&data)?);
    result.insert(
        M::TotalVariationDistance,
        stats::total_variation_distance(&data)?,
    );
    result.insert(M::KolmorogvSmirnov, stats::kolmogorov_smirnov(&data)?);

    Ok(result)
}

#[cfg(test)]
mod db_core {
    use super::super::statistics as stats;
    use super::*;
    use crate::data_handler::BiasSegmentationCriteria;
    use crate::data_handler::BiasSegmentationType;
    use crate::metrics::DataBiasMetric as M;

    #[test]
    fn test_ad_hoc_segmentation() {
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

        let lp_norm = stats::lp_norm(
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
        let mut result: HashMap<M, f32> = HashMap::with_capacity(7);
        result.insert(M::ClassImbalance, ci);
        result.insert(M::DifferenceInProportionOfLabels, dpl);
        result.insert(M::KlDivergence, kl);
        result.insert(M::JsDivergence, js);
        result.insert(M::LpNorm, lp_norm);
        result.insert(M::TotalVariationDistance, tvd);
        result.insert(M::KolmorogvSmirnov, ks);

        let pre_training = PreTraining::new_from_segmentation(
            &feature_data,
            &BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            &BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        assert_eq!(pre_training_bias(pre_training).unwrap(), result)
    }
}
