use numpy::dtype_bound;
use numpy::PyUntypedArrayMethods;
use numpy::{PyArrayDescrMethods, PyUntypedArray};
use pyo3::prelude::*;
use std::collections::HashMap;

mod data_bias;
use data_bias::{
    class_imbalance, diff_in_proportion_of_labels, kl_divergence, kolmorogv_smirnov, lp_norm,
    total_variation_distance, PreTraining, PreTrainingComputations,
};
mod model_bias;

use model_bias::{
    accuracy_difference, cond_dem_desp_in_pred_labels, diff_in_acceptance_rate,
    diff_in_cond_acceptance, diff_in_cond_rejection, diff_in_pos_proportion_in_pred_labels,
    diff_in_rejection_rate, disparate_impact, generalized_entropy, recall_difference,
    specailty_difference, treatment_equity, PostTrainingComputations, PostTrainingData,
};

mod data_handler;
use data_handler::{apply_label, perform_segmentation_data_bias, perform_segmentation_model_bias};

fn pre_training_bias(
    facet_a_trues: Vec<i16>,
    facet_d_trues: Vec<i16>,
) -> PyResult<HashMap<String, f32>> {
    let data = PreTraining {
        facet_a: facet_a_trues,
        facet_d: facet_d_trues,
    };

    let computed_data: PreTrainingComputations = data.generate();
    let mut result = HashMap::new();
    result.insert("CI".into(), class_imbalance(&data));
    result.insert("DPL".into(), diff_in_proportion_of_labels(&data));
    result.insert("KL".into(), kl_divergence(&computed_data));
    // do JS
    result.insert("LPNorm".into(), lp_norm(&computed_data));
    result.insert(
        "TotalVarationDistance".into(),
        total_variation_distance(&computed_data),
    );
    result.insert("KS".into(), kolmorogv_smirnov(&data));

    Ok(result)
}

fn post_training_bias(
    facet_a_scores: Vec<i16>,
    facet_d_scores: Vec<i16>,
    facet_a_trues: Vec<i16>,
    facet_d_trues: Vec<i16>,
) -> PyResult<HashMap<String, f32>> {
    let data = PostTrainingData {
        facet_a_scores,
        facet_d_scores,
        facet_a_trues,
        facet_d_trues,
    };
    let pre_computed_data: PostTrainingComputations = data.general_data_computations();
    let mut result = HashMap::new();
    result.insert("DDPL".into(), diff_in_pos_proportion_in_pred_labels(&data));
    result.insert("DI".into(), disparate_impact(&data));
    result.insert("AD".into(), accuracy_difference(&pre_computed_data, &data));
    result.insert("RD".into(), recall_difference(&pre_computed_data));
    result.insert("CDACC".into(), diff_in_cond_acceptance(&data));
    result.insert("DAR".into(), diff_in_acceptance_rate(&pre_computed_data));
    result.insert("SD".into(), specailty_difference(&pre_computed_data));
    result.insert("DCR".into(), diff_in_cond_rejection(&data));
    result.insert("DRR".into(), diff_in_rejection_rate(&pre_computed_data));
    result.insert("TE".into(), treatment_equity(&pre_computed_data));
    result.insert("CCDPL".into(), cond_dem_desp_in_pred_labels(&data));
    result.insert("GE".into(), generalized_entropy(&data));

    Ok(result)
}

#[pyfunction]
pub fn model_bias_analyzer<'py>(
    py: Python<'_>,
    feature_array: &Bound<'_, PyUntypedArray>,
    ground_truth_array: &Bound<'_, PyUntypedArray>,
    prediction_array: &Bound<'_, PyUntypedArray>,
    feature_label_or_threshold: Bound<'py, PyAny>, //fix
    ground_truth_label_or_threshold: Bound<'py, PyAny>, //fix
    prediction_label_or_threshold: Bound<'py, PyAny>, // fix
) -> PyResult<HashMap<String, f32>> {
    let labeled_predictions: Vec<i16> =
        apply_label(py, prediction_array, prediction_label_or_threshold);
    let labeled_ground_truth: Vec<i16> =
        apply_label(py, ground_truth_array, ground_truth_label_or_threshold);
    let labeled_features: Vec<i16> = apply_label(py, feature_array, feature_label_or_threshold);
    let (facet_a_trues, facet_a_scores, facet_d_trues, facet_d_scores) =
        perform_segmentation_model_bias(
            labeled_features,
            labeled_predictions,
            labeled_ground_truth,
        );

    post_training_bias(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues)
}

#[pyfunction]
fn data_bias_analyzer<'py>(
    py: Python<'_>,
    feature_array: &Bound<'_, PyUntypedArray>,
    ground_truth_array: &Bound<'_, PyUntypedArray>,
    feature_label_or_threshold: Bound<'py, PyAny>, //fix
    ground_truth_label_or_threshold: Bound<'py, PyAny>, //fix
) -> PyResult<HashMap<String, f32>> {
    let labeled_ground_truth = apply_label(py, ground_truth_array, ground_truth_label_or_threshold);
    let labeled_feature = apply_label(py, feature_array, feature_label_or_threshold);

    let (facet_a_trues, facet_d_trues) =
        perform_segmentation_data_bias(labeled_feature, labeled_ground_truth);

    pre_training_bias(facet_a_trues, facet_d_trues)
}

/// A Python module implemented in Rust.
#[pymodule]
fn ml_bias(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(model_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(data_bias_analyzer, m)?)?;

    Ok(())
}
