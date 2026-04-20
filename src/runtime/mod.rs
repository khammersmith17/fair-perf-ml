pub mod data_bias;
pub mod model_bias;
pub mod model_perf;

pub use data_bias::DataBiasRuntime;
pub use model_bias::ModelBiasRuntime;
pub use model_perf::{
    BinaryClassificationRuntime, LinearRegressionRuntime, LogisticRegressionRuntime,
};

pub(crate) const EQUALITY_ERROR_ALLOWANCE: f32 = 1e-5;

#[cfg(test)]
mod runtime_container_tests {
    use super::*;
    use crate::metrics::{
        ClassificationEvaluationMetric as C, DataBiasMetric, LinearRegressionEvaluationMetric as L,
    };

    // --- DataBiasRuntime ---

    #[test]
    fn data_bias_runtime_from_hashmap_happy_path() {
        let mut map = std::collections::HashMap::new();
        map.insert("ClassImbalance".to_string(), 0.1_f32);
        map.insert("DifferenceInProportionOfLabels".to_string(), 0.2_f32);
        map.insert("KlDivergence".to_string(), 0.3_f32);
        map.insert("JsDivergence".to_string(), 0.4_f32);
        map.insert("LpNorm".to_string(), 0.5_f32);
        map.insert("TotalVariationDistance".to_string(), 0.6_f32);
        map.insert("KolmogorovSmirnov".to_string(), 0.7_f32);

        let rt = DataBiasRuntime::try_from(map).unwrap();
        assert_eq!(rt.ci, 0.1_f32);
        assert_eq!(rt.dpl, 0.2_f32);
        assert_eq!(rt.kl, 0.3_f32);
        assert_eq!(rt.js, 0.4_f32);
        assert_eq!(rt.lpnorm, 0.5_f32);
        assert_eq!(rt.tvd, 0.6_f32);
        assert_eq!(rt.ks, 0.7_f32);
    }

    #[test]
    fn data_bias_runtime_from_hashmap_missing_key() {
        let mut map = std::collections::HashMap::new();
        map.insert("ClassImbalance".to_string(), 0.1_f32);
        // all other keys missing
        let res = DataBiasRuntime::try_from(map);
        assert!(res.is_err());
    }

    #[test]
    fn data_bias_runtime_generate_report_round_trip() {
        let rt = DataBiasRuntime {
            ci: 0.1_f32,
            dpl: 0.2_f32,
            kl: 0.3_f32,
            js: 0.4_f32,
            lpnorm: 0.5_f32,
            tvd: 0.6_f32,
            ks: 0.7_f32,
        };
        let report = rt.generate_report();
        let rt2 = DataBiasRuntime::try_from(report).unwrap();
        assert_eq!(rt, rt2);
    }

    #[test]
    fn data_bias_runtime_check_detects_drift() {
        let baseline = DataBiasRuntime {
            ci: 0.2_f32,
            dpl: 0.1_f32,
            kl: 0.1_f32,
            js: 0.1_f32,
            lpnorm: 0.1_f32,
            tvd: 0.1_f32,
            ks: 0.1_f32,
        };
        // ci=0.5 > 0.2 * 1.1 = 0.22 → flagged
        let runtime = DataBiasRuntime {
            ci: 0.5_f32,
            dpl: 0.1_f32,
            kl: 0.1_f32,
            js: 0.1_f32,
            lpnorm: 0.1_f32,
            tvd: 0.1_f32,
            ks: 0.1_f32,
        };
        let result = runtime.runtime_check(baseline, 0.1_f32, &[DataBiasMetric::ClassImbalance]);
        assert!(result.contains_key(&DataBiasMetric::ClassImbalance));
    }

    #[test]
    fn data_bias_runtime_check_no_drift_within_threshold() {
        let baseline = DataBiasRuntime {
            ci: 0.2_f32,
            dpl: 0.1_f32,
            kl: 0.1_f32,
            js: 0.1_f32,
            lpnorm: 0.1_f32,
            tvd: 0.1_f32,
            ks: 0.1_f32,
        };
        // ci=0.21 < 0.2 * 1.1 = 0.22 → not flagged
        let runtime = DataBiasRuntime {
            ci: 0.21_f32,
            ..baseline.clone()
        };
        let result = runtime.runtime_check(baseline, 0.1_f32, &[DataBiasMetric::ClassImbalance]);
        assert!(result.is_empty());
    }

    // --- BinaryClassificationRuntime ---

    #[test]
    fn binary_classification_runtime_new_perfect_predictions() {
        let y_true = vec![1_i32, 0, 1, 0, 1];
        let y_pred = vec![1_i32, 0, 1, 0, 1];
        let rt = BinaryClassificationRuntime::new(&y_true, &y_pred, &1_i32).unwrap();
        assert!((rt.accuracy - 1.0_f32).abs() < 1e-5_f32);
        assert!((rt.precision_positive - 1.0_f32).abs() < 1e-5_f32);
        assert!((rt.recall_positive - 1.0_f32).abs() < 1e-5_f32);
        assert!((rt.f1_score - 1.0_f32).abs() < 1e-5_f32);
    }

    #[test]
    fn binary_classification_runtime_compare_detects_drop() {
        let baseline = BinaryClassificationRuntime {
            balanced_accuracy: 0.9_f32,
            precision_positive: 0.9_f32,
            precision_negative: 0.9_f32,
            recall_positive: 0.9_f32,
            recall_negative: 0.9_f32,
            accuracy: 0.9_f32,
            f1_score: 0.9_f32,
        };
        // accuracy=0.5, 0.5 * 1.1 = 0.55 < 0.9 → flagged
        let runtime = BinaryClassificationRuntime {
            balanced_accuracy: 0.5_f32,
            precision_positive: 0.5_f32,
            precision_negative: 0.5_f32,
            recall_positive: 0.5_f32,
            recall_negative: 0.5_f32,
            accuracy: 0.5_f32,
            f1_score: 0.5_f32,
        };
        let result = runtime.compare_to_baseline(&[C::Accuracy, C::F1Score], &baseline, 0.1_f32);
        assert!(result.contains_key(&C::Accuracy));
        assert!(result.contains_key(&C::F1Score));
    }

    #[test]
    fn binary_classification_runtime_compare_no_drift_identical() {
        let baseline = BinaryClassificationRuntime {
            balanced_accuracy: 0.9_f32,
            precision_positive: 0.9_f32,
            precision_negative: 0.9_f32,
            recall_positive: 0.9_f32,
            recall_negative: 0.9_f32,
            accuracy: 0.9_f32,
            f1_score: 0.9_f32,
        };
        let runtime = BinaryClassificationRuntime {
            balanced_accuracy: 0.9_f32,
            precision_positive: 0.9_f32,
            precision_negative: 0.9_f32,
            recall_positive: 0.9_f32,
            recall_negative: 0.9_f32,
            accuracy: 0.9_f32,
            f1_score: 0.9_f32,
        };
        let all_metrics = vec![
            C::BalancedAccuracy,
            C::PrecisionPositive,
            C::PrecisionNegative,
            C::RecallPositive,
            C::RecallNegative,
            C::Accuracy,
            C::F1Score,
        ];
        let result = runtime.compare_to_baseline(&all_metrics, &baseline, 0.1_f32);
        assert!(result.is_empty());
    }

    #[test]
    fn binary_classification_runtime_report_round_trip() {
        let rt = BinaryClassificationRuntime {
            balanced_accuracy: 0.85_f32,
            precision_positive: 0.80_f32,
            precision_negative: 0.82_f32,
            recall_positive: 0.78_f32,
            recall_negative: 0.90_f32,
            accuracy: 0.84_f32,
            f1_score: 0.79_f32,
        };
        let report = rt.generate_report();
        let rt2 = BinaryClassificationRuntime::try_from(&report).unwrap();
        assert_eq!(rt, rt2);
    }

    // --- LogisticRegressionRuntime ---

    #[test]
    fn logistic_regression_log_loss_increase_flagged() {
        let baseline = LogisticRegressionRuntime {
            balanced_accuracy: 0.9_f32,
            precision_positive: 0.9_f32,
            precision_negative: 0.9_f32,
            recall_positive: 0.9_f32,
            recall_negative: 0.9_f32,
            accuracy: 0.9_f32,
            f1_score: 0.9_f32,
            log_loss: 0.2_f32,
        };
        // log_loss=0.5 > 0.2 * 1.1 = 0.22 → flagged
        let runtime = LogisticRegressionRuntime {
            log_loss: 0.5_f32,
            ..baseline.clone()
        };
        let result = runtime.compare_to_baseline(&[C::LogLoss], &baseline, 0.1_f32);
        assert!(result.contains_key(&C::LogLoss));
    }

    #[test]
    fn logistic_regression_log_loss_improvement_not_flagged() {
        let baseline = LogisticRegressionRuntime {
            balanced_accuracy: 0.9_f32,
            precision_positive: 0.9_f32,
            precision_negative: 0.9_f32,
            recall_positive: 0.9_f32,
            recall_negative: 0.9_f32,
            accuracy: 0.9_f32,
            f1_score: 0.9_f32,
            log_loss: 0.5_f32,
        };
        // log_loss=0.1, 0.1 > 0.5 * 1.1 = 0.55 → false, not flagged
        let runtime = LogisticRegressionRuntime {
            log_loss: 0.1_f32,
            ..baseline.clone()
        };
        let result = runtime.compare_to_baseline(&[C::LogLoss], &baseline, 0.1_f32);
        assert!(result.is_empty());
    }

    // --- LinearRegressionRuntime ---

    #[test]
    fn linear_regression_runtime_new_constant_error() {
        // y_pred = y_true + 1.0 for all → error=1.0 throughout
        // y_true_mean=3.0, ss_total=10.0, mse=1.0, r2=0.5
        let y_true = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![2.0_f32, 3.0, 4.0, 5.0, 6.0];
        let rt = LinearRegressionRuntime::new(&y_true, &y_pred).unwrap();
        assert!((rt.mse - 1.0_f32).abs() < 1e-4_f32);
        assert!((rt.rmse - 1.0_f32).abs() < 1e-4_f32);
        assert!((rt.mae - 1.0_f32).abs() < 1e-4_f32);
        assert!((rt.r_squared - 0.5_f32).abs() < 1e-4_f32);
    }

    #[test]
    fn linear_regression_r_squared_drop_detected() {
        let baseline = LinearRegressionRuntime {
            rmse: 0.1_f32,
            mse: 0.01_f32,
            mae: 0.1_f32,
            r_squared: 0.95_f32,
            max_error: 0.2_f32,
            msle: 0.01_f32,
            rmsle: 0.1_f32,
            mape: 0.05_f32,
        };
        // r_squared=0.5 < 0.95 * (1 - 0.1) = 0.855 → flagged
        let runtime = LinearRegressionRuntime {
            r_squared: 0.5_f32,
            ..baseline.clone()
        };
        let result = runtime.compare_to_baseline(&[L::RSquared], &baseline, 0.1_f32);
        assert!(result.contains_key(&L::RSquared));
    }

    #[test]
    fn linear_regression_r_squared_improvement_not_flagged() {
        let baseline = LinearRegressionRuntime {
            rmse: 0.1_f32,
            mse: 0.01_f32,
            mae: 0.1_f32,
            r_squared: 0.8_f32,
            max_error: 0.2_f32,
            msle: 0.01_f32,
            rmsle: 0.1_f32,
            mape: 0.05_f32,
        };
        // r_squared=0.95, 0.95 < 0.8 * (1 - 0.1) = 0.72 → false, not flagged
        let runtime = LinearRegressionRuntime {
            r_squared: 0.95_f32,
            ..baseline.clone()
        };
        let result = runtime.compare_to_baseline(&[L::RSquared], &baseline, 0.1_f32);
        assert!(result.is_empty());
    }

    #[test]
    fn linear_regression_runtime_report_round_trip() {
        let rt = LinearRegressionRuntime {
            rmse: 0.5_f32,
            mse: 0.25_f32,
            mae: 0.4_f32,
            r_squared: 0.85_f32,
            max_error: 1.2_f32,
            msle: 0.03_f32,
            rmsle: 0.17_f32,
            mape: 0.06_f32,
        };
        let report = rt.generate_report();
        let rt2 = LinearRegressionRuntime::try_from(&report).unwrap();
        assert_eq!(rt, rt2);
    }

    #[test]
    fn linear_regression_error_metrics_increase_flagged() {
        let baseline = LinearRegressionRuntime {
            rmse: 0.5_f32,
            mse: 0.25_f32,
            mae: 0.4_f32,
            r_squared: 0.85_f32,
            max_error: 1.0_f32,
            msle: 0.03_f32,
            rmsle: 0.17_f32,
            mape: 0.06_f32,
        };
        // mse=0.5 > 0.25 * 1.1 = 0.275 → flagged
        let runtime = LinearRegressionRuntime {
            mse: 0.5_f32,
            rmse: 0.5_f32.sqrt(),
            ..baseline.clone()
        };
        let result = runtime.compare_to_baseline(&[L::MeanSquaredError], &baseline, 0.1_f32);
        assert!(result.contains_key(&L::MeanSquaredError));
    }
}

#[cfg(test)]
mod runtime_coverage_tests {
    use super::*;
    use crate::metrics::{
        ClassificationEvaluationMetric as C, DataBiasMetric as D,
        LinearRegressionEvaluationMetric as L, ModelBiasMetric as M, FULL_MODEL_BIAS_METRICS,
    };
    use crate::reporting::ModelBiasAnalysisReport;
    use std::collections::HashMap;

    // --- helpers ---

    fn data_bias_runtime(v: f32) -> DataBiasRuntime {
        DataBiasRuntime {
            ci: v,
            dpl: v,
            kl: v,
            js: v,
            lpnorm: v,
            tvd: v,
            ks: v,
        }
    }

    fn model_bias_report(v: f32) -> ModelBiasAnalysisReport {
        let mut m = HashMap::with_capacity(12);
        m.insert(M::DifferenceInPositivePredictedLabels, v);
        m.insert(M::DisparateImpact, v);
        m.insert(M::AccuracyDifference, v);
        m.insert(M::RecallDifference, v);
        m.insert(M::DifferenceInConditionalAcceptance, v);
        m.insert(M::DifferenceInAcceptanceRate, v);
        m.insert(M::SpecialityDifference, v);
        m.insert(M::DifferenceInConditionalRejection, v);
        m.insert(M::DifferenceInRejectionRate, v);
        m.insert(M::TreatmentEquity, v);
        m.insert(M::ConditionalDemographicDesparityPredictedLabels, v);
        m.insert(M::GeneralizedEntropy, v);
        m
    }

    fn model_bias_string_map(v: f32) -> HashMap<String, f32> {
        let mut m = HashMap::with_capacity(12);
        m.insert("DifferenceInPositivePredictedLabels".into(), v);
        m.insert("DisparateImpact".into(), v);
        m.insert("AccuracyDifference".into(), v);
        m.insert("RecallDifference".into(), v);
        m.insert("DifferenceInConditionalAcceptance".into(), v);
        m.insert("DifferenceInAcceptanceRate".into(), v);
        m.insert("SpecialityDifference".into(), v);
        m.insert("DifferenceInConditionalRejection".into(), v);
        m.insert("DifferenceInRejectionRate".into(), v);
        m.insert("TreatmentEquity".into(), v);
        m.insert("ConditionalDemographicDesparityPredictedLabels".into(), v);
        m.insert("GeneralizedEntropy".into(), v);
        m
    }

    fn binary_classification_runtime(v: f32) -> BinaryClassificationRuntime {
        BinaryClassificationRuntime {
            balanced_accuracy: v,
            precision_positive: v,
            precision_negative: v,
            recall_positive: v,
            recall_negative: v,
            accuracy: v,
            f1_score: v,
        }
    }

    fn binary_classification_string_map(v: f32) -> HashMap<String, f32> {
        let mut m = HashMap::with_capacity(7);
        m.insert("BalancedAccuracy".into(), v);
        m.insert("PrecisionPositive".into(), v);
        m.insert("PrecisionNegative".into(), v);
        m.insert("RecallPositive".into(), v);
        m.insert("RecallNegative".into(), v);
        m.insert("Accuracy".into(), v);
        m.insert("F1Score".into(), v);
        m
    }

    fn logistic_regression_runtime(v: f32) -> LogisticRegressionRuntime {
        LogisticRegressionRuntime {
            balanced_accuracy: v,
            precision_positive: v,
            precision_negative: v,
            recall_positive: v,
            recall_negative: v,
            accuracy: v,
            f1_score: v,
            log_loss: v,
        }
    }

    fn logistic_regression_string_map(v: f32) -> HashMap<String, f32> {
        let mut m = HashMap::with_capacity(8);
        m.insert("BalancedAccuracy".into(), v);
        m.insert("PrecisionPositive".into(), v);
        m.insert("PrecisionNegative".into(), v);
        m.insert("RecallPositive".into(), v);
        m.insert("RecallNegative".into(), v);
        m.insert("Accuracy".into(), v);
        m.insert("F1Score".into(), v);
        m.insert("LogLoss".into(), v);
        m
    }

    fn linear_regression_runtime(v: f32) -> LinearRegressionRuntime {
        LinearRegressionRuntime {
            rmse: v,
            mse: v,
            mae: v,
            r_squared: v,
            max_error: v,
            msle: v,
            rmsle: v,
            mape: v,
        }
    }

    fn linear_regression_string_map(v: f32) -> HashMap<String, f32> {
        let mut m = HashMap::with_capacity(8);
        m.insert("RootMeanSquaredError".into(), v);
        m.insert("MeanSquaredError".into(), v);
        m.insert("MeanAbsoluteError".into(), v);
        m.insert("RSquared".into(), v);
        m.insert("MaxError".into(), v);
        m.insert("MeanSquaredLogError".into(), v);
        m.insert("RootMeanSquaredLogError".into(), v);
        m.insert("MeanAbsolutePercentageError".into(), v);
        m
    }

    // --- DataBiasRuntime::runtime_drift_report ---

    #[test]
    fn data_bias_runtime_drift_report_identical_is_zero() {
        let rt = data_bias_runtime(0.5);
        let report = rt.runtime_drift_report(&rt);
        assert_eq!(report.len(), 7);
        for v in report.values() {
            assert!(*v < 1e-5, "expected zero drift, got {v}");
        }
    }

    #[test]
    fn data_bias_runtime_drift_report_contains_all_metric_keys() {
        let baseline = data_bias_runtime(0.2);
        let runtime = data_bias_runtime(0.6);
        let report = runtime.runtime_drift_report(&baseline);
        assert!(report.contains_key(&D::ClassImbalance));
        assert!(report.contains_key(&D::DifferenceInProportionOfLabels));
        assert!(report.contains_key(&D::KlDivergence));
        assert!(report.contains_key(&D::JsDivergence));
        assert!(report.contains_key(&D::LpNorm));
        assert!(report.contains_key(&D::TotalVariationDistance));
        assert!(report.contains_key(&D::KolmogorovSmirnov));
    }

    #[test]
    fn data_bias_runtime_drift_report_nonzero_when_different() {
        let baseline = data_bias_runtime(0.2);
        let runtime = data_bias_runtime(0.6);
        let report = runtime.runtime_drift_report(&baseline);
        assert!(report.values().all(|v| *v > 0.0));
    }

    // --- DataBiasRuntime::runtime_check per metric ---

    // Baseline values are 0.2 throughout; runtime sets the tested metric to 0.5
    // so it exceeds 0.2 * 1.1 = 0.22, triggering a flag.
    fn baseline_02() -> DataBiasRuntime {
        data_bias_runtime(0.2)
    }

    #[test]
    fn data_bias_runtime_check_dpl_flagged() {
        let mut rt = baseline_02();
        rt.dpl = 0.5;
        let result = rt.runtime_check(baseline_02(), 0.1, &[D::DifferenceInProportionOfLabels]);
        assert!(result.contains_key(&D::DifferenceInProportionOfLabels));
    }

    #[test]
    fn data_bias_runtime_check_kl_flagged() {
        let mut rt = baseline_02();
        rt.kl = 0.5;
        let result = rt.runtime_check(baseline_02(), 0.1, &[D::KlDivergence]);
        assert!(result.contains_key(&D::KlDivergence));
    }

    #[test]
    fn data_bias_runtime_check_js_flagged() {
        let mut rt = baseline_02();
        rt.js = 0.5;
        let result = rt.runtime_check(baseline_02(), 0.1, &[D::JsDivergence]);
        assert!(result.contains_key(&D::JsDivergence));
    }

    #[test]
    fn data_bias_runtime_check_lpnorm_flagged() {
        let mut rt = baseline_02();
        rt.lpnorm = 0.5;
        let result = rt.runtime_check(baseline_02(), 0.1, &[D::LpNorm]);
        assert!(result.contains_key(&D::LpNorm));
    }

    #[test]
    fn data_bias_runtime_check_tvd_flagged() {
        let mut rt = baseline_02();
        rt.tvd = 0.5;
        let result = rt.runtime_check(baseline_02(), 0.1, &[D::TotalVariationDistance]);
        assert!(result.contains_key(&D::TotalVariationDistance));
    }

    #[test]
    fn data_bias_runtime_check_ks_flagged() {
        let mut rt = baseline_02();
        rt.ks = 0.5;
        let result = rt.runtime_check(baseline_02(), 0.1, &[D::KolmogorovSmirnov]);
        assert!(result.contains_key(&D::KolmogorovSmirnov));
    }

    #[test]
    fn data_bias_runtime_check_metric_not_in_subset_is_not_reported() {
        // runtime has large drift on CI, but we only check DPL — CI should be absent
        let mut rt = baseline_02();
        rt.ci = 0.9;
        let result = rt.runtime_check(baseline_02(), 0.1, &[D::DifferenceInProportionOfLabels]);
        assert!(!result.contains_key(&D::ClassImbalance));
    }

    // --- ModelBiasRuntime ---

    #[test]
    fn model_bias_runtime_from_analysis_report_happy_path() {
        let rt = ModelBiasRuntime::try_from(model_bias_report(0.5)).unwrap();
        assert_eq!(rt.ddpl, 0.5);
        assert_eq!(rt.ge, 0.5);
    }

    #[test]
    fn model_bias_runtime_from_analysis_report_missing_key_returns_error() {
        let mut report = model_bias_report(0.5);
        report.remove(&M::GeneralizedEntropy);
        assert!(ModelBiasRuntime::try_from(report).is_err());
    }

    #[test]
    fn model_bias_runtime_from_string_map_happy_path() {
        let rt = ModelBiasRuntime::try_from(model_bias_string_map(0.3)).unwrap();
        assert_eq!(rt.ddpl, 0.3);
        assert_eq!(rt.ge, 0.3);
    }

    #[test]
    fn model_bias_runtime_from_string_map_missing_key_returns_error() {
        let mut map = model_bias_string_map(0.3);
        map.remove("DisparateImpact");
        assert!(ModelBiasRuntime::try_from(map).is_err());
    }

    #[test]
    fn model_bias_runtime_generate_report_round_trip() {
        let rt = ModelBiasRuntime::try_from(model_bias_report(0.4)).unwrap();
        let report = rt.generate_report();
        let rt2 = ModelBiasRuntime::try_from(report).unwrap();
        assert_eq!(rt.ddpl, rt2.ddpl);
        assert_eq!(rt.ge, rt2.ge);
    }

    #[test]
    fn model_bias_runtime_check_detects_drift() {
        let baseline = ModelBiasRuntime::try_from(model_bias_report(0.2)).unwrap();
        // ddpl=0.5 > 0.2 * 1.1 → flagged
        let mut drifted = model_bias_report(0.2);
        drifted.insert(M::DifferenceInPositivePredictedLabels, 0.5);
        let runtime = ModelBiasRuntime::try_from(drifted).unwrap();
        let result =
            runtime.runtime_check(baseline, 0.1, &[M::DifferenceInPositivePredictedLabels]);
        assert!(result.contains_key(&M::DifferenceInPositivePredictedLabels));
    }

    #[test]
    fn model_bias_runtime_check_no_drift_within_threshold() {
        let baseline = ModelBiasRuntime::try_from(model_bias_report(0.2)).unwrap();
        // same values → no drift
        let runtime = ModelBiasRuntime::try_from(model_bias_report(0.2)).unwrap();
        let result = runtime.runtime_check(baseline, 0.1, &FULL_MODEL_BIAS_METRICS);
        assert!(result.is_empty());
    }

    #[test]
    fn model_bias_runtime_drift_report_identical_is_zero() {
        let rt = ModelBiasRuntime::try_from(model_bias_report(0.4)).unwrap();
        let rt2 = ModelBiasRuntime::try_from(model_bias_report(0.4)).unwrap();
        let report = rt.runtime_drift_report(&rt2);
        assert_eq!(report.len(), 12);
        for v in report.values() {
            assert!(*v < 1e-5, "expected zero drift, got {v}");
        }
    }

    // --- BinaryClassificationRuntime ---

    #[test]
    fn binary_classification_runtime_from_string_map_happy_path() {
        let rt =
            BinaryClassificationRuntime::try_from(binary_classification_string_map(0.8)).unwrap();
        assert!((rt.accuracy - 0.8).abs() < 1e-5);
    }

    #[test]
    fn binary_classification_runtime_from_string_map_missing_key_returns_error() {
        let mut map = binary_classification_string_map(0.8);
        map.remove("Accuracy");
        assert!(BinaryClassificationRuntime::try_from(map).is_err());
    }

    #[test]
    fn binary_classification_runtime_from_report_missing_key_returns_error() {
        let mut report = binary_classification_runtime(0.8).generate_report();
        report.remove(&C::F1Score);
        assert!(BinaryClassificationRuntime::try_from(&report).is_err());
    }

    #[test]
    fn binary_classification_runtime_drift_report_identical_is_zero() {
        let rt = binary_classification_runtime(0.9);
        let report = rt.runtime_drift_report(&rt);
        assert_eq!(report.len(), 7);
        for v in report.values() {
            assert!(v.abs() < 1e-5, "expected zero drift, got {v}");
        }
    }

    #[test]
    fn binary_classification_runtime_drift_report_reflects_degradation() {
        let baseline = binary_classification_runtime(0.9);
        let runtime = binary_classification_runtime(0.5);
        let report = runtime.runtime_drift_report(&baseline);
        for v in report.values() {
            assert!(*v > 0.0, "expected positive drift");
        }
    }

    // --- LogisticRegressionRuntime ---

    #[test]
    fn logistic_regression_runtime_new_length_mismatch_returns_error() {
        let y_true = vec![1.0_f32, 0.0, 1.0];
        let y_pred = vec![0.9_f32, 0.1];
        assert!(LogisticRegressionRuntime::new(&y_true, &y_pred, 0.5).is_err());
    }

    #[test]
    fn logistic_regression_runtime_generate_report_round_trip() {
        let rt = logistic_regression_runtime(0.7);
        let report = rt.generate_report();
        let rt2 = LogisticRegressionRuntime::try_from(&report).unwrap();
        assert_eq!(rt, rt2);
    }

    #[test]
    fn logistic_regression_runtime_from_report_missing_key_returns_error() {
        let mut report = logistic_regression_runtime(0.7).generate_report();
        report.remove(&C::LogLoss);
        assert!(LogisticRegressionRuntime::try_from(&report).is_err());
    }

    #[test]
    fn logistic_regression_runtime_from_string_map_happy_path() {
        let rt = LogisticRegressionRuntime::try_from(logistic_regression_string_map(0.6)).unwrap();
        assert!((rt.accuracy - 0.6).abs() < 1e-5);
        assert!((rt.log_loss - 0.6).abs() < 1e-5);
    }

    #[test]
    fn logistic_regression_runtime_from_string_map_missing_key_returns_error() {
        let mut map = logistic_regression_string_map(0.6);
        map.remove("LogLoss");
        assert!(LogisticRegressionRuntime::try_from(map).is_err());
    }

    #[test]
    fn logistic_regression_runtime_drift_report_identical_is_zero() {
        let rt = logistic_regression_runtime(0.8);
        let rt2 = logistic_regression_runtime(0.8);
        let report = rt.runtime_drift_report(&rt2);
        assert_eq!(report.len(), 8);
        for v in report.values() {
            assert!(v.abs() < 1e-5, "expected zero drift, got {v}");
        }
    }

    // --- LinearRegressionRuntime ---

    #[test]
    fn linear_regression_runtime_from_string_map_happy_path() {
        let rt = LinearRegressionRuntime::try_from(linear_regression_string_map(0.5)).unwrap();
        assert!((rt.mse - 0.5).abs() < 1e-5);
        assert!((rt.r_squared - 0.5).abs() < 1e-5);
    }

    #[test]
    fn linear_regression_runtime_from_string_map_missing_key_returns_error() {
        let mut map = linear_regression_string_map(0.5);
        map.remove("RSquared");
        assert!(LinearRegressionRuntime::try_from(map).is_err());
    }

    #[test]
    fn linear_regression_runtime_from_report_missing_key_returns_error() {
        let mut report = linear_regression_runtime(0.5).generate_report();
        report.remove(&L::MaxError);
        assert!(LinearRegressionRuntime::try_from(&report).is_err());
    }

    #[test]
    fn linear_regression_runtime_drift_report_identical_is_zero() {
        let rt = linear_regression_runtime(0.5);
        let rt2 = linear_regression_runtime(0.5);
        let report = rt.runtime_drift_report(&rt2);
        assert_eq!(report.len(), 8);
        for v in report.values() {
            assert!(v.abs() < 1e-5, "expected zero drift, got {v}");
        }
    }

    #[test]
    fn linear_regression_runtime_drift_report_nonzero_when_different() {
        let baseline = linear_regression_runtime(0.1);
        let runtime = linear_regression_runtime(0.5);
        let report = runtime.runtime_drift_report(&baseline);
        for v in report.values() {
            assert!(*v > 0.0, "expected nonzero drift");
        }
    }
}

#[cfg(test)]
mod test_runtime_containers {
    use super::*;
    use crate::data_bias::PreTraining;
    use crate::data_handler::{BiasSegmentationCriteria, BiasSegmentationType, ConfusionMatrix};
    use crate::model_bias::{PostTraining, PostTrainingDistribution};

    #[test]
    fn model_bias_runtime_from_parts() {
        /*
         * ddpl: (10 / 19) - (8 / 18)
         */
        let pt = PostTraining {
            confusion_a: ConfusionMatrix {
                true_p: 4_f32,
                true_n: 6_f32,
                false_p: 5_f32,
                false_n: 4_f32,
            },
            confusion_d: ConfusionMatrix {
                true_p: 5_f32,
                true_n: 4_f32,
                false_p: 3_f32,
                false_n: 6_f32,
            },
            dist_a: PostTrainingDistribution {
                len: 19,
                positive_pred: 10,
                positive_gt: 8,
            },
            dist_d: PostTrainingDistribution {
                len: 18,
                positive_pred: 8,
                positive_gt: 8,
            },
        };

        let mb_rt = super::ModelBiasRuntime::new_from_post_training(&pt, 1_f32).unwrap();
        assert_eq!(mb_rt.ddpl, 0.08187136_f32);
        assert_eq!(mb_rt.di, 1.184210526_f32);
        assert_eq!(mb_rt.ad, (10_f32 / 19_f32) - (9_f32 / 18_f32));
        assert_eq!(mb_rt.rd, (4_f32 / 8_f32) - (5_f32 / 11_f32));
        assert_eq!(mb_rt.cdacc, (10_f32 / 8_f32) - 1_f32);
        assert_eq!(mb_rt.dar, (4_f32 / 9_f32) - (5_f32 / 8_f32));
        assert_eq!(mb_rt.sd, (6_f32 / 11_f32) - (4_f32 / 7_f32));
    }

    #[test]
    fn data_bias_runtime_from_pretraining() {
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

        let pretraining = PreTraining::new_from_segmentation(
            &feature_data,
            &BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            &BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let test = DataBiasRuntime::new_from_pre_training(&pretraining).unwrap();
        assert_eq!(test, base);
    }
}
