use crate::{
    data_bias::PreTraining,
    data_handler::{bool_to_f32, ApplyThreshold, ConfusionMatrix},
    errors::{DataBiasRuntimeError, ModelBiasRuntimeError, ModelPerformanceError},
    metrics::{
        ClassificationEvaluationMetric, DataBiasMetric, LinearRegressionEvaluationMetric,
        ModelBiasMetric,
    },
    model_bias::PostTraining,
    model_perf::streaming::LinearRegressionErrorBuckets,
    reporting::{
        BinaryClassificationAnalysisReport, BinaryClassificationRuntimeReport,
        DataBiasAnalysisReport, DataBiasRuntimeReport, LinearRegressionAnalysisReport,
        LinearRegressionRuntimeReport, LogisticRegressionAnalysisReport,
        LogisticRegressionRuntimeReport, ModelBiasAnalysisReport, ModelBiasRuntimeReport,
    },
    zip_iters,
};
use std::collections::HashMap;

pub struct DataBiasRuntime {
    ci: f32,
    dpl: f32,
    kl: f32,
    js: f32,
    lpnorm: f32,
    tvd: f32,
    ks: f32,
}

impl DataBiasRuntime {
    pub(crate) fn new_from_pre_training(pre_training: &PreTraining) -> DataBiasRuntime {
        use crate::data_bias::statistics::inner as metrics;
        DataBiasRuntime {
            ci: metrics::class_imbalance(&pre_training),
            dpl: metrics::diff_in_proportion_of_labels(&pre_training),
            kl: metrics::kl_divergence(&pre_training),
            js: metrics::jensen_shannon(&pre_training),
            lpnorm: metrics::lp_norm(&pre_training),
            tvd: metrics::total_variation_distance(&pre_training),
            ks: metrics::kolmorogv_smirnov(&pre_training),
        }
    }
}

impl TryFrom<HashMap<String, f32>> for DataBiasRuntime {
    type Error = DataBiasRuntimeError;
    fn try_from(data: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let ci = match data.get("ClassImbalance") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::ClassImbalance),
        };
        let dpl = match data.get("DifferenceInProportionOfLabels") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::DifferenceInProportionOfLabels),
        };
        let kl = match data.get("KlDivergence") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::KlDivergence),
        };
        let js = match data.get("JsDivergence") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::JsDivergence),
        };
        let lpnorm = match data.get("LpNorm") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::LpNorm),
        };
        let tvd = match data.get("TotalVarationDistance") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::TotalVariationDistance),
        };
        let ks = match data.get("KolmorogvSmirnov") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::KolmorogvSmirnov),
        };

        Ok(DataBiasRuntime {
            ci,
            dpl,
            kl,
            js,
            lpnorm,
            tvd,
            ks,
        })
    }
}

impl TryFrom<DataBiasAnalysisReport> for DataBiasRuntime {
    type Error = DataBiasRuntimeError;
    fn try_from(data: DataBiasAnalysisReport) -> Result<Self, Self::Error> {
        use DataBiasMetric as D;
        let ci = match data.get(&D::ClassImbalance) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::ClassImbalance),
        };
        let dpl = match data.get(&D::DifferenceInProportionOfLabels) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::DifferenceInProportionOfLabels),
        };
        let kl = match data.get(&D::KlDivergence) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::KlDivergence),
        };
        let js = match data.get(&D::JsDivergence) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::JsDivergence),
        };
        let lpnorm = match data.get(&D::LpNorm) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::LpNorm),
        };
        let tvd = match data.get(&D::TotalVariationDistance) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::TotalVariationDistance),
        };
        let ks = match data.get(&D::KolmorogvSmirnov) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::KolmorogvSmirnov),
        };

        Ok(DataBiasRuntime {
            ci,
            dpl,
            kl,
            js,
            lpnorm,
            tvd,
            ks,
        })
    }
}

impl DataBiasRuntime {
    pub fn runtime_check(
        &self,
        baseline: DataBiasRuntime,
        threshold: f32,
        metrics: &[DataBiasMetric],
    ) -> DataBiasRuntimeReport {
        let mut result: HashMap<DataBiasMetric, f32> = HashMap::with_capacity(metrics.len());
        for m in metrics {
            match m {
                DataBiasMetric::ClassImbalance => {
                    if self.ci.abs() > baseline.ci.abs() * (1_f32 + threshold) {
                        result.insert(
                            DataBiasMetric::ClassImbalance,
                            (self.ci.abs() - baseline.ci.abs()).abs(),
                        );
                    }
                }
                DataBiasMetric::DifferenceInProportionOfLabels => {
                    if self.dpl.abs() > baseline.dpl.abs() * (1_f32 + threshold) {
                        result.insert(
                            DataBiasMetric::DifferenceInProportionOfLabels,
                            (self.dpl.abs() - baseline.dpl.abs()).abs(),
                        );
                    }
                }
                DataBiasMetric::KlDivergence => {
                    if self.kl > baseline.kl * (1_f32 + threshold) {
                        result.insert(DataBiasMetric::KlDivergence, self.kl - baseline.kl);
                    }
                }
                DataBiasMetric::JsDivergence => {
                    if self.js > baseline.js * (1_f32 + threshold) {
                        result.insert(DataBiasMetric::JsDivergence, self.kl - baseline.kl);
                    }
                }
                DataBiasMetric::LpNorm => {
                    if self.lpnorm > baseline.lpnorm * (1_f32 + threshold) {
                        result.insert(DataBiasMetric::LpNorm, self.lpnorm - baseline.lpnorm);
                    }
                }
                DataBiasMetric::TotalVariationDistance => {
                    if self.tvd > baseline.tvd * (1_f32 + threshold) {
                        result.insert(
                            DataBiasMetric::TotalVariationDistance,
                            self.tvd - baseline.tvd,
                        );
                    }
                }
                DataBiasMetric::KolmorogvSmirnov => {
                    if self.ks > baseline.ks * (1_f32 + threshold) {
                        result.insert(DataBiasMetric::KolmorogvSmirnov, self.tvd - baseline.tvd);
                    }
                }
            }
        }
        result
    }

    pub fn runtime_drift_report(&self, baseline: &DataBiasRuntime) -> DataBiasRuntimeReport {
        let mut result = DataBiasRuntimeReport::with_capacity(7);
        result.insert(
            DataBiasMetric::ClassImbalance,
            ((self.ci - baseline.ci).abs()) / baseline.ci.abs(),
        );
        result.insert(
            DataBiasMetric::DifferenceInProportionOfLabels,
            ((self.dpl - baseline.dpl).abs()) / baseline.dpl.abs(),
        );
        result.insert(
            DataBiasMetric::KlDivergence,
            (self.kl - baseline.kl).abs() / baseline.kl.abs(),
        );
        result.insert(
            DataBiasMetric::JsDivergence,
            (self.js - baseline.js).abs() / baseline.js.abs(),
        );
        result.insert(
            DataBiasMetric::LpNorm,
            (self.lpnorm - baseline.lpnorm).abs() / baseline.lpnorm.abs(),
        );
        result.insert(
            DataBiasMetric::TotalVariationDistance,
            (self.tvd - baseline.tvd).abs() / baseline.tvd.abs(),
        );
        result.insert(
            DataBiasMetric::KolmorogvSmirnov,
            (self.ks - baseline.ks).abs() / baseline.ks.abs(),
        );

        result
    }

    pub(crate) fn generate_report(&self) -> DataBiasAnalysisReport {
        let mut result = DataBiasAnalysisReport::with_capacity(7);
        result.insert(DataBiasMetric::ClassImbalance, self.ci);
        result.insert(DataBiasMetric::DifferenceInProportionOfLabels, self.dpl);
        result.insert(DataBiasMetric::KlDivergence, self.kl);
        result.insert(DataBiasMetric::JsDivergence, self.js);
        result.insert(DataBiasMetric::LpNorm, self.lpnorm);
        result.insert(DataBiasMetric::TotalVariationDistance, self.tvd);
        result.insert(DataBiasMetric::KolmorogvSmirnov, self.ks);
        result
    }
}

pub struct ModelBiasRuntime {
    ddpl: f32,
    di: f32,
    ad: f32,
    rd: f32,
    cdacc: f32,
    dar: f32,
    sd: f32,
    dcr: f32,
    drr: f32,
    te: f32,
    ccdpl: f32,
    ge: f32,
}

impl ModelBiasRuntime {
    pub(crate) fn new_from_post_training(
        post_training: &PostTraining,
        ge: f32,
    ) -> ModelBiasRuntime {
        use crate::model_bias::statistics::inner as stats;

        ModelBiasRuntime {
            ddpl: stats::diff_in_pos_proportion_in_pred_labels(post_training),
            di: stats::disparate_impact(post_training),
            ad: stats::accuracy_difference(post_training),
            rd: stats::recall_difference(post_training),
            cdacc: stats::diff_in_cond_acceptance(post_training),
            dar: stats::diff_in_acceptance_rate(post_training),
            sd: stats::specailty_difference(post_training),
            dcr: stats::diff_in_cond_rejection(post_training),
            drr: stats::diff_in_rejection_rate(post_training),
            te: stats::treatment_equity(post_training),
            ccdpl: stats::cond_dem_desp_in_pred_labels(post_training),
            ge,
        }
    }

    pub(crate) fn runtime_drift_report(&self, bl: &Self) -> ModelBiasRuntimeReport {
        let mut report = ModelBiasRuntimeReport::with_capacity(12);
        report.insert(
            ModelBiasMetric::DifferenceInPositivePredictedLabels,
            (self.ddpl - bl.ddpl).abs() / bl.ddpl.abs(),
        );
        report.insert(
            ModelBiasMetric::DisparateImpact,
            (self.di - bl.di).abs() / bl.di.abs(),
        );
        report.insert(
            ModelBiasMetric::AccuracyDifference,
            (self.ad - bl.ad).abs() / bl.ad.abs(),
        );
        report.insert(
            ModelBiasMetric::RecallDifference,
            (self.rd - bl.rd).abs() / bl.rd.abs(),
        );
        report.insert(
            ModelBiasMetric::ConditionalDemographicDesparityPredictedLabels,
            (self.cdacc - bl.cdacc).abs() / bl.cdacc.abs(),
        );
        report.insert(
            ModelBiasMetric::DifferenceInAcceptanceRate,
            (self.dar - bl.dar).abs() / bl.dar.abs(),
        );
        report.insert(
            ModelBiasMetric::SpecialityDifference,
            (self.sd - bl.sd).abs() / bl.sd.abs(),
        );
        report.insert(
            ModelBiasMetric::DifferenceInConditionalRejection,
            (self.dcr - bl.dcr).abs() / bl.dcr.abs(),
        );
        report.insert(
            ModelBiasMetric::DifferenceInRejectionRate,
            (self.drr - bl.drr).abs() / bl.drr.abs(),
        );
        report.insert(
            ModelBiasMetric::TreatmentEquity,
            (self.te - bl.te).abs() / bl.te.abs(),
        );
        report.insert(
            ModelBiasMetric::ConditionalDemographicDesparityPredictedLabels,
            (self.ccdpl - bl.ccdpl).abs() / bl.ccdpl.abs(),
        );
        report.insert(
            ModelBiasMetric::GeneralizedEntropy,
            (self.ge - bl.ge).abs() / bl.ge.abs(),
        );
        report
    }

    pub(crate) fn generate_report(&self) -> ModelBiasAnalysisReport {
        let mut report = ModelBiasAnalysisReport::with_capacity(12);
        report.insert(
            ModelBiasMetric::DifferenceInPositivePredictedLabels,
            self.ddpl,
        );
        report.insert(ModelBiasMetric::DisparateImpact, self.di);
        report.insert(ModelBiasMetric::AccuracyDifference, self.ad);
        report.insert(ModelBiasMetric::RecallDifference, self.rd);
        report.insert(
            ModelBiasMetric::ConditionalDemographicDesparityPredictedLabels,
            self.cdacc,
        );
        report.insert(ModelBiasMetric::DifferenceInAcceptanceRate, self.dar);
        report.insert(ModelBiasMetric::SpecialityDifference, self.sd);
        report.insert(ModelBiasMetric::DifferenceInConditionalRejection, self.dcr);
        report.insert(ModelBiasMetric::DifferenceInRejectionRate, self.drr);
        report.insert(ModelBiasMetric::TreatmentEquity, self.te);
        report.insert(
            ModelBiasMetric::ConditionalDemographicDesparityPredictedLabels,
            self.ccdpl,
        );
        report.insert(ModelBiasMetric::GeneralizedEntropy, self.ge);
        report
    }
}

impl TryFrom<ModelBiasAnalysisReport> for ModelBiasRuntime {
    type Error = ModelBiasRuntimeError;
    fn try_from(data: ModelBiasAnalysisReport) -> Result<Self, Self::Error> {
        use ModelBiasMetric as M;
        let ddpl = match data.get(&M::DifferenceInPositivePredictedLabels) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInPositivePredictedLabels),
        };
        let di = match data.get(&M::DisparateImpact) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DisparateImpact),
        };
        let ad = match data.get(&M::AccuracyDifference) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::AccuracyDifference),
        };
        let rd = match data.get(&M::RecallDifference) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::RecallDifference),
        };
        let cdacc = match data.get(&M::DifferenceInConditionalAcceptance) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInConditionalAcceptance),
        };
        let dar = match data.get(&M::DifferenceInAcceptanceRate) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInAcceptanceRate),
        };
        let sd = match data.get(&M::SpecialityDifference) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::SpecialityDifference),
        };
        let dcr = match data.get(&M::DifferenceInConditionalRejection) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInConditionalRejection),
        };
        let drr = match data.get(&M::DifferenceInRejectionRate) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInRejectionRate),
        };
        let te = match data.get(&M::TreatmentEquity) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::TreatmentEquity),
        };
        let ccdpl = match data.get(&M::ConditionalDemographicDesparityPredictedLabels) {
            Some(val) => *val,
            None => {
                return Err(ModelBiasRuntimeError::ConditionalDemographicDesparityPredictedLabels)
            }
        };
        let ge = match data.get(&M::GeneralizedEntropy) {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::GeneralizedEntropy),
        };
        Ok(ModelBiasRuntime {
            ddpl,
            di,
            ad,
            rd,
            cdacc,
            dar,
            sd,
            dcr,
            drr,
            te,
            ccdpl,
            ge,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for ModelBiasRuntime {
    type Error = ModelBiasRuntimeError;
    fn try_from(data: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let ddpl = match data.get("DifferenceInPositivePredictedLabels") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInPositivePredictedLabels),
        };
        let di = match data.get("DisparateImpact") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DisparateImpact),
        };
        let ad = match data.get("AccuracyDifference") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::AccuracyDifference),
        };
        let rd = match data.get("RecallDifference") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::RecallDifference),
        };
        let cdacc = match data.get("DifferenceInConditionalAcceptance") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInConditionalAcceptance),
        };
        let dar = match data.get("DifferenceInAcceptanceRate") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInAcceptanceRate),
        };
        let sd = match data.get("SpecialityDifference") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::SpecialityDifference),
        };
        let dcr = match data.get("DifferenceInConditionalRejection") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInConditionalRejection),
        };
        let drr = match data.get("DifferenceInRejectionRate") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::DifferenceInRejectionRate),
        };
        let te = match data.get("TreatmentEquity") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::TreatmentEquity),
        };
        let ccdpl = match data.get("ConditionalDemographicDesparityPredictedLabels") {
            Some(val) => *val,
            None => {
                return Err(ModelBiasRuntimeError::ConditionalDemographicDesparityPredictedLabels)
            }
        };
        let ge = match data.get("GeneralizedEntropy") {
            Some(val) => *val,
            None => return Err(ModelBiasRuntimeError::GeneralizedEntropy),
        };
        Ok(ModelBiasRuntime {
            ddpl,
            di,
            ad,
            rd,
            cdacc,
            dar,
            sd,
            dcr,
            drr,
            te,
            ccdpl,
            ge,
        })
    }
}

impl ModelBiasRuntime {
    pub fn runtime_check(
        &self,
        baseline: ModelBiasRuntime,
        threshold: f32,
        metrics: &[ModelBiasMetric],
    ) -> ModelBiasRuntimeReport {
        use ModelBiasMetric as M;
        let mut result: HashMap<ModelBiasMetric, f32> = HashMap::with_capacity(metrics.len());
        for m in metrics {
            match m {
                ModelBiasMetric::DifferenceInPositivePredictedLabels => {
                    if self.ddpl.abs() > baseline.ddpl.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::DifferenceInPositivePredictedLabels,
                            (self.ddpl.abs() - baseline.ddpl.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::DisparateImpact => {
                    if self.di > baseline.di * (1_f32 + threshold) {
                        result.insert(M::DisparateImpact, (self.di - baseline.di).abs());
                    }
                }
                ModelBiasMetric::AccuracyDifference => {
                    if self.ad.abs() > baseline.ad.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::AccuracyDifference,
                            (self.ad.abs() - baseline.ad.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::RecallDifference => {
                    if self.rd.abs() > baseline.rd.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::RecallDifference,
                            (self.rd.abs() - baseline.rd.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::DifferenceInConditionalAcceptance => {
                    if self.cdacc.abs() > baseline.cdacc.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::DifferenceInConditionalAcceptance,
                            (self.cdacc.abs() - baseline.cdacc.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::DifferenceInAcceptanceRate => {
                    if self.dar.abs() > baseline.dar.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::DifferenceInAcceptanceRate,
                            (self.dar.abs() - baseline.dar.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::SpecialityDifference => {
                    if self.sd.abs() > baseline.sd.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::SpecialityDifference,
                            (self.sd.abs() - baseline.sd.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::DifferenceInConditionalRejection => {
                    if self.dcr.abs() > baseline.dcr.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::DifferenceInConditionalRejection,
                            (self.dcr.abs() - baseline.dcr.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::DifferenceInRejectionRate => {
                    if self.drr.abs() > baseline.drr.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::DifferenceInRejectionRate,
                            (self.drr.abs() - baseline.drr.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::TreatmentEquity => {
                    if self.te.abs() > baseline.te.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::TreatmentEquity,
                            (self.te.abs() - baseline.te.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::ConditionalDemographicDesparityPredictedLabels => {
                    if self.ccdpl.abs() > baseline.ccdpl.abs() * (1_f32 + threshold) {
                        result.insert(
                            M::ConditionalDemographicDesparityPredictedLabels,
                            (self.ccdpl.abs() - baseline.ccdpl.abs()).abs(),
                        );
                    }
                }
                ModelBiasMetric::GeneralizedEntropy => {
                    if self.ge > baseline.ge * (1_f32 + threshold) {
                        result.insert(M::GeneralizedEntropy, (self.ge - baseline.ge).abs());
                    }
                }
            }
        }

        result
    }
}

pub struct BinaryClassificationRuntime {
    balanced_accuracy: f32,
    precision_positive: f32,
    precision_negative: f32,
    recall_positive: f32,
    recall_negative: f32,
    accuracy: f32,
    f1_score: f32,
}

impl BinaryClassificationRuntime {
    pub fn new<T>(
        y_true: &[T],
        y_pred: &[T],
        label: &T,
    ) -> Result<BinaryClassificationRuntime, ModelPerformanceError>
    where
        T: PartialOrd,
    {
        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        for (t, p) in zip_iters!(y_true, y_pred) {
            let is_positive = p.eq(label);
            let is_true = *p == *t;
            c_matrix.true_p += bool_to_f32(is_true && is_positive);
            c_matrix.false_p += bool_to_f32(!is_true && is_positive);
            c_matrix.true_n += bool_to_f32(is_true && !is_positive);
            c_matrix.false_n += bool_to_f32(!is_true && !is_positive);
        }

        let accuracy = metrics::accuracy(y_true, y_pred)?;
        let balanced_accuracy = metrics::balanced_accuracy(&c_matrix);
        let precision_positive = metrics::precision_positive(&c_matrix);
        let precision_negative = metrics::precision_negative(&c_matrix);
        let recall_positive = metrics::recall_positive(&c_matrix);
        let recall_negative = metrics::recall_negative(&c_matrix);
        let f1_score = metrics::f1_score(&c_matrix);

        Ok(BinaryClassificationRuntime {
            balanced_accuracy,
            precision_positive,
            precision_negative,
            recall_positive,
            recall_negative,
            accuracy,
            f1_score,
        })
    }

    // Utlity to easliy compute the current model performance runtime state from the bucketing
    // style containers used in the stream variants
    pub(crate) fn runtime_from_parts(
        c_matrix: &ConfusionMatrix,
        accuracy: f32,
    ) -> BinaryClassificationRuntime {
        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;

        let balanced_accuracy = metrics::balanced_accuracy(c_matrix);
        let precision_positive = metrics::precision_positive(c_matrix);
        let precision_negative = metrics::precision_negative(c_matrix);
        let recall_positive = metrics::recall_positive(c_matrix);
        let recall_negative = metrics::recall_negative(c_matrix);
        let f1_score = metrics::f1_score(c_matrix);

        BinaryClassificationRuntime {
            balanced_accuracy,
            precision_positive,
            precision_negative,
            recall_positive,
            recall_negative,
            accuracy,
            f1_score,
        }
    }

    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> BinaryClassificationRuntimeReport {
        use ClassificationEvaluationMetric as C;
        let mut res: HashMap<C, f32> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 + drift_threshold;
        // log loss should not be present here
        // so when log loss comes up, we return Err
        for m in metrics.iter() {
            match *m {
                C::BalancedAccuracy => {
                    if (self.balanced_accuracy * drift_factor) < baseline.balanced_accuracy {
                        res.insert(
                            C::BalancedAccuracy,
                            baseline.balanced_accuracy - self.balanced_accuracy,
                        );
                    }
                }
                C::PrecisionPositive => {
                    if (self.precision_positive * drift_factor) < baseline.precision_positive {
                        res.insert(
                            C::PrecisionPositive,
                            baseline.precision_positive - self.precision_positive,
                        );
                    }
                }
                C::PrecisionNegative => {
                    if (self.precision_negative * drift_factor) < baseline.precision_negative {
                        res.insert(
                            C::PrecisionNegative,
                            baseline.precision_negative - self.precision_negative,
                        );
                    }
                }
                C::RecallPositive => {
                    if (self.recall_positive * drift_factor) < baseline.recall_positive {
                        res.insert(
                            C::RecallPositive,
                            baseline.recall_positive - self.recall_positive,
                        );
                    }
                }
                C::RecallNegative => {
                    if (self.recall_negative * drift_factor) < baseline.recall_negative {
                        res.insert(
                            C::RecallNegative,
                            baseline.recall_negative - self.recall_negative,
                        );
                    }
                }
                C::Accuracy => {
                    if (self.accuracy * drift_factor) < baseline.accuracy {
                        res.insert(C::Accuracy, baseline.accuracy - self.accuracy);
                    }
                }
                C::F1Score => {
                    if (self.f1_score * drift_factor) < baseline.f1_score {
                        res.insert(C::F1Score, baseline.f1_score - self.f1_score);
                    }
                }
                _ => continue,
            }
        }

        res
    }

    pub(crate) fn runtime_drift_report(
        &self,
        baseline: &BinaryClassificationRuntime,
    ) -> BinaryClassificationRuntimeReport {
        use crate::metrics::ClassificationEvaluationMetric as C;
        let mut report = BinaryClassificationRuntimeReport::with_capacity(7);
        report.insert(
            C::BalancedAccuracy,
            baseline.balanced_accuracy - self.balanced_accuracy,
        );
        report.insert(
            C::PrecisionPositive,
            baseline.precision_positive - self.precision_positive,
        );
        report.insert(
            C::PrecisionNegative,
            baseline.precision_negative - self.precision_negative,
        );

        report.insert(
            C::RecallPositive,
            baseline.recall_positive - self.recall_positive,
        );
        report.insert(
            C::RecallNegative,
            baseline.recall_negative - self.recall_negative,
        );
        report.insert(C::Accuracy, baseline.accuracy - self.accuracy);
        report.insert(C::F1Score, baseline.f1_score - self.f1_score);

        report
    }
}

impl BinaryClassificationRuntime {
    pub fn generate_report(&self) -> BinaryClassificationAnalysisReport {
        use ClassificationEvaluationMetric as C;
        let mut map: HashMap<C, f32> = HashMap::with_capacity(7);
        map.insert(C::BalancedAccuracy, self.balanced_accuracy);
        map.insert(C::PrecisionPositive, self.precision_positive);
        map.insert(C::PrecisionNegative, self.precision_negative);
        map.insert(C::RecallPositive, self.recall_positive);
        map.insert(C::RecallNegative, self.recall_negative);
        map.insert(C::Accuracy, self.accuracy);
        map.insert(C::F1Score, self.f1_score);
        map
    }
}

impl TryFrom<&BinaryClassificationAnalysisReport> for BinaryClassificationRuntime {
    type Error = ModelPerformanceError;
    fn try_from(payload: &BinaryClassificationAnalysisReport) -> Result<Self, Self::Error> {
        use ClassificationEvaluationMetric as C;
        let value_fetcher = |p: &BinaryClassificationAnalysisReport, key: C| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };

            Ok(*v)
        };
        Ok(BinaryClassificationRuntime {
            balanced_accuracy: value_fetcher(&payload, C::BalancedAccuracy)?,
            precision_positive: value_fetcher(&payload, C::PrecisionPositive)?,
            precision_negative: value_fetcher(&payload, C::PrecisionNegative)?,
            recall_positive: value_fetcher(&payload, C::RecallPositive)?,
            recall_negative: value_fetcher(&payload, C::RecallNegative)?,
            accuracy: value_fetcher(&payload, C::Accuracy)?,
            f1_score: value_fetcher(&payload, C::F1Score)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for BinaryClassificationRuntime {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };

            Ok(v)
        };

        Ok(BinaryClassificationRuntime {
            balanced_accuracy: value_fetcher(&mut payload, "BalancedAccuracy")?,
            precision_positive: value_fetcher(&mut payload, "PrecisionPositive")?,
            precision_negative: value_fetcher(&mut payload, "PrecisionNegative")?,
            recall_positive: value_fetcher(&mut payload, "RecallPositive")?,
            recall_negative: value_fetcher(&mut payload, "RecallNegative")?,
            accuracy: value_fetcher(&mut payload, "Accuracy")?,
            f1_score: value_fetcher(&mut payload, "F1Score")?,
        })
    }
}

pub struct LogisticRegressionRuntime {
    balanced_accuracy: f32,
    precision_positive: f32,
    precision_negative: f32,
    recall_positive: f32,
    recall_negative: f32,
    accuracy: f32,
    f1_score: f32,
    log_loss: f32,
}

// assume that positive label is 1
impl LogisticRegressionRuntime {
    pub(crate) fn new(
        y_true: &[f32],
        y_pred: &[f32],
        threshold: f32,
    ) -> Result<LogisticRegressionRuntime, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        for (t, p) in zip_iters!(y_true, y_pred) {
            let label = p.apply_threshold(&threshold);
            let is_positive = label == 1_f32;
            let is_true = label == *t;

            c_matrix.true_p += bool_to_f32(is_true && is_positive);
            c_matrix.false_p += bool_to_f32(!is_true && is_positive);
            c_matrix.true_n += bool_to_f32(is_true && !is_positive);
            c_matrix.false_n += bool_to_f32(!is_true && !is_positive);
        }

        let accuracy = metrics::accuracy(y_true, y_pred)?;
        let balanced_accuracy = metrics::balanced_accuracy(&c_matrix);
        let precision_positive = metrics::precision_positive(&c_matrix);
        let precision_negative = metrics::precision_negative(&c_matrix);
        let recall_positive = metrics::recall_positive(&c_matrix);
        let recall_negative = metrics::recall_negative(&c_matrix);
        let f1_score = metrics::f1_score(&c_matrix);
        let log_loss = metrics::log_loss_score(y_true, y_pred)?;

        Ok(LogisticRegressionRuntime {
            balanced_accuracy,
            precision_positive,
            precision_negative,
            recall_positive,
            recall_negative,
            accuracy,
            f1_score,
            log_loss,
        })
    }
    // Utlity to easliy compute the current model performance runtime state from the bucketing
    // style containers used in the stream variants
    pub(crate) fn runtime_from_parts(
        c_matrix: &ConfusionMatrix,
        accuracy: f32,
        log_loss: f32,
    ) -> Result<LogisticRegressionRuntime, ModelPerformanceError> {
        if c_matrix.len() == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        };
        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;
        Ok(LogisticRegressionRuntime {
            balanced_accuracy: metrics::balanced_accuracy(c_matrix),
            precision_positive: metrics::precision_positive(c_matrix),
            precision_negative: metrics::precision_negative(c_matrix),
            recall_positive: metrics::recall_positive(c_matrix),
            recall_negative: metrics::recall_negative(c_matrix),
            f1_score: metrics::f1_score(c_matrix),
            log_loss,
            accuracy,
        })
    }

    pub(crate) fn runtime_drift_report(&self, bl: &Self) -> LogisticRegressionRuntimeReport {
        let mut report = LogisticRegressionAnalysisReport::with_capacity(8);
        report.insert(
            ClassificationEvaluationMetric::Accuracy,
            bl.accuracy - self.accuracy,
        );
        report.insert(
            ClassificationEvaluationMetric::BalancedAccuracy,
            bl.balanced_accuracy - self.balanced_accuracy,
        );
        report.insert(
            ClassificationEvaluationMetric::PrecisionPositive,
            bl.precision_positive - self.precision_positive,
        );
        report.insert(
            ClassificationEvaluationMetric::PrecisionNegative,
            bl.precision_negative - self.precision_negative,
        );
        report.insert(
            ClassificationEvaluationMetric::RecallPositive,
            bl.recall_positive - self.recall_positive,
        );
        report.insert(
            ClassificationEvaluationMetric::RecallNegative,
            bl.recall_negative - self.recall_negative,
        );
        report.insert(
            ClassificationEvaluationMetric::F1Score,
            bl.f1_score - self.f1_score,
        );
        report.insert(
            ClassificationEvaluationMetric::LogLoss,
            bl.log_loss - self.log_loss,
        );
        todo!()
    }
}

impl LogisticRegressionRuntime {
    pub fn generate_report(&self) -> LogisticRegressionAnalysisReport {
        use ClassificationEvaluationMetric as M;
        let mut map: HashMap<M, f32> = HashMap::with_capacity(8);
        map.insert(M::BalancedAccuracy, self.balanced_accuracy);
        map.insert(M::PrecisionPositive, self.precision_positive);
        map.insert(M::PrecisionNegative, self.precision_negative);
        map.insert(M::RecallPositive, self.recall_positive);
        map.insert(M::RecallNegative, self.recall_negative);
        map.insert(M::Accuracy, self.accuracy);
        map.insert(M::F1Score, self.f1_score);
        map.insert(M::LogLoss, self.log_loss);
        map
    }
}

impl TryFrom<&LogisticRegressionAnalysisReport> for LogisticRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(payload: &LogisticRegressionAnalysisReport) -> Result<Self, Self::Error> {
        use ClassificationEvaluationMetric as L;
        let value_fetcher = |p: &LogisticRegressionAnalysisReport, key: L| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(*v)
        };
        Ok(LogisticRegressionRuntime {
            balanced_accuracy: value_fetcher(&payload, L::BalancedAccuracy)?,
            precision_positive: value_fetcher(&payload, L::PrecisionPositive)?,
            precision_negative: value_fetcher(&payload, L::PrecisionNegative)?,
            recall_positive: value_fetcher(&payload, L::RecallPositive)?,
            recall_negative: value_fetcher(&payload, L::RecallNegative)?,
            accuracy: value_fetcher(&payload, L::Accuracy)?,
            f1_score: value_fetcher(&payload, L::F1Score)?,
            log_loss: value_fetcher(&payload, L::LogLoss)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for LogisticRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(v)
        };
        Ok(LogisticRegressionRuntime {
            balanced_accuracy: value_fetcher(&mut payload, "BalancedAccuracy")?,
            precision_positive: value_fetcher(&mut payload, "PrecisionPositive")?,
            precision_negative: value_fetcher(&mut payload, "PrecisionNegative")?,
            recall_positive: value_fetcher(&mut payload, "RecallPositive")?,
            recall_negative: value_fetcher(&mut payload, "RecallNegative")?,
            accuracy: value_fetcher(&mut payload, "Accuracy")?,
            f1_score: value_fetcher(&mut payload, "F1Score")?,
            log_loss: value_fetcher(&mut payload, "LogLoss")?,
        })
    }
}

impl LogisticRegressionRuntime {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> LogisticRegressionRuntimeReport {
        // all the metrics here are used, at this point we have
        // everything correct, thus no Result<T,E>
        use ClassificationEvaluationMetric as C;
        let mut res: HashMap<C, f32> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 - drift_threshold;
        for m in metrics.iter() {
            match *m {
                C::BalancedAccuracy => {
                    if self.balanced_accuracy < baseline.balanced_accuracy * drift_factor {
                        res.insert(
                            C::BalancedAccuracy,
                            baseline.balanced_accuracy - self.balanced_accuracy,
                        );
                    }
                }
                C::PrecisionPositive => {
                    if self.precision_positive < baseline.precision_positive * drift_factor {
                        res.insert(
                            C::PrecisionPositive,
                            baseline.precision_positive - self.precision_positive,
                        );
                    }
                }
                C::PrecisionNegative => {
                    if self.precision_negative < baseline.precision_negative * drift_factor {
                        res.insert(
                            C::PrecisionNegative,
                            baseline.precision_negative - self.precision_negative,
                        );
                    }
                }
                C::RecallPositive => {
                    if self.recall_positive < baseline.recall_positive * drift_factor {
                        res.insert(
                            C::RecallPositive,
                            baseline.recall_positive - self.recall_positive,
                        );
                    }
                }
                C::RecallNegative => {
                    if self.recall_negative < baseline.recall_negative * drift_factor {
                        res.insert(
                            C::RecallNegative,
                            baseline.recall_negative - self.recall_negative,
                        );
                    }
                }
                C::Accuracy => {
                    if self.accuracy < baseline.accuracy * drift_factor {
                        res.insert(C::Accuracy, baseline.accuracy - self.accuracy);
                    }
                }
                C::F1Score => {
                    if self.f1_score < baseline.f1_score * drift_factor {
                        res.insert(C::F1Score, baseline.f1_score - self.f1_score);
                    }
                }
                C::LogLoss => {
                    if self.log_loss < baseline.log_loss * drift_factor {
                        res.insert(C::F1Score, baseline.log_loss - self.log_loss);
                    }
                }
            }
        }
        res
    }
}

pub struct LinearRegressionRuntime {
    rmse: f32,
    mse: f32,
    mae: f32,
    r_squared: f32,
    max_error: f32,
    msle: f32,
    rmsle: f32,
    mape: f32,
}

impl LinearRegressionRuntime {
    pub fn new<T>(
        y_true: &[T],
        y_pred: &[T],
    ) -> Result<LinearRegressionRuntime, ModelPerformanceError>
    where
        T: Into<f64> + Copy,
    {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        if y_true.len() == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let n = y_true.len() as f64;

        // Computing the linear regression metrics inline here rather than pay for an O(n)
        // iteration for every error metric computation.

        let mut squared_error_sum = 0_f64;
        let mut abs_error_sum = 0_f64;
        let mut max_error = 0_f64;
        let mut sqaured_log_error_sum = 0_f64;
        let mut abs_percent_error_sum = 0_f64;
        let mut y_true_sum = 0_f64;

        for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
            let t: f64 = (*t_ref).into();
            let p: f64 = (*p_ref).into();

            y_true_sum += t;
            squared_error_sum += (t - p).powi(2);
            abs_error_sum += (t - p).abs();
            max_error = max_error.max((t - p).abs());
            sqaured_log_error_sum += ((1_f64 + t).log10() - (1_f64 + p).log10()).powi(2);
            abs_percent_error_sum += ((t - p) / t).abs();
        }

        let mut ss_total = 0_f64;
        let y_true_mean = y_true_sum / n;
        for t_ref in y_true.iter() {
            let t: f64 = (*t_ref).into();
            ss_total += (t - y_true_mean).powi(2);
        }

        let mse = squared_error_sum / n;
        let msle = sqaured_log_error_sum / n;

        Ok(LinearRegressionRuntime {
            r_squared: (1_f64 - (squared_error_sum / ss_total)) as f32,
            rmse: (mse).powf(0_f64) as f32,
            mse: mse as f32,
            mae: (abs_error_sum / n) as f32,
            max_error: max_error as f32,
            msle: msle as f32,
            rmsle: (msle.powf(0.5_f64)) as f32,
            mape: (abs_percent_error_sum / n) as f32,
        })
    }

    /// Utlity to easliy compute the current model performance runtime state from the bucketing
    /// style containers used in the stream variants. Acknowledging here the explicit cast from f64
    /// to f32 which may forgoe some precision here.
    pub(crate) fn runtime_from_parts(
        parts: &LinearRegressionErrorBuckets,
    ) -> Result<LinearRegressionRuntime, ModelPerformanceError> {
        let n = parts.len;
        if n == 0_f64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let mse = (parts.squared_error_sum / n) as f32;
        let msle = (parts.squared_log_error_sum / n) as f32;

        Ok(LinearRegressionRuntime {
            r_squared: parts.r2_snapshot() as f32,
            mse,
            rmse: mse.powf(0.5_f32),
            max_error: parts.max_error as f32,
            mae: (parts.abs_error_sum / n) as f32,
            msle,
            rmsle: msle.powf(0.5_f32),
            mape: (parts.abs_percent_error_sum / n) as f32,
        })
    }

    pub fn compare_to_baseline(
        &self,
        metrics: &[LinearRegressionEvaluationMetric],
        baseline: &LinearRegressionRuntime,
        drift_threshold: f32,
    ) -> LinearRegressionRuntimeReport {
        use LinearRegressionEvaluationMetric as L;
        let mut res: HashMap<L, f32> = HashMap::with_capacity(8);
        for m in metrics.iter() {
            // All values should be positive here, so all comparisons are greater than allowable
            // drift threshold define by the user.
            match *m {
                L::RootMeanSquaredError => {
                    if self.rmse > baseline.rmse * (1_f32 + drift_threshold) {
                        res.insert(L::RootMeanSquaredError, self.rmse - baseline.rmse);
                    }
                }
                L::MeanSquaredError => {
                    if self.mse > baseline.mse * (1_f32 + drift_threshold) {
                        res.insert(L::MeanSquaredError, self.mse - baseline.mse);
                    }
                }
                L::MeanAbsoluteError => {
                    if self.mae > baseline.mae * (1_f32 + drift_threshold) {
                        res.insert(L::MeanAbsoluteError, self.mae - baseline.mae);
                    }
                }
                L::RSquared => {
                    if self.r_squared > baseline.r_squared * (1_f32 + drift_threshold) {
                        res.insert(L::RSquared, self.r_squared - baseline.r_squared);
                    }
                }
                L::MaxError => {
                    if self.max_error > baseline.max_error * (1_f32 + drift_threshold) {
                        res.insert(L::MaxError, self.max_error - baseline.max_error);
                    }
                }
                L::MeanSquaredLogError => {
                    if self.msle > baseline.msle * (1_f32 + drift_threshold) {
                        res.insert(L::MeanSquaredLogError, self.msle - baseline.msle);
                    }
                }
                L::RootMeanSquaredLogError => {
                    if self.rmsle > baseline.rmsle * (1_f32 + drift_threshold) {
                        res.insert(L::RootMeanSquaredLogError, self.rmsle - baseline.rmsle);
                    }
                }
                L::MeanAbsolutePercentageError => {
                    if self.mape > baseline.mape * (1_f32 + drift_threshold) {
                        res.insert(L::MeanAbsolutePercentageError, self.mape - baseline.mape);
                    }
                }
            }
        }
        res
    }

    pub fn runtime_drift_report(&self, bl: &Self) -> LinearRegressionRuntimeReport {
        use crate::metrics::LinearRegressionEvaluationMetric as L;
        let mut result = LinearRegressionRuntimeReport::with_capacity(8);
        result.insert(L::RootMeanSquaredError, (bl.rmse - self.rmse).abs());
        result.insert(L::MeanSquaredError, (bl.mse - self.mse).abs());
        result.insert(L::MeanAbsoluteError, (bl.mae - self.mae).abs());
        result.insert(L::RSquared, (bl.r_squared - self.r_squared).abs());
        result.insert(L::MaxError, (bl.max_error - self.max_error).abs());
        result.insert(L::MeanSquaredLogError, (bl.msle - self.msle).abs());
        result.insert(L::RootMeanSquaredLogError, (bl.rmsle - self.rmsle).abs());
        result.insert(L::MeanAbsolutePercentageError, (bl.mape - self.mape).abs());
        result
    }
}

impl TryFrom<&LinearRegressionAnalysisReport> for LinearRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(payload: &LinearRegressionAnalysisReport) -> Result<Self, Self::Error> {
        use LinearRegressionEvaluationMetric as L;
        let value_fetcher = |p: &LinearRegressionAnalysisReport, key: L| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(*v)
        };

        Ok(LinearRegressionRuntime {
            rmse: value_fetcher(&payload, L::RootMeanSquaredError)?,
            mse: value_fetcher(&payload, L::MeanSquaredError)?,
            mae: value_fetcher(&payload, L::MeanAbsoluteError)?,
            r_squared: value_fetcher(&payload, L::RSquared)?,
            max_error: value_fetcher(&payload, L::MaxError)?,
            msle: value_fetcher(&payload, L::MeanSquaredLogError)?,
            rmsle: value_fetcher(&payload, L::RootMeanSquaredLogError)?,
            mape: value_fetcher(&payload, L::MeanAbsolutePercentageError)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for LinearRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(v)
        };

        Ok(LinearRegressionRuntime {
            rmse: value_fetcher(&mut payload, "RootMeanSquaredError")?,
            mse: value_fetcher(&mut payload, "MeanSquaredError")?,
            mae: value_fetcher(&mut payload, "MeanAbsoluteError")?,
            r_squared: value_fetcher(&mut payload, "RSquared")?,
            max_error: value_fetcher(&mut payload, "MaxError")?,
            msle: value_fetcher(&mut payload, "MeanSquaredLogError")?,
            rmsle: value_fetcher(&mut payload, "RootMeanSquaredLogError")?,
            mape: value_fetcher(&mut payload, "MeanAbsolutePercentageError")?,
        })
    }
}

impl LinearRegressionRuntime {
    pub fn generate_report(&self) -> LinearRegressionAnalysisReport {
        use LinearRegressionEvaluationMetric as L;
        let mut map: HashMap<L, f32> = HashMap::with_capacity(8);
        map.insert(L::RootMeanSquaredError, self.rmse);
        map.insert(L::MeanSquaredError, self.mse);
        map.insert(L::MeanAbsoluteError, self.mae);
        map.insert(L::RSquared, self.r_squared);
        map.insert(L::MaxError, self.max_error);
        map.insert(L::MeanSquaredLogError, self.msle);
        map.insert(L::RootMeanSquaredLogError, self.rmsle);
        map.insert(L::MeanAbsolutePercentageError, self.mape);
        map
    }
}

#[cfg(test)]
mod test_runtime_containers {
    use super::*;
    use crate::data_handler::ConfusionMatrix;
    use crate::model_bias::{PostTraining, PostTrainingDistribution};

    #[test]
    fn model_bias_runtime_from_parts() {
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
    }
}
