use crate::{
    data_bias::PreTraining,
    data_handler::{ApplyThreshold, ConfusionMatrix},
    errors::{
        BiasError, DataBiasRuntimeError, ModelBiasRuntimeError, ModelPerfResult,
        ModelPerformanceError,
    },
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
};
use std::collections::HashMap;

pub(crate) const EQUALITY_ERROR_ALLOWANCE: f32 = 1e-5;

#[derive(Debug, PartialEq, Clone)]
pub struct DataBiasRuntime {
    pub(crate) ci: f32,
    pub(crate) dpl: f32,
    pub(crate) kl: f32,
    pub(crate) js: f32,
    pub(crate) lpnorm: f32,
    pub(crate) tvd: f32,
    pub(crate) ks: f32,
}

impl DataBiasRuntime {
    pub(crate) fn new_from_pre_training(
        pre_training: &PreTraining,
    ) -> Result<DataBiasRuntime, BiasError> {
        use crate::data_bias::statistics::inner as metrics;
        Ok(DataBiasRuntime {
            ci: metrics::class_imbalance(pre_training),
            dpl: metrics::diff_in_proportion_of_labels(pre_training)?,
            kl: metrics::kl_divergence(pre_training)?,
            js: metrics::jensen_shannon(pre_training)?,
            lpnorm: metrics::lp_norm(pre_training)?,
            tvd: metrics::total_variation_distance(pre_training)?,
            ks: metrics::kolmogorov_smirnov(pre_training)?,
        })
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
        let ks = match data.get("KolmogorovSmirnov") {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::KolmogorovSmirnov),
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
        let ks = match data.get(&D::KolmogorovSmirnov) {
            Some(val) => *val,
            None => return Err(DataBiasRuntimeError::KolmogorovSmirnov),
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
                        result.insert(DataBiasMetric::JsDivergence, self.js - baseline.js);
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
                DataBiasMetric::KolmogorovSmirnov => {
                    if self.ks > baseline.ks * (1_f32 + threshold) {
                        result.insert(DataBiasMetric::KolmogorovSmirnov, self.ks - baseline.ks);
                    }
                }
            }
        }
        result
    }

    pub fn runtime_drift_report(&self, baseline: &DataBiasRuntime) -> DataBiasRuntimeReport {
        let eps = crate::metrics::get_stability_eps() as f32;
        let mut result = DataBiasRuntimeReport::with_capacity(7);
        result.insert(
            DataBiasMetric::ClassImbalance,
            ((self.ci - baseline.ci).abs()) / baseline.ci.abs().max(eps),
        );
        result.insert(
            DataBiasMetric::DifferenceInProportionOfLabels,
            ((self.dpl - baseline.dpl).abs()) / baseline.dpl.abs().max(eps),
        );
        result.insert(
            DataBiasMetric::KlDivergence,
            (self.kl - baseline.kl).abs() / baseline.kl.abs().max(eps),
        );
        result.insert(
            DataBiasMetric::JsDivergence,
            (self.js - baseline.js).abs() / baseline.js.abs().max(eps),
        );
        result.insert(
            DataBiasMetric::LpNorm,
            (self.lpnorm - baseline.lpnorm).abs() / baseline.lpnorm.abs().max(eps),
        );
        result.insert(
            DataBiasMetric::TotalVariationDistance,
            (self.tvd - baseline.tvd).abs() / baseline.tvd.abs().max(eps),
        );
        result.insert(
            DataBiasMetric::KolmogorovSmirnov,
            (self.ks - baseline.ks).abs() / baseline.ks.abs().max(eps),
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
        result.insert(DataBiasMetric::KolmogorovSmirnov, self.ks);
        result
    }
}

#[derive(Debug, Clone)]
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
    ) -> ModelPerfResult<ModelBiasRuntime> {
        use crate::model_bias::statistics::inner as stats;

        Ok(ModelBiasRuntime {
            ddpl: stats::diff_in_pos_proportion_in_pred_labels(post_training)?,
            di: stats::disparate_impact(post_training)?,
            ad: stats::accuracy_difference(post_training),
            rd: stats::recall_difference(post_training),
            cdacc: stats::diff_in_cond_acceptance(post_training)?,
            dar: stats::diff_in_acceptance_rate(post_training),
            sd: stats::specailty_difference(post_training),
            dcr: stats::diff_in_cond_rejection(post_training)?,
            drr: stats::diff_in_rejection_rate(post_training)?,
            te: stats::treatment_equity(post_training)?,
            ccdpl: stats::cond_dem_desp_in_pred_labels(post_training),
            ge,
        })
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
            ModelBiasMetric::DifferenceInConditionalAcceptance,
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
            ModelBiasMetric::DifferenceInConditionalAcceptance,
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

#[derive(Debug, Clone)]
pub struct BinaryClassificationRuntime {
    pub(crate) balanced_accuracy: f32,
    pub(crate) precision_positive: f32,
    pub(crate) precision_negative: f32,
    pub(crate) recall_positive: f32,
    pub(crate) recall_negative: f32,
    pub(crate) accuracy: f32,
    pub(crate) f1_score: f32,
}

impl PartialEq for BinaryClassificationRuntime {
    fn eq(&self, other: &Self) -> bool {
        if (self.balanced_accuracy - other.balanced_accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_positive - other.precision_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_negative - other.precision_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_positive - other.recall_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_negative - other.recall_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.accuracy - other.accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.f1_score - other.f1_score).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        true
    }
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

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.eq(label))?;

        Ok(BinaryClassificationRuntime {
            balanced_accuracy: metrics::balanced_accuracy(&c_matrix),
            precision_positive: metrics::precision_positive(&c_matrix),
            precision_negative: metrics::precision_negative(&c_matrix),
            recall_positive: metrics::recall_positive(&c_matrix),
            recall_negative: metrics::recall_negative(&c_matrix),
            accuracy: metrics::accuracy(&c_matrix),
            f1_score: metrics::f1_score(&c_matrix),
        })
    }

    // Utlity to easliy compute the current model performance runtime state from the bucketing
    // style containers used in the stream variants
    pub(crate) fn runtime_from_parts(c_matrix: &ConfusionMatrix) -> BinaryClassificationRuntime {
        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;

        BinaryClassificationRuntime {
            balanced_accuracy: metrics::balanced_accuracy(c_matrix),
            precision_positive: metrics::precision_positive(c_matrix),
            precision_negative: metrics::precision_negative(c_matrix),
            recall_positive: metrics::recall_positive(c_matrix),
            recall_negative: metrics::recall_negative(c_matrix),
            accuracy: c_matrix.accuracy(),
            f1_score: metrics::f1_score(c_matrix),
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
            balanced_accuracy: value_fetcher(payload, C::BalancedAccuracy)?,
            precision_positive: value_fetcher(payload, C::PrecisionPositive)?,
            precision_negative: value_fetcher(payload, C::PrecisionNegative)?,
            recall_positive: value_fetcher(payload, C::RecallPositive)?,
            recall_negative: value_fetcher(payload, C::RecallNegative)?,
            accuracy: value_fetcher(payload, C::Accuracy)?,
            f1_score: value_fetcher(payload, C::F1Score)?,
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

#[derive(Debug, Clone)]
pub struct LogisticRegressionRuntime {
    pub(crate) balanced_accuracy: f32,
    pub(crate) precision_positive: f32,
    pub(crate) precision_negative: f32,
    pub(crate) recall_positive: f32,
    pub(crate) recall_negative: f32,
    pub(crate) accuracy: f32,
    pub(crate) f1_score: f32,
    pub(crate) log_loss: f32,
}

impl PartialEq for LogisticRegressionRuntime {
    fn eq(&self, other: &Self) -> bool {
        if (self.balanced_accuracy - other.balanced_accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_positive - other.precision_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_negative - other.precision_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_positive - other.recall_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_negative - other.recall_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.accuracy - other.accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.f1_score - other.f1_score).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.log_loss - other.log_loss).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        true
    }
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

        let true_label = 1_f32;
        c_matrix.push_dataset(y_true, y_pred, |v: &f32| {
            v.apply_threshold(&threshold).eq(&true_label)
        })?;

        /*
                for (t, p) in zip_iters!(y_true, y_pred) {
                    let label = p.apply_threshold(&threshold);

                    c_matrix.push(ConfusionPushPayload {
                        true_gt: t.eq(&true_label),
                        true_pred: label.eq(&true_label),
                    });
                }
        */

        let accuracy = c_matrix.accuracy();
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
            accuracy: c_matrix.accuracy(),
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
        report
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
            balanced_accuracy: value_fetcher(payload, L::BalancedAccuracy)?,
            precision_positive: value_fetcher(payload, L::PrecisionPositive)?,
            precision_negative: value_fetcher(payload, L::PrecisionNegative)?,
            recall_positive: value_fetcher(payload, L::RecallPositive)?,
            recall_negative: value_fetcher(payload, L::RecallNegative)?,
            accuracy: value_fetcher(payload, L::Accuracy)?,
            f1_score: value_fetcher(payload, L::F1Score)?,
            log_loss: value_fetcher(payload, L::LogLoss)?,
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
                    if self.log_loss > baseline.log_loss * (1_f32 + drift_threshold) {
                        res.insert(C::LogLoss, self.log_loss - baseline.log_loss);
                    }
                }
            }
        }
        res
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegressionRuntime {
    pub(crate) rmse: f32,
    pub(crate) mse: f32,
    pub(crate) mae: f32,
    pub(crate) r_squared: f32,
    pub(crate) max_error: f32,
    pub(crate) msle: f32,
    pub(crate) rmsle: f32,
    pub(crate) mape: f32,
}

/// Implementing this by hand to allow for some small error.
impl PartialEq for LinearRegressionRuntime {
    fn eq(&self, other: &Self) -> bool {
        if (self.rmse - other.rmse).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.mse - other.mse).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.mae - other.mae).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.r_squared - other.r_squared).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.max_error - other.max_error).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.msle - other.msle).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.rmsle - other.rmsle).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.mape - other.mape).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        true
    }
}

impl LinearRegressionRuntime {
    pub fn new<T>(
        y_true: &[T],
        y_pred: &[T],
    ) -> Result<LinearRegressionRuntime, ModelPerformanceError>
    where
        T: Into<f64> + Copy,
    {
        let n = y_true.len() as f64;
        let error_buckets = LinearRegressionErrorBuckets::from_dataset(y_true, y_pred)?;

        let LinearRegressionErrorBuckets {
            squared_error_sum,
            abs_error_sum,
            max_error,
            squared_log_error_sum,
            abs_percent_error_sum,
            ..
        } = error_buckets;
        let r_squared = error_buckets.r2_snapshot() as f32;

        let mse = squared_error_sum / n;
        let msle = squared_log_error_sum / n;

        Ok(LinearRegressionRuntime {
            r_squared,
            rmse: (mse).powf(0.5_f64) as f32,
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
            rmsle: msle.sqrt(),
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
                    if self.r_squared < baseline.r_squared * (1_f32 - drift_threshold) {
                        res.insert(L::RSquared, baseline.r_squared - self.r_squared);
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
            rmse: value_fetcher(payload, L::RootMeanSquaredError)?,
            mse: value_fetcher(payload, L::MeanSquaredError)?,
            mae: value_fetcher(payload, L::MeanAbsoluteError)?,
            r_squared: value_fetcher(payload, L::RSquared)?,
            max_error: value_fetcher(payload, L::MaxError)?,
            msle: value_fetcher(payload, L::MeanSquaredLogError)?,
            rmsle: value_fetcher(payload, L::RootMeanSquaredLogError)?,
            mape: value_fetcher(payload, L::MeanAbsolutePercentageError)?,
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
        // intentional typos matching the TryFrom impl
        map.insert("TotalVarationDistance".to_string(), 0.6_f32);
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
