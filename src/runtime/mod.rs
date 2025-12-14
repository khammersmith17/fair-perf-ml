use crate::{
    data_bias::PreTraining,
    data_handler::{ApplyThreshold, ConfusionMatrix},
    errors::{DataBiasRuntimeError, ModelBiasRuntimeError, ModelPerformanceError},
    metrics::{
        ClassificationEvaluationMetric, DataBiasMetric, LinearRegressionEvaluationMetric,
        ModelBiasMetric,
    },
    model_bias::PostTraining,
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

    // TODO: determine how to compute the drift here, ie should it be based on abs diff etc...
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
            ddpl: stats::diff_in_pos_proportion_in_pred_labels(&post_training),
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
        label: T,
    ) -> Result<BinaryClassificationRuntime, ModelPerformanceError>
    where
        T: PartialOrd,
    {
        use crate::model_perf::statistics::classification_metrics as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        for (t, p) in zip_iters!(y_true, y_pred) {
            let is_positive = *p == label;
            let is_true = *p == *t;
            c_matrix.true_p += (is_true && is_positive) as i32 as f32;
            c_matrix.false_p += (!is_true && is_positive) as i32 as f32;
            c_matrix.true_n += (is_true && !is_positive) as i32 as f32;
            c_matrix.false_n += (!is_true && !is_positive) as i32 as f32;
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

    pub(crate) fn from_confusion_and_accuracy(
        c_matrix: &ConfusionMatrix,
        accuracy: f32,
    ) -> BinaryClassificationRuntime {
        use crate::model_perf::statistics::classification_metrics as metrics;

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

impl BinaryClassificationRuntime {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> BinaryClassificationRuntimeReport {
        use ClassificationEvaluationMetric as C;
        let mut res: HashMap<C, f32> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 - drift_threshold;
        // log loss should not be present here
        // so when log loss comes up, we return Err
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
                _ => continue,
            }
        }

        res
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

        use crate::model_perf::statistics::classification_metrics as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        for (t, p) in zip_iters!(y_true, y_pred) {
            let label = p.apply_threshold(threshold);
            let is_positive = label == 1_f32;
            let is_true = label == *t;

            c_matrix.true_p += (is_true && is_positive) as i32 as f32;
            c_matrix.false_p += (!is_true && is_positive) as i32 as f32;
            c_matrix.true_n += (is_true && !is_positive) as i32 as f32;
            c_matrix.false_n += (!is_true && !is_positive) as i32 as f32;
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

        let mut squared_error_sum = 0_f64;
        let mut abs_error_sum = 0_f64;
        let mut max_error = 0_f64;
        let mut log_error_sum = 0_f64;
        let mut abs_percent_error_sum = 0_f64;
        let mut y_true_sum = 0_f64;

        for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
            let t: f64 = (*t_ref).into();
            let p: f64 = (*p_ref).into();

            y_true_sum += t;
            squared_error_sum += (t - p).powi(2);
            abs_error_sum += (t - p).abs();
            max_error = max_error.max((t - p).abs());
            log_error_sum += (1_f64 + t).log10() - (1_f64 + p).log10();
            abs_percent_error_sum += (t - p).abs() / t;
        }

        let mut ss_total = 0_f64;
        let y_true_mean = y_true_sum / n;
        for t_ref in y_true.iter() {
            let t: f64 = (*t_ref).into();
            ss_total += (t - y_true_mean).powi(2);
        }

        let mse = squared_error_sum / n;
        let msle = log_error_sum / n;

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

impl LinearRegressionRuntime {
    pub fn compare_to_baseline(
        &self,
        metrics: &[LinearRegressionEvaluationMetric],
        baseline: &LinearRegressionRuntime,
        drift_threshold: f32,
    ) -> LinearRegressionRuntimeReport {
        use LinearRegressionEvaluationMetric as L;
        let mut res: HashMap<L, f32> = HashMap::with_capacity(8);
        for m in metrics.iter() {
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
}
