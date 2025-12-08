use crate::data_bias::PreTraining;
use crate::errors::{DataBiasRuntimeError, ModelBiasRuntimeError};
use crate::metrics::{DataBiasMetric, ModelBiasMetric};
use crate::reporting::{
    DataBiasAnalysisReport, DataBiasRuntimeReport, ModelBiasAnalysisReport, ModelBiasRuntimeReport,
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
            (self.ci.abs() - baseline.ci.abs()).abs(),
        );
        result.insert(
            DataBiasMetric::DifferenceInProportionOfLabels,
            (self.dpl.abs() - baseline.dpl.abs()).abs(),
        );
        result.insert(DataBiasMetric::KlDivergence, self.kl - baseline.kl);
        result.insert(DataBiasMetric::JsDivergence, self.js - baseline.js);
        result.insert(DataBiasMetric::LpNorm, self.lpnorm - baseline.lpnorm);
        result.insert(
            DataBiasMetric::TotalVariationDistance,
            self.tvd - baseline.tvd,
        );
        result.insert(DataBiasMetric::KolmorogvSmirnov, self.tvd - baseline.tvd);

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
