use super::data_bias::DataBiasMetrics;
use super::model_bias::ModelBiasMetrics;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataBiasRuntimeError {
    #[error("ClassImbalance not present")]
    ClassImbalance,
    #[error("DifferenceInProportionOfLabels not present")]
    DifferenceInProportionOfLabels,
    #[error("KlDivergence not present")]
    KlDivergence,
    #[error("JsDivergence not present")]
    JsDivergence,
    #[error("TotalVariationDistance not present")]
    TotalVariationDistance,
    #[error("KolmorogvSmirnov not present")]
    KolmorogvSmirnov,
    #[error("LpNorm not present")]
    LpNorm,
}

pub struct DataBiasRuntime {
    ci: f32,
    dpl: f32,
    kl: f32,
    js: f32,
    lpnorm: f32,
    tvd: f32,
    ks: f32,
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

impl DataBiasRuntime {
    pub fn runtime_check(
        &self,
        baseline: DataBiasRuntime,
        threshold: f32,
        metrics: &[DataBiasMetrics],
    ) -> HashMap<String, String> {
        let mut result: HashMap<String, String> = HashMap::with_capacity(metrics.len());
        for m in metrics {
            match m {
                DataBiasMetrics::ClassImbalance => {
                    if self.ci.abs() > baseline.ci.abs() * (1_f32 + threshold) {
                        result.insert(
                            "ClassImbalance".to_string(),
                            format!(
                                "Exceeded baseline by: {}",
                                (self.ci.abs() - baseline.ci.abs()).abs()
                            ),
                        );
                    }
                }
                DataBiasMetrics::DifferenceInProportionOfLabels => {
                    if self.dpl.abs() > baseline.dpl.abs() * (1_f32 + threshold) {
                        result.insert(
                            "DfferenceInProportionOfLabels".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.dpl.abs() - baseline.dpl.abs()).abs()
                            ),
                        );
                    }
                }
                DataBiasMetrics::KlDivergence => {
                    if self.kl > baseline.kl * (1_f32 + threshold) {
                        result.insert(
                            "KlDivergence".to_string(),
                            format!("Execeed baseline by: {}", self.kl - baseline.kl),
                        );
                    }
                }
                DataBiasMetrics::JsDivergence => {
                    if self.js > baseline.js * (1_f32 + threshold) {
                        result.insert(
                            "JsDivergance".to_string(),
                            format!("Execeed baseline by: {}", self.kl - baseline.kl),
                        );
                    }
                }
                DataBiasMetrics::LpNorm => {
                    if self.lpnorm > baseline.lpnorm * (1_f32 + threshold) {
                        result.insert(
                            "LpNorm".to_string(),
                            format!("Exceeded baseline by: {}", self.lpnorm - baseline.lpnorm),
                        );
                    }
                }
                DataBiasMetrics::TotalVariationDistance => {
                    if self.tvd > baseline.tvd * (1_f32 + threshold) {
                        result.insert(
                            "TotalVariationDistance".to_string(),
                            format!("Exceed baseline by: {}", self.tvd - baseline.tvd),
                        );
                    }
                }
                DataBiasMetrics::KolmorogvSmirnov => {
                    if self.ks > baseline.ks * (1_f32 + threshold) {
                        result.insert(
                            "KolmorogvSmirnov".to_string(),
                            format!("Exceed baseline by: {}", self.tvd - baseline.tvd),
                        );
                    }
                }
            }
        }
        result
    }
}

#[derive(Debug, Error)]
pub enum ModelBiasRuntimeError {
    #[error("DifferenceInPositivePredictedLabels not present")]
    DifferenceInPositivePredictedLabels,
    #[error("DisparateImpact not present")]
    DisparateImpact,
    #[error("AccuracyDifference not present")]
    AccuracyDifference,
    #[error("RecallDifference not present")]
    RecallDifference,
    #[error("DifferenceInConditionalAcceptance not present")]
    DifferenceInConditionalAcceptance,
    #[error("DifferenceInAcceptanceRate not present")]
    DifferenceInAcceptanceRate,
    #[error("SpecialityDifference not present")]
    SpecialityDifference,
    #[error("DifferenceInConditionalRejection not present")]
    DifferenceInConditionalRejection,
    #[error("TreatmentEquity not present")]
    TreatmentEquity,
    #[error("ConditionalDemographicDesparityPredictedLabels not present")]
    ConditionalDemographicDesparityPredictedLabels,
    #[error("DifferenceInRejectionRate not present")]
    DifferenceInRejectionRate,
    #[error("GeneralizedEntropy not present")]
    GeneralizedEntropy,
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
        metrics: &[ModelBiasMetrics],
    ) -> HashMap<String, String> {
        let mut result: HashMap<String, String> = HashMap::with_capacity(metrics.len());
        for m in metrics {
            match m {
                ModelBiasMetrics::DifferenceInPositivePredictedLabels => {
                    if self.ddpl.abs() > baseline.ddpl.abs() * (1_f32 + threshold) {
                        result.insert(
                            "DifferenceInPositivePredictedLabels".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.ddpl.abs() - baseline.ddpl.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::DisparateImpact => {
                    if self.di > baseline.di * (1_f32 + threshold) {
                        result.insert(
                            "DisparateImpact".to_string(),
                            format!("Exceed baseline by: {}", (self.di - baseline.di).abs()),
                        );
                    }
                }
                ModelBiasMetrics::AccuracyDifference => {
                    if self.ad.abs() > baseline.ad.abs() * (1_f32 + threshold) {
                        result.insert(
                            "AccuracyDifference".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.ad.abs() - baseline.ad.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::RecallDifference => {
                    if self.rd.abs() > baseline.rd.abs() * (1_f32 + threshold) {
                        result.insert(
                            "RecallDifference".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.rd.abs() - baseline.rd.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::DifferenceInConditionalAcceptance => {
                    if self.cdacc.abs() > baseline.cdacc.abs() * (1_f32 + threshold) {
                        result.insert(
                            "DifferenceInConditionalAcceptance".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.cdacc.abs() - baseline.cdacc.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::DifferenceInAcceptanceRate => {
                    if self.dar.abs() > baseline.dar.abs() * (1_f32 + threshold) {
                        result.insert(
                            "DifferenceInAcceptanceRate".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.dar.abs() - baseline.dar.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::SpecialityDifference => {
                    if self.sd.abs() > baseline.sd.abs() * (1_f32 + threshold) {
                        result.insert(
                            "SpecialityDifference".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.sd.abs() - baseline.sd.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::DifferenceInConditionalRejection => {
                    if self.dcr.abs() > baseline.dcr.abs() * (1_f32 + threshold) {
                        result.insert(
                            "DifferenceInConditionalRejection".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.dcr.abs() - baseline.dcr.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::DifferenceInRejectionRate => {
                    if self.drr.abs() > baseline.drr.abs() * (1_f32 + threshold) {
                        result.insert(
                            "DifferenceInRejectionRate".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.drr.abs() - baseline.drr.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::TreatmentEquity => {
                    if self.te.abs() > baseline.te.abs() * (1_f32 + threshold) {
                        result.insert(
                            "TreatmentEquity".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.te.abs() - baseline.te.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::ConditionalDemographicDesparityPredictedLabels => {
                    if self.ccdpl.abs() > baseline.ccdpl.abs() * (1_f32 + threshold) {
                        result.insert(
                            "ConditionalDemographicDesparityPredictedLabels".to_string(),
                            format!(
                                "Exceed baseline by: {}",
                                (self.ccdpl.abs() - baseline.ccdpl.abs()).abs()
                            ),
                        );
                    }
                }
                ModelBiasMetrics::GeneralizedEntropy => {
                    if self.ge > baseline.ge * (1_f32 + threshold) {
                        result.insert(
                            "GeneralizedEntropy".to_string(),
                            format!("Exceed baseline by: {}", (self.ge - baseline.ge).abs()),
                        );
                    }
                }
            }
        }

        result
    }
}
