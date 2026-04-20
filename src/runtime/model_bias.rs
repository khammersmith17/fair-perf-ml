use crate::{
    errors::{ModelBiasRuntimeError, ModelPerfResult},
    metrics::ModelBiasMetric,
    model_bias::PostTraining,
    reporting::{ModelBiasAnalysisReport, ModelBiasRuntimeReport},
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ModelBiasRuntime {
    pub(crate) ddpl: f32,
    pub(crate) di: f32,
    pub(crate) ad: f32,
    pub(crate) rd: f32,
    pub(crate) cdacc: f32,
    pub(crate) dar: f32,
    pub(crate) sd: f32,
    pub(crate) dcr: f32,
    pub(crate) drr: f32,
    pub(crate) te: f32,
    pub(crate) ccdpl: f32,
    pub(crate) ge: f32,
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
    /*
     * TODO: for dppl, cddpl, ad, rd, cdacc, dar, sd, dcr, drr, te
     *
     * need to decide if absolute magnitude or drift is the condition to check on
     * */
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
