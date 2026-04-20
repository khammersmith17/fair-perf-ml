use crate::{
    data_bias::PreTraining,
    errors::{BiasError, DataBiasRuntimeError},
    metrics::DataBiasMetric,
    reporting::{DataBiasAnalysisReport, DataBiasRuntimeReport},
};
use std::collections::HashMap;
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
        let tvd = match data.get("TotalVariationDistance") {
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

/*
* TODO: for class imbalance, dpl
*
* need to decide if absolute magnitude or drift is the condition to check on
* */
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
                    if self.ci.abs() > (baseline.ci.abs() * (1_f32 + threshold)).abs() {
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
                            (self.dpl - baseline.dpl).abs(),
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
