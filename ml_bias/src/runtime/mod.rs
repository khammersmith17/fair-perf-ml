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
    pub fn new(data: HashMap<String, f32>) -> Result<DataBiasRuntime, String> {
        let ci = match data.get("ClassImbalance") {
            Some(val) => *val,
            None => return Err("ClassImbalance not present".to_string()),
        };
        let dpl = match data.get("DifferenceInProportionOfLabels") {
            Some(val) => *val,
            None => return Err("DifferenceInProportionOfLabels not present".to_string()),
        };
        let kl = match data.get("KlDivergence") {
            Some(val) => *val,
            None => return Err("KlDivergence not present".to_string()),
        };
        let js = match data.get("JsDivergence") {
            Some(val) => *val,
            None => return Err("JsDivergence not present".to_string()),
        };
        let lpnorm = match data.get("LpNorm") {
            Some(val) => *val,
            None => return Err("LpNorm not present".to_string()),
        };
        let tvd = match data.get("TotalVarationDistance") {
            Some(val) => *val,
            None => return Err("TotalVariationDifference not present".to_string()),
        };
        let ks = match data.get("KolmorogvSmirnov") {
            Some(val) => *val,
            None => return Err("KolmogorvSmirnov is not present".to_string()),
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
    pub fn runtime_check(&self, baseline: DataBiasRuntime) -> HashMap<String, String> {
        let mut result: HashMap<String, String> = HashMap::new();
        if self.ci.abs() > baseline.ci.abs() {
            result.insert(
                "ClassImbalance".to_string(),
                format!(
                    "Exceeded baseline by: {}",
                    (self.ci.abs() - baseline.ci.abs()).abs()
                ),
            );
        }
        if self.dpl.abs() > baseline.dpl.abs() {
            result.insert(
                "DfferenceInProportionalOfLabels".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.dpl.abs() - baseline.dpl.abs()).abs()
                ),
            );
        }
        if self.kl > baseline.kl {
            result.insert(
                "KlDivergence".to_string(),
                format!("Execeed baseline by: {}", self.kl - baseline.kl),
            );
        }
        if self.js > baseline.js {
            result.insert(
                "JsDivergance".to_string(),
                format!("Execeed baseline by: {}", self.kl - baseline.kl),
            );
        }
        if self.lpnorm > baseline.lpnorm {
            result.insert(
                "LpNorm".to_string(),
                format!("Exceeded baseline by: {}", self.lpnorm - baseline.lpnorm),
            );
        }
        if self.tvd > baseline.tvd {
            result.insert(
                "TotalVariationDistance".to_string(),
                format!("Exceed baseline by: {}", self.tvd - baseline.tvd),
            );
        }
        if self.ks > baseline.ks {
            result.insert(
                "TotalVariationDistance".to_string(),
                format!("Exceed baseline by: {}", self.tvd - baseline.tvd),
            );
        }
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
    pub fn new(data: HashMap<String, f32>) -> Result<ModelBiasRuntime, String> {
        let ddpl = match data.get("DifferenceInPositivePredictedLabels") {
            Some(val) => *val,
            None => return Err("DifferenceInPositivePredictedLabels is not present".to_string()),
        };
        let di = match data.get("DisparateImpact") {
            Some(val) => *val,
            None => return Err("DisparateImpact is not present".to_string()),
        };
        let ad = match data.get("AccuracyDifference") {
            Some(val) => *val,
            None => return Err("AccuracyDifference is not present".to_string()),
        };
        let rd = match data.get("RecallDifference") {
            Some(val) => *val,
            None => return Err("RecallDifference is not present".to_string()),
        };
        let cdacc = match data.get("CDACC") {
            Some(val) => *val,
            None => return Err("CDACC is not present".to_string()),
        };
        let dar = match data.get("DifferenceInAcceptanceRates") {
            Some(val) => *val,
            None => return Err("DifferenceInAcceptanceRates is not present".to_string()),
        };
        let sd = match data.get("SpecailityDifference") {
            Some(val) => *val,
            None => return Err("SpecailityDifference not present".to_string()),
        };
        let dcr = match data.get("DifferenceInConditionalRejection") {
            Some(val) => *val,
            None => return Err("DifferenceInConditionalRejection not present".to_string()),
        };
        let drr = match data.get("DifferenceInRejectionRate") {
            Some(val) => *val,
            None => return Err("DifferenceInRejectionRate is not present".to_string()),
        };
        let te = match data.get("TreatmentEquity") {
            Some(val) => *val,
            None => return Err("TreatmentEquity is not present".to_string()),
        };
        let ccdpl = match data.get("ConditionalDemographicDesparityPredictedLabels") {
            Some(val) => *val,
            None => {
                return Err(
                    "ConditionalDemographicDesparityPredictedLabels is not present".to_string(),
                )
            }
        };
        let ge = match data.get("GeneralizedEntropy") {
            Some(val) => *val,
            None => return Err("GeneralizedEntropy is not present".to_string()),
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
    pub fn runtime_check(&self, baseline: ModelBiasRuntime) -> HashMap<String, String> {
        let mut result: HashMap<String, String> = HashMap::new();
        if self.ddpl.abs() > baseline.ddpl.abs() {
            result.insert(
                "DDPL".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.ddpl.abs() - baseline.ddpl.abs()).abs()
                ),
            );
        }

        if self.di > baseline.di {
            result.insert(
                "DI".to_string(),
                format!("Exceed baseline by: {}", (self.di - baseline.di).abs()),
            );
        }
        if self.ad.abs() > baseline.ad.abs() {
            result.insert(
                "AD".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.ad.abs() - baseline.ad.abs()).abs()
                ),
            );
        }
        if self.rd.abs() > baseline.rd.abs() {
            result.insert(
                "RD".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.rd.abs() - baseline.rd.abs()).abs()
                ),
            );
        }
        if self.cdacc.abs() > baseline.cdacc.abs() {
            result.insert(
                "CDACC".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.cdacc.abs() - baseline.cdacc.abs()).abs()
                ),
            );
        }
        if self.dar.abs() > baseline.dar.abs() {
            result.insert(
                "DAR".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.dar.abs() - baseline.dar.abs()).abs()
                ),
            );
        }
        if self.sd.abs() > baseline.sd.abs() {
            result.insert(
                "SD".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.sd.abs() - baseline.sd.abs()).abs()
                ),
            );
        }
        if self.dcr.abs() > baseline.dcr.abs() {
            result.insert(
                "DCR".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.dcr.abs() - baseline.dcr.abs()).abs()
                ),
            );
        }
        if self.drr.abs() > baseline.drr.abs() {
            result.insert(
                "DRR".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.drr.abs() - baseline.drr.abs()).abs()
                ),
            );
        }
        if self.te.abs() > baseline.te.abs() {
            result.insert(
                "TE".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.te.abs() - baseline.te.abs()).abs()
                ),
            );
        }
        if self.ccdpl.abs() > baseline.ccdpl.abs() {
            result.insert(
                "CCDPL".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.ccdpl.abs() - baseline.ccdpl.abs()).abs()
                ),
            );
        }
        if self.ge > baseline.ge {
            result.insert(
                "GE".to_string(),
                format!("Exceed baseline by: {}", (self.ge - baseline.ge).abs()),
            );
        }

        result
    }
}
