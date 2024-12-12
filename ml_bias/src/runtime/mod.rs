use std::collections::HashMap;

pub struct DataBiasRuntime {
    ci: f32,
    dpl: f32,
    kl: f32,
    lpnorm: f32,
    tvd: f32,
}

impl DataBiasRuntime {
    pub fn new(data: HashMap<String, f32>) -> Result<DataBiasRuntime, String> {
        let ci = match data.get("CI") {
            Some(val) => *val,
            None => return Err("CI not present".to_string()),
        };
        let dpl = match data.get("DPL") {
            Some(val) => *val,
            None => return Err("DPL not present".to_string()),
        };
        let kl = match data.get("KL") {
            Some(val) => *val,
            None => return Err("KL not present".to_string()),
        };
        let lpnorm = match data.get("LPNorm") {
            Some(val) => *val,
            None => return Err("LPNorm not present".to_string()),
        };
        let tvd = match data.get("TVD") {
            Some(val) => *val,
            None => return Err("TVD not present".to_string()),
        };

        Ok(DataBiasRuntime {
            ci,
            dpl,
            kl,
            lpnorm,
            tvd,
        })
    }
    pub fn runtime_check(&self, baseline: DataBiasRuntime) -> HashMap<String, String> {
        let mut result: HashMap<String, String> = HashMap::new();
        if self.ci.abs() > baseline.ci.abs() {
            result.insert(
                "CI".to_string(),
                format!(
                    "Exceeded baseline by: {}",
                    (self.ci.abs() - baseline.ci.abs()).abs()
                ),
            );
        }
        if self.dpl.abs() > baseline.dpl.abs() {
            result.insert(
                "DPL".to_string(),
                format!(
                    "Exceed baseline by: {}",
                    (self.dpl.abs() - baseline.dpl.abs()).abs()
                ),
            );
        }
        if self.kl > baseline.kl {
            result.insert(
                "KL".to_string(),
                format!("Execeed baseline by: {}", self.kl - baseline.kl),
            );
        }
        if self.lpnorm > baseline.lpnorm {
            result.insert(
                "LPNorm".to_string(),
                format!("Exceeded baseline by: {}", self.lpnorm - baseline.lpnorm),
            );
        }
        if self.tvd > baseline.tvd {
            result.insert(
                "TVD".to_string(),
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
        let ddpl = match data.get("DDPL") {
            Some(val) => *val,
            None => return Err("DDPL is not present".to_string()),
        };
        let di = match data.get("DI") {
            Some(val) => *val,
            None => return Err("DI is not present".to_string()),
        };
        let ad = match data.get("AD") {
            Some(val) => *val,
            None => return Err("AD is not present".to_string()),
        };
        let rd = match data.get("RD") {
            Some(val) => *val,
            None => return Err("RD is not present".to_string()),
        };
        let cdacc = match data.get("CDACC") {
            Some(val) => *val,
            None => return Err("CDACC is not present".to_string()),
        };
        let dar = match data.get("DAR") {
            Some(val) => *val,
            None => return Err("DAR is not present".to_string()),
        };
        let sd = match data.get("SD") {
            Some(val) => *val,
            None => return Err("SD not present".to_string()),
        };
        let dcr = match data.get("DCR") {
            Some(val) => *val,
            None => return Err("DCR not present".to_string()),
        };
        let drr = match data.get("DRR") {
            Some(val) => *val,
            None => return Err("DRR is not present".to_string()),
        };
        let te = match data.get("TE") {
            Some(val) => *val,
            None => return Err("TE is not present".to_string()),
        };
        let ccdpl = match data.get("CCDPL") {
            Some(val) => *val,
            None => return Err("CCDPL is not present".to_string()),
        };
        let ge = match data.get("GE") {
            Some(val) => *val,
            None => return Err("GE is not present".to_string()),
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
