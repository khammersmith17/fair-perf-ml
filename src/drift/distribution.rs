#[non_exhaustive]
#[derive(Default, Debug)]
pub enum QuantileType {
    #[default]
    FreedmanDiaconis,
    Scott,
    Sturges,
}

impl TryFrom<&str> for QuantileType {
    type Error = crate::errors::DriftError;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "FreedmanDiaconis" => Ok(Self::FreedmanDiaconis),
            "Scott" => Ok(Self::Scott),
            "Sturges" => Ok(Self::Sturges),
            _ => Err(Self::Error::UnsupportedDriftType),
        }
    }
}

impl QuantileType {
    pub fn compute_num_bins(self, baseline_distribution: &[f64]) -> usize {
        match self {
            QuantileType::FreedmanDiaconis => freedman_diaconis(baseline_distribution),
            QuantileType::Scott => scott(baseline_distribution),
            QuantileType::Sturges => sturges(baseline_distribution),
        }
    }
}

const SCOTT_CONSTANT: f64 = 3.49;

/// Compute the optimal number of bins using Scott's method.
fn scott(dataset: &[f64]) -> usize {
    let n = dataset.len() as f64;
    let mean = dataset.iter().sum::<f64>() / n;
    let deviation_term: f64 = dataset
        .iter()
        .map(|v| (*v - mean).powi(2))
        .sum::<f64>()
        .sqrt();
    // use the sample standard deviation
    let std_dev = 1_f64 / (n - 1.0) * deviation_term;
    ((SCOTT_CONSTANT * std_dev * n.powf(-1_f64 / 3_f64)).floor() as usize).max(3_usize)
}

/// Compute the optimal number of bins using Sturges method.
fn sturges(dataset: &[f64]) -> usize {
    ((dataset.len() as f64).ln().floor() as usize + 1_usize).max(3_usize)
}

/// Compute the optimal number of bins using Freedman Diaconis method.
fn freedman_diaconis(sorted_data: &[f64]) -> usize {
    let n = sorted_data.len() as f64;
    if n == 1_f64 {
        return 1;
    }
    let iqr = sorted_data[(0.75 * n).floor() as usize] - sorted_data[(0.25 * n).floor() as usize];
    let width = 2_f64 * iqr * n.powf(-0.33);
    let dmax = sorted_data[n as usize - 1];
    let dmin = sorted_data[0];
    // clamp to [2, ...)
    ((dmax - dmin) / width).ceil().max(3_f64) as usize
}
