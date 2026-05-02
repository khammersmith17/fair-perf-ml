use super::baseline::ContinuousBinEdges;
use super::opt;

pub(crate) fn compute_dataset_from_bins_continuous(
    dataset: &[f64],
    edges: &ContinuousBinEdges,
) -> Vec<f64> {
    opt::continuous::parallel_approx_dataset(dataset, edges)
}
