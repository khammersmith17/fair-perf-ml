use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, pyclass, pymethods};
use std::cmp::Ordering;

#[pyclass]
pub struct ContinuousPSI {
    pub bin_edges: Vec<f64>,
    pub baseline: Vec<f64>,
    n_bins: usize,
}

#[pymethods]
impl ContinuousPSI {
    #[new]
    pub fn new(n_bins: usize) -> ContinuousPSI {
        let bin_edges: Vec<f64> = Vec::with_capacity(n_bins + 1);

        ContinuousPSI {
            bin_edges,
            baseline: Vec::new(),
            n_bins,
        }
    }

    pub fn compute_baseline<'py>(
        &mut self,
        baseline_data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        self.define_bins(baseline_data.clone())?;
        self.baseline = self.compute_hist(baseline_data)?;

        Ok(())
    }

    fn compute_hist<'py>(&self, data: PyReadonlyArray1<'py, f64>) -> PyResult<Vec<f64>> {
        /*
         * for each item in data
         *   1. get an iterator over the bins
         */

        let data_slice = data.as_slice()?;
        let bl_count = data_slice.len();
        let mut hist = vec![0_usize; self.bin_edges.len() - 1];
        let n_bins = self.bin_edges.len() - 1;
        for item in data_slice {
            let mut idx: usize = n_bins - 1;
            for i in 0..n_bins {
                if *item < self.bin_edges[i + 1] {
                    idx = i;
                    break;
                }
            }
            hist[idx] += 1;
        }

        let bl_hist: Vec<f64> = hist
            .into_iter()
            .map(|n| n as f64 / bl_count as f64)
            .collect();
        Ok(bl_hist)
    }

    fn define_bins<'py>(&mut self, data: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        let mut sorted_baseline: Vec<f64> = data.as_slice()?.to_vec();
        if sorted_baseline.len() <= 1 {
            return Err(PyValueError::new_err("Baseline array requires > 1 value"));
        }
        sorted_baseline.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));

        self.bin_edges.clear();

        // safe unwrap because we know there are items in the array
        if sorted_baseline.first().unwrap() == sorted_baseline.last().unwrap() {
            self.bin_edges
                .extend(vec![sorted_baseline[0], sorted_baseline[0]]);
            return Ok(());
        }

        let n_bl_samples = sorted_baseline.len();
        let n_bins = self.n_bins.min(n_bl_samples - 1).max(1);
        let bin_size = sorted_baseline.len() / n_bins;

        self.bin_edges.push(sorted_baseline[0]);
        for i in 1..(n_bins) {
            let idx = i * bin_size;
            if idx < n_bl_samples {
                self.bin_edges.push(sorted_baseline[idx - 1]);
            }
        }
        self.bin_edges.push(sorted_baseline[n_bl_samples - 1]);
        Ok(())
    }
}

pub struct RealTimePSI {}
