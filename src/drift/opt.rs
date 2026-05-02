use super::baseline::ContinuousBinEdges;

/// Get the available number of threads.
/// If the dataset size is less than the number of threads, use a single thread.
fn get_thread_count(n: usize) -> usize {
    let Ok(nz_count) = std::thread::available_parallelism() else {
        return 1;
    };
    let c = nz_count.get();

    if c > n {
        return 1;
    }
    c
}

/*
* Methods to approximate the distribution histogram across threads.
* Derive an approximate number of threads.
* Compute thread local distribution.
* Reduce local element wise count into global bins by folding the local bins.
*/

pub(crate) mod continuous {
    pub(crate) fn parallel_approx_dataset(
        dataset: &[f64],
        bin_edges: &super::ContinuousBinEdges,
    ) -> Vec<f64> {
        let n = dataset.len();
        let thread_count = super::get_thread_count(n);
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = bin_edges.n_bins();

        let hist = std::thread::scope(|s| {
            dataset
                .chunks(chunk_size)
                .map(|dataset_chunk| {
                    s.spawn(|| {
                        let mut local_hist = vec![0_f64; n_bins];
                        for ex in dataset_chunk.iter() {
                            local_hist[bin_edges.resolve_bin(*ex)] += 1_f64;
                        }
                        local_hist
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(vec![0_f64; n_bins], |mut acc, local| {
                    acc.iter_mut().zip(local.iter()).for_each(|(a, b)| *a += b);
                    acc
                })
        });
        hist
    }
}

pub(crate) mod categorical {
    use std::hash::Hash;

    pub(crate) struct CategoricalBinEdges<T: Hash + Ord + Clone>(
        std::collections::HashMap<T, usize>,
    );

    impl<T: Hash + Ord + Clone> CategoricalBinEdges<T> {
        fn resolve_bin<Q>(&self, key: &Q) -> usize
        where
            T: std::borrow::Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            if let Some(idx) = self.0.get(key) {
                *idx
            } else {
                self.0.len() - 1
            }
        }

        fn n_bins(&self) -> usize {
            self.0.len() + 1
        }
    }
    pub(crate) fn parallel_approx_dataset<T: Hash + Ord + Clone + Sync>(
        dataset: &[T],
        baseline: &CategoricalBinEdges<T>,
    ) -> Vec<f64> {
        let n = dataset.len();
        let thread_count = super::get_thread_count(n);
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = baseline.n_bins();

        let hist = std::thread::scope(|s| {
            dataset
                .chunks(chunk_size)
                .map(|dataset_chunk| {
                    s.spawn(|| {
                        let mut local_hist = vec![0_f64; n_bins];
                        for ex in dataset_chunk.iter() {
                            local_hist[baseline.resolve_bin(ex)] += 1_f64;
                        }
                        local_hist
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(vec![0_f64; n_bins], |mut acc, local| {
                    acc.iter_mut().zip(local.iter()).for_each(|(a, b)| *a += b);
                    acc
                })
        });
        hist
    }
}
