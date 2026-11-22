#[macro_export]
macro_rules! zip_iters {
    ($first:expr $(,)?) => {
        $first.iter()
    };

    // Recursive case: zip first with zip of the rest
    ($first:expr, $($rest:expr),+ $(,)?) => {
        $first.iter().zip($crate::zip_iters!($($rest),+))
    };
}

/// Coerce a numpy array owned by the interpreter to a slice.
/// Interpolates the readonly -> slice dance.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! numpy_slice1d {
    ($name:ident = $arr:expr $(,)?) => {
        let guard = ::numpy::PyArrayMethods::readonly(&$arr);
        let $name: &[_] = guard.as_slice()?;
    };
}

/// Create an interned table of the metrics, lazily on demand.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! impl_metrics_table {
    ($metric: ty, $table: expr) => {
        impl PyMachineLearningMetric for $metric {
            fn py_interned_table(py: Python<'_>) -> &'static [Py<PyString>] {
                static TABLE: GILOnceCell<Vec<Py<PyString>>> = GILOnceCell::new();
                // Unbind the static table from the GIL.
                TABLE.get_or_init(py, || {
                    $table
                        .iter()
                        .map(|m| PyString::intern(py, m.as_ref()).unbind())
                        .collect()
                })
            }

            fn idx(&self) -> usize {
                *self as usize
            }
        }
    };
}
