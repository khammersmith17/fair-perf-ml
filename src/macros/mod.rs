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
