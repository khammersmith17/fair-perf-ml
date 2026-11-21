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

// Bind a contiguous slice view of a 1d numpy array in the caller's scope.
// Must be used in statement position: the readonly guard has to live in the
// caller's block for the slice borrow to remain valid.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! numpy_slice1d {
    // Map the NotContiguousError into the caller's error type
    ($name:ident = $arr:expr, map_err = $f:expr $(,)?) => {
        let guard = ::numpy::PyArrayMethods::readonly(&$arr);
        let $name: &[_] = guard.as_slice().map_err($f)?;
    };

    ($name:ident = $arr:expr $(,)?) => {
        let guard = ::numpy::PyArrayMethods::readonly(&$arr);
        let $name: &[_] = guard.as_slice()?;
    };

    // Recursive case: bind first, then the rest
    ($name:ident = $arr:expr, $($rest_name:ident = $rest_arr:expr),+ $(,)?) => {
        $crate::numpy_slice1d!($name = $arr);
        $crate::numpy_slice1d!($($rest_name = $rest_arr),+);
    };
}
