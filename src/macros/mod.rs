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
