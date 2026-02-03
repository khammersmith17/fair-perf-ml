use crate::errors::ModelPerformanceError;
use crate::zip_iters;

/// All methods accept any type that implements Into<f64>, to account for scenarios where scores and
/// ground truth values may overflow on f32, and to allow the most flexibility in terms of typing.
/// Copy is also a requirement, internally the function performs the computation on f64 values,
/// thus requiring copy for performance related implications on deref and into f64. All slices must
/// be the same length and be of the same type.

/// Root Mean Squared Error.
pub fn root_mean_squared_error<T>(y_true: &[T], y_pred: &[T]) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let n = y_true.len() as f64;
    let mut errors = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        errors += (t - p).powi(2);
    }
    Ok((errors / n).powf(0.5_f64))
}

/// Mean Squared Error
pub fn mean_squared_error<T>(y_true: &[T], y_pred: &[T]) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let n = y_true.len() as f64;
    let mut errors = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        errors += (t - p).powi(2);
    }
    Ok(errors / n)
}

/// Mean Absolute Error
pub fn mean_absolute_error<T>(y_true: &[T], y_pred: &[T]) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let n = y_true.len() as f64;
    let mut errors = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        errors += (t - p).abs();
    }
    Ok(errors / n)
}

/// Mean Absolute Error
pub fn r_squared<T>(y_true: &[T], y_pred: &[T]) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let n = y_true.len() as f64;
    let mut y_true_sum = 0_f64;
    let mut ss_regression = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        ss_regression += (t - p).powi(2);
        y_true_sum += t;
    }

    let y_true_mean = y_true_sum / n;
    let mut ss_total = 0_f64;
    for t_ref in y_true.iter() {
        let t: f64 = (*t_ref).into();
        ss_total += (t - y_true_mean).powi(2);
    }

    Ok(1_f64 - (ss_regression / ss_total))
}

/// Mean Absolute Error
pub fn max_error<T>(y_true: &[T], y_pred: &[T]) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let mut res = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        res = f64::max(t - p, res);
    }
    Ok(res)
}

/// Mean sqaured Log Error
pub fn mean_squared_log_error<T>(y_true: &[T], y_pred: &[T]) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let n = y_true.len() as f64;
    let mut sum = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        sum += (1_f64 + t).log10() - (1_f64 + p).log10();
    }
    Ok(sum.powi(2) / n)
}

/// Root Mean Sqaured Log Error
pub fn root_mean_squared_log_error<T>(
    y_true: &[T],
    y_pred: &[T],
) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let n = y_true.len() as f64;
    let mut sum = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        sum += (1_f64 + t).log10() - (1_f64 + p).log10();
    }
    Ok(sum.powi(2).sqrt() / n)
}

/// Ad hoc method to compute the mean absolute percentage error Linear Regression metric. pub fn mean_absolute_percentage_error<T>(
pub fn mean_absolute_percentage_error<T>(
    y_true: &[T],
    y_pred: &[T],
) -> Result<f64, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(ModelPerformanceError::DataVectorLengthMismatch);
    }
    let n: f64 = y_true.len() as f64;
    let mut sum = 0_f64;
    for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
        let t: f64 = (*t_ref).into();
        let p: f64 = (*p_ref).into();
        sum += (t - p).abs() / t;
    }
    Ok((sum / n) * 100_f64)
}

pub(crate) mod classification_metrics {
    use crate::data_handler::ConfusionMatrix;
    use crate::errors::ModelPerformanceError;
    use crate::zip_iters;

    /// TP / TP + FP
    #[inline]
    pub(crate) fn precision_positive(confusion_matrix: &ConfusionMatrix) -> f32 {
        confusion_matrix.true_p / (confusion_matrix.true_p + confusion_matrix.false_p)
    }

    /// TN / TN + FN
    #[inline]
    pub(crate) fn precision_negative(confusion_matrix: &ConfusionMatrix) -> f32 {
        confusion_matrix.true_n / (confusion_matrix.true_n + confusion_matrix.false_n)
    }

    /// TP / TP + FN
    #[inline]
    pub(crate) fn recall_positive(confusion_matrix: &ConfusionMatrix) -> f32 {
        confusion_matrix.true_p / (confusion_matrix.true_p + confusion_matrix.false_n)
    }

    /// TN / TN + FP
    #[inline]
    pub(crate) fn recall_negative(confusion_matrix: &ConfusionMatrix) -> f32 {
        confusion_matrix.true_n / (confusion_matrix.true_n + confusion_matrix.false_p)
    }

    #[inline]
    pub(crate) fn f1_score(confusion_matrix: &ConfusionMatrix) -> f32 {
        let rp = recall_positive(confusion_matrix);
        let pp = precision_positive(confusion_matrix);
        2_f32 * rp * pp / (rp + pp)
    }

    #[inline]
    pub(crate) fn balanced_accuracy(confusion_matrix: &ConfusionMatrix) -> f32 {
        let rp = recall_positive(confusion_matrix);
        let rn = recall_negative(confusion_matrix);
        rp * rn * 0.5_f32
    }

    #[inline]
    pub(crate) fn accuracy<T>(y_true: &[T], y_pred: &[T]) -> Result<f32, ModelPerformanceError>
    where
        T: PartialOrd,
    {
        if y_pred.len() != y_true.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        let n = y_true.len();
        let mut correct: f32 = 0_f32;
        for (t, p) in zip_iters!(y_true, y_pred) {
            if t == p {
                correct += 1_f32;
            }
        }
        Ok(correct / n as f32)
    }

    #[inline]
    pub(crate) fn log_loss_score(
        y_true: &[f32],
        y_proba: &[f32],
    ) -> Result<f32, ModelPerformanceError> {
        if y_proba.len() != y_true.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let n = y_proba.len() as f32;
        let mut penalties = 0_f32;
        for (t, p) in zip_iters!(y_true, y_proba) {
            penalties += t * f32::log10(*p) + (1_f32 - *t) * f32::log10(1_f32 - *p);
        }
        let res = (-1_f32 * penalties) / n;

        if res.is_nan() {
            Ok(0_f32)
        } else {
            Ok(res)
        }
    }
}
