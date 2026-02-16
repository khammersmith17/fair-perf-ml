pub mod classification_metrics {
    use super::classification_metrics_from_parts as CMetrics;
    /// Methods to perform ad hoc loss function computations.
    /// Each method aside from log loss has two variants, one for binary labeled data and one for
    /// logisitc regression style labeling via a threshold. The only exception is the log loss
    /// score, which only provides a single variant.

    /// Generic types can used that implement the associated [std::cmp::Ordering] trait. The label
    /// methods require [PartialEq], and the threshold methods take [ParitalOrd].
    use crate::data_handler::ConfusionMatrix;
    use crate::errors::ModelPerformanceError;

    pub fn precision_from_label<T: PartialEq>(
        y_true: &[T],
        y_pred: &[T],
        positive_label: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.eq(&positive_label));
        Ok(CMetrics::precision_positive(&c_matrix))
    }

    pub fn precision_from_threshold<T: PartialOrd>(
        y_true: &[T],
        y_pred: &[T],
        positive_threshold: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.ge(&positive_threshold));
        Ok(CMetrics::precision_positive(&c_matrix))
    }

    pub fn recall_from_label<T: PartialEq>(
        y_true: &[T],
        y_pred: &[T],
        positive_label: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.eq(&positive_label));
        Ok(CMetrics::recall_positive(&c_matrix))
    }

    pub fn recall_from_threshold<T: PartialOrd>(
        y_true: &[T],
        y_pred: &[T],
        positive_threshold: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.ge(&positive_threshold));
        Ok(CMetrics::recall_positive(&c_matrix))
    }

    pub fn accuracy_from_label<T: PartialEq>(
        y_true: &[T],
        y_pred: &[T],
    ) -> Result<f32, ModelPerformanceError> {
        CMetrics::accuracy(y_true, y_pred)
    }

    pub fn accuracy_from_threshold<T: PartialOrd>(
        y_true: &[T],
        y_pred: &[T],
        positive_threshold: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.ge(&positive_threshold));
        Ok(c_matrix.accuracy())
    }

    pub fn log_loss_score(y_true: &[f32], y_pred: &[f32]) -> Result<f32, ModelPerformanceError> {
        CMetrics::log_loss_score(y_true, y_pred)
    }

    pub fn f1_score_from_label<T: PartialEq>(
        y_true: &[T],
        y_pred: &[T],
        positive_label: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.eq(&positive_label));
        Ok(CMetrics::f1_score(&c_matrix))
    }

    pub fn f1_score_from_threshold<T: PartialOrd>(
        y_true: &[T],
        y_pred: &[T],
        positive_threshold: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.ge(&positive_threshold));
        Ok(CMetrics::f1_score(&c_matrix))
    }

    pub fn balanced_accuracy_from_label<T: PartialEq>(
        y_true: &[T],
        y_pred: &[T],
        positive_label: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.eq(&positive_label));
        Ok(CMetrics::balanced_accuracy(&c_matrix))
    }

    pub fn balanced_accuracy_from_threshold<T: PartialOrd>(
        y_true: &[T],
        y_pred: &[T],
        positive_threshold: T,
    ) -> Result<f32, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.ge(&positive_threshold));
        Ok(CMetrics::balanced_accuracy(&c_matrix))
    }
}

pub mod linear_regression_metric {
    /// All methods accept any type that implements Into<f64>, to account for scenarios where scores and
    /// ground truth values may overflow on f32, and to allow the most flexibility in terms of typing.
    /// Copy is also a requirement, internally the function performs the computation on f64 values,
    /// thus requiring copy for performance related implications on deref and into f64. All slices must
    /// be the same length and be of the same type.
    use crate::errors::ModelPerformanceError;
    use crate::zip_iters;

    /// Root Mean Squared Error. This will error when the y_true and y_pred slices are not the same
    /// length.
    pub fn root_mean_squared_error<T>(
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
        let mut errors = 0_f64;
        for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
            let t: f64 = (*t_ref).into();
            let p: f64 = (*p_ref).into();
            errors += (t - p).powi(2);
        }
        Ok((errors / n).powf(0.5_f64))
    }

    /// Mean Squared Error. This will error when the y_true and y_pred slices are not the same
    /// length.

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

    /// Mean Absolute Error. This will error when the y_true and y_pred slices are not the same
    /// length.

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

    /// Mean Absolute Error. This will error when the y_true and y_pred slices are not the same
    /// length.

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

    /// Mean Absolute Error. This will error when the y_true and y_pred slices are not the same
    /// length.

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
            res = f64::max((t - p).abs(), res);
        }
        Ok(res)
    }

    /// Mean sqaured Log Error. This will error when the y_true and y_pred slices are not the same
    /// length.

    pub fn mean_squared_log_error<T>(
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
            if t == -1_f64 || p == -1_f64 {
                return Err(ModelPerformanceError::InvalidDataValue);
            }
            sum += ((1_f64 + t).ln() - (1_f64 + p).ln()).powi(2);
        }
        Ok(sum / n)
    }

    /// Root Mean Sqaured Log Error. This will error when the y_true and y_pred slices are not the same
    /// length.

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
            sum += ((1_f64 + t).ln() - (1_f64 + p).ln()).powi(2);
        }
        Ok((sum / n).sqrt())
    }

    /// Ad hoc method to compute the mean absolute percentage error Linear Regression metric. This will error
    /// when the y_true and y_pred slices are not the same length.
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
        Ok(sum / n)
    }
}

pub(crate) mod classification_metrics_from_parts {
    use crate::data_handler::ConfusionMatrix;
    use crate::errors::ModelPerformanceError;
    use crate::metrics::get_stability_eps;
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
        (rp + rn) * 0.5_f32
    }

    #[inline]
    pub(crate) fn accuracy<T>(y_true: &[T], y_pred: &[T]) -> Result<f32, ModelPerformanceError>
    where
        T: PartialEq,
    {
        if y_pred.len() != y_true.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        let n = y_true.len();
        let mut correct: f32 = 0_f32;
        for (t, p) in zip_iters!(y_true, y_pred) {
            if t.eq(p) {
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
        let eps = get_stability_eps() as f32;
        let mut penalties = 0_f32;
        for (t, p) in zip_iters!(y_true, y_proba) {
            let t = *t;
            let mut p = *p;
            p = p.clamp(eps, 1_f32 - eps);
            penalties += t * p.ln() + (1_f32 - t) * (1_f32 - p).ln();
        }
        let res = (-1_f32 * penalties) / n;

        if res.is_nan() {
            return Err(ModelPerformanceError::InvalidDataValue);
        }
        Ok(res)
    }
}

mod test_model_perf_stats {
    #[test]
    fn test_regression_ad_hoc_metrics_zero() {
        use super::linear_regression_metric as metrics;
        let y_pred = vec![11.1, 12.2, 13.4, 10.7, 15.8, 16.3, 14.5, 12.3, 11.0];
        let y_true = vec![11.1, 12.2, 13.4, 10.7, 15.8, 16.3, 14.5, 12.3, 11.0];
        let rmse = metrics::root_mean_squared_error(&y_true, &y_pred).unwrap();
        assert!(rmse.abs() < 1e-5_f64);
        let mse = metrics::mean_squared_error(&y_true, &y_pred).unwrap();
        assert!(mse.abs() < 1e-5_f64);
        let mae = metrics::mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!(mae.abs() < 1e-5_f64);
        let r2 = metrics::r_squared(&y_true, &y_pred).unwrap();
        assert!((1_f64 - r2).abs() < 1e-5_f64);
        let max_error = metrics::max_error(&y_true, &y_pred).unwrap();
        assert!(max_error.abs() < 1e-5_f64);
        let msle = metrics::mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert!(msle.abs() < 1e-5_f64);
        let rmsle = metrics::root_mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert!(rmsle.abs() < 1e-5_f64);
        let mape = metrics::mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        assert!(mape.abs() < 1e-5_f64);
    }

    #[test]
    fn test_regression_ad_hoc_metrics_nonzero() {
        use super::linear_regression_metric as metrics;
        /*
         * {
         *       'rmse': 0.7781745019952505,
         *       'mse': 0.6055555555555561,
         *       'msle': 0.0030593723018136412,
         *       'rmsle': 0.055311592833814156,
         *       'r2': 0.7435160008366448,
         *       'mape': 0.053999057284547014,
         *       'max_error': 1.3000000000000007,
         *       'mae': 0.7000000000000003
         *   }
         * */
        let y_pred = vec![11.1, 12.2, 13.4, 10.7, 15.8, 16.3, 14.5, 12.3, 11.0];
        let y_true = vec![11.0, 12.5, 14.0, 11.7, 15.1, 15.4, 13.2, 11.5, 11.6];
        let rmse = metrics::root_mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((0.77817_f64 - rmse).abs() < 1e-5_f64);
        let mse = metrics::mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((0.605555 - mse).abs() < 1e-5_f64);
        let mae = metrics::mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!((0.7000 - mae).abs() < 1e-5_f64);
        let r2 = metrics::r_squared(&y_true, &y_pred).unwrap();
        assert!((0.743516_f64 - r2).abs() < 1e-5_f64);
        let max_error = metrics::max_error(&y_true, &y_pred).unwrap();
        assert!((1.3 - max_error).abs() < 1e-5_f64);
        let msle = metrics::mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert!((0.003059_f64 - msle).abs() < 1e-5_f64);
        let rmsle = metrics::root_mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert!((0.055311 - rmsle).abs() < 1e-5_f64);
        let mape = metrics::mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        assert!((0.05399_f64 - mape).abs() < 1e-5_f64);
    }

    #[test]
    fn test_add_hoc_classification_from_label() {
        use super::classification_metrics as metrics;
        /*
         * {
         *       'precision': 0.75,
         *       'recall': 0.6666666666666666,
         *       'f1': 0.7058823529411765,
         *       'accuracy': 0.6875,
         *       'balanced_accuracy': 0.6904761904761905
         *   }
         * */

        let y_pred = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let y_true = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1];
        let computed_prec = metrics::precision_from_label(&y_true, &y_pred, 1).unwrap();
        assert!((computed_prec - 0.75).abs() < 1e-5);

        let computed_recall = metrics::recall_from_label(&y_true, &y_pred, 1).unwrap();
        assert!((computed_recall - 0.66666).abs() < 1e-5);

        let computed_f1 = metrics::f1_score_from_label(&y_true, &y_pred, 1).unwrap();
        assert!((computed_f1 - 0.70588).abs() < 1e-5);

        let computed_acc = metrics::accuracy_from_label(&y_true, &y_pred).unwrap();
        assert!((computed_acc - 0.6875).abs() < 1e-5);

        let computed_bal_acc = metrics::balanced_accuracy_from_label(&y_true, &y_pred, 1).unwrap();
        assert!((computed_bal_acc - 0.6904762).abs() < 1e-5);
    }

    #[test]
    fn test_add_hoc_classification_from_threshold() {
        use super::classification_metrics as metrics;
        let y_pred = [
            0.7, 0.3, 0.65, 0.55, 0.1, 0.2, 0.25, 0.66, 0.12, 0.98, 0.23, 0.34, 0.67, 0.77, 0.45,
            0.88,
        ];
        let y_true = [
            0_f32, 0_f32, 1_f32, 1_f32, 1_f32, 0_f32, 0_f32, 1_f32, 1_f32, 1_f32, 0_f32, 0_f32,
            1_f32, 0_f32, 1_f32, 1_f32,
        ];
        let computed_prec = metrics::precision_from_threshold(&y_true, &y_pred, 0.5_f32).unwrap();
        assert!((computed_prec - 0.75).abs() < 1e-5);

        let computed_recall = metrics::recall_from_threshold(&y_true, &y_pred, 0.5_f32).unwrap();
        assert!((computed_recall - 0.66666).abs() < 1e-5);

        let computed_f1 = metrics::f1_score_from_threshold(&y_true, &y_pred, 0.5_f32).unwrap();
        assert!((computed_f1 - 0.70588).abs() < 1e-5);

        let computed_acc = metrics::accuracy_from_threshold(&y_true, &y_pred, 0.5_f32).unwrap();
        assert!((computed_acc - 0.6875).abs() < 1e-5);

        let computed_bal_acc =
            metrics::balanced_accuracy_from_threshold(&y_true, &y_pred, 0.5_f32).unwrap();
        assert!((computed_bal_acc - 0.6904762).abs() < 1e-5);

        let computed_log_loss = metrics::log_loss_score(&y_true, &y_pred).unwrap();
        println!("computed log loss: {computed_log_loss}");
        assert!((computed_log_loss - 0.7145021801144907).abs() < 1e-5)
    }
}
