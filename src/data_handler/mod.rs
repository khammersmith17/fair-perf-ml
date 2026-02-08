#[cfg(feature = "python")]
pub(crate) mod py_types_handler {
    use super::{apply_label_continuous, apply_label_discrete};
    use numpy::dtype;
    use numpy::PyUntypedArrayMethods;
    use numpy::{PyArrayDescrMethods, PyUntypedArray};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyDictMethods, PyFloat, PyInt, PyString};
    use std::collections::HashMap;
    use std::error::Error;

    pub type PyDictResult<'py> = PyResult<Bound<'py, PyDict>>;

    // Coerce analysis/runtime report into Python Dictionary
    pub(crate) fn report_to_py_dict<'py, T>(
        py: Python<'py>,
        report: HashMap<T, f32>,
    ) -> Bound<'py, PyDict>
    where
        T: ToString,
    {
        let dict = PyDict::new(py);
        for (key, val) in report.into_iter() {
            let _ = dict.set_item(key.to_string(), val);
        }
        dict
    }

    /// The purpose of this type is to be able to use floating point values to create a HashSet.
    /// This is used below when attempting to determine which labeling functionality should be
    /// dispached to based on the number of unique values in the dataset.
    /// Floating point values are typically not supported because they do not implement Eq. This is
    /// simply a wrapper type to manually override the convetional edge case behavior such as
    /// comparison of NaNs. This type uses the same convention as the ordered_float package,
    /// and diverges from the IEEE standard for some NaN properties.
    #[derive(PartialEq)]
    struct OrderedFloat(f64);

    impl PartialOrd for OrderedFloat {
        /// None will ever be returned here. The only time the interanl floating point values are
        /// comapred here is when it is validated that neither are NaN.
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            match (self.0.is_nan(), other.0.is_nan()) {
                (true, true) => Some(std::cmp::Ordering::Equal),
                (true, false) => Some(std::cmp::Ordering::Greater),
                (false, true) => Some(std::cmp::Ordering::Less),
                (false, false) => self.0.partial_cmp(&other.0),
            }
        }
    }

    impl Ord for OrderedFloat {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // The PartialOrd implmenetation does not present a case where None is returned.
            self.partial_cmp(other).unwrap()
        }
    }

    impl Eq for OrderedFloat {}

    impl std::hash::Hash for OrderedFloat {
        fn hash<H: std::hash::Hasher>(&self, hash_state: &mut H) {
            hash_state.write_u64(self.0.to_bits())
        }
    }

    #[derive(PartialEq)]
    pub enum PassedType {
        Float,
        Integer,
        String,
    }

    /// Utility to extract the type from a PyUntypedArray to work with the dyanmic typed semantics
    /// in Python. Here we attempt to extract the type so we can then generate a Rust data
    /// container that is owned from the data passed by the Python wrapper function.
    pub(crate) fn determine_type(py: Python<'_>, array: &Bound<'_, PyUntypedArray>) -> PassedType {
        let element_type = array.dtype();

        // Accepted types here are basic primitives, other "object" types are not supported.
        if element_type.is_equiv_to(&dtype::<f64>(py)) | element_type.is_equiv_to(&dtype::<f32>(py))
        {
            PassedType::Float
        } else if element_type.is_equiv_to(&dtype::<i32>(py))
            | element_type.is_equiv_to(&dtype::<i64>(py))
            | element_type.is_equiv_to(&dtype::<i16>(py))
        {
            PassedType::Integer
        } else {
            PassedType::String
        }
    }

    fn copy_into_rust_type<'py, T: Clone + FromPyObject<'py>>(
        arr: &Bound<'py, PyUntypedArray>,
    ) -> PyResult<Vec<T>> {
        let mut copied_data: Vec<T> = Vec::with_capacity(arr.len());
        for item in arr.try_iter()? {
            copied_data.push(item?.extract()?);
        }
        Ok(copied_data)
    }

    // Utility to resolve the number of unique values in the dataset passed to determine which
    // labeling logic to dispatch to. Takes in a closure to map the values to an OrderedFloat so the data
    // can be hashed into a HashSet. OrderedFloat is used to allow for the precision of floating
    // point values.
    fn resolve_num_unique_values<T>(arr: &[T], f: &dyn Fn(&T) -> OrderedFloat) -> usize {
        let arr_set: std::collections::HashSet<OrderedFloat> =
            arr.iter().map(|v| (*f)(v)).collect();
        arr_set.len()
    }

    // Handles untyped nature of python data. Determines type and labels accordingly. This function
    // will error if the underlying type identification is incorrect.
    pub(crate) fn apply_label<'py>(
        py: Python<'_>,
        array: &Bound<'_, PyUntypedArray>,
        label: Bound<'py, PyAny>,
    ) -> Result<Vec<i16>, Box<dyn Error>> {
        let pred_type = determine_type(py, &array);

        // Based on the observed type of the data in the PyUntypedArray, copy the data into an
        // owned rust container. Given that these come from numpy arrays, the type in the array
        // should be uniform.
        let labeled_array: Vec<i16> = match pred_type {
            PassedType::String => {
                let data_vec: Vec<String> = copy_into_rust_type(array)?;
                if !label.is_instance_of::<PyString>() {
                    return Err("string".into());
                }

                let data_label: String = label.extract::<String>()?;
                apply_label_discrete(&data_vec, &data_label)
            }
            PassedType::Float => {
                let data_vec: Vec<f64> = copy_into_rust_type(array)?;

                // handling users passing float vs int as label_or_threshold
                let data_label: f64 = if label.is_instance_of::<PyFloat>() {
                    label.extract::<f64>()?
                } else if label.is_instance_of::<PyInt>() {
                    label.extract::<i64>()? as f64
                } else {
                    return Err("float".into());
                };

                let f = Box::new(|v: &f64| OrderedFloat(*v));
                let num_unique = resolve_num_unique_values(&data_vec, &f);

                if num_unique == 2 {
                    apply_label_discrete(&data_vec, &data_label)
                } else {
                    apply_label_continuous(&data_vec, &data_label)
                }
            }
            PassedType::Integer => {
                let data_vec: Vec<i64> = copy_into_rust_type(array)?;

                // handling users passing float vs int as label_or_threshold
                let data_label: i64 = if label.is_instance_of::<PyFloat>() {
                    label.extract::<f64>()? as i64
                } else if label.is_instance_of::<PyInt>() {
                    label.extract::<i64>()?
                } else {
                    return Err("float".into());
                };

                let f = Box::new(|v: &i64| OrderedFloat(*v as f64));
                let num_unique = resolve_num_unique_values(&data_vec, &f);

                if num_unique == 2 {
                    apply_label_discrete(&data_vec, &data_label)
                } else {
                    apply_label_continuous(&data_vec, &data_label)
                }
            }
        };
        Ok(labeled_array)
    }
}

#[inline]
pub(crate) fn bool_to_f32(v: bool) -> f32 {
    v as usize as f32
}

pub(crate) trait ApplyThreshold {
    fn apply_threshold(&self, threshold: &Self) -> Self;
}

impl ApplyThreshold for f32 {
    fn apply_threshold(&self, threshold: &f32) -> f32 {
        bool_to_f32(self.ge(threshold))
    }
}

pub(crate) struct ConfusionPushPayload {
    pub(crate) true_gt: bool,
    pub(crate) true_pred: bool,
}

pub(crate) struct ConditionalConfusionPushPayload {
    pub(crate) true_gt: bool,
    pub(crate) true_pred: bool,
    pub(crate) cond: bool,
}
// Type to hold confusion matrix for binary type classification. This allows for much cheaper
// computation of many of the classic classification metrics
#[derive(Default, Debug, PartialEq)]
pub(crate) struct ConfusionMatrix {
    pub(crate) true_p: f32,
    pub(crate) false_p: f32,
    pub(crate) true_n: f32,
    pub(crate) false_n: f32,
}

impl ConfusionMatrix {
    pub(crate) fn clear(&mut self) {
        self.true_p = 0_f32;
        self.false_p = 0_f32;
        self.false_n = 0_f32;
        self.true_n = 0_f32;
    }

    pub(crate) fn len(&self) -> f32 {
        self.true_p + self.false_p + self.false_n + self.true_n
    }

    /// Push a single record update to the confusion matrix.
    #[inline]
    pub(crate) fn push(&mut self, payload: ConfusionPushPayload) {
        let ConfusionPushPayload { true_gt, true_pred } = payload;
        self.true_p += bool_to_f32(true_gt && true_pred);
        self.false_p += bool_to_f32(!true_gt && true_pred);
        self.true_n += bool_to_f32(!true_gt && !true_pred);
        self.false_n += bool_to_f32(true_gt && !true_pred);
    }

    /// Method to conditionally update the confusion matrix state. The intention here to allow
    /// upstream code to avoid branching based on which demographic class belongs to. Allows for
    /// inline branchless computations to be maintained, and all update logic stays in the same
    /// place.
    #[inline]
    pub(crate) fn conditional_push(&mut self, payload: ConditionalConfusionPushPayload) {
        let ConditionalConfusionPushPayload {
            cond,
            true_gt,
            true_pred,
        } = payload;
        self.true_p += bool_to_f32(cond && (true_gt && true_pred));
        self.false_p += bool_to_f32(cond && (!true_gt && true_pred));
        self.true_n += bool_to_f32(cond && (!true_gt && !true_pred));
        self.false_n += bool_to_f32(cond && (true_gt && !true_pred));
    }

    /// Pushes a batch dataset of inference and ground truth examples into the confusion matrix.
    /// The true/false label is applied by the closure passed to compute the result. This allows
    /// for dynamic labeling, and this method can be used accross binary classification and
    /// logistic regression contexts.
    pub(crate) fn push_dataset<T, F: Fn(&T) -> bool>(
        &mut self,
        y_true: &[T],
        y_pred: &[T],
        label_f: F,
    ) {
        for (y_true, y_pred) in crate::zip_iters!(y_true, y_pred) {
            self.push(ConfusionPushPayload {
                true_gt: label_f(y_true),
                true_pred: label_f(y_pred),
            });
        }
    }
}

/// Enum to differentiate between an array that is going to be segmented by label versus by a
/// threshold. Dicrete data should be segmented by a label and continuous data should be segmented
/// by a threshold. For example a categorical dataset should use the label variant, where as a
/// linear regression prediction dataset should use the threshold.
#[derive(PartialEq, Clone)]
pub enum BiasSegmentationType {
    Label,
    Threshold,
}

/// Type to identify how bias data will be segmented into positive and negative classes. This can
/// be applied to feature classes, prediction classes, and ground truth classes. All bias analysis
/// in this crate is based on segmentation of all data into a positive and negative class. The
/// feature data is segmented into "advantaged" and "disadvantaged" groups, where predictions and
/// ground truth examples are segmented into "positive" and "negative" outcomes.
///
/// Continuous data is segmented into positive and negative classes based on some threshold, and
/// discrete data is segmented based on some label, this labeling can either dictate the advantaged
/// or disadvantaged group. To gain analysis on multiple classes, the same data can be segmented in
/// multiple ways for different dimensions of analysis/montioring. Given the labeling involved, the
/// type that performs the segmentation must implement 'PartialOrd' and 'PartialEq'. The
/// segmentation operation is dictated by the 'BiasSegmentationType' passed on construction.
///
/// This intentionally does implement Clone, to avoid limiting possible types that can be used.
/// This type is cheap enough to reconstuct when needed
pub struct BiasSegmentationCriteria<T>
where
    T: PartialOrd,
{
    value: T,
    stype: BiasSegmentationType,
}

impl<T> BiasSegmentationCriteria<T>
where
    T: PartialOrd + PartialEq,
{
    pub fn new(value: T, stype: BiasSegmentationType) -> BiasSegmentationCriteria<T> {
        BiasSegmentationCriteria { value, stype }
    }

    /// Compute the class segmentation of the value passed. true refers to the abritrary label associated
    /// with the favored class and false refers to the disfavored class.
    #[inline]
    pub(crate) fn label(&self, value: &T) -> bool {
        if self.stype == BiasSegmentationType::Label {
            self.value == *value
        } else {
            self.value <= *value
        }
    }
}

/// Type to organize bias monitor inputs and differentiate between a continuous and a discrete
/// feature. This can be defined once if bias is performed across multiple features, the baseline
/// data just needs to live as long as all analysis computations. This is also cheap to clone, for
/// easy resuse.
pub struct BiasDataPayload<'a, T>
where
    T: PartialOrd,
{
    data: &'a [T],
    segmentation_criteria: BiasSegmentationCriteria<T>,
}

impl<'a, T> BiasDataPayload<'a, T>
where
    T: PartialOrd,
{
    /// Constructor for BiasDataPayload. The type in the slice must be the same as the type passed
    /// for the segmentation criteria.
    pub fn new_from_parts(
        data: &'a [T],
        value: T,
        stype: BiasSegmentationType,
    ) -> BiasDataPayload<'a, T> {
        let segmentation_criteria = BiasSegmentationCriteria { value, stype };

        BiasDataPayload {
            data,
            segmentation_criteria,
        }
    }

    /// Constructor from already formed 'BiasSegmentationCriteria'.
    pub fn new_from_criteria(
        data: &'a [T],
        segmentation_criteria: BiasSegmentationCriteria<T>,
    ) -> BiasDataPayload<'a, T> {
        BiasDataPayload {
            data,
            segmentation_criteria,
        }
    }

    pub(crate) fn generate_labeled_data(&self) -> Vec<i16> {
        if self.segmentation_criteria.stype == BiasSegmentationType::Label {
            apply_label_discrete(self.data, &self.segmentation_criteria.value)
        } else {
            apply_label_continuous(self.data, &self.segmentation_criteria.value)
        }
    }
}

#[inline]
fn apply_label_discrete<T>(array: &[T], label: &T) -> Vec<i16>
where
    T: PartialEq<T>,
{
    let labeled_array: Vec<i16> = array
        .iter()
        .map(|value| if value == label { 1_i16 } else { 0_i16 })
        .collect();
    labeled_array
}

#[inline]
fn apply_label_continuous<T>(array: &[T], threshold: &T) -> Vec<i16>
where
    T: PartialOrd<T>,
{
    let labeled_array: Vec<i16> = array
        .iter()
        .map(|value| if value >= threshold { 1_i16 } else { 0_i16 })
        .collect();
    labeled_array
}

#[cfg(test)]
mod data_handler_tests {
    use super::*;

    #[test]
    fn test_label_bias_seg() {
        // test that the correct label is assigned when the segmentation criteria is label
        let seg = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        assert_eq!(seg.label(&0), false);
        assert_eq!(seg.label(&1), true);
    }

    #[test]
    fn test_threshold_bias_seg() {
        // test that the correct label is assigned when the segmentation criteria is label
        let seg = BiasSegmentationCriteria::new(0.5_f32, BiasSegmentationType::Threshold);
        assert_eq!(seg.label(&0.1_f32), false);
        assert_eq!(seg.label(&0.5_f32), true);
        assert_eq!(seg.label(&0.6_f32), true);
    }

    #[test]
    fn test_confusion_push() {
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push(ConfusionPushPayload {
            true_gt: false,
            true_pred: false,
        });
        c_matrix.push(ConfusionPushPayload {
            true_gt: true,
            true_pred: true,
        });
        c_matrix.push(ConfusionPushPayload {
            true_gt: false,
            true_pred: true,
        });
        c_matrix.push(ConfusionPushPayload {
            true_gt: true,
            true_pred: false,
        });
        dbg!(&c_matrix);
        assert!(
            c_matrix.true_n == 1_f32
                && c_matrix.true_p == 1_f32
                && c_matrix.false_n == 1_f32
                && c_matrix.false_p == 1_f32
        );

        c_matrix.push(ConfusionPushPayload {
            true_gt: true,
            true_pred: false,
        });
        assert!(
            c_matrix.true_n == 1_f32
                && c_matrix.true_p == 1_f32
                && c_matrix.false_n == 2_f32
                && c_matrix.false_p == 1_f32
        );

        c_matrix.push(ConfusionPushPayload {
            true_gt: true,
            true_pred: true,
        });
        assert!(
            c_matrix.true_n == 1_f32
                && c_matrix.true_p == 2_f32
                && c_matrix.false_n == 2_f32
                && c_matrix.false_p == 1_f32
        );
        c_matrix.push(ConfusionPushPayload {
            true_gt: false,
            true_pred: true,
        });
        assert!(
            c_matrix.true_n == 1_f32
                && c_matrix.true_p == 2_f32
                && c_matrix.false_n == 2_f32
                && c_matrix.false_p == 2_f32
        );
        c_matrix.push(ConfusionPushPayload {
            true_gt: false,
            true_pred: false,
        });
        assert!(
            c_matrix.true_n == 2_f32
                && c_matrix.true_p == 2_f32
                && c_matrix.false_n == 2_f32
                && c_matrix.false_p == 2_f32
        );
    }
}
