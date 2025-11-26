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

    pub fn report_to_py_dict<'py, T>(py: Python<'py>, report: HashMap<T, f32>) -> Bound<'py, PyDict>
    where
        T: ToString,
    {
        let dict = PyDict::new(py);
        for (key, val) in report.into_iter() {
            let _ = dict.set_item(key.to_string(), val);
        }
        dict
    }

    #[derive(PartialEq)]
    pub enum PassedType {
        Float,
        Integer,
        String,
    }
    pub fn determine_type(py: Python<'_>, array: &Bound<'_, PyUntypedArray>) -> PassedType {
        let element_type = array.dtype();

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

    pub fn apply_label<'py>(
        py: Python<'_>,
        array: &Bound<'_, PyUntypedArray>,
        label: Bound<'py, PyAny>,
    ) -> Result<Vec<i16>, Box<dyn Error>> {
        let pred_type = determine_type(py, &array);
        let arr_len = array.len();
        let iter = &array.try_iter()?;

        let labeled_array: Vec<i16> = match pred_type {
            PassedType::String => {
                let mut data_vec: Vec<String> = Vec::with_capacity(arr_len);
                for item in iter {
                    let data = item?.extract::<String>()?;
                    data_vec.push(data);
                }
                if !label.is_instance_of::<PyString>() {
                    return Err("string".into());
                }

                let data_label: String = label.extract::<String>()?;
                apply_label_discrete(&data_vec, &data_label)
            }
            PassedType::Float => {
                let mut data_vec: Vec<f64> = Vec::with_capacity(arr_len);
                for item in iter {
                    let data = item?.extract::<f64>()?;
                    data_vec.push(data);
                }
                // handling users passing float vs int as label_or_threshold
                let data_label: f64 = if label.is_instance_of::<PyFloat>() {
                    label.extract::<f64>()?
                } else if label.is_instance_of::<PyInt>() {
                    label.extract::<i64>()? as f64
                } else {
                    return Err("float".into());
                };

                let data_set: std::collections::HashSet<i32> = data_vec
                    .iter()
                    .map(|value| *value as i32)
                    .collect::<std::collections::HashSet<_>>();

                if data_set.len() == 2 {
                    apply_label_discrete(&data_vec, &data_label)
                } else {
                    apply_label_continuous(&data_vec, &data_label)
                }
            }
            PassedType::Integer => {
                let mut data_vec: Vec<i64> = Vec::with_capacity(arr_len);
                for item in iter {
                    let data = item?.extract::<i64>()?;
                    data_vec.push(data);
                }

                let data_set: std::collections::HashSet<i32> = data_vec
                    .iter()
                    .map(|value| *value as i32)
                    .collect::<std::collections::HashSet<_>>();

                // handling users passing float vs int as label_or_threshold
                let data_label: i64 = if label.is_instance_of::<PyFloat>() {
                    label.extract::<f64>()? as i64
                } else if label.is_instance_of::<PyInt>() {
                    label.extract::<i64>()?
                } else {
                    return Err("float".into());
                };

                if data_set.len() == 2 {
                    apply_label_discrete(&data_vec, &data_label)
                } else {
                    apply_label_continuous(&data_vec, &data_label)
                }
            }
        };
        Ok(labeled_array)
    }
}

#[derive(PartialEq)]
pub enum BiasSegmentationType {
    Continuous,
    Discrete,
}

pub struct BiasDataPayload<'a, T> {
    data: &'a [T],
    segmentation_criteria: T,
    segmentation_type: BiasSegmentationType,
}

impl<'a, T> BiasDataPayload<'a, T>
where
    T: PartialEq + PartialOrd,
{
    pub fn new(
        data: &'a [T],
        segmentation_criteria: T,
        segmentation_type: BiasSegmentationType,
    ) -> BiasDataPayload<'a, T> {
        BiasDataPayload {
            data,
            segmentation_criteria,
            segmentation_type,
        }
    }

    pub(crate) fn generate_labeled_data(&self) -> Vec<i16> {
        if self.segmentation_type == BiasSegmentationType::Discrete {
            apply_label_discrete(self.data, &self.segmentation_criteria)
        } else {
            apply_label_continuous(self.data, &self.segmentation_criteria)
        }
    }
}

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
