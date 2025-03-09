# fair-ml
Custom implementation of bias analysis for machine learning models. Based on the AWS SageMaker bias models, though this accommodates bias based on protected classes not used in model training.

[Background](#background)
[Modules](#modules)
[Usage](#usage)

## Background
Governance in AI systems is becoming more important. Many large cloud providers and other vendors provide services for these analyses, but they are expensive and sometimes over-engineered. 

Bias analysis works by seperating a feature into two demographic groups, and predictions and outcomes into positive and negative outcomes. The goal in this analysis is to quantify the divergance between how the model and true outcomes favors one demographic.

To do so everything is segmented into two distinct groups representing favored and disfavord groups.

Additionally, there is a module to monitor overall performance at runtime. The nature of ML makes it difficult to have unit tests, and to ensure performance at runtime. ML deployments are different from other software deployments given the inability to ensure accurate results. Our assertions need to be done after the fact. Though, most deploy the model and let it run. There often is not a consistent effort to ensure accuracy in the model predictions over the entire lifetime of the model. There are services available to do this with different vendors (ie AWS SageMaker), but this requires significant cost and compute; they also tend to be slow. 

The goal of this package is to offer that kind of ML observability at no cost, and limited resources needed.

The logic is written in Rust, with a python interface to let users pass in some different types (ie not only numpy arrays but also python lists, if for whatever reason someones likes to use python lists instead of numpy arrays), and an easy to use interface. The performance penalty there is minimal and makes use quite a bit easier. I generally feel that users do not need to pay someone a lot of money for services that do not require it.

This package would not be possbile without the great work done by the contributors of PYO3, that work is wonderdul.

## Modules
There are two modules that have very similar structure.

### data_bias
- perform_anaylsis
    - Arguments
        - feature: Union[List[int, float, string], NDArray]
            - an array of the feature values
        - ground_truth: Union[List[int, float, string], NDArray]
            - an array of the ground truth values
        - feature_label_or_threshold: Union[str, int, float]
            - the feature label or threshold for segmentation into facets
        - ground_truth_label_or_threshold: Union[str, int, float]
            - the ground truth label or threshold for positive/negative outcome labeling
    - Returns
        - dict
        - the analysis results
- runtime_comparison
    - Arguments
        - baseline: dict
            - the result of analysis at training time
        - current: dict
            - the result of the current data being evaluated
        - threshold: Optional[float] = 0.10
            - The allowable difference threshold between baseline divergence between facets and current data
    - Returns
        - dict
        - runtime check results

### model_bias
- perform_anaylsis
    - Arguments
        - feature: Union[List[int, float, string], NDArray]
            - an array of the feature values
        - ground_truth: Union[List[int, float, string], NDArray]
            - an array of the ground truth values
        - predictions: Union[List[int, float, string], NDArray]
            - an array of the predictions
        - feature_label_or_threshold: Union[str, int, float]
            - the feature label or threshold for segmentation into facets
        - ground_truth_label_or_threshold: Union[str, int, float]
            - the ground truth label or threshold for positive/negative outcome labeling
        - prediction_label_or_threshold: Union[str, int, float]
            - the prediction label or threshold for positive/negative outcome labeling
    - Returns
        - dict
        - the analysis results
- runtime_comparison
    - Arguments
        - baseline: dict
            - the result of analysis at training time
        - current: dict
            - the result of the current data being evaluated
        - threshold: Optional[float] = 0.10
            - The allowable difference threshold between baseline divergence between facets and current data
    - Returns
        - dict
        - runtime check results

## Usage
The intended usage for this package is for monitoring machine learning models for bias. The high level principle is that users perform bias analysis at training time, preferrably on a holdout set, and this serves as the baseline data.
When a model is deployed, users save the data used to score, the predictions, and collect the ground truth as it becomes available. At some cadence, depending on how quickly ground truth becomes available, analysis is then performed on the features of intereset for bias, the ground truth, and the predictions. The result of the comparison then can be used for comparison.

At runtime something may look like this:
```python
from fair_ml import data_bias, model_bias

"""baseline"""
data_bias_baseline = data_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...
)

model_bias_baseline = model_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    predictions=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...,
    prediction_label_or_threshold=...
)

"""save these somewhere"""


"""At runtime"""

# saved runtime data
data_bias_curr = data_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...
)

model_bias_curr = model_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    predictions=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...,
    prediction_label_or_threshold=...
)

# load in baseline
data_bias_baseline = ...
model_bias_baseline = ...

data_bias_monitor = data_bias.runtime_comparison(
    baseline=data_bias_baseline,
    current=data_bias_curr
)

if data_bias_monitor.get("passed"):
    "Data bias passed"
else:
    "Data bias failed"
    # may need to check to see if the model needs to be retrained

model_bias_monitor = data_bias.runtime_comparison(
    baseline=model_bias_baseline,
    current=model_bias_curr
)

if model_bias_monitor.get("passed"):
    "Model bias passed"
else:
    "Model bias failed"
    # may need to check to see if the model needs to be retrained


```

When the results fail a check, this indicates we may need to retrain the model depending on the severity of the divergence. 

The evaluation best serves as a background job.
