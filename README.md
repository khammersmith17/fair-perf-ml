# fair-ml
Custom implementation of bias analysis for machine learning models. Based on the AWS SageMaker bias models, though this accommodates bias based on protected classes not used in model training.

[Background](#background)
[Modules](#modules)
[Usage](#usage)
[Future](#future)

## Background
Governance in AI systems is becoming more important. Many large cloud providers and other vendors provide services for these analyses, but they are expensive and sometimes over-engineered. The overall goal of this project is to provide a lightweight monitoring framework for machine learning models, that is hopefully easy to use.

The core idea of this project is to provide composable pieces to create you own ML observavility service. The tools available on the market force users into a rather rigid pattern of use, and are built in such a way that they are not "editable" after definition, at least the tools I have used in my experience. For example, once the "schema" is defined for a model, you cannot later add a non model feature to the monitoring set for bias, where logically, this should not be a constraint as long as there is an ability to define a mapping between inference scores, ground truth, and feature values.

Bias analysis works by seperating a feature into two demographic groups, and predictions and outcomes into positive and negative outcomes. The goal in this analysis is to quantify the divergance between how the model and true outcomes favors one demographic.

To do so everything is segmented into two distinct groups representing favored and disfavord groups, though labeling one group as "favored" is a semantic detail. In reality, it does not matter which group is which, only that there is logical segmentation logic to distinctly define 2 groups.

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
        - dict: runtime check results

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
        - dict: the analysis results
- runtime_comparison
    - Arguments
        - baseline: dict
            - the result of analysis at training time
        - current: dict
            - the result of the current data being evaluated
        - threshold: Optional[float] = 0.10
            - The allowable difference threshold between baseline divergence between facets and current data
    - Returns
        - dict: runtime check results
### model_perf
- linear_regression_analysis
    - Arguments
        - y_true: Union[NDArray, List[Union[int, float]]]
            - the ground truth values
        - y_pred: Union[NDArray, List[Union[int, float]]]
            - the predictions
    - Returns
        - dict
            - the results of the analysis
- logistic_regression_analysis
    - Arguments
        - y_true: Union[NDArray, List[Union[int, float]]]
            - the ground truth values
        - y_pred: Union[NDArray, List[Union[int, float]]]
            - the predictions
        - decision_threshold: float
            - the decision threshold of the logisitc regression model
    - Returns
        - dict
            - the results of the analysis
- binary_classification_analysis
    - Arguments
        - y_true: Union[NDArray, List[Union[int, float]]]
            - the ground truth values
        - y_pred: Union[NDArray, List[Union[int, float]]]
            - the predictions
    - Returns
        - dict
            - the results of the analysis
- runtime_check_full
    - Arguments
        - latest
            - the result from the analysis of the data to be compared to the baseline
        - baseline
            - the baseline analysis results
        - threshold
            - the drift threshold to evaluate model health
    - Returns
        - dict
            - the runtime check results
- partial_runtime_check
    - Arguments
        - latest
            - the result from the analysis of the data to be compared to the baseline
        - baseline
            - the baseline analysis results
        - threshold
            - the drift threshold to evaluate model healh
        - metrics: List[str]
            - The list of metrics to evaluate on
            - This is useful when you are only interested in a subset of metrics for model health
            - The metrics need to be aligned for the model type
            - LinearRegression accepted values
                - RootMeanSquaredError
                - MeanSquaredError
                - MeanAbsoluteError
                - RSquared
                - MaxError
                - MeanSquaredLogError
                - RootMeanSquaredLogError
                - MeanAbsolutePercentageError
            - BinaryClassification accepted values
                - BalancedAccuracy
                - PrecisionPositive
                - PrecisionNegative
                - RecallPositive
                - RecallNegative
                - Accuracy
                - F1Score
                - LogLoss
            - LogisticRegression accepted values
                - BalancedAccuracy
                - PrecisionPositive
                - PrecisionNegative
                - RecallPositive
                - RecallNegative
                - Accuracy
                - F1Score
                - LogLoss

    - Returns
        - dict
            - the runtime check results

## Usage
- The intended usage for this package is for monitoring machine learning models for bias. 
    - The high level principle is that users perform bias and performance analysis at training time, preferrably on a holdout set, and this serves as the baseline data.
    - See usage/usage.py in the repo for a rough example of the what the flow would look like
- When a model is deployed, users save the data used to score, the predictions, and collect the ground truth as it becomes available. 
- At some cadence, depending on how quickly ground truth becomes available, analysis is then performed on the features of intereset for bias, the ground truth, and the predictions.
- The result of the evaluation then can be used for comparison.
- Where this fits in a system architecture depends on the nature of the model deloyment.
    - Model is being served on an API
        - This process best fits in as a background job
        - Over time, data is collected and persisted
        - When ground truth data becomes available, a job is kicked off
        - Using some cron scheduler is the most straight forward approach
    - Model scores are generated in a batch job
        - This process can be run along with batch inference job
        - In this case, we would be evaulting new data from a previous run, where we have ground truth, not the data apart of the scoring run
- Whether the job is done as a background job on a server or a batch scoring job, the logic looks the same
- Before the model is deployed we need to generate our baseline for every monitor
    - Some prerequisite work is required for the bias monitors generaly, to identify features and thresholds to evaluate on
- These should be saved somewhere, ie some persistent storage
- During deployment, retrive the baseline data, retrieve the saved inference data for you records, and the ground truth data
    - The inference data (ie features and inference score) should be persisted as well
- Use the ground truth and inference data to run analysis for runtime
- Run the methods to compare to the baseline results
- You should probably know what you want to do in the case of drift


### Bias Evaluations
For the bias evaluations, it important to persist the feature data used to make with the prediction in addition to the inference score. To save storage cost or runtime costs, the features that are being evaluated for bias can be stored, though it is probably good practice to store all the data for additional training on the model.

Additionally, some pre work is required to determine first, the features to be evaluated for bias, and the logic to partition the data into advantaged and disadvtanged groups, and positive and negative outcomes.

At runtime something may look like this:
```python
from fair_perf_ml import data_bias, model_bias

"""baseline"""
data_bias_baseline = data_bias.perform_analysis(
    feature=[...], # array of feature data 
    ground_truth=[...], # array of the ground truth data
    feature_label_or_threshold=..., # the label or segmentation threshold for the feature
    ground_truth_label_or_threshold=... # teh array or segmentation threshold for the ground truth
)

model_bias_baseline = model_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    predictions=[...], # an array of the predictions
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...,
    prediction_label_or_threshold=...
)

##### Save these somewhere to be used during model deployment #####


##### At runtime #####

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

The json schema for the Data Bias evaluations will look as such:
```json
{
    'ClassImbalance': float,
    'DifferenceInProportionOfLabels': float,
    'KlDivergence': float,
    'JsDivergence': float,
    'LpNorm': float,
    'TotalVarationDistance': float,
    'KolmorogvSmirnov': float
}
```

And for the Model Bias evaluations:
```json
{
    'DifferenceInPositivePredictedLabels': float,
    'DisparateImpact': float,
    'AccuracyDifference': float,
    'RecallDifference': float,
    'DifferenceInConditionalAcceptance': float,
    'DifferenceInAcceptanceRate': float,
    'SpecialityDifference': float,
    'DifferenceInConditionalRejection': float,
    'DifferenceInRejectionRate': float,
    'TreatmentEquity': float,
    'ConditionalDemographicDesparityPredictedLabels': float,
    'GeneralizedEntropy': float
}

```


### Model Performance
When the results fail a check, this indicates we may need to retrain the model depending on the severity of the divergence.


```python
from fair_perf_ml import model_perf

# depending on your model type
# one of the following
bl = model_perf.linear_regression_analysis(
    y_true=bl_true,
    y_pred=bl_pred
)

bl = model_perf.binary_classification_analysis(
    y_true=bl_true,
    y_pred=bl_pred
)

bl = model_perf.logistic_regression_analysis(
    y_true=bl_true,
    y_pred=bl_pred,
    decision_threshold=0.5
)

##### SAVE THESE RESULTS SOMEWHERE ######


##### AT RUNTIME #####


runtime = model_perf.linear_regression_analysis(
    y_true=rt_true,
    y_pred=rt_pred
)


# for full check
eval_results = model_perf.runtime_check_full(
    baseline=bl,
    latest=runtime
)


# for partial chekc

eval_results = model_perf.partial_runtime_check(
    baseline=bl,
    latest=runtime,
    metrics=[
        "RootMeanSquaredError",
        "MeanSquaredError",
        "MeanAbsoluteError",
        "RSquared"
    ]
)

```


The evalution result body for Linear Regeression:
```json
{
    "modelType": "LinearRegression",
    "performanceData": {
        "RootMeanSquaredError": float,
        "MeanSquaredError": float,
        "MeanAbsoluteError": float,
        "RSquared": float,
        "MaxError": float,
        "MeanSquaredLogError": float,
        "RootMeanSquaredLogError": float,
        "MeanAbsolutePercentageError": float
    }
}
```


Evaluation reults for BinaryClassification:
```json
{
    "modelType": "BinaryClassification",
    "performanceData": {
        "BalancedAccuracy": float,
        "PrecisionPositive": float,
        "PrecisionNegative": float,
        "RecallPositive": float,
        "RecallNegative": float,
        "Accuracy": float,
        "F1Score":float
    }
}
```

Evaluation results for LogisticRegression:
```json
{
    "modelType": "LogisticRegression",
    "performanceData": {
        "BalancedAccuracy": float,
        "PrecisionPositive": float,
        "PrecisionNegative": float,
        "RecallPositive": float,
        "RecallNegative": float,
        "Accuracy": float,
        "F1Score": float,
        "LogLoss": float
    }
}


```

All evaluation jobs will have the same structure:
```json
{
    "passed": bool,
    "fail_report": {
        // all the metrics that did not meet thresholds
    }
}

```

Following the evaluation job, one might have some to handle when a comparison job returns some metric failures. This may be some alerting logic, some automated model retraining logic, or some other remediation.


## Future
- Going forward, there are some more things I would like to add
- These include
    - multi-dimensional segmenter
        - This can be done as is, but requires more effort from the user
    - batteries included methods to perform runtime analysis and do baseline comparison in a single API
