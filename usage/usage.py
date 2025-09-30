"""
Demo of fair-perf-ml using an abalone dataset example.
Work through the logic of computing baseline evaluations.
"""

from typing import Tuple
import pandas as pd
from numpy.typing import NDArray
from fair_perf_ml import data_bias, model_bias, model_perf


class TestData:
    def __init__(self, split: float = 0.7):
        self.data = pd.read_csv("test_data.csv")
        size = self.data.shape[0]
        self.baseline = self.data.iloc[int(size * split) :, :]
        self.runtime = self.data.iloc[: int(size * split), :]

    @property
    def baseline_ground_truth(self):
        return self.baseline["true"].to_numpy()

    @property
    def baseline_predictions(self):
        return self.baseline["pred"].to_numpy()

    @property
    def baseline_feature_sex(self):
        return self.baseline["sex"].to_numpy()

    @property
    def runtime_ground_truth(self):
        return self.runtime["true"].to_numpy()

    @property
    def runtime_predictions(self):
        return self.runtime["pred"].to_numpy()

    @property
    def runtime_feature_sex(self):
        return self.baseline["sex"].to_numpy()


def run_baseline(test_data: TestData) -> Tuple[dict, dict, dict]:
    """
    Runs a baseline analysis for Data Bias, Model Performance, and Model Bias.
    """
    data_bias_baseline = data_bias.perform_analysis(
        feature=test_data.baseline_feature_sex,
        ground_truth=test_data.baseline_ground_truth,
        feature_label_or_threshold="M",
        ground_truth_label_or_threshold=15,
    )

    model_bias_baseline = model_bias.perform_analysis(
        feature=test_data.baseline_feature_sex,
        ground_truth=test_data.baseline_ground_truth,
        feature_label_or_threshold="M",
        ground_truth_label_or_threshold=15,
        predictions=test_data.baseline_predictions,
        prediction_label_or_threshold=15.0,
    )

    model_perf_baseline = model_perf.linear_regression_analysis(
        y_true=test_data.baseline_ground_truth, y_pred=test_data.baseline_predictions
    )
    return data_bias_baseline, model_bias_baseline, model_perf_baseline



def runtime_model_performance(runtime_gt: NDArray, runtime_preds: NDArray, baseline: dict) -> dict:
    """
    An example showing the use
    """
    runtime_analysis =  model_perf.linear_regression_analysis(
        y_true=runtime_gt, y_pred=runtime_preds
    )
    check_results = model_perf.runtime_check_full(
        baseline=baseline,
        latest=runtime_analysis,
    )

    return check_results


def runtime_model_bias(features: NDArray, runtime_gt: NDArray, runtime_preds: NDArray, baseline: dict) -> dict:
    model_bias_runtime = model_bias.perform_analysis(
        feature=features,
        ground_truth=runtime_gt,
        feature_label_or_threshold="M",
        ground_truth_label_or_threshold=15,
        predictions=runtime_preds,
        prediction_label_or_threshold=15.0,
    )

    runtime_check = model_bias.runtime_comparison(
        baseline=baseline, comparison=model_bias_runtime
    )
    
    return runtime_check


def runtime_data_bias(features: NDArray, runtime_gt: NDArray, baseline: dict) -> dict:
    data_bias_runtime = data_bias.perform_analysis(
        feature=features,
        ground_truth=runtime_gt,
        feature_label_or_threshold="M",
        ground_truth_label_or_threshold=15,
    )

    runtime_check = data_bias.runtime_comparison(
        baseline=baseline, latest=data_bias_runtime
    )

    return runtime_check


def fetch_runtime_data(test_data: TestData) -> Tuple[NDArray, NDArray, NDArray]:
    return test_data.runtime_feature_sex, test_data.runtime_ground_truth, test_data.runtime_predictions


def evaluate_perf(runtime_perf_check: dict): # pyright: ignore
    raise NotImplementedError

def evaluate_model_bias(runtime_check_mb: dict): # pyright: ignore
    raise NotImplementedError

def evaluate_data_bias(runtime_check_db: dict): # pyright: ignore
    raise NotImplementedError



if __name__ == "__main__":
    """
    This script is to simulate behavior and how this module might be used in production.
    1. Before deployment of the model, generate all baselines for the model performance,
        and any features to be monitored for bias.
        - Save these baseline evaluations somewhere that can be fetched at runtime
            - ie in a database, object store, disk, etc...
    2. At runtime, log all features used in inference along with the inference score along with some key
    3. Collect ground truth (actual outcomes) and join on the feature and inference set that has been logged.
    4. Follow the below procedure
        1. Fetch baselines
        2. Perform analysis on the logged features, inference, and ground truth
        3. Pass both the baseline and runtime performance evalutions to the associated comparison functions
        4. Analyze results and implement your own logic to determine if model performance is accpetable for your application
    """

    # LOADING IN TEST DATA...
    test_data = TestData()

    # COMPUTE BASELINES
    db_baseline, mb_baseline, perf_baseline = run_baseline(test_data)
    # STORE BASELINE SOMEWHERE THAT CAN FETCHED AT RUNTIME


    # RUNTIME
    # SIMULATING FETCHING RUNTIME DATA
    features, gt, preds = fetch_runtime_data(test_data)


    # PERFORM MONITORING
    runtime_perf = runtime_model_performance(gt, preds, perf_baseline)

    # EVALUATE MONITORING RESULTS
    evaluate_perf(runtime_perf)

    runtime_mb = runtime_model_bias(features, gt, preds, mb_baseline)
    evaluate_model_bias(runtime_mb)

    runtime_db = runtime_data_bias(features, gt, db_baseline)
    evaluate_data_bias(runtime_db)
