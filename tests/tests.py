import pandas as pd
import numpy as np
from fair_ml import (
    data_bias,
    model_bias,
    model_perf
)
from numpy.typing import NDArray
from typing import Tuple


def generate_binary_data(len: int) -> Tuple[NDArray, NDArray, NDArray]:
    np.random.seed(1)
    proba = np.random.rand(len)
    pred = np.where(proba >= 0.5, 1.0, 0.0)
    np.random.seed(57)
    true = np.where(np.random.rand(len) >= 0.5, 1.0, 0.0)
    return true, pred, proba

def get_data() -> pd.DataFrame:
    headers = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
        "rings",
    ]
    """
    Using abalone dataset to test
    """
    return pd.read_csv("abalone.data", names=headers)

def generate_synthetic_scores(row: int) -> float:
    op = 1 if np.random.random() > 0.5 else -1
    return row + (2 * np.random.random() * op)

def test_db_numpy(bl_df, runtime_test) -> None:
    """
    testing using a numpy array with the db methods
    """
    db_bl = data_bias.perform_analysis(
        bl_df["sex"].to_numpy(),
        bl_df["rings"].to_numpy(),
        "M",
        15
    )

    db_runtime = data_bias.perform_analysis(
        runtime_test["sex"].to_numpy(),
        runtime_test["rings"].to_numpy(),
        "M",
        15
    )

    runtime_check = data_bias.runtime_comparison(
        db_bl,
        db_runtime,
        0.15
    )

    print(f"baseline:\n{db_bl}")
    print("\n")
    print(f"runtime:\n{db_runtime}")
    print("\n")
    print(f"runtime check:\n{runtime_check}")


def test_db_list(bl_df, runtime_test) -> None:
    """
    testing using a numpy array with the db methods
    """
    db_bl = data_bias.perform_analysis(
        bl_df["sex"].to_list(),
        bl_df["rings"].to_list(),
        "M",
        15
    )

    db_runtime = data_bias.perform_analysis(
        runtime_test["sex"].to_list(),
        runtime_test["rings"].to_list(),
        "M",
        15
    )

    runtime_check = data_bias.runtime_comparison(
        db_bl,
        db_runtime
    )

    print(f"baseline\n{db_bl}")
    print("\n")
    print(f"runtime\n:{db_runtime}")
    print("\n")
    print(f"runtime check:{runtime_check}")

def test_mb_numpy(bl_df, runtime_test):
    bl = model_bias.perform_analysis(
        bl_df["sex"].to_numpy(),
        bl_df["rings"].to_numpy(),
        bl_df["preds"].to_numpy(),
        "M",
        15,
        15.0
    )

    runtime = model_bias.perform_analysis(
        runtime_test["sex"].to_numpy(),
        runtime_test["rings"].to_numpy(),
        runtime_test["preds"].to_numpy(),
        "M",
        15,
        15.0
    )

    runtime_check = model_bias.runtime_comparison(
        bl,
        runtime,
        0.15
    )

    print(f"bl:\n{bl}" )
    print("\n")
    print(f"runtime:\n{runtime}")
    print("\n")
    print(f"check:\n{runtime_check}")


def test_perf_reg_numpy(y_pred, y_true):
    l = int(y_pred.size * 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.linear_regression_analysis(
        y_true=bl_true,
        y_pred=bl_pred
    )

    runtime = model_perf.linear_regression_analysis(
        y_true=rt_true,
        y_pred=rt_pred
    )

    partial = model_perf.partial_runtime_check(
        baseline=bl,
        latest=runtime,
        metrics=[
            "RootMeanSquaredError",
            "MeanSquaredError",
            "MeanAbsoluteError",
            "RSquared"
        ]
    )


    res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    print(res)

def test_perf_reg_list(y_pred, y_true):
    l = int(len(y_pred) * 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.linear_regression_analysis(
        y_true=bl_true,
        y_pred=bl_pred
    )
    print(bl)

    runtime = model_perf.linear_regression_analysis(
        y_true=rt_true,
        y_pred=rt_pred
    )
    print(runtime)

    res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    print(res)


def test_perf_reg_classification_numpy(y_pred, y_true):
    l = int(y_pred.size * 0.7)
    bl_true = y_true[:l]
    pred_true = y_true[l:]
    bl_pred = y_pred[:l]
    pred_pred = y_pred[l:]

    bl = model_perf.binary_classification_analysis(
        y_true=bl_true,
        y_pred=bl_pred
    )
    print(bl)

    runtime = model_perf.binary_classification_analysis(
        y_true=pred_true,
        y_pred=pred_pred
    )
    print(runtime)

    res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    print(res)

def test_perf_reg_classification_list(y_pred, y_true):
    l = int(len(y_pred)* 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.binary_classification_analysis(
        y_true=bl_true,
        y_pred=bl_pred
    )
    print(bl)

    runtime = model_perf.binary_classification_analysis(
        y_true=rt_true,
        y_pred=rt_pred
    )
    print(runtime)

    res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    print(res)


def test_perf_logisitc_reg_numpy(y_pred, y_true):
    l = int(y_pred.size* 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.logistic_regression_analysis(
        y_true=bl_true,
        y_pred=bl_pred
    )
    print(bl)

    runtime = model_perf.logistic_regression_analysis(
        y_true=rt_true,
        y_pred=rt_pred
    )
    print(runtime)

    res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    print(res)


def test_perf_logisitc_reg_list(y_pred, y_true):
    l = int(len(y_pred)* 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.logistic_regression_analysis(
        y_true=bl_true,
        y_pred=bl_pred
    )
    print(bl)

    runtime = model_perf.logistic_regression_analysis(
        y_true=rt_true,
        y_pred=rt_pred
    )
    print(runtime)

    res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    print(res)


def test_mb_list(bl_df, runtime_test):
    bl = model_bias.perform_analysis(
        bl_df["sex"].to_list(),
        bl_df["rings"].to_list(),
        bl_df["preds"].to_list(),
        "M",
        15,
        15.0
    )

    runtime = model_bias.perform_analysis(
        runtime_test["sex"].to_list(),
        runtime_test["rings"].to_list(),
        runtime_test["preds"].to_list(),
        "M",
        15,
        15.0
    )

    runtime_check = model_bias.runtime_comparison(
        bl,
        runtime,
        0.15
    )

    print(f"bl:\n{bl}" )
    print("\n")
    print(f"runtime:\n{runtime}")
    print("\n")
    print(f"check:\n{runtime_check}")

if __name__ == "__main__":
    df = get_data()
    df["preds"] = df.rings.apply(
        generate_synthetic_scores
    )

    bin_true, bin_pred, bin_proba = generate_binary_data(10000)
    reg_true = df["rings"].to_numpy().copy()
    reg_pred = df["preds"].to_numpy().copy()

    bl_df = df.iloc[:int(.60 * df.shape[0]), :]
    runtime_test = df.iloc[int(.60 * df.shape[0]) + 1:, :]
    print("TESTING DATA BIAS WITH NUMPY ARRAYS...")
    test_db_numpy(bl_df, runtime_test)
    print("\n")

    print("TESTING DATA BIAS WITH LIST ARRAYS...")
    test_db_list(bl_df, runtime_test)

    print("\n")
    print("TESTING MB with numpy...")
    test_mb_numpy(bl_df, runtime_test)

    print("\n")
    print("TESTING MB with lists...")
    test_mb_list(bl_df, runtime_test)

    print("\n")
    print("TESTING PERF WITH NUMPY ARRAYS")
    test_perf_reg_numpy(reg_pred, reg_true)

    print("\n")
    print("TESTING PERF WITH LISTS")
    test_perf_reg_list(reg_pred.tolist(), reg_true.tolist())

    print("\n")
    print("TESTING Classification PERF WITH NUMPY")
    test_perf_reg_classification_numpy(bin_pred, bin_true) 

    print("\n")
    print("TESTING Classification PERF WITH LISTS")
    test_perf_reg_classification_list(bin_pred.tolist(), bin_true.tolist())

    print("\n")
    print("TESTING logisitc PERF WITH numpy")
    test_perf_logisitc_reg_numpy(bin_proba, bin_true)


    print("\n")
    print("TESTING logisitc PERF WITH list")
    test_perf_logisitc_reg_list(bin_proba.tolist(), bin_true.tolist())
