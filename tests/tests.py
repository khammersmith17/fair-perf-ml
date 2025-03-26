import pandas as pd
import numpy as np
from fair_perf_ml import data_bias, model_bias, model_perf
from numpy.typing import NDArray
from typing import Tuple
import argparse


def generate_binary_data(len: int) -> Tuple[NDArray, NDArray, NDArray]:
    np.random.seed(1)
    proba = np.random.rand(len)
    pred = np.where(proba >= 0.5, 1.0, 0.0)
    np.random.seed(57)
    true = np.where(np.random.rand(len) >= 0.5, 1.0, 0.0)
    df = pd.concat(
        [
            pd.Series(data=true, name="true"),
            pd.Series(data=pred, name="pred"),
            pd.Series(data=proba, name="proba"),
        ],
        axis=1,
    )
    df.to_csv("perf_data.csv", header=True, index=False)
    return true, pred, proba


def generate_synthetic_scores(row: int) -> float:
    op = 1 if np.random.random() > 0.5 else -1
    return row + (2 * np.random.random() * op)


def generate_test_set(df: pd.DataFrame) -> pd.DataFrame:
    df["preds"] = df.rings.apply(generate_synthetic_scores)
    df.to_csv("test_data.csv", header=True, index=False)
    return df


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


def test_db_numpy(bl_df, runtime_test) -> bool:
    """
    testing using a numpy array with the db methods
    """
    res = [False] * 3
    db_bl = data_bias.perform_analysis(
        bl_df["sex"].to_numpy(), bl_df["rings"].to_numpy(), "M", 15
    )

    res[0] = db_bl == {
        "ClassImbalance": 0.2545889914035797,
        "DifferenceInProportionOfLabels": 0.039049260318279266,
        "KlDivergence": 0.007977087050676346,
        "JsDivergence": 0.002021726220846176,
        "LpNorm": 0.23192767798900604,
        "TotalVarationDistance": 0.039049260318279266,
        "KolmorogvSmirnov": 0.039049260318279266,
    }

    db_runtime = data_bias.perform_analysis(
        runtime_test["sex"].to_numpy(), runtime_test["rings"].to_numpy(), "M", 15
    )
    res[1] = db_runtime == {
        "ClassImbalance": 0.28862276673316956,
        "DifferenceInProportionOfLabels": 0.026097401976585388,
        "KlDivergence": 0.006829630583524704,
        "JsDivergence": 0.0016257409006357193,
        "LpNorm": 0.11823293566703796,
        "TotalVarationDistance": 0.026097401976585388,
        "KolmorogvSmirnov": 0.026097401976585388,
    }

    runtime_check = data_bias.runtime_comparison(db_bl, db_runtime, 0.15)

    res[2] = runtime_check == {"passed": True}

    return all(res)


def test_db_numpy_partial(bl_df, runtime_test) -> bool:
    """
    testing using a numpy array with the db methods
    """
    res = [False] * 3
    db_bl = data_bias.perform_analysis(
        bl_df["sex"].to_numpy(), bl_df["rings"].to_numpy(), "M", 15
    )

    res[0] = db_bl == {
        "ClassImbalance": 0.2545889914035797,
        "DifferenceInProportionOfLabels": 0.039049260318279266,
        "KlDivergence": 0.007977087050676346,
        "JsDivergence": 0.002021726220846176,
        "LpNorm": 0.23192767798900604,
        "TotalVarationDistance": 0.039049260318279266,
        "KolmorogvSmirnov": 0.039049260318279266,
    }

    db_runtime = data_bias.perform_analysis(
        runtime_test["sex"].to_numpy(), runtime_test["rings"].to_numpy(), "M", 15
    )

    res[1] = db_runtime == {
        "ClassImbalance": 0.28862276673316956,
        "DifferenceInProportionOfLabels": 0.026097401976585388,
        "KlDivergence": 0.006829630583524704,
        "JsDivergence": 0.0016257409006357193,
        "LpNorm": 0.11823293566703796,
        "TotalVarationDistance": 0.026097401976585388,
        "KolmorogvSmirnov": 0.026097401976585388,
    }

    runtime_check = data_bias.partial_runtime_comparison(
        db_bl,
        db_runtime,
        [
            "ClassImbalance",
            "DifferenceInProportionOfLabels",
            "KlDivergence",
            "JsDivergence",
        ],
        0.15,
    )

    res[2] = runtime_check == {"passed": True}

    return all(res)


def test_db_list(bl_df, runtime_test) -> bool:
    """
    testing using a numpy array with the db methods
    """
    res = [False] * 3
    db_bl = data_bias.perform_analysis(
        bl_df["sex"].to_numpy(), bl_df["rings"].to_numpy(), "M", 15
    )

    res[0] = db_bl == {
        "ClassImbalance": 0.2545889914035797,
        "DifferenceInProportionOfLabels": 0.039049260318279266,
        "KlDivergence": 0.007977087050676346,
        "JsDivergence": 0.002021726220846176,
        "LpNorm": 0.23192767798900604,
        "TotalVarationDistance": 0.039049260318279266,
        "KolmorogvSmirnov": 0.039049260318279266,
    }

    db_runtime = data_bias.perform_analysis(
        runtime_test["sex"].to_numpy(), runtime_test["rings"].to_numpy(), "M", 15
    )
    res[1] = db_runtime == {
        "ClassImbalance": 0.28862276673316956,
        "DifferenceInProportionOfLabels": 0.026097401976585388,
        "KlDivergence": 0.006829630583524704,
        "JsDivergence": 0.0016257409006357193,
        "LpNorm": 0.11823293566703796,
        "TotalVarationDistance": 0.026097401976585388,
        "KolmorogvSmirnov": 0.026097401976585388,
    }

    runtime_check = data_bias.runtime_comparison(db_bl, db_runtime, 0.15)

    res[2] = runtime_check == {"passed": True}

    return all(res)


def test_mb_numpy(bl_df, runtime_test) -> bool:
    res = [False] * 3

    bl = model_bias.perform_analysis(
        bl_df["sex"].to_numpy(),
        bl_df["rings"].to_numpy(),
        bl_df["preds"].to_numpy(),
        "M",
        15,
        15.0,
    )

    res[0] = bl == {
        "DifferenceInPositivePredictedLabels": 0.002093970775604248,
        "DisparateImpact": 0.8409091234207153,
        "AccuracyDifference": -8.034706115722656e-05,
        "RecallDifference": 0.04676508903503418,
        "DifferenceInConditionalAcceptance": -0.0026617050170898438,
        "DifferenceInAcceptanceRate": 0.05057328939437866,
        "SpecialityDifference": -0.0018883943557739258,
        "DifferenceInConditionalRejection": 0.005205392837524414,
        "DifferenceInRejectionRate": 0.0032941699028015137,
        "TreatmentEquity": -0.8666665554046631,
        "ConditionalDemographicDesparityPredictedLabels": 0.09311360120773315,
        "GeneralizedEntropy": 139933.03125,
    }

    runtime = model_bias.perform_analysis(
        runtime_test["sex"].to_numpy(),
        runtime_test["rings"].to_numpy(),
        runtime_test["preds"].to_numpy(),
        "M",
        15,
        15.0,
    )

    res[1] = runtime == {
        "DifferenceInPositivePredictedLabels": -0.09821432828903198,
        "DisparateImpact": 0.7872340083122253,
        "AccuracyDifference": -0.00829547643661499,
        "RecallDifference": -0.0654761791229248,
        "DifferenceInConditionalAcceptance": 0.11385858058929443,
        "DifferenceInAcceptanceRate": 0.025301873683929443,
        "SpecialityDifference": 0.0005710124969482422,
        "DifferenceInConditionalRejection": 0.008004844188690186,
        "DifferenceInRejectionRate": 0.008531749248504639,
        "TreatmentEquity": -1.4666666984558105,
        "ConditionalDemographicDesparityPredictedLabels": 0.0892782211303711,
        "GeneralizedEntropy": 27914.095703125,
    }

    runtime_check = model_bias.runtime_comparison(bl, runtime, 0.15)

    res[2] = runtime_check == {
        "passed": False,
        "failReport": {
            "DifferenceInPositivePredictedLabels": "Exceed baseline by: 0.09612036",
            "RecallDifference": "Exceed baseline by: 0.01871109",
            "AccuracyDifference": "Exceed baseline by: 0.008215129",
            "DifferenceInConditionalRejection": "Exceed baseline by: 0.0027994514",
            "DifferenceInConditionalAcceptance": "Exceed baseline by: 0.111196876",
            "DifferenceInRejectionRate": "Exceed baseline by: 0.0052375793",
            "TreatmentEquity": "Exceed baseline by: 0.60000014",
        },
    }
    return all(res)


def test_mb_numpy_partial(bl_df, runtime_test) -> bool:
    res = [False] * 3
    bl = model_bias.perform_analysis(
        bl_df["sex"].to_numpy(),
        bl_df["rings"].to_numpy(),
        bl_df["preds"].to_numpy(),
        "M",
        15,
        15.0,
    )

    res[0] = bl == {
        "DifferenceInPositivePredictedLabels": 0.002093970775604248,
        "DisparateImpact": 0.8409091234207153,
        "AccuracyDifference": -8.034706115722656e-05,
        "RecallDifference": 0.04676508903503418,
        "DifferenceInConditionalAcceptance": -0.0026617050170898438,
        "DifferenceInAcceptanceRate": 0.05057328939437866,
        "SpecialityDifference": -0.0018883943557739258,
        "DifferenceInConditionalRejection": 0.005205392837524414,
        "DifferenceInRejectionRate": 0.0032941699028015137,
        "TreatmentEquity": -0.8666665554046631,
        "ConditionalDemographicDesparityPredictedLabels": 0.09311360120773315,
        "GeneralizedEntropy": 139933.03125,
    }

    runtime = model_bias.perform_analysis(
        runtime_test["sex"].to_numpy(),
        runtime_test["rings"].to_numpy(),
        runtime_test["preds"].to_numpy(),
        "M",
        15,
        15.0,
    )

    res[1] = runtime == {
        "DifferenceInPositivePredictedLabels": -0.09821432828903198,
        "DisparateImpact": 0.7872340083122253,
        "AccuracyDifference": -0.00829547643661499,
        "RecallDifference": -0.0654761791229248,
        "DifferenceInConditionalAcceptance": 0.11385858058929443,
        "DifferenceInAcceptanceRate": 0.025301873683929443,
        "SpecialityDifference": 0.0005710124969482422,
        "DifferenceInConditionalRejection": 0.008004844188690186,
        "DifferenceInRejectionRate": 0.008531749248504639,
        "TreatmentEquity": -1.4666666984558105,
        "ConditionalDemographicDesparityPredictedLabels": 0.0892782211303711,
        "GeneralizedEntropy": 27914.095703125,
    }

    runtime_check = model_bias.partial_runtime_comparison(
        bl,
        runtime,
        [
            "DifferenceInPositivePredictedLabels",
            "DisparateImpact",
            "AccuracyDifference",
            "RecallDifference",
            "DifferenceInConditionalAcceptance",
        ],
        0.15,
    )

    res[2] = runtime_check == {
        "passed": False,
        "failReport": {
            "DifferenceInPositivePredictedLabels": "Exceed baseline by: 0.09612036",
            "AccuracyDifference": "Exceed baseline by: 0.008215129",
            "DifferenceInConditionalAcceptance": "Exceed baseline by: 0.111196876",
            "RecallDifference": "Exceed baseline by: 0.01871109",
        },
    }
    return all(res)


def test_perf_reg_numpy(y_pred, y_true) -> bool:
    res = [False] * 4
    l = int(y_pred.size * 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.linear_regression_analysis(y_true=bl_true, y_pred=bl_pred)

    res[0] = bl == {
        "modelType": "LinearRegression",
        "performanceData": {
            "RootMeanSquaredError": 1.1617724895477295,
            "MeanSquaredError": 1.349715232849121,
            "MeanAbsoluteError": 1.0100986957550049,
            "RSquared": 0.1202625185251236,
            "MaxError": 1.999868392944336,
            "MeanSquaredLogError": 204887.609375,
            "RootMeanSquaredLogError": 24472.158203125,
            "MeanAbsolutePercentageError": 11.345174789428711,
        },
    }
    runtime = model_perf.linear_regression_analysis(y_true=rt_true, y_pred=rt_pred)

    res[1] = runtime == {
        "modelType": "LinearRegression",
        "performanceData": {
            "RootMeanSquaredError": 1.133068323135376,
            "MeanSquaredError": 1.283843755722046,
            "MeanAbsoluteError": 0.9810680150985718,
            "RSquared": 0.15180453658103943,
            "MaxError": 1.999338150024414,
            "MeanSquaredLogError": 54827.40234375,
            "RootMeanSquaredLogError": 8291.77734375,
            "MeanAbsolutePercentageError": 10.8325834274292,
        },
    }
    partial = model_perf.partial_runtime_check(
        baseline=bl,
        latest=runtime,
        metrics=[
            "RootMeanSquaredError",
            "MeanSquaredError",
            "MeanAbsoluteError",
            "RSquared",
        ],
    )


    res[2] = partial == {
        "passed": False,
        "failReport": {"RSquared": "Exceeded threshold by 0.031542018"},
    }
    full_res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    res[3] = full_res == {
        "passed": False,
        "failReport": {"RSquared": "Exceeded threshold by 0.031542018"},
    }
    return all(res)


def test_perf_reg_list(y_pred, y_true):
    res = [False] * 3
    l = int(len(y_pred) * 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.linear_regression_analysis(y_true=bl_true, y_pred=bl_pred)

    res[0] = bl == {
        "modelType": "LinearRegression",
        "performanceData": {
            "RootMeanSquaredError": 1.1617724895477295,
            "MeanSquaredError": 1.349715232849121,
            "MeanAbsoluteError": 1.0100986957550049,
            "RSquared": 0.1202625185251236,
            "MaxError": 1.999868392944336,
            "MeanSquaredLogError": 204887.609375,
            "RootMeanSquaredLogError": 24472.158203125,
            "MeanAbsolutePercentageError": 11.345174789428711,
        },
    }

    runtime = model_perf.linear_regression_analysis(y_true=rt_true, y_pred=rt_pred)

    res[1] = runtime == {
        "modelType": "LinearRegression",
        "performanceData": {
            "RootMeanSquaredError": 1.133068323135376,
            "MeanSquaredError": 1.283843755722046,
            "MeanAbsoluteError": 0.9810680150985718,
            "RSquared": 0.15180453658103943,
            "MaxError": 1.999338150024414,
            "MeanSquaredLogError": 54827.40234375,
            "RootMeanSquaredLogError": 8291.77734375,
            "MeanAbsolutePercentageError": 10.8325834274292,
        },
    }

    full_res = model_perf.runtime_check_full(baseline=bl, latest=runtime)
    res[2] = full_res == {
        "passed": False,
        "failReport": {"RSquared": "Exceeded threshold by 0.031542018"},
    }

    return all(res)

def test_perf_reg_classification_numpy(y_pred, y_true) -> bool:
    res = [False] * 3
    l = int(y_pred.size * 0.7)
    bl_true = y_true[:l]
    pred_true = y_true[l:]
    bl_pred = y_pred[:l]
    pred_pred = y_pred[l:]

    bl = model_perf.binary_classification_analysis(y_true=bl_true, y_pred=bl_pred)
    res[0] = bl == {
        "modelType": "BinaryClassification",
        "performanceData": {
            "BalancedAccuracy": 0.12866388261318207,
            "PrecisionPositive": 0.5059422850608826,
            "PrecisionNegative": 0.5086554884910583,
            "RecallPositive": 0.5121741890907288,
            "RecallNegative": 0.5024223327636719,
            "Accuracy": 0.5072857141494751,
            "F1Score": 0.5090391635894775,
        },
    }

    runtime = model_perf.binary_classification_analysis(
        y_true=pred_true, y_pred=pred_pred
    )


    res[1] = runtime == {
        "modelType": "BinaryClassification",
        "performanceData": {
            "BalancedAccuracy": 0.11985205113887787,
            "PrecisionPositive": 0.48915988206863403,
            "PrecisionNegative": 0.49015748500823975,
            "RecallPositive": 0.48165443539619446,
            "RecallNegative": 0.49766820669174194,
            "Accuracy": 0.4896666705608368,
            "F1Score": 0.4853781461715698,
        },
    }

    eval_res = model_perf.runtime_check_full(baseline=bl, latest=runtime)

    res[2] = eval_res == {"passed": True}

    return all(res)


def test_perf_reg_classification_list(y_pred, y_true) -> bool:
    res = [False] * 3
    l = int(len(y_pred) * 0.7)
    bl_true = y_true[:l]
    pred_true = y_true[l:]
    bl_pred = y_pred[:l]
    pred_pred = y_pred[l:]

    bl = model_perf.binary_classification_analysis(y_true=bl_true, y_pred=bl_pred)
    res[0] = bl == {
        "modelType": "BinaryClassification",
        "performanceData": {
            "BalancedAccuracy": 0.12866388261318207,
            "PrecisionPositive": 0.5059422850608826,
            "PrecisionNegative": 0.5086554884910583,
            "RecallPositive": 0.5121741890907288,
            "RecallNegative": 0.5024223327636719,
            "Accuracy": 0.5072857141494751,
            "F1Score": 0.5090391635894775,
        },
    }

    runtime = model_perf.binary_classification_analysis(
        y_true=pred_true, y_pred=pred_pred
    )


    res[1] = runtime == {
        "modelType": "BinaryClassification",
        "performanceData": {
            "BalancedAccuracy": 0.11985205113887787,
            "PrecisionPositive": 0.48915988206863403,
            "PrecisionNegative": 0.49015748500823975,
            "RecallPositive": 0.48165443539619446,
            "RecallNegative": 0.49766820669174194,
            "Accuracy": 0.4896666705608368,
            "F1Score": 0.4853781461715698,
        },
    }

    eval_res = model_perf.runtime_check_full(baseline=bl, latest=runtime)

    res[2] = eval_res == {"passed": True}

    return all(res)


def test_perf_logisitc_reg_numpy(y_pred, y_true) -> bool:
    res = [False] * 3
    l = int(y_pred.size * 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.logistic_regression_analysis(y_true=bl_true, y_pred=bl_pred)
    res[0] = bl == {
        "modelType": "LogisticRegression",
        "performanceData": {
            "BalancedAccuracy": 0.12866388261318207,
            "PrecisionPositive": 0.5059422850608826,
            "PrecisionNegative": 0.5086554884910583,
            "RecallPositive": 0.5121741890907288,
            "RecallNegative": 0.5024223327636719,
            "Accuracy": 0.5072857141494751,
            "F1Score": 0.5090391635894775,
            "LogLoss": 0.4283928871154785,
        },
    }

    runtime = model_perf.logistic_regression_analysis(y_true=rt_true, y_pred=rt_pred)

    res[1] = runtime == {
        "modelType": "LogisticRegression",
        "performanceData": {
            "BalancedAccuracy": 0.11985205113887787,
            "PrecisionPositive": 0.48915988206863403,
            "PrecisionNegative": 0.49015748500823975,
            "RecallPositive": 0.48165443539619446,
            "RecallNegative": 0.49766820669174194,
            "Accuracy": 0.4896666705608368,
            "F1Score": 0.4853781461715698,
            "LogLoss": 0.4418107867240906,
        },
    }

    eval_res = model_perf.runtime_check_full(baseline=bl, latest=runtime)

    res[2] = eval_res == {"passed": True}
    return all(res)


def test_perf_logisitc_reg_list(y_pred, y_true):
    res = [False] * 3
    l = int(len(y_pred) * 0.7)
    bl_true = y_true[:l]
    rt_true = y_true[l:]
    bl_pred = y_pred[:l]
    rt_pred = y_pred[l:]

    bl = model_perf.logistic_regression_analysis(y_true=bl_true, y_pred=bl_pred)
    res[0] = bl == {
        "modelType": "LogisticRegression",
        "performanceData": {
            "BalancedAccuracy": 0.12866388261318207,
            "PrecisionPositive": 0.5059422850608826,
            "PrecisionNegative": 0.5086554884910583,
            "RecallPositive": 0.5121741890907288,
            "RecallNegative": 0.5024223327636719,
            "Accuracy": 0.5072857141494751,
            "F1Score": 0.5090391635894775,
            "LogLoss": 0.4283928871154785,
        },
    }

    runtime = model_perf.logistic_regression_analysis(y_true=rt_true, y_pred=rt_pred)

    res[1] = runtime == {
        "modelType": "LogisticRegression",
        "performanceData": {
            "BalancedAccuracy": 0.11985205113887787,
            "PrecisionPositive": 0.48915988206863403,
            "PrecisionNegative": 0.49015748500823975,
            "RecallPositive": 0.48165443539619446,
            "RecallNegative": 0.49766820669174194,
            "Accuracy": 0.4896666705608368,
            "F1Score": 0.4853781461715698,
            "LogLoss": 0.4418107867240906,
        },
    }

    eval_res = model_perf.runtime_check_full(baseline=bl, latest=runtime)

    res[2] = eval_res == {"passed": True}
    return all(res)


def test_mb_list(bl_df, runtime_test):
    res = [False] * 3

    bl = model_bias.perform_analysis(
        bl_df["sex"].to_list(),
        bl_df["rings"].to_list(),
        bl_df["preds"].to_list(),
        "M",
        15,
        15.0,
    )


    res[0] = bl == {
        "DifferenceInPositivePredictedLabels": 0.002093970775604248,
        "DisparateImpact": 0.8409091234207153,
        "AccuracyDifference": -8.034706115722656e-05,
        "RecallDifference": 0.04676508903503418,
        "DifferenceInConditionalAcceptance": -0.0026617050170898438,
        "DifferenceInAcceptanceRate": 0.05057328939437866,
        "SpecialityDifference": -0.0018883943557739258,
        "DifferenceInConditionalRejection": 0.005205392837524414,
        "DifferenceInRejectionRate": 0.0032941699028015137,
        "TreatmentEquity": -0.8666665554046631,
        "ConditionalDemographicDesparityPredictedLabels": 0.09311360120773315,
        "GeneralizedEntropy": 139933.03125,
    }
    runtime = model_bias.perform_analysis(
        runtime_test["sex"].to_list(),
        runtime_test["rings"].to_list(),
        runtime_test["preds"].to_list(),
        "M",
        15,
        15.0,
    )


    res[1] = runtime == {
        "DifferenceInPositivePredictedLabels": -0.09821432828903198,
        "DisparateImpact": 0.7872340083122253,
        "AccuracyDifference": -0.00829547643661499,
        "RecallDifference": -0.0654761791229248,
        "DifferenceInConditionalAcceptance": 0.11385858058929443,
        "DifferenceInAcceptanceRate": 0.025301873683929443,
        "SpecialityDifference": 0.0005710124969482422,
        "DifferenceInConditionalRejection": 0.008004844188690186,
        "DifferenceInRejectionRate": 0.008531749248504639,
        "TreatmentEquity": -1.4666666984558105,
        "ConditionalDemographicDesparityPredictedLabels": 0.0892782211303711,
        "GeneralizedEntropy": 27914.095703125,
    }
    runtime_check = model_bias.runtime_comparison(bl, runtime, 0.15)

    res[2] = runtime_check == {
        "passed": False,
        "failReport": {
            "DifferenceInPositivePredictedLabels": "Exceed baseline by: 0.09612036",
            "AccuracyDifference": "Exceed baseline by: 0.008215129",
            "DifferenceInConditionalRejection": "Exceed baseline by: 0.0027994514",
            "RecallDifference": "Exceed baseline by: 0.01871109",
            "TreatmentEquity": "Exceed baseline by: 0.60000014",
            "DifferenceInConditionalAcceptance": "Exceed baseline by: 0.111196876",
            "DifferenceInRejectionRate": "Exceed baseline by: 0.0052375793",
        },
    }

    return all(res)


def eval_str_to_bool(v: str) -> bool:
    if v.lower() not in ["true", "false"]:
        raise ValueError("Invalid value")
    return v.lower() == "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-data", type=eval_str_to_bool, default=False)
    args = parser.parse_args()
    generate = args.generate_data
    if generate:
        df = get_data()
        df = generate_test_set(df)
        bin_true, bin_pred, bin_proba = generate_binary_data(10000)
    else:
        df = pd.read_csv("test_data.csv")
        perf_data = pd.read_csv("perf_data.csv")
        bin_true, bin_pred, bin_proba = (
            perf_data["true"].to_numpy(),
            perf_data["pred"].to_numpy(),
            perf_data["proba"].to_numpy(),
        )

    reg_true = df["rings"].to_numpy().copy()
    reg_pred = df["preds"].to_numpy().copy()

    bl_df = df.iloc[: int(0.60 * df.shape[0]), :]
    runtime_test = df.iloc[int(0.60 * df.shape[0]) + 1 :, :]
    print("TESTING DATA BIAS WITH NUMPY ARRAYS...")
    assert test_db_numpy(bl_df, runtime_test)
    print("passed...")

    print("TESTING DATA BIAS WITH NUMPY ARRAYS partial...")
    assert test_db_numpy_partial(bl_df, runtime_test)
    print("passed...")

    print("TESTING DATA BIAS WITH LIST ARRAYS...")
    assert test_db_list(bl_df, runtime_test)
    print("passed...")

    print("TESTING MB with numpy...")
    assert test_mb_numpy(bl_df, runtime_test)
    print("passed...")

    print("TESTING MB with numpy PARTIAL...")
    assert test_mb_numpy_partial(bl_df, runtime_test)
    print("passed...")

    print("TESTING MB with lists...")
    assert test_mb_list(bl_df, runtime_test)
    print("passed...")

    print("TESTING PERF WITH NUMPY ARRAYS")
    assert test_perf_reg_numpy(reg_pred, reg_true)
    print("passed...")

    print("TESTING PERF WITH LISTS")
    assert test_perf_reg_list(reg_pred.tolist(), reg_true.tolist())
    print("passed...")

    print("TESTING Classification PERF WITH NUMPY")
    assert test_perf_reg_classification_numpy(bin_pred, bin_true)
    print("passed...")

    print("TESTING Classification PERF WITH LISTS")
    assert test_perf_reg_classification_list(bin_pred.tolist(), bin_true.tolist())
    print("passed...")

    print("TESTING logisitc PERF WITH numpy")
    assert test_perf_logisitc_reg_numpy(bin_proba, bin_true)
    print("passed...")

    print("TESTING logisitc PERF WITH list")
    assert test_perf_logisitc_reg_list(bin_proba.tolist(), bin_true.tolist())
    print("passed...")
