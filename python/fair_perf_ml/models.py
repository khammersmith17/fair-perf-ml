from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Union


class ModelType(str, Enum):
    LinearRegression = "LinearRegression"
    LogisticRegression = "LogisticRegression"
    BinaryClassification = "BinaryClassification"


class ModelBiasBaseline(BaseModel):
    """data model for consistently formatted returns to users"""

    model_config = ConfigDict(extra="forbid", strict=True)
    DifferenceInPositivePredictedLabels: float
    DisparateImpact: float
    AccuracyDifference: float
    RecallDifference: float
    DifferenceInConditionalAcceptance: float
    DifferenceInAcceptanceRate: float
    SpecialityDifference: float
    DifferenceInConditionalRejection: float
    DifferenceInRejectionRate: float
    TreatmentEquity: float
    ConditionalDemographicDesparityPredictedLabels: float
    GeneralizedEntropy: float


class DataBiasBaseline(BaseModel):
    """data model for consistently formatted returns to users"""

    model_config = ConfigDict(extra="forbid", strict=True)
    ClassImbalance: float
    DifferenceInProportionOfLabels: float
    KlDivergence: float
    JsDivergence: float
    LpNorm: float
    TotalVarationDistance: float
    KolmorogvSmirnov: float


# some models for conistent formatting on metric objects
class LinearRegressionReport(BaseModel):
    RootMeanSquaredError: float
    MeanSquaredError: float
    MeanAbsoluteError: float
    RSquared: float
    MaxError: float
    MeanSquaredLogError: float
    RootMeanSquaredLogError: float
    MeanAbsolutePercentageError: float


class LogisticRegressionReport(BaseModel):
    BalancedAccuracy: float
    PrecisionPositive: float
    PrecisionNegative: float
    RecallPositive: float
    RecallNegative: float
    Accuracy: float
    F1Score: float
    LogLoss: float


class BinaryClassificationReport(BaseModel):
    BalancedAccuracy: float
    PrecisionPositive: float
    PrecisionNegative: float
    RecallPositive: float
    RecallNegative: float
    Accuracy: float
    F1Score: float


class ModelPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)
    modelType: ModelType
    performanceData: Union[
        LinearRegressionReport, LogisticRegressionReport, BinaryClassificationReport
    ]
