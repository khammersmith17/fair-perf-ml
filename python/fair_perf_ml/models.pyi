from enum import Enum
from typing import TypedDict, TypeAlias
from pydantic import BaseModel, ConfigDict

class DriftReport(TypedDict):
    passed: bool
    failed_report: dict[str, float]

type PerformanceSnapshot = dict[str, float]
type DriftSnapshot = dict[str, float]

class ModelType(str, Enum):
    LinearRegression: str
    LogisticRegression: str
    BinaryClassification: str

class DataBiasMetric(str, Enum):
    ClassImbalance: str
    DifferenceInProportionOfLabels: str
    KlDivergence: str
    JsDivergence: str
    LpNorm: str
    TotalVariationDistance: str
    KolmogorovSmirnov: str

class ModelBiasMetric(str, Enum):
    DifferenceInPositivePredictedLabels: str
    DisparateImpact: str
    AccuracyDifference: str
    RecallDifference: str
    DifferenceInConditionalAcceptance: str
    DifferenceInAcceptanceRate: str
    SpecialityDifference: str
    DifferenceInConditionalRejection: str
    DifferenceInRejectionRate: str
    TreatmentEquity: str
    ConditionalDemographicDesparityPredictedLabels: str
    GeneralizedEntropy: str

class ClassificationEvaluationMetric(str, Enum):
    BalancedAccuracy: str
    PrecisionPositive: str
    PrecisionNegative: str
    RecallPositive: str
    RecallNegative: str
    Accuracy: str
    F1Score: str
    LogLoss: str

class LinearRegressionEvaluationMetric(str, Enum):
    RootMeanSquaredError: str
    MeanSquaredError: str
    MeanAbsoluteError: str
    RSquared: str
    MaxError: str
    MeanSquaredLogError: str
    RootMeanSquaredLogError: str
    MeanAbsolutePercentageError: str

ModelPerformanceMetric: TypeAlias = LinearRegressionEvaluationMetric | ClassificationEvaluationMetric
MachineLearningMetric: TypeAlias = ModelBiasMetric | DataBiasMetric | ModelPerformanceDriftMetric

class ModelBiasBaseline(BaseModel):
    model_config: ConfigDict
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
    model_config: ConfigDict
    ClassImbalance: float
    DifferenceInProportionOfLabels: float
    KlDivergence: float
    JsDivergence: float
    LpNorm: float
    TotalVarationDistance: float
    KolmogorovSmirnov: float

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

class ModelPerformanceReport(BaseModel):
    model_config: ConfigDict
    model_type: ModelType
    performance_data: LinearRegressionReport | LogisticRegressionReport | BinaryClassificationReport

type DataBiasDriftMetric = DataBiasMetric | str
type ModelBiasDriftMetric = ModelBiasMetric | str
type ClassificationDriftMetric = ClassificationEvaluationMetric | str
type LinearRegressionDriftMetric = LinearRegressionEvaluationMetric | str
type ModelPerformanceDriftMetric = ModelPerformanceMetric | str
