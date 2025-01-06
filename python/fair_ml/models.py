from pydantic import BaseModel, ConfigDict
from typing import Optional, Union


class DataBiasRuntimeReturn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )
    DfferenceInProportionalOfLabels: Optional[str]
    KlDivergence: Optional[str]
    JsDivergance: Optional[str]
    LpNorm: Optional[str]
    TotalVariationDistance: Optional[str]
    TotalVariationDistance: Optional[str]


class ModelBiasRuntimeReturn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )
    DifferenceInPositivePredictedLabels: Optional[str]
    DisparateImpact: Optional[str]
    AccuracyDifference: Optional[str]
    RecallDifference: Optional[str]
    DifferenceInConditionalAcceptance: Optional[str]
    DifferenceInAcceptanceRate: Optional[str]
    SpecailityDifference: Optional[str]
    DifferenceInConditionalRejection: Optional[str]
    DifferenceInRejectionRate: Optional[str]
    TreatmentEquity: Optional[str]
    ConditionalDemographicDesparityPredictedLabels: Optional[str]
    GeneralizedEntropy: Optional[str]


class BaseRuntimeReturn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )
    passed: bool
    failed_report: Optional[Union[DataBiasRuntimeReturn, ModelBiasRuntimeReturn]]


class ModelBiasBaseline(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )
    DifferenceInPositivePredictedLabels: float
    DisparateImpact: float
    AccuracyDifference: float
    RecallDifference: float
    DifferenceInConditionalAcceptance: float
    DifferenceInAcceptanceRate: float
    SpecailityDifference: float
    DifferenceInConditionalRejection: float
    DifferenceInRejectionRate: float
    TreatmentEquity: float
    ConditionalDemographicDesparityPredictedLabels: float
    GeneralizedEntropy: float

class DataBiasBaseline(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )
    ClassImbalance: float
    DifferenceInProportionOfLabels: float
    KlDivergence: float
    JsDivergence: float
    LpNorm: float
    TotalVarationDistance: float
    KolmorogvSmirnov: float
