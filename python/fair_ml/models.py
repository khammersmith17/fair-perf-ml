from pydantic import BaseModel, ConfigDict


class ModelBiasBaseline(BaseModel):
    """data model for consistently formatted returns to users"""
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
    """data model for consistently formatted returns to users"""
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
