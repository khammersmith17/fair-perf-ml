from typing import List, Iterable, Any, Union, NamedTuple
from .base import ContinuousDataDrift, CategoricalDataDrift
from .streaming import StreamingContinuousDataDrift, StreamingCategoricalDataDrift

StreamingDriftAgent = Union[StreamingContinuousDataDrift, StreamingCategoricalDataDrift]
DiscreteDriftAgent = Union[ContinuousDataDrift, CategoricalDataDrift]


class StreamingDriftManagerEntry(NamedTuple):
    name: str
    drift_agent: StreamingDriftAgent


class DriftManagerEntry(NamedTuple):
    name: str
    drift_agent: DiscreteDriftAgent


class StreamingDataDriftManager:
    """
    A managed container for storing and maintaining a set of streaming DataDrift monitoring agents.
    Designed for long running services where high volume runtime inference data is accumulated.
    """

    __slots__ = ["_state_table", "_drift_table"]

    def __init__(self, monitor_entries: List[StreamingDriftManagerEntry]):
        raise NotImplementedError("StreamingDataDriftManager not implemented")


class DiscretePsiManager:
    """
    Manager for discrete DataDrift computations across a given feature group.
    Provides a discrete snapshot of the data state compared against a baseline set.
    """

    __slots__ = ["_state_table", "_drift_table"]

    def __init__(self, monitor_entries: List[DriftManagerEntry]):
        raise NotImplementedError("DataDriftManager not implemented")
