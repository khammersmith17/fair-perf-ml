"""
Provides runtime drift monitoring utilites and managers for such utilities.
These drift techniques can be used for data drift or a proxy for model drift
when ground truth feedback loop is slow.
"""

from enum import Enum
from typing import Tuple, Union, List, Iterable, Any
import numpy as np
from numpy.typing import NDArray
from ._fair_perf_ml import (
    StreamingContinuousPSI,
    ContinuousPSI,
    StreamingCategoricalPSI,
    CategoricalPSI,
)


class PsiType(str, Enum):
    CONTINUOUS = "Continuous"
    CATEGORICAL = "Categoical"


PsiUtil = Union[
    StreamingContinuousPSI, ContinuousPSI, StreamingCategoricalPSI, CategoricalPSI
]
PSIRegisterRequest = Tuple[str, Union[str, PsiType], Union[NDArray, List[str]]]
ContinuousPSIRegisterEntry = Tuple[str, NDArray]
CategoricalRegisterEntry = Tuple[str, List[str]]


class PsiParamValidationError(Exception):
    """
    Exception for when users pass invalid data in
    """


def _coerce_data_to_np_float(data: Iterable[Any]) -> NDArray:
    """
    Utility to convert to np float64 array.
    Will throw an exception when data is not numeric and cannot be casted to float.
    """
    try:
        return np.array([float(item) for item in data], dtype=np.float64)
    except ValueError:
        raise TypeError("StreamingContinuousPSI data must be numeric")


def _coerce_data_to_string_list(data: Iterable[Any]) -> List[str]:
    """
    Utility to convert data into string type for categorical analysis.
    """
    return [str(item) for item in data]


def _coerce_data(agent: PsiUtil, data: Iterable[Any]) -> Union[NDArray, List[str]]:
    if isinstance(agent, Union[StreamingContinuousPSI, ContinuousPSI]):
        return _coerce_data_to_np_float(data)
    else:
        return _coerce_data_to_string_list(data)


def smooth_continuous_register_entry(
    register_entry: PSIRegisterRequest,
) -> ContinuousPSIRegisterEntry:
    """
    Perform required data coersions.
    """
    if len(register_entry) != 3:
        raise PsiParamValidationError(
            "Register entry must be length 3 (column name, PsiType | str, baseline data)"
        )

    col_name = register_entry[0]

    # coerce data into numpy float numpy array
    try:
        bl_data = _coerce_data_to_np_float(register_entry[2])
    except ValueError:
        raise PsiParamValidationError("Invalid data for continuous baseline data")

    return (col_name, bl_data)


def smooth_categorical_register_entry(
    register_entry: PSIRegisterRequest,
) -> CategoricalRegisterEntry:
    """
    Perform required data coersions.
    """
    if len(register_entry) != 3:
        raise PsiParamValidationError(
            "Register entry must be length 3 (column name, PsiType | str, baseline data)"
        )

    col_name = register_entry[0]
    bl_data = _coerce_data_to_string_list(register_entry[2])
    return (col_name, bl_data)


def resolve_psi_type(register_entry: PSIRegisterRequest) -> PsiType:
    try:
        return PsiType(register_entry[1])
    except ValueError:
        raise PsiParamValidationError(
            "Register entry must be length 3 (column name, PsiType, baseline data)"
        )


class StreamingPsiManager:
    """
    A managed container for storing and maintaining a set of streaming PSI monitoring agents.
    Designed for long running services where high volume runtime inference data is accumulated.
    """

    __slots__ = ["_state_table", "_drift_table"]

    def __init__(self, monitor_entries: List[PSIRegisterRequest]):
        self._state_table = {}
        self._drift_table = {}

        psi_types = map(resolve_psi_type, monitor_entries)

        for t, entry in zip(psi_types, monitor_entries):
            self._register_agent(t, entry)

    def register_new(self, entry: PSIRegisterRequest) -> None:
        """
        Utility to register new feature for PSI monitoring after initialization.
        """
        psi_type = resolve_psi_type(entry)
        self._register_agent(psi_type, entry)

    def _register_agent(self, psi_type: PsiType, entry: PSIRegisterRequest) -> None:
        """
        Internal utility to resolve PSI type and initialized a monitoring agent.
        """
        if psi_type == PsiType.CONTINUOUS:
            name, bl_data = smooth_continuous_register_entry(entry)
            agent = StreamingContinuousPSI(bl_data)
        else:
            name, bl_data = smooth_categorical_register_entry(entry)
            agent = StreamingCategoricalPSI(bl_data)

        self._state_table[name] = agent

    def push_data_to_stream(self, name: str, data: Iterable[Any]) -> float:
        """
        Push data to a monitoring agent managed by this data structure.
        args:
            name: the name of the registered feature
            data: runtime data to push to the stream
        returns:
            float - the latest PSI drift value
        """
        agent = self._state_table.get(name)

        if agent is None:
            raise KeyError(f"Monitoring agent not registered for {name}")

        data = _coerce_data(agent, data)
        curr_drift = agent.update_stream(data)
        self._drift_table[name] = curr_drift

        return curr_drift

    def latest_feature_drift(self, name: str) -> float:
        """
        Fetch the latest computed PSI drift value for a particular feature.
        """
        curr_drift = self._drift_table.get(name)

        if curr_drift is None:
            raise KeyError(
                f"Monitoring agent not registered for {name} or no data has been pushed to stream"
            )

        return curr_drift

    def flush_feature(self, name: str):
        """
        Flush the runtime data for a particular monitoring agent.
        """
        agent = self._state_table.get(name)

        if agent is None:
            raise KeyError(f"Monitoring agent not registered for {name}")

        agent.flush()

    def flush_all(self):
        """
        Flush runtime data in all monitoring agents.
        """
        for agent in self._state_table.values():
            agent.flush()

    def reset_baseline(self, name: str, data: Iterable[Any]):
        agent = self._state_table.get(name)
        if agent is None:
            raise KeyError(f"Monitoring agent not registered for {name}")

        data = _coerce_data(agent, data)
        agent.reset_baseline(data)


class DiscretePsiManager:
    """
    Manager for discrete PSI computations across a given feature group.
    Provides a discrete snapshot of the data state compared against a baseline set.
    """

    __slots__ = ["_state_table", "_drift_table"]

    def __init__(self, monitor_entries: List[PSIRegisterRequest]):
        self._state_table = {}
        self._drift_table = {}

        psi_types = map(resolve_psi_type, monitor_entries)

        for t, entry in zip(psi_types, monitor_entries):
            self._register_agent(t, entry)

    def register_new(self, entry: PSIRegisterRequest) -> None:
        """
        Utility to register new feature for PSI monitoring after initialization.
        """
        psi_type = resolve_psi_type(entry)
        self._register_agent(psi_type, entry)

    def _register_agent(self, psi_type: PsiType, entry: PSIRegisterRequest) -> None:
        """
        Internal utility to resolve PSI type and initialized a monitoring agent.
        """
        if psi_type == PsiType.CONTINUOUS:
            name, bl_data = smooth_continuous_register_entry(entry)
            agent = ContinuousPSI(bl_data)
        else:
            name, bl_data = smooth_categorical_register_entry(entry)
            agent = CategoricalPSI(bl_data)

        self._state_table[name] = agent

    def compute_psi_drift(self, name: str, data: Iterable[Any]) -> float:
        agent = self._state_table.get(name)

        if agent is None:
            raise KeyError(f"No monitoring agent registered for {name}")

        data = _coerce_data(agent, data)
        return agent.compute_psi(data)

    def reset_baseline(self, name: str, data: Iterable[Any]):
        agent = self._state_table.get(name)
        if agent is None:
            raise KeyError(f"Monitoring agent not registered for {name}")

        data = _coerce_data(agent, data)
        agent.reset_baseline(data)
