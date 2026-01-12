from typing import List, Iterable, Any, Union, NamedTuple
from .base import (
    DataDriftRegisterRequest,
    resolve_drift_type,
    DriftType,
    _coerce_data,
    smooth_continuous_register_entry,
    smooth_categorical_register_entry,
)
from .streaming import StreamingContinuousDataDrift, StreamingCategoricalDataDrift

StreamingDriftAgent = Union[StreamingContinuousDataDrift, StreamingCategoricalDataDrift]


class DriftManagerEntry(NamedTuple):
    name: str
    drift_agent: StreamingDriftAgent


class StreamingDataDriftManager:
    """
    A managed container for storing and maintaining a set of streaming DataDrift monitoring agents.
    Designed for long running services where high volume runtime inference data is accumulated.
    """

    __slots__ = ["_state_table", "_drift_table"]

    def __init__(self, monitor_entries: List[DriftManagerEntry]):
        self._state_table = {}
        self._drift_table = {}

        for entry in monitor_entries:
            self._register_agent(entry)

    def register_new(self, entry: DriftManagerEntry) -> None:
        """
        Utility to register new feature for DataDrift monitoring after initialization.
        """
        self._register_agent(entry)

    def _register_agent(self, entry: DriftManagerEntry) -> None:
        """
        Internal utility to resolve DataDrift type and initialized a monitoring agent.
        """
        self._state_table[entry.name] = entry.drift_agent

    def push_data_to_stream(self, name: str, data: Iterable[Any]) -> float:
        """
        Push data to a monitoring agent managed by this data structure.
        args:
            name: the name of the registered feature
            data: runtime data to push to the stream
        returns:
            float - the latest DataDrift drift value
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
        Fetch the latest computed DataDrift drift value for a particular feature.
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
    Manager for discrete DataDrift computations across a given feature group.
    Provides a discrete snapshot of the data state compared against a baseline set.
    """

    __slots__ = ["_state_table", "_drift_table"]

    def __init__(self, monitor_entries: List[DataDriftRegisterRequest]):
        self._state_table = {}
        self._drift_table = {}

        psi_types = map(resolve_drift_type, monitor_entries)

        for t, entry in zip(psi_types, monitor_entries):
            self._register_agent(t, entry)

    def register_new(self, entry: DataDriftRegisterRequest) -> None:
        """
        Utility to register new feature for DataDrift monitoring after initialization.
        """
        psi_type = resolve_drift_type(entry)
        self._register_agent(psi_type, entry)

    def _register_agent(
        self, psi_type: DriftType, entry: DataDriftRegisterRequest
    ) -> None:
        """
        Internal utility to resolve DataDrift type and initialized a monitoring agent.
        """
        if psi_type == DriftType.CONTINUOUS:
            name, bl_data = smooth_continuous_register_entry(entry)
            agent = ContinuousDataDrift(bl_data)
        else:
            name, bl_data = smooth_categorical_register_entry(entry)
            agent = CategoricalDataDrift(bl_data)

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
