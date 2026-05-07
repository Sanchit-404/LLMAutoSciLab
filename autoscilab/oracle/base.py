from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OracleResult:
    params: dict[str, float]
    measurement: float
    domain: str
    noise_level: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseOracle(ABC):
    """Abstract oracle: receives parameter dict, returns scalar measurement."""

    @property
    @abstractmethod
    def domain(self) -> str:
        """Return domain identifier string, e.g. 'm0_gravity'."""
        ...

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Ordered list of input parameter names for this domain."""
        ...

    @property
    @abstractmethod
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Log-scale bounds for each parameter: {name: (min, max)}."""
        ...

    @property
    @abstractmethod
    def function_signature(self) -> str:
        """Expected function signature for discovered_law, e.g. 'def discovered_law(mass1, mass2, distance):'"""
        ...

    @property
    @abstractmethod
    def param_description(self) -> str:
        """Human-readable description of parameters."""
        ...

    @property
    def objective_type(self) -> str:
        """Type of downstream objective: 'equation' (default) or 'optimization'."""
        return "equation"

    @property
    def objective_direction(self) -> str:
        """Direction of optimization-style objectives: 'maximize' (default) or 'minimize'."""
        return "maximize"

    @property
    def objective_profile(self) -> dict[str, Any]:
        """
        Optional structured objective metadata for prompt conditioning.

        Examples:
          {"name": "yield", "direction": "maximize", "type": "single_objective"}
          {"name": "validation_rmsle", "direction": "minimize", "type": "equation_fit"}
        """
        return {}

    @abstractmethod
    def run(self, params: dict[str, float]) -> OracleResult:
        """Execute one experiment, return OracleResult."""
        ...

    @abstractmethod
    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        """
        Score a discovered law string against ground truth.
        Returns dict with keys: rmsle, exact_accuracy, symbolic_equivalent, error
        """
        ...

    def evaluate_run(self, results: list[OracleResult]) -> dict[str, Any] | None:
        """
        Optional run-level evaluation for non-equation objectives.

        For optimization domains this can report metrics like best/mean value,
        improvement ratio, regret, target-hit, etc.
        """
        return None
