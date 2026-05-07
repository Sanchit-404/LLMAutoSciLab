"""
ExperimentStore: accumulates (params, measurement) pairs across all oracle calls.
The GP is always re-fitted on the full history — GP transfer is solved by keeping all data.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autoscilab.oracle.base import OracleResult


class ExperimentStore:
    def __init__(self, domain: str, parameter_names: list[str]):
        self._domain = domain
        self._param_names = parameter_names
        self._results: list[OracleResult] = []

    def add(self, result: OracleResult) -> None:
        self._results.append(result)

    def get_all(self) -> list[OracleResult]:
        return list(self._results)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return X (N, D) and y (N,) arrays for GP fitting."""
        if not self._results:
            return np.empty((0, len(self._param_names))), np.empty(0)
        X = np.array([[r.params[p] for p in self._param_names] for r in self._results])
        y = np.array([r.measurement for r in self._results])
        return X, y

    def __len__(self) -> int:
        return len(self._results)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "domain": self._domain,
            "parameter_names": self._param_names,
            "results": [
                {
                    "params": r.params,
                    "measurement": r.measurement,
                    "noise_level": r.noise_level,
                    "metadata": r.metadata,
                }
                for r in self._results
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ExperimentStore":
        data = json.loads(Path(path).read_text())
        store = cls(domain=data["domain"], parameter_names=data["parameter_names"])
        for r in data["results"]:
            store.add(
                OracleResult(
                    params=r["params"],
                    measurement=r["measurement"],
                    domain=data["domain"],
                    noise_level=r["noise_level"],
                    metadata=r["metadata"],
                )
            )
        return store

    def summary(self) -> str:
        """Return a brief text summary of what's in the store."""
        if not self._results:
            return "No experiments yet."
        _, y = self.to_arrays()
        return (
            f"Domain: {self._domain} | "
            f"N={len(self._results)} experiments | "
            f"y range: [{y.min():.3e}, {y.max():.3e}] | "
            f"y mean: {y.mean():.3e}"
        )

    def to_text_table(self, max_rows: int = 20) -> str:
        """Format store as a plain-text table for LLM prompts.

        Shows both raw values and log10 values so the LLM can identify
        power-law exponents by comparing log columns directly.
        """
        if not self._results:
            return "No experiments yet."

        log_header = " | ".join(
            [f"log10({p})" for p in self._param_names] + ["log10(|measurement|)"]
        )
        raw_header = " | ".join(self._param_names + ["measurement"])

        rows = []
        results_to_show = self._results[-max_rows:]
        for r in results_to_show:
            raw_vals = [f"{r.params[p]:.4g}" for p in self._param_names]
            raw_vals.append(f"{r.measurement:.6g}")

            import math
            log_vals = []
            for p in self._param_names:
                v = r.params[p]
                log_vals.append(f"{math.log10(abs(v)):.2f}" if v != 0 else "—")
            m = r.measurement
            log_vals.append(f"{math.log10(abs(m)):.2f}" if m != 0 else "—")

            rows.append(" | ".join(raw_vals) + "    [" + " | ".join(log_vals) + "]")

        header = f"{raw_header}    [{log_header}]"
        table = f"{header}\n" + "\n".join(rows)
        if len(self._results) > max_rows:
            table = f"[Showing last {max_rows} of {len(self._results)} experiments]\n" + table
        return table
