from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class PipelineConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def columns(self) -> dict[str, str]:
        defaults = {
            "timestamp": "timestamp",
            "label": "label",
            "transaction_id": "transaction_id",
            "account_id": "account_id",
            "counterparty_account_id": "counterparty_account_id",
        }
        defaults.update(self.raw.get("columns", {}))
        return defaults

    @property
    def supervised(self) -> dict[str, Any]:
        return self.raw.get("supervised", {})

    @property
    def anomaly(self) -> dict[str, Any]:
        return self.raw.get("anomaly", {})

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw.get("risk", {})

    @property
    def graph(self) -> dict[str, Any]:
        return self.raw.get("graph", {})

    @property
    def components(self) -> dict[str, bool]:
        """Return component switches, preserving the original all-on behavior."""
        defaults = {
            "supervised": True,
            "anomaly": True,
            "graph": True,
        }
        defaults.update(self.raw.get("components", {}))
        return {name: bool(enabled) for name, enabled in defaults.items()}


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f) or {}
        elif yaml is not None:
            data = yaml.safe_load(f) or {}
        else:
            raise ModuleNotFoundError(
                "PyYAML is not installed. Use a .json config file or install pyyaml."
            )
    return PipelineConfig(raw=data)
