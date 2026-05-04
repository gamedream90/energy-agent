from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .models import Constraints, Weights


@dataclass
class AgentConfig:
    backend: str
    simulation_mode: bool
    default_task: str
    build_local_artifacts: bool
    artifact_dir: str
    calibration_data_dir: str | None
    constraints: Constraints
    weights: Weights
    candidates: List[Dict[str, Any]]



def load_config(path: str) -> AgentConfig:
    payload: Dict[str, Any] = json.loads(Path(path).read_text())
    constraints = Constraints(**payload["constraints"])
    weights = Weights(**payload.get("weights", {}))

    return AgentConfig(
        backend=payload.get("backend", "openrouter"),
        simulation_mode=bool(payload.get("simulation_mode", True)),
        default_task=payload.get("default_task", "general"),
        build_local_artifacts=bool(payload.get("build_local_artifacts", False)),
        artifact_dir=payload.get("artifact_dir", "artifacts"),
        calibration_data_dir=payload.get("calibration_data_dir"),
        constraints=constraints,
        weights=weights,
        candidates=payload["candidates"],
    )
