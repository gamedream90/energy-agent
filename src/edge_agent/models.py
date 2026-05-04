from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DeviceProfile:
    name: str
    cpu_cores: int
    ram_mb: int
    has_gpu: bool = False
    supports_fp16: bool = False
    supports_int8: bool = True


@dataclass
class Constraints:
    max_p95_latency_ms: float
    max_peak_memory_mb: int
    min_quality_score: float
    max_energy_mj: Optional[float] = None


@dataclass
class Weights:
    quality: float = 1.0
    latency: float = 0.02
    memory: float = 0.003
    energy: float = 0.001


@dataclass
class ModelConfig:
    name: str
    source_path: str
    source_format: str = "onnx"
    accuracy_baseline: float = 0.85


@dataclass
class Candidate:
    name: str
    provider: str
    model: str
    quantization: str
    precision: str
    runtime: str
    source_model_path: Optional[str] = None
    artifact_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateMetrics:
    quality_score: float
    p95_latency_ms: float
    peak_memory_mb: int
    energy_mj: float
    est_cost_per_1k: float


@dataclass
class EvaluationResult:
    candidate: Candidate
    metrics: CandidateMetrics
    passed: bool
    reasons: List[str]
    score: Optional[float] = None




@dataclass
class InteractiveConfig:
    device: DeviceProfile
    constraints: Constraints
    weights: Weights
    models: List[ModelConfig]
    simulation_mode: bool = True
    build_local_artifacts: bool = False
    artifact_dir: str = "artifacts"
    calibration_data_dir: Optional[str] = None

    def save(self, path: str) -> None:
        data = {
            "device": {
                "name": self.device.name,
                "cpu_cores": self.device.cpu_cores,
                "ram_mb": self.device.ram_mb,
                "has_gpu": self.device.has_gpu,
                "supports_fp16": self.device.supports_fp16,
                "supports_int8": self.device.supports_int8,
            },
            "constraints": {
                "max_p95_latency_ms": self.constraints.max_p95_latency_ms,
                "max_peak_memory_mb": self.constraints.max_peak_memory_mb,
                "min_quality_score": self.constraints.min_quality_score,
                "max_energy_mj": self.constraints.max_energy_mj,
            },
            "weights": {
                "quality": self.weights.quality,
                "latency": self.weights.latency,
                "memory": self.weights.memory,
                "energy": self.weights.energy,
            },
            "models": [
                {
                    "name": m.name,
                    "source_path": m.source_path,
                    "source_format": m.source_format,
                    "accuracy_baseline": m.accuracy_baseline,
                }
                for m in self.models
            ],
            "simulation_mode": self.simulation_mode,
            "build_local_artifacts": self.build_local_artifacts,
            "artifact_dir": self.artifact_dir,
            "calibration_data_dir": self.calibration_data_dir,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> InteractiveConfig:
        data = json.loads(Path(path).read_text())
        device = DeviceProfile(**data["device"])
        constraints = Constraints(**data["constraints"])
        weights = Weights(**data["weights"])
        models = [ModelConfig(**m) for m in data["models"]]
        return cls(
            device=device,
            constraints=constraints,
            weights=weights,
            models=models,
            simulation_mode=data.get("simulation_mode", True),
            build_local_artifacts=data.get("build_local_artifacts", False),
            artifact_dir=data.get("artifact_dir", "artifacts"),
            calibration_data_dir=data.get("calibration_data_dir"),
        )
@dataclass
class RunResult:
    device: DeviceProfile
    constraints: Constraints
    weights: Weights
    evaluations: List[EvaluationResult]
    winner: Optional[EvaluationResult]
    artifact_logs: List[str] = field(default_factory=list)
    feasibility_analysis: Optional[Dict[str, Any]] = None
    winner_explanation: str = ""
