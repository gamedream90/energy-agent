from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .models import Candidate, DeviceProfile



def generate_candidates_from_config(
    candidate_specs: List[Dict[str, Any]],
    device: DeviceProfile,
    artifact_dir: str = "artifacts",
) -> List[Candidate]:
    candidates: List[Candidate] = []

    for spec in candidate_specs:
        precision = spec.get("precision", "fp16")
        if precision == "fp16" and not device.supports_fp16:
            continue
        if precision == "int8" and not device.supports_int8:
            continue

        runtime = spec.get("runtime", "openrouter-api")
        source_model_path = spec.get("source_model_path")
        artifact_path = spec.get("artifact_path")
        if artifact_path is None and source_model_path:
            suffix = _runtime_suffix(runtime, source_model_path)
            artifact_name = f"{spec['name'].replace(' ', '_')}_{spec.get('quantization', 'none')}{suffix}"
            artifact_path = str(Path(artifact_dir) / artifact_name)

        candidates.append(
            Candidate(
                name=spec["name"],
                provider=spec.get("provider", "openrouter"),
                model=spec["model"],
                quantization=spec.get("quantization", "none"),
                precision=precision,
                runtime=runtime,
                source_model_path=source_model_path,
                artifact_path=artifact_path,
                config=spec.get("config", {}),
            )
        )

    return candidates


def _runtime_suffix(runtime: str, source_model_path: str) -> str:
    runtime_l = runtime.lower()
    source_path = Path(source_model_path)
    if "tflite" in runtime_l:
        return ".tflite"
    if source_path.suffix:
        return source_path.suffix
    return ".onnx"
