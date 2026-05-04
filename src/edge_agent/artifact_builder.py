from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List

from .models import Candidate


class LocalArtifactBuilder:
    def __init__(self, artifact_dir: str = "artifacts", calibration_data_dir: str | None = None) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.calibration_data_dir = Path(calibration_data_dir) if calibration_data_dir else None

    def build(self, candidates: Iterable[Candidate]) -> List[str]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        logs: List[str] = []

        for candidate in candidates:
            if candidate.provider != "local":
                continue

            if not candidate.source_model_path:
                logs.append(f"SKIP {candidate.name}: missing source_model_path")
                continue

            runtime = candidate.runtime.lower()
            if "onnx" in runtime:
                logs.extend(self._build_onnx(candidate))
            elif "tflite" in runtime:
                logs.extend(self._build_tflite(candidate))
            else:
                logs.append(f"SKIP {candidate.name}: unsupported runtime '{candidate.runtime}'")

        return logs

    def _build_onnx(self, candidate: Candidate) -> List[str]:
        logs: List[str] = []
        source = Path(candidate.source_model_path or "")
        target = Path(candidate.artifact_path or self.artifact_dir / f"{candidate.name}.onnx")
        target.parent.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            candidate.artifact_path = None
            return [f"FAIL {candidate.name}: source model not found at {source}"]

        quantization = candidate.quantization.lower()
        if quantization == "none":
            shutil.copy2(source, target)
            candidate.artifact_path = str(target)
            return [f"OK {candidate.name}: copied ONNX artifact -> {target}"]

        try:
            if quantization == "float16":
                import onnx
                from onnxconverter_common import float16

                model = onnx.load(str(source))
                fp16_model = float16.convert_float_to_float16(model)
                onnx.save(fp16_model, str(target))
                candidate.artifact_path = str(target)
                logs.append(f"OK {candidate.name}: ONNX FP16 artifact -> {target}")
                return logs

            from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static

            if quantization == "dynamic":
                quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
                candidate.artifact_path = str(target)
                logs.append(f"OK {candidate.name}: ONNX INT8 dynamic artifact -> {target}")
                return logs

            if quantization == "static":
                calib_file = self._resolve_calibration_file(candidate)
                if calib_file is None:
                    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
                    candidate.artifact_path = str(target)
                    logs.append(
                        f"WARN {candidate.name}: no calibration data, fallback to ONNX INT8 dynamic -> {target}"
                    )
                    return logs

                data_reader = _NumpyCalibrationDataReader(
                    model_path=str(source),
                    calibration_file=str(calib_file),
                )
                quantize_static(
                    model_input=str(source),
                    model_output=str(target),
                    calibration_data_reader=data_reader,
                    quant_format=QuantFormat.QOperator,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                )
                candidate.artifact_path = str(target)
                logs.append(f"OK {candidate.name}: ONNX INT8 static artifact -> {target}")
                return logs

            logs.append(f"WARN {candidate.name}: unsupported ONNX quantization '{candidate.quantization}'")
            return logs
        except Exception as exc:
            candidate.artifact_path = None
            logs.append(f"FAIL {candidate.name}: ONNX build failed ({exc})")
            return logs

    def _build_tflite(self, candidate: Candidate) -> List[str]:
        logs: List[str] = []
        source = Path(candidate.source_model_path or "")
        target = Path(candidate.artifact_path or self.artifact_dir / f"{candidate.name}.tflite")
        target.parent.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            candidate.artifact_path = None
            return [f"FAIL {candidate.name}: source model not found at {source}"]

        quantization = candidate.quantization.lower()

        if source.suffix == ".tflite" and quantization == "none":
            shutil.copy2(source, target)
            candidate.artifact_path = str(target)
            return [f"OK {candidate.name}: copied TFLite artifact -> {target}"]

        try:
            import numpy as np
            import tensorflow as tf

            if source.is_dir():
                converter = tf.lite.TFLiteConverter.from_saved_model(str(source))
            else:
                candidate.artifact_path = None
                return [
                    f"FAIL {candidate.name}: TFLite conversion expects SavedModel directory or .tflite copy source"
                ]

            if quantization in {"dynamic", "static", "float16"}:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if quantization == "float16":
                converter.target_spec.supported_types = [tf.float16]

            if quantization == "static":
                calib_file = self._resolve_calibration_file(candidate)
                if calib_file is None:
                    logs.append(
                        f"WARN {candidate.name}: no calibration data, fallback to TFLite dynamic range quantization"
                    )
                else:
                    calibration_data = np.load(str(calib_file), allow_pickle=False)

                    def representative_dataset():
                        for row in calibration_data[: min(128, len(calibration_data))]:
                            yield [row.astype(np.float32)]

                    converter.representative_dataset = representative_dataset

            tflite_model = converter.convert()
            target.write_bytes(tflite_model)
            candidate.artifact_path = str(target)
            logs.append(f"OK {candidate.name}: TFLite artifact -> {target}")
            return logs
        except Exception as exc:
            candidate.artifact_path = None
            logs.append(f"FAIL {candidate.name}: TFLite build failed ({exc})")
            return logs

    def _resolve_calibration_file(self, candidate: Candidate) -> Path | None:
        candidate_calib = candidate.config.get("calibration_file")
        if candidate_calib:
            path = Path(candidate_calib)
            return path if path.exists() else None

        if self.calibration_data_dir and self.calibration_data_dir.exists():
            files = sorted(self.calibration_data_dir.glob("*.npy"))
            if files:
                return files[0]

        return None


class _NumpyCalibrationDataReader:
    def __init__(self, model_path: str, calibration_file: str) -> None:
        import numpy as np
        import onnx

        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        self.data = np.load(calibration_file, allow_pickle=False)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        sample = self.data[self.index]
        self.index += 1
        return {self.input_name: sample}

    def rewind(self) -> None:
        self.index = 0