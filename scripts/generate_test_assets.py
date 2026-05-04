from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CALIB_DIR = ROOT / "calibration"


def build_tiny_onnx_model(path: Path) -> None:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    weights = np.array(
        [
            [0.2, -0.1, 0.4],
            [0.7, 0.3, -0.2],
            [-0.5, 0.6, 0.1],
            [0.9, -0.4, 0.8],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.05, -0.03, 0.1], dtype=np.float32)

    w_init = numpy_helper.from_array(weights, name="W")
    b_init = numpy_helper.from_array(bias, name="B")

    matmul_node = helper.make_node("MatMul", ["input", "W"], ["mm_out"])
    add_node = helper.make_node("Add", ["mm_out", "B"], ["output"])

    graph = helper.make_graph(
        nodes=[matmul_node, add_node],
        name="TinyLinearModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[w_init, b_init],
    )

    model = helper.make_model(graph, producer_name="edge-agent-test")
    model.opset_import[0].version = 11
    model.ir_version = 7
    onnx.save(model, str(path))


def build_calibration_samples(path: Path) -> None:
    rng = np.random.default_rng(seed=42)
    samples = rng.normal(loc=0.0, scale=1.0, size=(128, 1, 4)).astype(np.float32)
    np.save(str(path), samples)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "model.onnx"
    calib_path = CALIB_DIR / "samples.npy"

    build_tiny_onnx_model(model_path)
    build_calibration_samples(calib_path)

    print(f"Wrote ONNX model: {model_path}")
    print(f"Wrote calibration data: {calib_path}")


if __name__ == "__main__":
    main()
