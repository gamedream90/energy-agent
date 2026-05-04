# Energy Agent: Edge Model Optimization & Quantization

**Energy Agent** is an intelligent optimization framework for edge deployment decisioning. It evaluates quantization candidates (FP32/FP16/INT8 dynamic/static) for local ONNX/TFLite models, applies SLA constraints (latency, memory, quality, energy), and selects the best configuration for your device.

## Features

- 🎯 **Interactive CLI** for device, constraint, and model definition (no JSON editing required)
- 📦 **Local ONNX Artifact Generation** (FP16, INT8 dynamic, INT8 static)
- 🔧 **TFLite Support** (SavedModel → TFLite with optimization)
- 📊 **Multi-Model Optimization** in a single run
- 🚀 **OpenRouter Integration** for live remote model probing (optional)
- ⚖️ **Constraint Filtering** (latency p95, peak memory, quality, energy per inference)
- 🏆 **Weighted Scoring** for winner selection
- 📈 **Human-Readable + JSON Output**
- 🐳 **Docker Support** with hardened security defaults
- ⚡ **Simulation Mode** for testing (no API key required)

## What It Does

### ONNX Quantization Flow (Default)
1. **Takes** a source ONNX model
2. **Generates** quantization candidates:
   - FP32 (baseline, no quantization)
   - FP16 (half-precision)
   - INT8 dynamic quantization
   - INT8 static quantization (with calibration data)
3. **Builds** all variants in `artifacts/`
4. **Evaluates** constraints and metrics for each variant
5. **Ranks** by weighted score: `quality - latency*w - memory*w - energy*w`
6. **Selects** the best configuration that meets all SLAs

### OpenRouter Live Probe Flow (Optional)
1. Define OpenRouter model candidates
2. Set `simulation_mode: false` + provide `OPENROUTER_API_KEY`
3. Agent probes each model via API in multiple rounds
4. Measures real latency and token usage
5. Ranks and selects best candidate

## Quick Start

### Install

```bash
cd energy_agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional: Install quantization backends (for ONNX/TFLite):

```bash
pip install -e .[quant]
```

### Run (ONNX Mode)

```bash
edge-agent run --config configs/local_onnx_test_config.json
```

### Run (Interactive Setup)

```bash
edge-agent interactive --build --save my_config.json
```

This launches a guided wizard:
- Device profile (CPU cores, RAM, GPU support)
- Constraints (latency, memory, quality, energy)
- Models to optimize
- Weights for scoring
- Build and simulation settings

## Commands

### `edge-agent run`

Run optimization with a config file or interactive mode.

```bash
edge-agent run --config configs/local_onnx_test_config.json [OPTIONS]
```

**Options:**
- `--config FILE` — Load config from JSON (optional; prompts interactive if omitted)
- `--device NAME` — Device name or "auto" (default: auto-detect)
- `--build` — Force build local ONNX/TFLite artifacts
- `--json` — Output full JSON (default: human-readable)

**Examples:**

```bash
# ONNX quantization with artifact building
edge-agent run --config configs/local_onnx_test_config.json --build

# JSON output for scripting
edge-agent run --config configs/local_onnx_test_config.json --json

# Interactive + save result
edge-agent interactive --save my_run.json
```

### `edge-agent interactive`

Guided setup wizard.

```bash
edge-agent interactive [OPTIONS]
```

**Options:**
- `--save FILE` — Save config to JSON after setup
- `--build` — Build artifacts after setup
- `--json` — Output full JSON

## Config Structure

### Basic Config (ONNX Quantization)

```json
{
  "backend": "hybrid",
  "simulation_mode": true,
  "build_local_artifacts": true,
  "artifact_dir": "artifacts",
  "calibration_data_dir": "calibration",
  "constraints": {
    "max_p95_latency_ms": 220,
    "max_peak_memory_mb": 3000,
    "min_quality_score": 0.82,
    "max_energy_mj": 140
  },
  "weights": {
    "quality": 1.0,
    "latency": 0.02,
    "memory": 0.003,
    "energy": 0.001
  },
  "candidates": [
    {
      "name": "local-onnx-fp32",
      "provider": "local",
      "model": "local/onnx-base",
      "precision": "fp32",
      "quantization": "none",
      "runtime": "onnxruntime-local",
      "source_model_path": "models/model.onnx"
    },
    {
      "name": "local-onnx-int8-dynamic",
      "provider": "local",
      "model": "local/onnx-base",
      "precision": "int8",
      "quantization": "dynamic",
      "runtime": "onnxruntime-local",
      "source_model_path": "models/model.onnx"
    }
  ]
}
```

### OpenRouter Config (Live API Probing)

```json
{
  "backend": "openrouter",
  "simulation_mode": false,
  "build_local_artifacts": false,
  "constraints": { ... },
  "weights": { ... },
  "candidates": [
    {
      "name": "gpt-4o-mini-fp32",
      "provider": "openrouter",
      "model": "openai/gpt-4o-mini",
      "precision": "fp32",
      "quantization": "none",
      "runtime": "openrouter-api"
    }
  ]
}
```

**Config Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `backend` | string | "openrouter" or "hybrid" |
| `simulation_mode` | bool | True = estimated metrics; False = live/measured |
| `build_local_artifacts` | bool | Generate ONNX/TFLite files |
| `artifact_dir` | string | Output directory for built models |
| `calibration_data_dir` | string | Path to .npy calibration data (for INT8 static) |
| `constraints` | object | SLA limits for latency, memory, quality, energy |
| `weights` | object | Scoring weights (higher = more important) |
| `candidates` | array | Model variants to evaluate |

## Output

### Human-Readable (Default)

```
============================================================
  Model: default
============================================================

Device: linux-edge (8c, 7778MB RAM)

Constraints:
  p95 latency: <= 220.0ms
  peak memory: <= 3000MB
  min quality: >= 0.820
  max energy: <= 140.0mJ

Candidates (4 variants):
  ✓ PASS  local-onnx-fp32               q=0.891  p95=219.9ms  mem=2722MB  score=-11.771128
  ✓ PASS  local-onnx-fp16               q=0.891  p95=146.7ms  mem=2177MB  score= -8.640199
  ✓ PASS  local-onnx-int8-dynamic       q=0.869  p95=120.4ms  mem=1687MB  score= -6.653708
  ✓ PASS  local-onnx-int8-static        q=0.889  p95=134.4ms  mem=1687MB  score= -6.920522

Artifact Build Logs:
  [OK] OK local-onnx-fp32: copied ONNX artifact -> artifacts/local-onnx-fp32_none.onnx
  [OK] OK local-onnx-fp16: ONNX FP16 artifact -> artifacts/local-onnx-fp16_float16.onnx
  [OK] OK local-onnx-int8-dynamic: ONNX INT8 dynamic artifact -> artifacts/local-onnx-int8-dynamic_dynamic.onnx
  [OK] OK local-onnx-int8-static: ONNX INT8 static artifact -> artifacts/local-onnx-int8-static_static.onnx

Feasibility Analysis:
  ✓ All candidates are feasible
  Reasoning: Analyzed all candidates against device constraints.

============================================================
  🏆 Winner: local-onnx-int8-dynamic
============================================================
  Model: local/onnx-base
  Precision: int8 | Quantization: dynamic
  Quality: 0.8691
  Latency (p95): 120.38ms
  Memory: 1687MB
  Energy: 54.17mJ
  Score: -6.653708

  Explanation:
  local-onnx-int8-dynamic is optimal: meets all SLAs (latency 120ms, memory 1687MB) while maintaining 86.9% quality.
```

### JSON Output

```bash
edge-agent run --config configs/local_onnx_test_config.json --json
```

Returns full structured result with all metrics, evaluations, and reasoning.

## Environment Variables

### OpenRouter Integration

Set your API key for live OpenRouter probing:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

Or in `.env`:

```bash
cp .env.example .env
# Edit .env with your key
```

## Examples

### Example 1: ONNX Quantization (Simulation)

```bash
edge-agent run --config configs/local_onnx_test_config.json
```

Uses simulated metrics; no API calls. Builds and evaluates local ONNX variants.

### Example 2: OpenRouter Live Probing

```bash
export OPENROUTER_API_KEY="sk-..."
edge-agent run --config configs/live_openrouter_config.json
```

Requires API key. Probes actual latency/performance from OpenRouter.

### Example 3: Interactive Setup

```bash
edge-agent interactive --build --save my_setup.json
```

Guided wizard → build artifacts → save config for later reuse.

### Example 4: Multi-Model Optimization

```bash
edge-agent interactive
# Select: 2 models to optimize
# Then build and compare all quantization variants
```

## Architecture

### Core Components

- **`agent.py`** — Main orchestration (single/multi-model flows)
- **`evaluator.py`** — Candidate evaluation and scoring
- **`artifact_builder.py`** — ONNX/TFLite artifact generation
- **`openrouter_client.py`** — OpenRouter API client
- **`reasoner.py`** — Feasibility analysis & LLM-based explanations
- **`device.py`** — Device auto-detection
- **`cli.py`** — CLI commands and output formatting

### Data Models

- **`Candidate`** — Model variant (precision, quantization, provider)
- **`DeviceProfile`** — Target device specs
- **`Constraints`** — SLA limits
- **`EvaluationResult`** — Metrics + pass/fail for one candidate
- **`RunResult`** — Final result with winner, feasibility, explanation

## Docker

Build and run with hardened security:

```bash
docker compose up --build
```

**Security features:**
- Non-root container user
- Read-only root filesystem
- No new privileges
- All Linux capabilities dropped
- Mounts: `configs/` (ro), `artifacts/` (rw) only

## Troubleshooting

### "OPENROUTER_API_KEY missing" error

**Cause:** Running in live mode (`simulation_mode: false`) without OpenRouter key.

**Solutions:**
1. Set `OPENROUTER_API_KEY` environment variable
2. Set `simulation_mode: true` in config for estimated metrics
3. Use local ONNX config instead

### OpenRouter candidates skipped with warnings

**Cause:** Live mode enabled but API key not found.

**Behavior:** Agent skips OpenRouter candidates and continues with local ones (if any).

### "Source model not found"

**Cause:** `source_model_path` in config points to non-existent file.

**Solution:** Verify path exists and is relative to current working directory.

### INT8 static fallback to dynamic

**Cause:** No calibration data found at specified path.

**Notice:** Agent warns and falls back to INT8 dynamic quantization.

## Development

### Install Dev Dependencies

```bash
pip install -e .[dev]
```

### Run Tests

```bash
pytest
```

### Format & Lint

```bash
black src/
mypy src/
```

## Limitations & Future Work

- ✗ ONNX model metrics currently use heuristics, not real runtime benchmarking
- ✗ No built-in model accuracy validation on labeled datasets
- ✗ Limited to single-batch inference profiling
- ✓ TODO: Real per-artifact inference loop with validation data
- ✓ TODO: Multi-image ONNX model support
- ✓ TODO: Streaming quantization data reader

## License

MIT

## Contributing

Issues and pull requests welcome.

## References

- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization/quantization-overview.html)
- [OpenRouter API](https://openrouter.ai/docs)
- [ONNX Converter Common](https://github.com/microsoft/onnxconverter-common)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
