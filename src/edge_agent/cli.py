from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import typer

from .agent import EdgeOptimizationAgent
from .config import load_config
from .device import detect_device
from .models import (
    Constraints,
    DeviceProfile,
    InteractiveConfig,
    ModelConfig,
    RunResult,
    Weights,
)

app = typer.Typer(
    name="edge-agent",
    help="Edge model optimization agent with interactive constraint definition.",
    no_args_is_help=True,
)


def interactive_setup() -> InteractiveConfig:
    """Interactively gather device, constraints, models, and weights."""

    typer.echo("\n" + "=" * 60)
    typer.echo("  Edge Optimizer Agent - Interactive Setup")
    typer.echo("=" * 60)

    # === Device Profile ===
    typer.echo("\n[1/4] Device Profile")
    typer.echo("-" * 40)

    device_name = typer.prompt("Device name (e.g., 'rpi5', 'jetson-nano')", default="linux-edge")
    cpu_cores = typer.prompt("CPU cores", type=int, default=detect_device("auto").cpu_cores)
    ram_mb = typer.prompt("RAM (MB)", type=int, default=detect_device("auto").ram_mb)
    has_gpu = typer.confirm("Has GPU support?", default=False)

    device = DeviceProfile(
        name=device_name,
        cpu_cores=cpu_cores,
        ram_mb=ram_mb,
        has_gpu=has_gpu,
        supports_fp16=cpu_cores >= 2,
        supports_int8=True,
    )

    # === Constraints ===
    typer.echo("\n[2/4] Constraints (SLAs)")
    typer.echo("-" * 40)

    p95_latency_ms = typer.prompt("Max p95 latency (ms)", type=float, default=50.0)
    max_memory_mb = typer.prompt("Max memory usage (MB)", type=int, default=512)
    min_quality = typer.prompt("Min quality/accuracy (0-1)", type=float, default=0.85)
    max_energy_mj = typer.prompt("Max energy per inference (mJ)", type=float, default=10.0)

    constraints = Constraints(
        max_p95_latency_ms=p95_latency_ms,
        max_peak_memory_mb=max_memory_mb,
        min_quality_score=min_quality,
        max_energy_mj=max_energy_mj,
    )

    # === Models ===
    typer.echo("\n[3/4] Models to Optimize")
    typer.echo("-" * 40)

    num_models = typer.prompt("How many models to optimize?", type=int, default=1)
    models: List[ModelConfig] = []

    for i in range(num_models):
        typer.echo(f"\n  Model {i + 1}/{num_models}:")
        model_name = typer.prompt(f"    Name (e.g., 'mobilenet-v3')")
        source_path = typer.prompt(f"    Path (e.g., 'models/model.onnx')")
        source_format = typer.prompt(f"    Format (onnx/saved_model)", default="onnx")
        accuracy_baseline = typer.prompt(f"    Baseline accuracy (0-1)", type=float, default=0.85)

        models.append(
            ModelConfig(
                name=model_name,
                source_path=source_path,
                source_format=source_format,
                accuracy_baseline=accuracy_baseline,
            )
        )

    # === Weights ===
    typer.echo("\n[4/4] Optimization Weights")
    typer.echo("-" * 40)

    w_quality = typer.prompt("Quality weight", type=float, default=1.0)
    w_latency = typer.prompt("Latency weight", type=float, default=0.02)
    w_memory = typer.prompt("Memory weight", type=float, default=0.003)
    w_energy = typer.prompt("Energy weight", type=float, default=0.001)

    weights = Weights(quality=w_quality, latency=w_latency, memory=w_memory, energy=w_energy)

    # === Build settings ===
    typer.echo("\n[Bonus] Build Settings")
    typer.echo("-" * 40)
    build_artifacts = typer.confirm("Build local ONNX/TFLite artifacts?", default=False)
    simulation_mode = typer.confirm("Simulation mode (no OpenRouter calls)?", default=True)

    config = InteractiveConfig(
        device=device,
        constraints=constraints,
        weights=weights,
        models=models,
        simulation_mode=simulation_mode,
        build_local_artifacts=build_artifacts,
        artifact_dir="artifacts",
        calibration_data_dir="calibration",
    )

    typer.echo("\n✓ Configuration complete!\n")
    return config


def _print_human(result: RunResult | dict) -> None:
    """Pretty-print single or multi-model results."""

    if isinstance(result, dict):
        # Multi-model results
        for model_name, run_result in result.items():
            _print_single_result(model_name, run_result)
    else:
        # Single model result
        _print_single_result("default", result)


def _print_single_result(model_name: str, result: RunResult) -> None:
    """Pretty-print a single model's results."""

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Model: {model_name}")
    typer.echo(f"{'='*60}")

    typer.echo(f"\nDevice: {result.device.name} ({result.device.cpu_cores}c, {result.device.ram_mb}MB RAM)")

    typer.echo(f"\nConstraints:")
    typer.echo(f"  p95 latency: <= {result.constraints.max_p95_latency_ms:.1f}ms")
    typer.echo(f"  peak memory: <= {result.constraints.max_peak_memory_mb}MB")
    typer.echo(f"  min quality: >= {result.constraints.min_quality_score:.3f}")
    if result.constraints.max_energy_mj:
        typer.echo(f"  max energy: <= {result.constraints.max_energy_mj:.1f}mJ")

    typer.echo(f"\nCandidates ({len(result.evaluations)} variants):")
    for r in result.evaluations:
        m = r.metrics
        status = "✓ PASS" if r.passed else "✗ FAIL"
        score_str = f"{r.score:.6f}" if r.score is not None else "n/a"
        typer.echo(
            f"  {status}  {r.candidate.name:<28}  "
            f"q={m.quality_score:.3f}  p95={m.p95_latency_ms:6.1f}ms  "
            f"mem={m.peak_memory_mb:5d}MB  "
            f"score={score_str:>10}"
        )
        if r.reasons:
            typer.echo(f"         └─ {', '.join(r.reasons)}")

    if result.artifact_logs:
        has_runtime_skip_logs = any("skipped (" in line for line in result.artifact_logs)
        has_artifact_build_logs = any(
            token in line.lower()
            for line in result.artifact_logs
            for token in ["artifact", "onnx", "tflite", "copied"]
        )

        if has_runtime_skip_logs and has_artifact_build_logs:
            log_section_title = "Build & Runtime Logs"
        elif has_runtime_skip_logs:
            log_section_title = "Runtime Logs"
        else:
            log_section_title = "Artifact Build Logs"

        typer.echo(f"\n{log_section_title}:")
        for line in result.artifact_logs:
            status = "OK" if line.startswith("OK") else "FAIL" if line.startswith("FAIL") else "WARN"
            typer.echo(f"  [{status}] {line}")

    if result.feasibility_analysis:
        typer.echo(f"\nFeasibility Analysis:")
        fa = result.feasibility_analysis
        if len(result.evaluations) == 0:
            typer.echo(f"  ⚠️  No candidates were evaluated after runtime checks")
        elif fa.get("all_feasible"):
            typer.echo(f"  ✓ All candidates are feasible")
        else:
            typer.echo(f"  ✗ Some candidates violate constraints:")
            for vc in fa.get("violated_constraints", []):
                typer.echo(f"     - {vc.get('constraint')}: {vc.get('gap_pct', 0):.1f}% over limit")
            if fa.get("suggested_trades"):
                typer.echo(f"  💡 Suggested trades:")
                for trade in fa.get("suggested_trades", []):
                    typer.echo(f"     - {trade}")
        typer.echo(f"  Reasoning: {fa.get('reasoning', 'N/A')}")

    if result.winner:
        w = result.winner
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  🏆 Winner: {w.candidate.name}")
        typer.echo(f"{'='*60}")
        typer.echo(f"  Model: {w.candidate.model}")
        typer.echo(f"  Precision: {w.candidate.precision} | Quantization: {w.candidate.quantization}")
        typer.echo(f"  Quality: {w.metrics.quality_score:.4f}")
        typer.echo(f"  Latency (p95): {w.metrics.p95_latency_ms:.2f}ms")
        typer.echo(f"  Memory: {w.metrics.peak_memory_mb}MB")
        typer.echo(f"  Energy: {w.metrics.energy_mj:.2f}mJ")
        typer.echo(f"  Score: {w.score:.6f}")
        if result.winner_explanation:
            typer.echo(f"\n  Explanation:")
            typer.echo(f"  {result.winner_explanation}")
    else:
        typer.echo(f"\n⚠️  No candidate satisfies all constraints.")


@app.command()
def interactive(
    save: Optional[str] = typer.Option(None, "--save", help="Save config to JSON file"),
    build: bool = typer.Option(False, "--build", help="Build local artifacts"),
    json_output: bool = typer.Option(False, "--json", help="Output full JSON"),
) -> None:
    """Run agent in interactive mode (guided setup)."""

    config = interactive_setup()

    if save:
        config.save(save)
        typer.echo(f"✓ Config saved to {save}\n")

    # Run agent
    agent = EdgeOptimizationAgent(config)
    result = agent.run(build_local_artifacts=(True if build else None))

    if json_output:
        if isinstance(result, dict):
            output = {k: asdict(v) for k, v in result.items()}
        else:
            output = asdict(result)
        typer.echo(json.dumps(output, indent=2))
    else:
        _print_human(result)


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", help="Load config from JSON file"),
    device: str = typer.Option("auto", "--device", help="Device name or 'auto'"),
    build: bool = typer.Option(False, "--build", help="Build local artifacts"),
    json_output: bool = typer.Option(False, "--json", help="Output full JSON"),
) -> None:
    """Run with config file or launch interactive mode if no config provided."""

    if config:
        # Load from JSON
        cfg = load_config(config) if config.endswith(".json") else None
        if cfg is None:
            typer.echo(f"Error: Could not load config from {config}")
            raise typer.Exit(1)
        agent = EdgeOptimizationAgent(cfg)
    else:
        # Interactive mode
        cfg_interactive = interactive_setup()
        agent = EdgeOptimizationAgent(cfg_interactive)

    result = agent.run(device_name=device, build_local_artifacts=(True if build else None))

    if json_output:
        if isinstance(result, dict):
            output = {k: asdict(v) for k, v in result.items()}
        else:
            output = asdict(result)
        typer.echo(json.dumps(output, indent=2))
    else:
        _print_human(result)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
