from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .artifact_builder import LocalArtifactBuilder
from .config import AgentConfig
from .device import detect_device
from .evaluator import CandidateEvaluator
from .models import DeviceProfile, EvaluationResult, InteractiveConfig, ModelConfig, RunResult
from .openrouter_client import OpenRouterClient
from .quantization import generate_candidates_from_config
from .reasoner import ConstraintReasoner


class EdgeOptimizationAgent:
    def __init__(self, config: AgentConfig | InteractiveConfig) -> None:
        self.config = config
        self.client = OpenRouterClient()
        self.evaluator = CandidateEvaluator(
            client=self.client,
            simulation_mode=getattr(config, "simulation_mode", True),
        )
        artifact_dir = getattr(config, "artifact_dir", "artifacts")
        calib_dir = getattr(config, "calibration_data_dir", None)
        self.artifact_builder = LocalArtifactBuilder(
            artifact_dir=artifact_dir,
            calibration_data_dir=calib_dir,
        )
        self.reasoner = ConstraintReasoner(self.client)

    def run(
        self,
        device_name: str = "auto",
        task_prompt: Optional[str] = None,
        build_local_artifacts: Optional[bool] = None,
    ) -> Dict[str, RunResult] | RunResult:
        if isinstance(self.config, InteractiveConfig) and device_name == "auto":
            device = self.config.device
        else:
            device = detect_device(device_name)
        prompt = task_prompt or "Summarize edge model optimization in one sentence."

        # Check if this is multi-model config (InteractiveConfig) or single-model (AgentConfig)
        if isinstance(self.config, InteractiveConfig):
            return self._run_multi_model(device, prompt, build_local_artifacts)
        else:
            return self._run_single_model(device, prompt, build_local_artifacts)

    def _run_single_model(
        self,
        device: DeviceProfile,
        prompt: str,
        build_local_artifacts: Optional[bool],
    ) -> RunResult:
        candidates = generate_candidates_from_config(
            self.config.candidates,
            device,
            artifact_dir=self.config.artifact_dir,
        )
        candidates, runtime_skip_logs = self._filter_runtime_candidates(candidates)
        should_build = self.config.build_local_artifacts if build_local_artifacts is None else build_local_artifacts
        artifact_logs = self.artifact_builder.build(candidates) if should_build else []
        artifact_logs.extend(runtime_skip_logs)

        evaluations = [
            self.evaluator.evaluate(
                candidate=c,
                device=device,
                constraints=self.config.constraints,
                weights=self.config.weights,
                task_prompt=prompt,
            )
            for c in candidates
        ]

        # Analyze feasibility with LLM
        feasibility = self._feasibility_or_empty(
            evaluations,
            self.config.constraints,
            "default_model",
        )

        winner = self._pick_winner(evaluations)

        # Get explanation
        explanation = ""
        if winner:
            explanation = self.reasoner.explain_decision(
                winner,
                self.config.constraints,
                "default_model",
            )

        return RunResult(
            device=device,
            constraints=self.config.constraints,
            weights=self.config.weights,
            evaluations=evaluations,
            winner=winner,
            artifact_logs=artifact_logs,
            feasibility_analysis=feasibility,
            winner_explanation=explanation,
        )

    def _run_multi_model(
        self,
        device: DeviceProfile,
        prompt: str,
        build_local_artifacts: Optional[bool],
    ) -> Dict[str, RunResult]:
        config: InteractiveConfig = self.config
        should_build = config.build_local_artifacts if build_local_artifacts is None else build_local_artifacts

        results: Dict[str, RunResult] = {}

        for model_cfg in config.models:
            source_path = Path(model_cfg.source_path)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Source model not found for '{model_cfg.name}': {model_cfg.source_path}"
                )

            # Generate candidates for this model
            candidates = self._generate_multi_model_candidates(model_cfg, device, config.artifact_dir)
            candidates, runtime_skip_logs = self._filter_runtime_candidates(candidates)

            # Build artifacts if needed
            artifact_logs: List[str] = []
            if should_build:
                artifact_logs = self.artifact_builder.build(candidates)
            artifact_logs.extend(runtime_skip_logs)

            # Evaluate
            evaluations = [
                self.evaluator.evaluate(
                    candidate=c,
                    device=device,
                    constraints=config.constraints,
                    weights=config.weights,
                    task_prompt=prompt,
                )
                for c in candidates
            ]

            # Analyze feasibility with LLM
            feasibility = self._feasibility_or_empty(
                evaluations,
                config.constraints,
                model_cfg.name,
            )

            # Pick winner
            winner = self._pick_winner(evaluations)

            # Get explanation
            explanation = ""
            if winner:
                explanation = self.reasoner.explain_decision(
                    winner,
                    config.constraints,
                    model_cfg.name,
                )

            results[model_cfg.name] = RunResult(
                device=device,
                constraints=config.constraints,
                weights=config.weights,
                evaluations=evaluations,
                winner=winner,
                artifact_logs=artifact_logs,
                feasibility_analysis=feasibility,
                winner_explanation=explanation,
            )

        return results

    def _generate_multi_model_candidates(
        self,
        model_cfg: ModelConfig,
        device: DeviceProfile,
        artifact_dir: str,
    ) -> list:
        """Generate quantization candidates for a specific model."""
        from .models import Candidate

        candidates = []

        # Base precision / quantization strategies
        strategies = [
            ("fp32", "none"),
            ("fp16", "float16") if device.supports_fp16 else None,
            ("int8", "dynamic") if device.supports_int8 else None,
            ("int8", "static") if device.supports_int8 else None,
        ]

        for strategy in strategies:
            if strategy is None:
                continue

            precision, quantization = strategy
            name = f"{model_cfg.name}-{precision}-{quantization}"

            candidate = Candidate(
                name=name,
                provider="local",
                model=model_cfg.name,
                quantization=quantization,
                precision=precision,
                runtime="onnxruntime-local" if model_cfg.source_format == "onnx" else "tflite-local",
                source_model_path=model_cfg.source_path,
                artifact_path=f"{artifact_dir}/{name}.{model_cfg.source_format}",
                config={},
            )
            candidates.append(candidate)

        return candidates

    def _pick_winner(self, evaluations: List[EvaluationResult]) -> Optional[EvaluationResult]:
        passing = [result for result in evaluations if result.passed and result.score is not None]
        if not passing:
            return None

        return sorted(passing, key=lambda result: result.score, reverse=True)[0]

    def _filter_runtime_candidates(self, candidates: list) -> Tuple[list, List[str]]:
        """Filter candidates unsupported by the current runtime prerequisites."""
        if self.evaluator.simulation_mode or self.client.is_configured():
            return candidates, []

        filtered = []
        logs: List[str] = []
        for candidate in candidates:
            if candidate.provider == "openrouter":
                logs.append(
                    f"WARN {candidate.name}: skipped (OPENROUTER_API_KEY missing; live OpenRouter disabled)"
                )
                continue
            filtered.append(candidate)

        return filtered, logs

    def _feasibility_or_empty(
        self,
        evaluations: List[EvaluationResult],
        constraints,
        model_name: str,
    ) -> dict:
        if evaluations:
            return self.reasoner.is_feasible(evaluations, constraints, model_name)

        return {
            "feasible_candidates": [],
            "all_feasible": False,
            "best_candidate": None,
            "violated_constraints": [],
            "suggested_trades": [],
            "reasoning": "No candidates were evaluated after runtime prerequisite checks.",
        }
