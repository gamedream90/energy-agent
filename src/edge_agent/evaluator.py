from __future__ import annotations

import random
import statistics
from typing import List

from .models import Candidate, CandidateMetrics, Constraints, DeviceProfile, EvaluationResult, Weights
from .openrouter_client import OpenRouterClient


class CandidateEvaluator:
    def __init__(self, client: OpenRouterClient, simulation_mode: bool = True) -> None:
        self.client = client
        self.simulation_mode = simulation_mode

    def evaluate(
        self,
        candidate: Candidate,
        device: DeviceProfile,
        constraints: Constraints,
        weights: Weights,
        task_prompt: str,
        rounds: int = 3,
    ) -> EvaluationResult:
        metrics = self._metrics_for_candidate(candidate, device, task_prompt, rounds)
        reasons: List[str] = []

        if metrics.p95_latency_ms > constraints.max_p95_latency_ms:
            reasons.append(
                f"latency {metrics.p95_latency_ms:.2f}ms > {constraints.max_p95_latency_ms:.2f}ms"
            )
        if metrics.peak_memory_mb > constraints.max_peak_memory_mb:
            reasons.append(
                f"memory {metrics.peak_memory_mb}MB > {constraints.max_peak_memory_mb}MB"
            )
        if metrics.quality_score < constraints.min_quality_score:
            reasons.append(
                f"quality {metrics.quality_score:.3f} < {constraints.min_quality_score:.3f}"
            )
        if constraints.max_energy_mj is not None and metrics.energy_mj > constraints.max_energy_mj:
            reasons.append(
                f"energy {metrics.energy_mj:.2f}mJ > {constraints.max_energy_mj:.2f}mJ"
            )

        passed = len(reasons) == 0
        score = None
        if passed:
            score = (
                weights.quality * metrics.quality_score
                - weights.latency * metrics.p95_latency_ms
                - weights.memory * metrics.peak_memory_mb
                - weights.energy * metrics.energy_mj
            )

        return EvaluationResult(
            candidate=candidate,
            metrics=metrics,
            passed=passed,
            reasons=reasons,
            score=score,
        )

    def _metrics_for_candidate(
        self,
        candidate: Candidate,
        device: DeviceProfile,
        task_prompt: str,
        rounds: int,
    ) -> CandidateMetrics:
        if self.simulation_mode:
            return self._simulate_metrics(candidate, device)

        if candidate.provider == "openrouter" and not self.client.is_configured():
            raise RuntimeError(
                "Live mode is enabled but OPENROUTER_API_KEY is missing. "
                "Set OPENROUTER_API_KEY or enable simulation_mode."
            )

        if candidate.provider != "openrouter":
            return self._simulate_metrics(candidate, device)

        latencies: List[float] = []
        token_totals: List[float] = []
        for _ in range(max(1, rounds)):
            m = self.client.chat_once(model=candidate.model, prompt=task_prompt)
            latencies.append(m["latency_ms"])
            token_totals.append(m["total_tokens"])

        p95 = max(latencies) if len(latencies) < 20 else statistics.quantiles(latencies, n=20)[-1]
        avg_tokens = sum(token_totals) / len(token_totals)

        quality = self._estimate_quality(candidate)
        memory = self._estimate_memory(candidate, device)
        energy = self._estimate_energy(candidate, p95)
        cost = self._estimate_cost(candidate, avg_tokens)

        return CandidateMetrics(
            quality_score=quality,
            p95_latency_ms=p95,
            peak_memory_mb=memory,
            energy_mj=energy,
            est_cost_per_1k=cost,
        )

    def _simulate_metrics(self, candidate: Candidate, device: DeviceProfile) -> CandidateMetrics:
        base_latency = 220.0
        base_memory = min(max(256, int(device.ram_mb * 0.35)), 4096)

        precision_factor = {
            "fp32": 1.0,
            "fp16": 0.72,
            "int8": 0.58,
        }.get(candidate.precision, 1.0)

        quantization_quality_penalty = {
            "none": 0.0,
            "float16": 0.01,
            "dynamic": 0.03,
            "static": 0.02,
        }.get(candidate.quantization, 0.0)

        jitter = random.uniform(0.92, 1.08)
        p95_latency = base_latency * precision_factor * jitter
        memory = int(base_memory * (0.8 if candidate.precision == "fp16" else 0.62 if candidate.precision == "int8" else 1.0))
        quality = max(0.0, 0.90 - quantization_quality_penalty + random.uniform(-0.01, 0.01))
        energy = p95_latency * 0.45
        cost = max(0.03, p95_latency / 10000)

        return CandidateMetrics(
            quality_score=quality,
            p95_latency_ms=p95_latency,
            peak_memory_mb=memory,
            energy_mj=energy,
            est_cost_per_1k=cost,
        )

    def _estimate_quality(self, candidate: Candidate) -> float:
        baseline = 0.90
        penalty = {
            "none": 0.0,
            "float16": 0.01,
            "dynamic": 0.03,
            "static": 0.02,
        }.get(candidate.quantization, 0.0)
        return max(0.0, baseline - penalty)

    def _estimate_memory(self, candidate: Candidate, device: DeviceProfile) -> int:
        base = min(max(256, int(device.ram_mb * 0.35)), 4096)
        if candidate.precision == "fp16":
            return int(base * 0.8)
        if candidate.precision == "int8":
            return int(base * 0.62)
        return base

    def _estimate_energy(self, candidate: Candidate, p95_latency_ms: float) -> float:
        factor = 1.0 if candidate.precision == "fp32" else 0.85 if candidate.precision == "fp16" else 0.72
        return p95_latency_ms * 0.45 * factor

    def _estimate_cost(self, candidate: Candidate, avg_tokens: float) -> float:
        multiplier = 1.0 if candidate.precision == "fp32" else 0.9 if candidate.precision == "fp16" else 0.8
        return (avg_tokens / 1000.0) * multiplier
