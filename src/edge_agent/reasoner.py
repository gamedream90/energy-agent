from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .models import Candidate, CandidateMetrics, Constraints, EvaluationResult
from .openrouter_client import OpenRouterClient


class ConstraintReasoner:
    """Uses LLM to reason about constraint feasibility and explain optimization decisions."""

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def is_feasible(
        self,
        evaluations: List[EvaluationResult],
        constraints: Constraints,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Ask LLM: Are these candidates feasible? Which meets constraints best?
        In simulation mode, returns mock reasoning.

        Returns:
        {
            "feasible_candidates": ["name1", "name2"],
            "all_feasible": bool,
            "best_candidate": str or None,
            "violated_constraints": [{"constraint": "latency", "gap_pct": 15.2}],
            "suggested_trades": ["relax_latency_10pct", "accept_2pct_accuracy_loss"],
            "reasoning": "detailed explanation"
        }
        """
        if not self.client.is_configured():
            return self._simulate_feasibility(evaluations, constraints, model_name)

        candidates_summary = self._format_candidates(evaluations)

        prompt = f"""You are an edge ML optimization expert. Analyze these quantization candidates 
for model '{model_name}' against device constraints.

Candidates:
{candidates_summary}

Constraints (hard limits):
- p95 latency: <= {constraints.max_p95_latency_ms}ms
- peak memory: <= {constraints.max_peak_memory_mb}MB
- min quality: >= {constraints.min_quality_score:.3f}
- max energy: <= {constraints.max_energy_mj:.1f}mJ

Instructions:
1. Identify which candidates are FEASIBLE (meet ALL hard constraints)
2. If NONE are feasible, identify which constraint(s) are violated most
3. Suggest trade-offs that could make optimization feasible
4. Name the BEST candidate overall

Respond ONLY in valid JSON (no markdown, no extra text):
{{
    "feasible_candidates": ["list", "of", "names"],
    "all_feasible": true/false,
    "best_candidate": "name or null",
    "violated_constraints": [{{"constraint": "name", "gap_pct": 15.2}} ...],
    "suggested_trades": ["trade1", "trade2"],
    "reasoning": "1-2 sentence explanation"
}}"""

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )

            result = json.loads(response)
            return result
        except Exception as exc:
            # Fallback to deterministic analysis on error
            return self._simulate_feasibility(evaluations, constraints, model_name)

    def explain_decision(
        self,
        winner: EvaluationResult,
        constraints: Constraints,
        model_name: str,
    ) -> str:
        """
        Generate human-readable explanation of why this candidate was chosen.
        """
        if not self.client.is_configured():
            return self._simulate_explanation(winner, constraints, model_name)

        prompt = f"""You are explaining an ML optimization decision.

Selected Config: {winner.candidate.name}
- Quality: {winner.metrics.quality_score:.3f}
- p95 Latency: {winner.metrics.p95_latency_ms:.1f}ms
- Peak Memory: {winner.metrics.peak_memory_mb}MB
- Energy/inf: {winner.metrics.energy_mj:.2f}mJ

Device Constraints:
- Max latency: {constraints.max_p95_latency_ms}ms
- Max memory: {constraints.max_peak_memory_mb}MB
- Min quality: {constraints.min_quality_score:.3f}
- Max energy: {constraints.max_energy_mj:.1f}mJ

Model: {model_name}

Generate a 1-2 sentence explanation of why this is the best choice for this device.
Keep it concise and technical."""

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=128,
            )
            return response.strip()
        except Exception:
            return self._simulate_explanation(winner, constraints, model_name)

    def _format_candidates(self, evaluations: List[EvaluationResult]) -> str:
        """Format candidates for LLM prompt."""
        lines = []
        for r in evaluations:
            m = r.metrics
            status = "✓" if r.passed else "✗"
            lines.append(
                f"{status} {r.candidate.name}: quality={m.quality_score:.3f}, "
                f"p95={m.p95_latency_ms:.1f}ms, mem={m.peak_memory_mb}MB, "
                f"energy={m.energy_mj:.2f}mJ"
            )
        return "\n".join(lines)

    def _simulate_feasibility(
        self,
        evaluations: List[EvaluationResult],
        constraints: Constraints,
        model_name: str,
    ) -> Dict[str, Any]:
        """Deterministic feasibility analysis (no LLM)."""
        feasible = [r for r in evaluations if r.passed]
        violated: List[Dict[str, float]] = []

        if not feasible:
            max_latency_gap = 0.0
            max_memory_gap = 0.0
            max_quality_gap = 0.0
            max_energy_gap = 0.0

            for r in evaluations:
                m = r.metrics

                if constraints.max_p95_latency_ms > 0 and m.p95_latency_ms > constraints.max_p95_latency_ms:
                    gap = ((m.p95_latency_ms - constraints.max_p95_latency_ms) / constraints.max_p95_latency_ms) * 100
                    max_latency_gap = max(max_latency_gap, gap)

                if constraints.max_peak_memory_mb > 0 and m.peak_memory_mb > constraints.max_peak_memory_mb:
                    gap = ((m.peak_memory_mb - constraints.max_peak_memory_mb) / constraints.max_peak_memory_mb) * 100
                    max_memory_gap = max(max_memory_gap, gap)

                if constraints.min_quality_score > 0 and m.quality_score < constraints.min_quality_score:
                    gap = ((constraints.min_quality_score - m.quality_score) / constraints.min_quality_score) * 100
                    max_quality_gap = max(max_quality_gap, gap)

                if constraints.max_energy_mj is not None and constraints.max_energy_mj > 0 and m.energy_mj > constraints.max_energy_mj:
                    gap = ((m.energy_mj - constraints.max_energy_mj) / constraints.max_energy_mj) * 100
                    max_energy_gap = max(max_energy_gap, gap)

            if max_latency_gap > 0:
                violated.append({"constraint": "latency", "gap_pct": round(max_latency_gap, 1)})
            if max_memory_gap > 0:
                violated.append({"constraint": "memory", "gap_pct": round(max_memory_gap, 1)})
            if max_quality_gap > 0:
                violated.append({"constraint": "quality", "gap_pct": round(max_quality_gap, 1)})
            if max_energy_gap > 0:
                violated.append({"constraint": "energy", "gap_pct": round(max_energy_gap, 1)})

        best = sorted(feasible, key=lambda r: r.score or 0, reverse=True)[0] if feasible else None

        return {
            "feasible_candidates": [r.candidate.name for r in feasible],
            "all_feasible": len(feasible) == len(evaluations),
            "best_candidate": best.candidate.name if best else None,
            "violated_constraints": violated,
            "suggested_trades": self._suggest_trades(evaluations, constraints),
            "reasoning": "Analyzed all candidates against device constraints.",
        }

    def _suggest_trades(
        self,
        evaluations: List[EvaluationResult],
        constraints: Constraints,
    ) -> List[str]:
        """Suggest possible trade-offs."""
        trades = []

        # Check if relaxing latency slightly would help
        best_by_quality = sorted(evaluations, key=lambda r: r.metrics.quality_score, reverse=True)[0]
        if best_by_quality.metrics.p95_latency_ms <= constraints.max_p95_latency_ms * 1.15:
            trades.append("Relax latency by 10% to gain better quality")

        # Check if accepting lower quality would help
        best_by_latency = sorted(evaluations, key=lambda r: r.metrics.p95_latency_ms)[0]
        if best_by_latency.metrics.quality_score >= constraints.min_quality_score * 0.98:
            trades.append("Accept 1-2% accuracy loss for faster inference")

        return trades

    def _simulate_explanation(
        self,
        winner: EvaluationResult,
        constraints: Constraints,
        model_name: str,
    ) -> str:
        """Generate deterministic explanation."""
        return (
            f"{winner.candidate.name} is optimal: meets all SLAs "
            f"(latency {winner.metrics.p95_latency_ms:.0f}ms, memory {winner.metrics.peak_memory_mb}MB) "
            f"while maintaining {winner.metrics.quality_score:.1%} quality."
        )
