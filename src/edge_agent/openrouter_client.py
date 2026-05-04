from __future__ import annotations

import os
import time
from typing import Dict, Optional

import requests


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_name: Optional[str] = None,
        site_url: Optional[str] = None,
        timeout_sec: int = 60,
        model: str = "google/gemma-4-26b-a4b-it:free",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME", "edge-openrouter-agent")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.timeout_sec = timeout_sec
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat_once(self, model: Optional[str] = None, prompt: str = "", max_tokens: int = 64) -> Dict[str, float]:
        model = model or self.model
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }

        start = time.perf_counter()
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        completion_tokens = float(usage.get("completion_tokens", 0))
        prompt_tokens = float(usage.get("prompt_tokens", 0))

        return {
            "latency_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def chat_completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 512,
    ) -> str:
        """Complete a chat conversation and return the response text."""
        model = model or self.model
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return ""
