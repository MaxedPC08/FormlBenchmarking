import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm_benchmarking import config


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    temperature: float = 0.0
    max_tokens: int = 512
    context_window: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class LLMProvider(ABC):
    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        raise NotImplementedError

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = f"HTTP {exc.code} {exc.reason}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(
                f"Provider request failed for {self.model.name}: {message}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Provider request failed for {self.model.name}: {exc}") from exc


class OllamaProvider(LLMProvider):
    def generate(self, prompt: str) -> GenerationResult:
        payload = {
            "model": self.model.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.model.temperature,
                "num_predict": self.model.max_tokens,
            },
        }
        data = self._post_json(f"{config.OLLAMA_BASE_URL}/api/generate", payload)
        input_tokens = _optional_int(data.get("prompt_eval_count"))
        output_tokens = _optional_int(data.get("eval_count"))
        return GenerationResult(
            text=str(data.get("response", "")).strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=_sum_optional(input_tokens, output_tokens),
        )


class OpenRouterProvider(LLMProvider):
    def generate(self, prompt: str) -> GenerationResult:
        api_key = _get_env(config.OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.OPENROUTER_API_KEY_ENV} in your shell or .env file "
                "to use OpenRouter models."
            )
        payload = {
            "model": self.model.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
        }
        data = self._post_json(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": getattr(config, "OPENROUTER_SITE_URL", ""),
                "X-Title": getattr(config, "APP_NAME", "LLM Benchmarking"),
            },
        )
        usage = data.get("usage") or {}
        input_tokens = _optional_int(usage.get("prompt_tokens"))
        output_tokens = _optional_int(usage.get("completion_tokens"))
        total_tokens = _optional_int(usage.get("total_tokens"))
        return GenerationResult(
            text=str(data["choices"][0]["message"]["content"]).strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens or _sum_optional(input_tokens, output_tokens),
        )


def build_provider(model_config: dict[str, Any]) -> LLMProvider:
    spec = ModelSpec(**model_config)
    if spec.provider == "ollama":
        return OllamaProvider(spec)
    if spec.provider == "openrouter":
        return OpenRouterProvider(spec)
    raise ValueError(f"Unsupported provider {spec.provider!r}; use 'ollama' or 'openrouter'.")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _get_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    env_path = getattr(config, "ENV_FILE", ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, raw_value = stripped.split("=", 1)
                if key.strip() == name:
                    return raw_value.strip().strip('"').strip("'")
    except FileNotFoundError:
        return None

    return None
