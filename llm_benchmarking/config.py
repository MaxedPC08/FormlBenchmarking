"""User-editable benchmark configuration.

Flip benchmark switches here. The runner only executes entries where
``enabled`` is true.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# BENCHMARK SWITCHBOARD
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "longbench_v2": {
        "enabled": False,
        "data_path": "data/longbench_v2/train.jsonl",
        "limit": 40,
    },
    "loogle": {
        "enabled": True,
        "data_path": "data/loogle/longdep_qa.jsonl",
        "limit": 40,
    },
    "ruler": {
        "enabled": False,
        "data_path": "data/ruler/niah_single_1_128k.jsonl",
        "limit": 40,
    },
    "babilong": {
        "enabled": False,
        "data_path": "data/babilong/qa1_16k.jsonl",
        "limit": 40,
    },
    "truthfulqa": {
        "enabled": False,
        "data_path": "data/truthfulqa/multiple_choice.jsonl",
        "limit": 40,
    },
    "stanfordfacts": {
        "enabled": False,
        "data_path": "data/stanfordfacts/facts_grounding_public.jsonl",
        "limit": 40,
    },
    "humaneval": {
        "enabled": False,
        "data_path": "data/humaneval/HumanEval.jsonl.gz",
        "limit": 20,
    },
    "universalner": {
        "enabled": False,
        "data_path": "data/universalner/test.jsonl",
        "limit": 40,
    },
}


# Provider must be either "ollama" or "openrouter".
MODELS = [
    {
        "name": "mistralai/mistral-medium-3-5",
        "provider": "openrouter",
        "temperature": 0.0,
        "max_tokens": 512,
        # Optional. If omitted, DEFAULT_CONTEXT_WINDOW_TOKENS is used.
        # Total prompt tokens plus max_tokens must fit inside this window.
        "context_window": None,
    },

]


OLLAMA_BASE_URL = "http://localhost:11434"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_SITE_URL = "http://localhost"
APP_NAME = "LLM Benchmarking"

RESULTS_DIR = Path("results")
REQUEST_TIMEOUT_SECONDS = 120

# Used to skip prompts that are too large for a model's context window.
# Set this conservatively if you are unsure. Per-model "context_window" overrides it.
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000

# Rough tokenizer-free estimate used before calling a model.
# English benchmark text is usually around 3.5-4 chars/token.
CHARS_PER_TOKEN_ESTIMATE = 4.0

# Extra safety margin so provider chat formatting does not push a request over the limit.
CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS = 256
