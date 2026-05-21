# LLM Benchmarking

Benchmark models served by Ollama or OpenRouter across LongBench V2, LooGLE,
RULER, BABILong, TruthfulQA, StanfordFACTS, HumanEval, and UniversalNER.

## Setup

```bash
source .venv/bin/activate
python main.py --list
```

The default dummy model is `qwen3:4b` on Ollama. Pull it before real runs:

```bash
ollama pull qwen3:4b
ollama serve
```

## Configure

Edit `llm_benchmarking/config.py`. The `BENCHMARKS` dictionary at the top is
the switchboard for turning benchmark adapters on and off. Set `enabled` to
`False` to skip a benchmark, and set `data_path` to a JSON or JSONL file for
real benchmark data.

If `data_path` is `None`, each adapter runs a tiny built-in smoke sample so the
harness can be checked without downloading datasets.

## Download Public Datasets

Install export dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Download and normalize the public datasets that can be fetched directly:

```bash
python scripts/download_datasets.py
```

Or download a subset:

```bash
python scripts/download_datasets.py --benchmarks humaneval truthfulqa babilong
```

The script writes:

- `data/longbench_v2/train.jsonl`
- `data/loogle/longdep_qa.jsonl`
- `data/ruler/niah_single_1_128k.jsonl`
- `data/babilong/qa1_16k.jsonl`
- `data/truthfulqa/multiple_choice.jsonl`
- `data/humaneval/HumanEval.jsonl.gz`
- `data/stanfordfacts/facts_grounding_public.jsonl`
- `data/universalner/test.jsonl`

Then copy the printed paths into the `data_path` values in
`llm_benchmarking/config.py`.

RULER can be exported from the pre-generated
`self-long/RULER-llama3-1M` Hugging Face dataset. By default, the downloader
writes `niah_single_1` at `128k` context. To export a different slice:

```bash
python scripts/download_datasets.py --benchmarks ruler --ruler-task qa_1 --ruler-length 8k
```

The official NVIDIA generator is still available at
<https://github.com/NVIDIA/RULER> if you want to create custom synthetic tasks.

LooGLE defaults to the `longdep_qa` subset. To export another subset:

```bash
python scripts/download_datasets.py --benchmarks loogle --loogle-subset summarization
```

Then update `llm_benchmarking/config.py` if you want to point `loogle` at the
new file, for example `data/loogle/summarization.jsonl`.

The `stanfordfacts` adapter currently exports Google FACTS Grounding public
examples. Those examples are best scored with a judge model; the local harness
can generate responses from them, but exact automatic scoring depends on
whether a reference answer field is present.

The `universalner` adapter exports the Universal NER test splits from Hugging
Face and scores exact entity spans with entity-level F1 over the PER, ORG, and
LOC labels.

Models are configured in the same file. Each model must use provider
`"ollama"` or `"openrouter"`. OpenRouter runs require:

```bash
export OPENROUTER_API_KEY=...
```

You can also create a local `.env` file:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Do not put the API key in `llm_benchmarking/config.py`; that file only names
the environment variable to read.

## Context Limits

The runner estimates prompt tokens before each model call. If a prompt plus
the configured output budget will not fit in the model context window, the case
is skipped and recorded in the JSONL/report as `skip`.

Set a default window in `llm_benchmarking/config.py`:

```python
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
```

Or override per model:

```python
{
    "name": "example/model",
    "provider": "openrouter",
    "max_tokens": 512,
    "context_window": 32_768,
}
```

You can also override for a single run:

```bash
python main.py --context-window 32768
```

## Run

Dry-run dataset loading:

```bash
python main.py --dry-run
```

Run all enabled benchmarks:

```bash
python main.py
```

Run a subset:

```bash
python main.py --only truthfulqa humaneval --limit 5
```

Results are written to `results/run_<timestamp>.jsonl`, summaries to
`results/summary_<timestamp>.json`, an HTML dashboard with SVG graphs to
`results/report_<timestamp>.html`, and a standalone grouped bar chart to
`results/chart_<timestamp>.svg`.

Generate a standalone grouped bar chart from any run JSONL:

```bash
python scripts/chart_run.py results/run_20260508T183302Z_with_184701Z_babilong.jsonl \
  --output charts/combined_mean_scores.svg
```

Generate one chart from multiple raw result JSONL files:

```bash
python scripts/chart_runs.py results/run_a.jsonl results/run_b.jsonl \
  --output charts/multi_run_mean_scores.svg
```

You can also pass `summary_*.json` files. Do not pass `report_*.html` or
`chart_*.svg` files; those are outputs, not data sources.

By default, matching model names are merged across files. Add `--split-by-run`
to show the same model from different runs as separate groups.

## Dataset Format

Most QA-style adapters accept JSON/JSONL rows with common fields:

- `prompt`, or `context` plus `question`
- `answer`, `answers`, `target`, `targets`, `output`, or `outputs`
- optional `id`

TruthfulQA supports rows with `question` and `choices`/`labels`, including the
Hugging Face-style `mc1_targets` shape. HumanEval supports OpenAI-style rows
with `task_id`, `prompt`, `test`, and `entry_point`.

UniversalNER supports rows with `text`, `tokens`, and BIO-style `ner_tags`.
LooGLE supports rows with `context`, `question`, `answer`, plus optional
`title`, `evidence`, `task`, `type`, and `doc_id`.
# FormlBenchmarking
