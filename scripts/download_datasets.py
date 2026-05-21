#!/usr/bin/env python3
"""Download and normalize public benchmark datasets into local JSONL files."""

import argparse
import json
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/export public benchmark datasets for the harness."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(DOWNLOADERS),
        default=sorted(DOWNLOADERS),
        help="Benchmarks to download. Defaults to every supported downloader.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory where normalized JSONL files should be written.",
    )
    parser.add_argument(
        "--ruler-task",
        default="niah_single_1",
        help="RULER task config to export from the pre-generated HF dataset.",
    )
    parser.add_argument(
        "--ruler-length",
        default="128k",
        help="RULER context length to export from the pre-generated HF dataset.",
    )
    parser.add_argument(
        "--loogle-subset",
        choices=LOOGLE_SUBSETS,
        default="longdep_qa",
        help="LooGLE subset to export.",
    )
    args = parser.parse_args()

    for benchmark in args.benchmarks:
        if benchmark == "ruler":
            output = download_ruler(args.data_dir, args.ruler_task, args.ruler_length)
        elif benchmark == "loogle":
            output = download_loogle(args.data_dir, args.loogle_subset)
        else:
            output = DOWNLOADERS[benchmark](args.data_dir)
        print(f"{benchmark}: wrote {output}")

    print("\nUpdate llm_benchmarking/config.py with these data_path values:")
    for benchmark in args.benchmarks:
        path = DEFAULT_OUTPUTS[benchmark]
        print(f'  {benchmark}: "{path.as_posix()}"')


def download_longbench_v2(data_dir: Path) -> Path:
    dataset = _load_dataset("zai-org/LongBench-v2", split="train")
    output = data_dir / "longbench_v2" / "train.jsonl"
    rows = []
    for row in dataset:
        rows.append(
            {
                "id": row.get("_id"),
                "context": row.get("context", ""),
                "question": row.get("question", ""),
                "choice_A": row.get("choice_A", ""),
                "choice_B": row.get("choice_B", ""),
                "choice_C": row.get("choice_C", ""),
                "choice_D": row.get("choice_D", ""),
                "answer": str(row.get("answer", "")).strip().upper(),
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
            }
        )
    _write_jsonl(output, rows)
    return output


def download_loogle(data_dir: Path, subset: str = "longdep_qa") -> Path:
    dataset = _load_dataset("bigai-nlco/LooGLE", subset, split="test")
    output = data_dir / "loogle" / f"{subset}.jsonl"
    rows = []
    for index, row in enumerate(dataset):
        answer = row.get("answer", "")
        if isinstance(answer, list):
            normalized_answer: str | list[str] = [str(item) for item in answer]
        else:
            normalized_answer = str(answer)
        rows.append(
            {
                "id": row.get("id", f"{subset}_{index}"),
                "doc_id": row.get("doc_id"),
                "task": row.get("task", subset),
                "type": row.get("type"),
                "title": row.get("title", ""),
                "context": row.get("context", ""),
                "question": row.get("question", ""),
                "answer": normalized_answer,
                "evidence": list(row.get("evidence") or []),
                "metadata": row.get("metadata"),
            }
        )
    _write_jsonl(output, rows)
    return output


def download_babilong(data_dir: Path) -> Path:
    dataset = _load_dataset("RMT-team/babilong", "16k", split="qa1")
    output = data_dir / "babilong" / "qa1_16k.jsonl"
    rows = []
    for index, row in enumerate(dataset):
        rows.append(
            {
                "id": row.get("id", f"qa1_16k_{index}"),
                "context": row.get("input", ""),
                "question": row.get("question", ""),
                "answer": row.get("target", ""),
            }
        )
    _write_jsonl(output, rows)
    return output


def download_truthfulqa(data_dir: Path) -> Path:
    dataset = _load_dataset("truthful_qa", "multiple_choice", split="validation")
    output = data_dir / "truthfulqa" / "multiple_choice.jsonl"
    rows = []
    for index, row in enumerate(dataset):
        mc1_targets = row.get("mc1_targets") or {}
        choices = list(mc1_targets.get("choices") or [])
        labels = list(mc1_targets.get("labels") or [])
        rows.append(
            {
                "id": row.get("id", f"truthfulqa_{index}"),
                "question": row.get("question", ""),
                "choices": choices,
                "labels": labels,
                "category": row.get("category"),
            }
        )
    _write_jsonl(output, rows)
    return output


def download_humaneval(data_dir: Path) -> Path:
    output = data_dir / "humaneval" / "HumanEval.jsonl.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    urllib.request.urlretrieve(url, output)
    return output


def download_ruler(
    data_dir: Path,
    task: str = "niah_single_1",
    length: str = "128k",
) -> Path:
    dataset = _load_dataset(
        "self-long/RULER-llama3-1M",
        f"{task}_{length}",
        split="validation",
    )
    output = data_dir / "ruler" / f"{task}_{length}.jsonl"
    rows = []
    for index, row in enumerate(dataset):
        prompt = _first_present(row, "input", "prompt", "context")
        answers = (
            row.get("outputs")
            or row.get("answers")
            or row.get("answer")
            or row.get("target")
            or []
        )
        if not isinstance(answers, list):
            answers = [answers]
        rows.append(
            {
                "id": row.get("index", row.get("id", f"{task}_{length}_{index}")),
                "prompt": prompt,
                "answers": [str(answer) for answer in answers],
                "task": task,
                "length": length,
            }
        )
    _write_jsonl(output, rows)
    return output


def download_facts_grounding_public(data_dir: Path) -> Path:
    dataset = _load_dataset("google/FACTS-grounding-public", split="public")
    output = data_dir / "stanfordfacts" / "facts_grounding_public.jsonl"
    rows = []
    for index, row in enumerate(dataset):
        prompt = _first_present(
            row,
            "prompt",
            "user_prompt",
            "user_request",
            "request",
            "question",
            "input",
        )
        document = _first_present(
            row,
            "document",
            "context",
            "source",
            "reference",
            "passage",
        )
        full_prompt = (
            "Answer the request using only the provided document.\n\n"
            f"Document:\n{document}\n\n"
            f"Request:\n{prompt}"
        )
        rows.append(
            {
                "id": row.get("id", row.get("example_id", f"facts_grounding_{index}")),
                "prompt": full_prompt,
                "answer": row.get("answer", row.get("reference_answer", "")),
                "raw": row,
            }
        )
    _write_jsonl(output, rows)
    return output


def download_universalner(data_dir: Path) -> Path:
    output = data_dir / "universalner" / "test.jsonl"
    raw_dir = data_dir / "universalner" / "raw"
    rows = []
    for config, relative_url in UNIVERSALNER_TEST_FILES.items():
        source = raw_dir / Path(relative_url).name
        source.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UNIVERSALNER_RAW_PREFIX + relative_url, source)
        for index, row in enumerate(_read_universalner_iob2(source)):
            rows.append(
                {
                    "id": f"{config}_{row.get('idx', index)}",
                    "idx": row.get("idx"),
                    "dataset": config,
                    "split": "test",
                    "text": row.get("text", ""),
                    "tokens": list(row.get("tokens", [])),
                    "ner_tags": list(row.get("ner_tags", [])),
                }
            )
    _write_jsonl(output, rows)
    return output


def _read_universalner_iob2(path: Path) -> list[dict[str, Any]]:
    rows = []
    metadata: dict[str, str] = {}
    tokens = []
    tags = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                _append_universalner_row(rows, metadata, tokens, tags)
                metadata = {}
                tokens = []
                tags = []
                continue
            if stripped.startswith("#"):
                key, separator, value = stripped[1:].partition("=")
                if separator:
                    metadata[key.strip()] = value.strip()
                continue

            columns = stripped.split("\t")
            if len(columns) < 3:
                columns = stripped.split()
            if len(columns) < 3:
                continue
            tokens.append(columns[1])
            tags.append(_normalize_universalner_tag(columns[2]))

    _append_universalner_row(rows, metadata, tokens, tags)
    return rows


def _append_universalner_row(
    rows: list[dict[str, Any]],
    metadata: dict[str, str],
    tokens: list[str],
    tags: list[str],
) -> None:
    if not tokens:
        return
    rows.append(
        {
            "idx": metadata.get("sent_id", str(len(rows))),
            "text": metadata.get("text", " ".join(tokens)),
            "tokens": tokens,
            "ner_tags": tags,
        }
    )


def _normalize_universalner_tag(tag: str) -> str:
    if "OTH" in tag or tag == "B-O":
        return "O"
    return tag


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install dataset export dependencies first:\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt"
        ) from exc
    return load_dataset(*args, **kwargs)


def _first_present(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


DOWNLOADERS: dict[str, Callable[[Path], Path]] = {
    "babilong": download_babilong,
    "humaneval": download_humaneval,
    "loogle": download_loogle,
    "longbench_v2": download_longbench_v2,
    "ruler": download_ruler,
    "stanfordfacts": download_facts_grounding_public,
    "truthfulqa": download_truthfulqa,
    "universalner": download_universalner,
}

DEFAULT_OUTPUTS = {
    name: Path("data") / relative
    for name, relative in {
        "babilong": Path("babilong") / "qa1_16k.jsonl",
        "humaneval": Path("humaneval") / "HumanEval.jsonl.gz",
        "loogle": Path("loogle") / "longdep_qa.jsonl",
        "longbench_v2": Path("longbench_v2") / "train.jsonl",
        "ruler": Path("ruler") / "niah_single_1_128k.jsonl",
        "stanfordfacts": Path("stanfordfacts") / "facts_grounding_public.jsonl",
        "truthfulqa": Path("truthfulqa") / "multiple_choice.jsonl",
        "universalner": Path("universalner") / "test.jsonl",
    }.items()
}

LOOGLE_SUBSETS = ("longdep_qa", "shortdep_qa", "shortdep_cloze", "summarization")

UNIVERSALNER_RAW_PREFIX = "https://raw.githubusercontent.com/UniversalNER/"

UNIVERSALNER_TEST_FILES = {
    "ceb_gja": "UNER_Cebuano-GJA/master/ceb_gja-ud-test.iob2",
    "zh_gsd": "UNER_Chinese-GSD/master/zh_gsd-ud-test.iob2",
    "zh_gsdsimp": "UNER_Chinese-GSDSIMP/master/zh_gsdsimp-ud-test.iob2",
    "zh_pud": "UNER_Chinese-PUD/master/zh_pud-ud-test.iob2",
    "hr_set": "UNER_Croatian-SET/main/hr_set-ud-test.iob2",
    "da_ddt": "UNER_Danish-DDT/main/da_ddt-ud-test.iob2",
    "en_ewt": "UNER_English-EWT/master/en_ewt-ud-test.iob2",
    "en_pud": "UNER_English-PUD/master/en_pud-ud-test.iob2",
    "de_pud": "UNER_German-PUD/master/de_pud-ud-test.iob2",
    "pt_bosque": "UNER_Portuguese-Bosque/master/pt_bosque-ud-test.iob2",
    "pt_pud": "UNER_Portuguese-PUD/master/pt_pud-ud-test.iob2",
    "ru_pud": "UNER_Russian-PUD/master/ru_pud-ud-test.iob2",
    "sr_set": "UNER_Serbian-SET/main/sr_set-ud-test.iob2",
    "sk_snk": "UNER_Slovak-SNK/master/sk_snk-ud-test.iob2",
    "sv_pud": "UNER_Swedish-PUD/master/sv_pud-ud-test.iob2",
    "sv_talbanken": "UNER_Swedish-Talbanken/master/sv_talbanken-ud-test.iob2",
    "tl_trg": "UNER_Tagalog-TRG/master/tl_trg-ud-test.iob2",
    "tl_ugnayan": "UNER_Tagalog-Ugnayan/master/tl_ugnayan-ud-test.iob2",
}


if __name__ == "__main__":
    main()
