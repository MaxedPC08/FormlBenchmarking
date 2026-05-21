from llm_benchmarking.benchmarks.babilong import BABILongBenchmark
from llm_benchmarking.benchmarks.humaneval import HumanEvalBenchmark
from llm_benchmarking.benchmarks.loogle import LooGLEBenchmark
from llm_benchmarking.benchmarks.longbench_v2 import LongBenchV2Benchmark
from llm_benchmarking.benchmarks.ruler import RulerBenchmark
from llm_benchmarking.benchmarks.stanfordfacts import StanfordFACTSBenchmark
from llm_benchmarking.benchmarks.truthfulqa import TruthfulQABenchmark
from llm_benchmarking.benchmarks.universalner import UniversalNERBenchmark


BENCHMARK_REGISTRY = {
    "longbench_v2": LongBenchV2Benchmark,
    "loogle": LooGLEBenchmark,
    "ruler": RulerBenchmark,
    "babilong": BABILongBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "stanfordfacts": StanfordFACTSBenchmark,
    "humaneval": HumanEvalBenchmark,
    "universalner": UniversalNERBenchmark,
}
