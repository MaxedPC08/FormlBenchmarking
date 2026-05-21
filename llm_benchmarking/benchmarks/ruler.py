from llm_benchmarking.benchmarks.generic_qa import GenericQABenchmark
from llm_benchmarking.cases import BenchmarkCase


class RulerBenchmark(GenericQABenchmark):
    name = "ruler"
    scoring = "contains"
    sample_cases = [
        BenchmarkCase(
            id="ruler_sample",
            prompt=(
                "A long-context retrieval marker appears below.\n\n"
                "Noise: alpha beta gamma delta epsilon.\n"
                "Needle: The pass key is LIME-742.\n"
                "Noise: theta iota kappa lambda.\n\n"
                "Question: What is the pass key?"
            ),
            answers=["LIME-742"],
        )
    ]
