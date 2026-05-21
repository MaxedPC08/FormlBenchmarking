from llm_benchmarking.benchmarks.generic_qa import GenericQABenchmark
from llm_benchmarking.cases import BenchmarkCase


class StanfordFACTSBenchmark(GenericQABenchmark):
    name = "stanfordfacts"
    scoring = "contains"
    sample_cases = [
        BenchmarkCase(
            id="stanfordfacts_sample",
            prompt=(
                "Answer the factual question concisely.\n\n"
                "Question: What gas do plants primarily absorb from the atmosphere "
                "during photosynthesis?"
            ),
            answers=["carbon dioxide", "CO2"],
        )
    ]
