from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    dataset_path: Path = Path("data/eval/qa_dataset.json")
    chunks_path: Path = Path("data/processed/chunks.json")
    results_dir: Path = Path("data/eval/results")

    # Retrieval
    retrieval_k: int = 20
    final_k: int = 5

    evaluation_ks: tuple = (1, 3, 5, 10)

    # Generation
    context_k: int = 3

    # Dataset generation
    samples_per_topic: int = 20
    context_window_size: int = 3

    # LLM judge
    enable_llm_judge: bool = True

    # RAGAS
    enable_ragas: bool = False


CONFIG = EvaluationConfig()