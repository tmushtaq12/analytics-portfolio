import json
import re
import string
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "squad" / "dev-v2.0.json"
OUTPUT_DIR = BASE_DIR / "projects" / "genai-evaluation-lab"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-cased-distilled-squad"
EVALUATION_LIMIT = 100


def load_answerable_examples(limit: int) -> List[dict]:
    """Load a deterministic sample of answerable SQuAD questions."""
    with DATA_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    examples = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                if question["is_impossible"] or not question["answers"]:
                    continue
                examples.append(
                    {
                        "question": question["question"],
                        "context": paragraph["context"],
                        "answers": [answer["text"] for answer in question["answers"]],
                    }
                )
                if len(examples) == limit:
                    return examples
    return examples


def normalize_text(text: str) -> str:
    """Normalize text for fair exact-match comparison."""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    """Calculate token-level F1 for one prediction/reference pair."""
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    common_tokens = set(prediction_tokens) & set(reference_tokens)
    overlap = sum(min(prediction_tokens.count(token), reference_tokens.count(token)) for token in common_tokens)

    if not prediction_tokens or not reference_tokens or overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, references: List[str]) -> Tuple[int, float]:
    """Score against the best of the accepted reference answers."""
    exact_match = max(normalize_text(prediction) == normalize_text(reference) for reference in references)
    f1 = max(token_f1(prediction, reference) for reference in references)
    return int(exact_match), f1


def save_metric_chart(metrics: Dict[str, object]) -> None:
    labels = ["Exact match", "Token F1"]
    values = [metrics["exact_match"], metrics["token_f1"]]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#e76f51", "#2a9d8f"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Extractive QA evaluation")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "genai_qa_evaluation.png", dpi=180)
    plt.close()


def main():
    examples = load_answerable_examples(EVALUATION_LIMIT)
    qa_model = pipeline("question-answering", model=MODEL_NAME, framework="pt")
    exact_matches = []
    f1_scores = []
    latencies_ms = []
    error_examples = []

    for example in examples:
        started = time.perf_counter()
        result = qa_model(question=example["question"], context=example["context"])
        latencies_ms.append((time.perf_counter() - started) * 1_000)
        exact_match, f1 = score_prediction(result["answer"], example["answers"])
        exact_matches.append(exact_match)
        f1_scores.append(f1)

        if f1 < 1.0 and len(error_examples) < 5:
            error_examples.append(
                {
                    "question": example["question"],
                    "prediction": result["answer"],
                    "reference": example["answers"][0],
                    "f1": round(f1, 3),
                }
            )

    metrics = {
        "model": MODEL_NAME,
        "dataset": "SQuAD v2 dev, first 100 answerable examples",
        "examples": len(examples),
        "exact_match": sum(exact_matches) / len(exact_matches),
        "token_f1": sum(f1_scores) / len(f1_scores),
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "p95_latency_ms": sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1],
        "error_examples": error_examples,
    }

    (OUTPUT_DIR / "experiment.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_metric_chart(metrics)

    print("GenAI Extractive-QA Evaluation")
    print("=" * 60)
    print(f"Model: {metrics['model']}")
    print(f"Examples: {metrics['examples']}")
    print(f"Exact match: {metrics['exact_match']:.1%}")
    print(f"Token F1: {metrics['token_f1']:.1%}")
    print(f"Mean latency: {metrics['mean_latency_ms']:.1f} ms")
    print(f"P95 latency: {metrics['p95_latency_ms']:.1f} ms")
    print("\nSample errors:")
    for error in error_examples[:3]:
        print(f"- {error['question']}")
        print(f"  predicted: {error['prediction']}")
        print(f"  reference: {error['reference']}")
    print("\nGenerated: projects/genai-evaluation-lab/experiment.json")
    print("Generated: images/genai_qa_evaluation.png")


if __name__ == "__main__":
    main()
