import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "squad" / "dev-v2.0.json"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)


def load_squad_questions():
    """Return answerable questions, answers, and their source contexts."""
    with DATA_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    questions = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for item in paragraph["qas"]:
                if item["is_impossible"] or not item["answers"]:
                    continue
                questions.append(
                    {
                        "question": item["question"],
                        "answer": item["answers"][0]["text"],
                        "context": context,
                    }
                )
    return questions


def build_retriever(contexts):
    """Index contexts with a lightweight lexical retrieval baseline."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(contexts)
    return vectorizer, matrix


def retrieve(question, vectorizer, matrix, contexts, history=None, top_k=5):
    """Retrieve the most similar contexts, optionally using conversation history."""
    query = " ".join(history or []) + " " + question
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indexes = np.argsort(scores)[::-1][:top_k]
    return [(contexts[index], scores[index]) for index in ranked_indexes]


def evaluate_recall(questions, contexts, vectorizer, matrix):
    """Measure whether the gold answer appears in a retrieved context."""
    hits = {1: 0, 5: 0}
    for item in questions:
        retrieved = retrieve(item["question"], vectorizer, matrix, contexts, top_k=5)
        for k in hits:
            if any(item["answer"].lower() in context.lower() for context, _ in retrieved[:k]):
                hits[k] += 1

    return {k: value / len(questions) for k, value in hits.items()}


def save_recall_chart(recall):
    labels = ["Recall@1", "Recall@5"]
    values = [recall[1], recall[5]]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#e76f51", "#2a9d8f"])
    plt.ylim(0, 1)
    plt.ylabel("Questions with answer-bearing context")
    plt.title("SQuAD retrieval recall")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "squad_retrieval_recall.png", dpi=180)
    plt.close()


def main():
    questions = load_squad_questions()
    contexts = list(dict.fromkeys(item["context"] for item in questions))
    vectorizer, matrix = build_retriever(contexts)
    recall = evaluate_recall(questions, contexts, vectorizer, matrix)

    example = questions[0]
    retrieved = retrieve(example["question"], vectorizer, matrix, contexts, top_k=3)
    print("SQuAD RAG Retrieval Evaluation")
    print("=" * 60)
    print(f"Answerable questions: {len(questions):,}")
    print(f"Unique indexed contexts: {len(contexts):,}")
    print(f"Recall@1: {recall[1]:.1%}")
    print(f"Recall@5: {recall[5]:.1%}")
    print("\nExample question:")
    print(example["question"])
    print(f"Gold answer: {example['answer']}")
    print("Top retrieved scores:")
    for rank, (_, score) in enumerate(retrieved, start=1):
        print(f"{rank}. {score:.3f}")

    save_recall_chart(recall)
    print("\nGenerated: images/squad_retrieval_recall.png")


if __name__ == "__main__":
    main()
