import re
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "tweet_eval_offensive"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def load_split(name: str) -> Tuple[pd.Series, pd.Series]:
    """Read one TweetEval split and return text and binary labels."""
    data = pd.read_parquet(DATA_DIR / f"{name}.parquet")
    return data[TEXT_COLUMN].astype(str), data[LABEL_COLUMN].astype(int)


def build_classifier() -> Pipeline:
    """Build a transparent text-classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    min_df=2,
                    ngram_range=(1, 2),
                    max_features=120_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )


def mask_sensitive_data(text: str) -> str:
    """Mask common contact details before text reaches a downstream model."""
    masked = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    masked = re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", "[PHONE]", masked)
    masked = re.sub(r"@([A-Za-z0-9_]+)", "@[USER]", masked)
    return masked


def save_confusion_matrix(actual: pd.Series, predicted):
    matrix = confusion_matrix(actual, predicted)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not offensive", "Offensive"],
        yticklabels=["Not offensive", "Offensive"],
    )
    plt.title("Offensive-speech classifier confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "offensive_speech_confusion_matrix.png", dpi=180)
    plt.close()


def main():
    train_text, train_labels = load_split("train")
    validation_text, validation_labels = load_split("validation")
    test_text, test_labels = load_split("test")

    model = build_classifier()
    model.fit(train_text, train_labels)

    validation_predictions = model.predict(validation_text)
    test_predictions = model.predict(test_text)

    print("TweetEval Offensive-Speech Detection")
    print("=" * 60)
    print(f"Training rows: {len(train_text):,}")
    print(f"Validation rows: {len(validation_text):,}")
    print(f"Test rows: {len(test_text):,}")
    print("\nValidation metrics:")
    print(f"Macro F1:  {f1_score(validation_labels, validation_predictions, average='macro'):.3f}")
    print(f"Offensive precision: {precision_score(validation_labels, validation_predictions):.3f}")
    print(f"Offensive recall:    {recall_score(validation_labels, validation_predictions):.3f}")
    print("\nHeld-out test classification report:")
    print(classification_report(test_labels, test_predictions, target_names=["not_offensive", "offensive"]))

    save_confusion_matrix(test_labels, test_predictions)

    examples = [
        "Contact me at analyst@example.com or +1 555 123 4567 @customer",
        "This is a normal product question from a customer",
    ]
    print("Masked examples:")
    for example in examples:
        print(f"- {mask_sensitive_data(example)}")

    print("\nGenerated: images/offensive_speech_confusion_matrix.png")


if __name__ == "__main__":
    main()
