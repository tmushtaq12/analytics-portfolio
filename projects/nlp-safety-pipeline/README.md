# NLP Safety Pipeline: Offensive-Speech Detection

## Why This Project Exists

This project maps directly to production NLP safety work: detect potentially offensive conversational input, protect contact information before downstream processing, and evaluate the classifier with class-aware metrics.

It uses the real [TweetEval offensive-language dataset](https://huggingface.co/datasets/cardiffnlp/tweet_eval), not hand-written examples.

## Dataset

TweetEval provides labeled tweets for offensive-language detection:

- `0` — not offensive
- `1` — offensive

The local files contain:

- 11,916 training rows
- validation and test splits from the same benchmark
- raw text plus binary labels

The dataset is stored as parquet under `data/raw/tweet_eval_offensive/`.

## Pipeline

1. Load the fixed train, validation, and test splits
2. Convert text to word-level TF-IDF features using unigrams and bigrams
3. Train a class-balanced logistic regression classifier
4. Report macro F1, offensive precision, offensive recall, and a full test classification report
5. Generate a confusion matrix
6. Mask emails, phone numbers, and usernames before downstream use

The implementation is deliberately small and readable. It uses a scikit-learn pipeline so feature extraction and classification stay together and the test set remains untouched until evaluation.

## Why These Metrics

For safety classification, accuracy alone is not enough:

- **Macro F1** prevents the larger class from hiding weak minority-class performance
- **Precision** measures how often offensive predictions are correct
- **Recall** measures how much offensive content the detector catches
- The confusion matrix makes false positives and false negatives visible

## Run It

```powershell
python projects/nlp-safety-pipeline/analysis.py
```

Required packages:

```powershell
pip install pandas pyarrow scikit-learn matplotlib seaborn
```

## Output

![Offensive speech confusion matrix](../../images/offensive_speech_confusion_matrix.png)

The script prints validation metrics, a held-out test classification report, and examples of PII masking.

## Production Next Steps

This is a classical, explainable baseline rather than a claim of production readiness. A production version would add:

- transformer fine-tuning and calibration
- threshold selection based on safety cost tradeoffs
- multilingual evaluation
- adversarial and drift test sets
- human review for uncertain predictions
- model and data versioning
- latency monitoring and containerized serving

## Files

- `analysis.py` — end-to-end dataset loading, training, evaluation, and masking
- `../../data/raw/tweet_eval_offensive/` — real benchmark data
- `../../images/offensive_speech_confusion_matrix.png` — generated evaluation visual
