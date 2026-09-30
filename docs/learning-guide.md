# Learning Guide: NLP, RAG, and GenAI Engineering

This guide explains the ideas behind the portfolio projects and gives you a practical order for learning them. You do not need to understand everything at once. Run one project, read the matching section, change one thing, and run it again.

## Recommended Learning Order

1. [Retail Transaction Analytics Capstone](../projects/retail-analytics/README.md) — data quality, SQL, customer analysis, and reporting
2. [Regression Models](../projects/linear-regression/README.md) — best-fit lines, transformations, errors, and an MLP
3. [NLP Safety Pipeline](../projects/nlp-safety-pipeline/README.md)
4. [RAG Retrieval Evaluation](../projects/rag-retrieval-evaluation/README.md)
5. [GenAI Evaluation Lab](../projects/genai-evaluation-lab/README.md)

The analytics capstone shows how to define trustworthy measures before modeling. Regression introduces controlled model comparison. The final three projects build toward NLP and GenAI work.

## 1. Foundations: Python and Machine Learning

Before going deep into LLMs, become comfortable with:

- functions, modules, classes, and virtual environments
- pandas for tabular data
- NumPy arrays and vectorized operations
- train/validation/test splits
- overfitting and generalization
- precision, recall, F1, and confusion matrices
- reproducibility with fixed random seeds

Start with the [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html). Read the model evaluation and text feature extraction sections first.

### Exercise
Change the NLP classifier's `ngram_range` from `(1, 2)` to `(1, 3)`. Run it again and record whether macro F1, offensive precision, and offensive recall improve or worsen. Write down why a metric tradeoff might matter for a safety system.

## 2. Retail Analytics: Define the Population Before the KPI

The [retail capstone](../projects/retail-analytics/README.md) uses UCI Online Retail, a public transaction workbook with 541,909 invoice lines across one year. It is a stronger analysis foundation than a tiny practice CSV because it includes invoice IDs, stock codes, quantity, unit price, time, customer IDs, and country.

The row-level value is `quantity * unit price`, in GBP. That does not automatically mean profit or net accounting revenue. The source also contains cancellations, negative quantities, exact duplicate rows, missing customer IDs, and a partial final month. The capstone keeps every source row for inspection, flags quality issues, excludes extra exact copies from primary totals, defines positive sales explicitly, and reports credits separately.

The customer work answers different questions at different grains:

- **Invoice grain:** aggregate line amounts by invoice before calculating average invoice value.
- **Customer grain:** count distinct invoices, sum positive sale value, and measure days since the last observed purchase.
- **Cohort grain:** group identified customers by first purchase month and count how many return in each later month.
- **Product/country grain:** aggregate sales and distinct invoices without treating line count as customer count.

RFM means recency, frequency, and monetary value. This project converts each measure to a 1-to-5 quintile score and adds the scores. The labels such as `Champions` and `At risk` are heuristics for exploration, not validated customer types or a churn model.

### Exercise
Run the capstone, then query `data/processed/online_retail.sqlite` with `projects/retail-analytics/queries.sql`. Compare monthly positive sales with signed credit activity. Quantify how totals change if exact duplicates are retained, and explain why customer-level results exclude rows without a customer ID while total country sales do not. Do not describe the result as margin: the data does not provide reliable costs.

## 3. Regression: Lines, Errors, Log Targets, and Neural Networks

The [regression project](../projects/linear-regression/README.md) predicts Titanic fare using information available before the voyage. It compares a training-mean baseline, an age-only best-fit line, multiple linear regression, log-target linear regression, and a small multilayer perceptron (MLP) neural network.

### Best-fit line and residuals

For one input `x` and target `y`, a line predicts `y_hat = slope * x + intercept`. Ordinary least squares chooses the slope and intercept to minimize the sum of squared residuals. A residual is `y - y_hat`: positive means the observation is above the prediction; negative means below it. A pattern in residuals can show curvature, unequal error spread, or outliers that a single score hides.

The Titanic fare model uses several columns, so it is a multivariable linear model rather than one line on a two-dimensional plot. The project also fits an illustrative one-feature age-to-fare line. Its coefficient describes an association in this historical sample, not a causal effect.

### Error metrics

For actual values `y` and predictions `y_hat`:

- `MAE = mean(abs(y - y_hat))` is the average absolute miss and remains in GBP.
- `MSE = mean((y - y_hat) ** 2)` squares misses and gives large errors more weight; its units are GBP squared.
- `RMSE = sqrt(MSE)` also emphasizes large misses but returns to GBP.
- `R2 = 1 - sum((y - y_hat) ** 2) / sum((y - mean(y)) ** 2)` compares the model with a constant prediction at the evaluated target mean. It is not percent accuracy and can be negative.

Always identify which split a score describes. Here, the fixed 20% test score estimates this fitted pipeline on one held-out split. Five shuffled folds summarize training-data variability; they are not five new test sets and do not guarantee future performance.

### Log transforms

When a positive target has a long right tail, `log1p(y)` compresses large values. Fit the model on the transformed training target, then use `expm1(prediction)` to convert predictions back before calculating MAE or RMSE in GBP. This changes the loss geometry and model emphasis; it does not make errors disappear. The log-target model is one candidate to compare, not an automatic upgrade.

The implementation uses scikit-learn's `TransformedTargetRegressor` so inverse transformation is part of prediction rather than a forgotten manual step. Preprocessing remains inside the pipeline and is fitted on training data only.

### Neural-network basics

A feed-forward network computes weighted sums plus biases, applies nonlinear activations in hidden layers, and produces an output. Training adjusts weights to reduce a loss. More hidden units increase flexibility, but also make overfitting and interpretation more difficult. An MLP is not automatically superior to a linear model, particularly on a small dataset.

The project uses one hidden layer with 16 units, L-BFGS optimization, regularization (`alpha=0.01`), a fixed random seed, and the same preprocessing and log-target wrapper as the comparison requires. The small example makes the network concrete without presenting it as production experience.

### Reproduced comparison

| Model | Test MAE | Test RMSE | Test R2 | Five-fold mean MAE +/- SD |
| --- | ---: | ---: | ---: | ---: |
| Training-mean baseline | GBP 25.69 | GBP 39.38 | -0.002 | Not applicable |
| Multiple linear | GBP 20.65 | GBP 30.31 | 0.406 | GBP 20.59 +/- 2.36 |
| Log-target linear | GBP 10.79 | GBP 25.08 | 0.594 | GBP 13.38 +/- 3.47 |
| Log-target MLP | GBP 12.61 | GBP 28.48 | 0.476 | GBP 12.52 +/- 2.52 |

The best test-split MAE and best cross-validation mean are not the same model. Treat that as evidence of split/model-selection uncertainty, not a reason to tune against the held-out test. The exact model settings and generated values are in [`model_comparison.csv`](../projects/linear-regression/model_comparison.csv).

### Exercise
Inspect three largest absolute test residuals and note their passenger class, ticket group, and fare. Then propose a group-aware split by ticket. State in advance which metric should decide a model comparison, fit and tune using training data only, and open the test set once for the final result.

## 2. NLP Classification

The safety project converts text into numerical features with TF-IDF:

- TF measures how often a term appears in one document.
- IDF reduces the influence of terms that appear in almost every document.
- N-grams capture short phrases, not just individual words.

The logistic regression model then learns a boundary between offensive and non-offensive examples.

Important distinction:

- **Precision** matters when false positives are costly.
- **Recall** matters when missing unsafe content is costly.
- **Macro F1** gives both classes equal importance.

The correct threshold depends on product risk. There is no universally best metric.

### Exercise
Change `class_weight="balanced"` to `None`. Compare the minority-class recall. This demonstrates why class imbalance changes the behavior of a safety classifier.

## 3. Retrieval and RAG

A RAG system usually has two separate problems:

1. **Retrieval:** find relevant evidence.
2. **Generation:** answer using that evidence.

The RAG project currently evaluates retrieval with TF-IDF and cosine similarity. That is a useful baseline because it is simple and explainable. It is not an embedding model or a vector database.

The main retrieval metric is Recall@k:

- Recall@1 asks whether the first result contains the answer.
- Recall@5 asks whether any of the first five results contains the answer.

If retrieval fails, a generator cannot recover the missing evidence reliably.

The next retrieval progression is:

1. TF-IDF lexical search
2. Dense embeddings
3. FAISS or another vector index
4. Hybrid lexical + dense retrieval
5. Reranking
6. Conversation-aware query rewriting

Read the [FAISS documentation](https://faiss.ai/) and the [Sentence Transformers documentation](https://www.sbert.net/) when you are ready for dense retrieval.

### Exercise
Change the RAG project's `top_k` from 5 to 1 and compare Recall@1. Then test a larger value such as 10. Explain the quality/latency tradeoff.

## 4. GenAI Evaluation

The GenAI Evaluation Lab evaluates a real Hugging Face model rather than trusting a few examples by eye.

It records:

- exact match
- token F1
- mean latency
- P95 latency
- representative errors
- model and dataset identity

This is the beginning of experiment tracking. A serious evaluation should also record:

- prompt or template version
- model version
- retrieval configuration
- temperature and token limits
- cost
- hardware/runtime
- dataset version
- failure categories

The [experiment.json](../projects/genai-evaluation-lab/experiment.json) file is intentionally lightweight. In production, the same fields could be stored in MLflow, Weights & Biases, a warehouse, or an internal evaluation service.

### Exercise
Increase `EVALUATION_LIMIT` from 100 to 250. Compare the metrics and latency. Ask whether the first 100 records are representative enough for a model decision.

## 5. Prompt Engineering and API Integration

Prompt engineering is not just writing a clever instruction. A useful prompt should define:

- the task
- the available evidence
- the required output format
- what the model must do when evidence is missing
- safety and privacy constraints
- examples, when examples reduce ambiguity

For an API-backed model, keep the provider behind a small adapter. The evaluation code should receive a consistent result such as:

```python
{
    "answer": "...",
    "model": "provider-model-name",
    "latency_ms": 123.4,
    "usage": {"input_tokens": 100, "output_tokens": 25}
}
```

That design lets you compare a local Hugging Face model, an OpenAI-compatible endpoint, Azure OpenAI, or Bedrock without rewriting the scoring layer.

Do not commit API keys. Use environment variables or a secret manager:

```powershell
$env:MODEL_API_KEY = "your-key"
```

Useful official documentation:

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)
- [OpenAI API documentation](https://platform.openai.com/docs)
- [Azure OpenAI documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/)

## 6. Quality, Bias, and Safety

A model can have good average metrics and still fail important groups or input types. Build evaluation slices such as:

- language or locale
- short versus long inputs
- spelling variation
- dialect or demographic indicators, when ethically and legally appropriate
- direct versus indirect requests
- adversarial prompts
- PII-containing inputs
- ambiguous or unanswerable questions

For every slice, compare both quality and error type. Do not infer sensitive attributes casually from text. Use documented, consented, and ethically reviewed test metadata.

## 7. Production Engineering Topics

After the local projects, study these areas in order:

### Serving
- FastAPI or a similar Python service framework
- request validation with Pydantic
- authentication and authorization
- structured logging
- timeouts and retries

### Packaging
- Docker images
- dependency pinning
- health checks
- configuration through environment variables

### Observability
- latency and token usage
- error rates
- retrieval quality
- model/version metadata
- traces for multi-step agent workflows

### Deployment
- CI checks and tests
- model registry and versioning
- Kubernetes basics
- one cloud platform first: AWS, Azure, or GCP

The portfolio does not claim that these are all implemented yet. It gives you a measured local foundation to build toward them honestly.

## Suggested Eight-Week Plan

### Weeks 1-2: Python and evaluation
Run every project. Read every function. Add tests for normalization, F1, masking, and metric calculations.

### Weeks 3-4: NLP and transformers
Fine-tune or compare a small transformer on the TweetEval task. Track training configuration and validation metrics.

### Weeks 5-6: RAG
Replace TF-IDF with Sentence Transformers plus FAISS. Compare Recall@1, Recall@5, latency, and index size.

### Week 7: API service
Wrap one model behind FastAPI. Add request validation, a health endpoint, structured errors, and a Dockerfile.

### Week 8: Evaluation and deployment
Add a small regression test suite, experiment metadata, cost/latency tracking, and a CI workflow that runs the checks on every push.

## How to Use This Repository to Learn

For each change:

1. Create a hypothesis.
2. Change one variable.
3. Run the project.
4. Compare metrics with the previous experiment.
5. Inspect at least three failures.
6. Record what changed and why.

That workflow will teach you more than adding libraries without measuring their effect.
