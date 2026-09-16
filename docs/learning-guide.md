# Learning Guide: NLP, RAG, and GenAI Engineering

This guide explains the ideas behind the portfolio projects and gives you a practical order for learning them. You do not need to understand everything at once. Run one project, read the matching section, change one thing, and run it again.

## Recommended Learning Order

1. [NLP Safety Pipeline](../projects/nlp-safety-pipeline/README.md)
2. [RAG Retrieval Evaluation](../projects/rag-retrieval-evaluation/README.md)
3. [GenAI Evaluation Lab](../projects/genai-evaluation-lab/README.md)
4. [Linear Regression](../projects/linear-regression/README.md)

The first three projects are the most relevant to NLP/GenAI roles. Linear regression gives you a clean foundation for model training and evaluation.

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
