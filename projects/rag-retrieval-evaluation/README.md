# RAG Retrieval Evaluation: SQuAD

## Objective
Evaluate the retrieval stage of a retrieval-augmented generation system before adding an LLM. The project measures whether a retriever returns a context containing the answer to a user question.

## Dataset
This project uses the real [SQuAD v2 dataset](https://rajpurkar.github.io/SQuAD-explorer/), including answerable and unanswerable questions. For a clean retrieval benchmark, the script evaluates the answerable development questions and indexes their source contexts.

## Retrieval Pipeline
1. Load the SQuAD JSON file
2. Exclude unanswerable questions from the retrieval score
3. Deduplicate source contexts
4. Build a TF-IDF bigram index
5. Rank contexts with cosine similarity
6. Measure whether the gold answer appears in the top 1 or top 5 contexts

The TF-IDF index is a transparent lexical baseline. It is intentionally not described as an embedding model or production vector database.

## Metrics
- **Recall@1**: percentage of questions where the first retrieved context contains the answer
- **Recall@5**: percentage where at least one of the five retrieved contexts contains the answer

Recall is the important first-stage metric here: if the answer-bearing context is never retrieved, a later generator cannot produce a grounded answer from it.

## Run It

```powershell
python projects/rag-retrieval-evaluation/analysis.py
```

Required packages:

```powershell
pip install numpy scikit-learn matplotlib
```

## Output

![SQuAD retrieval recall](../../images/squad_retrieval_recall.png)

The script prints dataset size, Recall@1, Recall@5, and an example query with its top retrieval scores.

## Production Next Steps

A production RAG implementation would compare this baseline with dense embeddings and a vector index such as FAISS, then add hybrid retrieval, reranking, conversation-aware query rewriting, citation checks, latency measurements, and answer-level metrics such as exact match, faithfulness, and hallucination rate.

## Files

- `analysis.py` — readable retrieval and evaluation workflow
- `../../data/raw/squad/dev-v2.0.json` — real SQuAD evaluation data
- `../../images/squad_retrieval_recall.png` — generated retrieval metric visual
