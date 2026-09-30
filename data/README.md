# Data Folder

`raw/` contains source datasets used by the projects. `processed/` is reserved for generated local databases and is ignored by Git.

## Tracked Sources

- `raw/titanic.csv` — passenger survival and fare data used by the Kaggle and regression projects.
- `raw/squad/dev-v2.0.json` — SQuAD v2 development examples used by retrieval and QA evaluation.
- `raw/tweet_eval_offensive/` — fixed TweetEval train, validation, and test splits used by the NLP safety project.
- `raw/retail_sales.csv` — a 48-row retail exercise file. It has no source provenance documented here and 38 rows do not satisfy `revenue = units_sold * unit_price * (1 - discount_percent / 100)`. Treat it as a small practice fixture, not as evidence for commercial conclusions.

## Downloaded Source

The [Retail Transaction Analytics capstone](../projects/retail-analytics/README.md) downloads the 22.6 MB UCI Online Retail workbook on first run. The source workbook and generated SQLite database are intentionally not tracked; the UCI source, CC BY 4.0 license, download behavior, and reproducible findings are documented in that project.
