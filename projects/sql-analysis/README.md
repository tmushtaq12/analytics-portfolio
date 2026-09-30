# SQL Sales Analysis Project

> **Scope:** This is an introductory six-query exercise written against a conceptual `retail_sales` table. It does not load the repository's retail CSV or create that table. For a full-scale, reproducible analysis with a real SQLite database, use the [Retail Transaction Analytics Capstone](../retail-analytics/README.md) and its eight executable queries.

## Overview
This project uses SQL to answer commercial questions about a retail sales dataset. The goal is to move from raw transaction data to actionable business insight using measurable KPIs and trend analysis.

## Business Questions
- Which regions generated the most revenue?
- Which products contributed the largest sales totals?
- How did revenue change over time?
- Which customer segments delivered the strongest business value?
- How did channel and discounting strategy affect performance?

## Methodology
The SQL workflow includes:
- filtering and date grouping
- revenue aggregation by region and category
- customer and product-level analysis
- order ranking
- trend analysis over time

## Query Skills Practiced
- `GROUP BY` aggregations and sorting
- Monthly date grouping
- Multi-dimensional segment comparisons
- Top-N ranking

## Files
- `queries.sql` — SQL analysis script with real business questions and answers

The queries demonstrate common grouping, ranking, and date-trend patterns. The older 48-row retail file has no source provenance documented; its category and region findings are not presented as verified commercial results.

## Business Impact
This is SQL practice, not independently validated commercial analysis. Use the transaction capstone for source-backed findings and a database that the project builds itself.
