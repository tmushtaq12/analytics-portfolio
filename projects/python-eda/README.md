# Python EDA Project

> **Scope:** This is a small practice script that reads the 48-row `data/raw/retail_sales.csv` fixture and writes charts to the repository's top-level `images/` folder. It is not the source of the portfolio's verified retail findings. Use the [Retail Transaction Analytics Capstone](../retail-analytics/README.md) for full-scale data cleaning, transaction-quality flags, monthly analysis, customer cohorts, reproducible results, and the generated dashboard.

## Overview
This project uses Python to analyze a retail sales dataset and turn raw transaction data into business insight. The goal is to identify the strongest-performing areas, categories, customer segments, and trends that can support strategic decisions.

## Business Question
Which products, regions, and customer groups are driving the most revenue, and how can this information inform resource allocation and marketing priorities?

## Methodology
1. Load the sales dataset from the raw data folder
2. Convert date fields and sort records in time order
3. Clean missing values and standardize categories
4. Aggregate revenue by region, category, and customer segment
5. Produce charts and summary metrics to communicate the findings

## Questions This Practice Script Illustrates
- How can a table be grouped by month, region, product category, and customer segment?
- What chart types make those group totals easy to compare?
- How should a recommendation change when data provenance or a key calculation is uncertain?

The rankings printed by this script are descriptive outputs of the small practice fixture. They are deliberately not repeated here as verified business findings.

## Files
- `analysis.py` — full analysis workflow
- `../../data/raw/retail_sales.csv` — 48-row practice input
- `../../images/` — generated practice charts

## Business Impact
This project demonstrates the mechanics of a short EDA workflow. Its input's provenance and discount calculation are not established, so findings should be treated as practice output rather than recommendations. The capstone provides the broader evidence-backed workflow.
