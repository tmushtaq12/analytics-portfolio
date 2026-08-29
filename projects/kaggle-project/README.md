# Kaggle-Style Retail Performance Analysis

## Overview
This project follows a realistic Kaggle-style workflow using a retail dataset to identify the main drivers of performance and translate them into a business-ready summary.

## Business Problem
A retail business needs to understand:
- which regions and categories generate the most revenue
- which customer segments are the most valuable
- how channel mix affects sales outcomes
- where the business should concentrate performance investment

## Dataset
The project uses a transaction-level retail dataset with fields such as:
- `order_id`
- `order_date`
- `region`
- `channel`
- `category`
- `product`
- `units_sold`
- `revenue`
- `discount_percent`
- `customer_segment`

## Methodology
1. Load the dataset
2. Clean and normalize the data
3. Group performance by region, category, channel, and customer segment
4. Visualize the strongest patterns
5. Summarize the commercial opportunity and strategic recommendations

## Key Findings
- Electronics is the strongest category by revenue
- West and East lead revenue generation across the dataset
- Corporate customers contribute the largest share of sales value
- Online sales are the strongest channel
- A small number of combinations drive the bulk of revenue, which suggests clear commercial focus areas

## Deliverables
- `analysis.py` — Python analysis script
- `summary.md` — final written summary
- `images/kaggle_region_category.png` — final chart output

## Business Value
This project shows the full analytics cycle: investigate, quantify, visualize, and explain the most important business patterns in a way that supports decision-making.
