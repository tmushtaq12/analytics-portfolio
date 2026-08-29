# Kaggle-Style Retail Performance Analysis

## Overview
This project follows a typical Kaggle-style analysis workflow using a retail dataset to understand sales performance, customer behavior, and business opportunities.

## Business Problem
A retail business wants to identify:
- which regions are generating the strongest revenue
- which product categories are performing best
- whether discounting is influencing sales volume
- which customer segments are most valuable
- where operational or pricing improvements may be needed

## Dataset
The dataset in this project includes purchase-level records with fields such as:
- order_id
- order_date
- region
- channel
- product_category
- units_sold
- revenue
- discount_percent
- customer_segment
- customer_id

## Questions to Answer
- Which region generated the highest revenue?
- What product categories are driving most of the sales?
- Are discounts helping or reducing margin quality?
- Which customer segment contributes the most revenue?
- Is there a relationship between discounting and sales growth?

## Methodology
1. Load the dataset and inspect quality issues
2. Review missing values and data consistency
3. Summarize descriptive statistics
4. Group results by region, segment, and product category
5. Visualize the strongest trends and variations
6. Turn the analysis into a clear recommendation summary

## Deliverables
- `sales_dataset.csv` — sample retail dataset
- `analysis.py` — Python script that calculates key insights
- `summary.md` — final business summary and recommendations

## Example Insights
- Revenue is concentrated in a few regions and product categories
- Discounts may increase volume but can hurt overall margin efficiency
- Repeat customers and high-value segments contribute disproportionately to sales

## Business Value
This project demonstrates the full analytics process: data cleaning, exploration, metric calculation, and communication of findings in a way that supports business decisions.
