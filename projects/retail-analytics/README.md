# Retail Transaction Analytics Capstone

A reproducible analysis of the UCI Online Retail transaction dataset. This capstone combines data quality checks, Python EDA, SQL analysis, customer segmentation, cohort retention, and a decision-oriented visual summary around one real source dataset.

## Business Questions

- How did recorded sales, invoice activity, and identified-customer counts move over time?
- Which countries and products contributed the most recorded positive sales, and how concentrated was the mix?
- How much cancellation and negative-quantity activity appears in the source, and how does it differ from positive sales?
- What repeat-purchase patterns appear among customers with IDs?
- How do customers compare on recency, invoice frequency, and recorded purchase value?
- What data-quality limitations should a stakeholder know before acting on these metrics?

## Dataset and Attribution

The project uses [Online Retail](https://doi.org/10.24432/C5BW33), donated by Daqing Chen to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail). It contains 541,909 transaction lines for a UK-based online retailer, dated December 1, 2010 through December 9, 2011. UCI lists the dataset under CC BY 4.0, the Creative Commons Attribution 4.0 International license.

The analysis downloads UCI's official ZIP archive on first run and extracts `Online Retail.xlsx` into the ignored `data/raw/uci_online_retail/` directory. The raw workbook and generated SQLite database are not committed to this repository; result CSVs and the dashboard are committed so the findings can be inspected without downloading the source.

## Reproduce

From the repository root, install the dependencies:

```powershell
python -m pip install pandas numpy matplotlib openpyxl
```

Run the analysis. The first run requires internet access to download the source file; later runs use the cached workbook.

```powershell
python projects/retail-analytics/analysis.py
```

To use an existing local copy instead:

```powershell
python projects/retail-analytics/analysis.py --data-path "C:\path\to\Online Retail.xlsx"
```

The script writes a queryable database to `data/processed/online_retail.sqlite`, summary tables to `projects/retail-analytics/results/`, and the dashboard to `images/retail_analytics_dashboard.png`.

Run the focused behavior tests with:

```powershell
python -m unittest discover -s projects/retail-analytics -p "test_*.py" -v
```

## Analysis Contract

The source contains sales, cancellations, returns, missing customer IDs, repeated source rows, and nonpositive unit prices. The code keeps all 541,909 original rows in SQLite and adds flags so exclusions remain inspectable.

The primary positive-sales population requires a non-cancellation invoice, positive quantity, positive unit price, and non-null line amount. Exact extra copies of source rows are excluded from analytical totals after retaining the first copy. The line amount is `quantity * unit price` in GBP. Cancellation and negative-quantity lines are analyzed separately as credits; they are not silently netted into gross positive sales.

Customer-level and cohort analyses use positive-sale rows with an identified customer ID. Unknown customer IDs stay in country and overall sales totals, but are not guessed or imputed for customer behavior. Duplicate, missing, cancellation, and invalid-price counts refer to the source unless the metric name says otherwise.

## Findings From the Reproduced Run

| Measure | Result |
| --- | ---: |
| Source transaction lines | 541,909 |
| Positive sale lines after duplicate removal | 524,878 |
| Recorded gross positive sales | GBP 10,642,110.80 |
| Positive-sale invoices | 19,960 |
| Identified customers with positive sales | 4,338 |
| Countries | 38 |
| Distinct stock codes | 4,070 |
| Exact duplicate excess rows excluded | 5,268 |
| Rows with no customer ID | 135,080 |
| Raw cancellation-invoice lines | 9,288 |
| Signed value of deduplicated credit lines | GBP -893,979.73 |

The United Kingdom contributes GBP 9,001,744.09 of recorded positive sales and dominates this dataset. Country and product rankings are descriptive; the source does not contain product cost, profit, validated discount, or customer acquisition cost. The last calendar month is incomplete, ending December 9, 2011, so it should not be compared with full months as if coverage were equal.

RFM scores divide identified customers into quintiles for recency, invoice frequency, and recorded purchase value. The named segments are transparent heuristics, not ground-truth customer types. Cohort retention counts identified customers with a positive purchase in each month after their first observed positive purchase; it does not prove why customers returned or stopped buying.

## Project Depth

### Data quality and definition
- Validate required source fields and parse dates, prices, quantities, invoice IDs, and customer IDs.
- Preserve source records and separately flag cancellations, negative quantities, nonpositive prices, missing IDs, missing descriptions, and exact duplicate groups.
- Define the primary sales population once and use it across Python outputs, SQLite flags, SQL queries, and the dashboard.

### Commercial analysis
- Compare month-level sales, unique invoices, units, and identified customers.
- Measure country and product performance with sales, order counts, unit counts, and concentration share.
- Keep returns and cancellation activity visible as a separate signed measure.

### Customer analysis
- Build repeat-purchase cohorts by first identified purchase month.
- Calculate retention as a share of each cohort returning in later months.
- Create interpretable RFM quintiles and summarize the heuristic customer segments.

### Reproducible delivery
- Store the normalized, flagged line-level data in SQLite for SQL exploration.
- Run eight portable SQLite queries, including month-over-month changes, customer RFM, invoice median, and cohort retention.
- Export CSV tables and a four-panel sales, market, product, and retention dashboard.
- Test population rules and customer/cohort calculations on a controlled workbook fixture.

## Files

- `analysis.py` — source download, validation, feature flags, KPI calculations, SQLite build, CSV exports, and chart generation
- `queries.sql` — eight SQLite analyses for quality, trends, credits, country/product mix, invoice values, RFM, and cohorts
- `test_analysis.py` — behavior tests for filtering, duplicate handling, missing customer IDs, RFM, and cohorts
- `results/` — generated quality, sales, credit, customer, product, and retention tables
- `../../images/retail_analytics_dashboard.png` — generated dashboard
- `../../data/processed/online_retail.sqlite` — local database created on run; intentionally ignored by Git

## Limitations

This is an educational, historical, descriptive analysis, not a current trading recommendation or causal study. The dataset records line-level quantity and unit price but does not provide reliable product costs or margin. Credit handling and exact-row deduplication are explicit analytical choices; inspect the flags and rerun alternate definitions when the decision warrants a sensitivity analysis. Customer analyses cover only identified customers, and the UK-heavy mix limits broad international comparisons.
