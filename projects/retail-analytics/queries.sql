-- UCI Online Retail analysis. Run analysis.py first to build data/processed/online_retail.sqlite.
-- Base sales definition: positive quantity, positive unit price, and non-cancellation invoice.
-- Monetary values are recorded line value in GBP, not profit or net revenue after all credits.

-- 1. Source coverage, data-quality flags, and the declared sales population.
SELECT
    COUNT(*) AS source_rows,
    SUM(is_positive_sale_line) AS positive_sales_lines,
    COUNT(DISTINCT CASE WHEN is_positive_sale_line = 1 THEN invoice_no END) AS positive_invoices,
    COUNT(DISTINCT CASE WHEN is_positive_sale_line = 1 THEN customer_id END) AS identified_customers,
    SUM(is_cancellation_invoice) AS cancellation_invoice_lines,
    SUM(is_exact_duplicate) AS rows_in_duplicate_groups,
    SUM(is_duplicate_excess) AS duplicate_excess_rows,
    SUM(customer_id IS NULL) AS rows_without_customer_id,
    SUM(description IS NULL OR description = '') AS rows_without_description
FROM order_lines;

-- 2. Monthly positive sales, invoice count, and month-over-month change.
WITH monthly AS (
    SELECT
        substr(invoice_date, 1, 7) AS month,
        SUM(line_amount_gbp) AS gross_sales_gbp,
        COUNT(DISTINCT invoice_no) AS invoices,
        COUNT(DISTINCT customer_id) AS identified_customers
    FROM order_lines
    WHERE is_positive_sale_line = 1
    GROUP BY substr(invoice_date, 1, 7)
), with_previous AS (
    SELECT
        *,
        LAG(gross_sales_gbp) OVER (ORDER BY month) AS previous_month_sales_gbp
    FROM monthly
)
SELECT
    month,
    ROUND(gross_sales_gbp, 2) AS gross_sales_gbp,
    invoices,
    identified_customers,
    ROUND(100.0 * (gross_sales_gbp - previous_month_sales_gbp) / NULLIF(previous_month_sales_gbp, 0), 2) AS mom_change_percent
FROM with_previous
ORDER BY month;

-- 3. Positive sale and credit activity by month. Credits are kept separate from gross sales.
SELECT
    substr(invoice_date, 1, 7) AS month,
    SUM(is_positive_sale_line) AS positive_sales_lines,
    ROUND(SUM(CASE WHEN is_positive_sale_line = 1 THEN line_amount_gbp ELSE 0 END), 2) AS gross_positive_sales_gbp,
    SUM(is_credit_line) AS credit_lines,
    ROUND(SUM(CASE WHEN is_credit_line = 1 THEN line_amount_gbp ELSE 0 END), 2) AS signed_credit_value_gbp
FROM order_lines
GROUP BY substr(invoice_date, 1, 7)
ORDER BY month;

-- 4. Country mix: sales share, invoice count, and identified-customer count.
WITH country_sales AS (
    SELECT
        country,
        SUM(line_amount_gbp) AS gross_sales_gbp,
        COUNT(DISTINCT invoice_no) AS invoices,
        COUNT(DISTINCT customer_id) AS customers
    FROM order_lines
    WHERE is_positive_sale_line = 1
    GROUP BY country
), total AS (
    SELECT SUM(gross_sales_gbp) AS all_sales_gbp FROM country_sales
)
SELECT
    country,
    ROUND(gross_sales_gbp, 2) AS gross_sales_gbp,
    invoices,
    customers,
    ROUND(100.0 * gross_sales_gbp / all_sales_gbp, 2) AS sales_share_percent
FROM country_sales CROSS JOIN total
ORDER BY gross_sales_gbp DESC;

-- 5. Product performance with both value and demand measures.
SELECT
    stock_code,
    COALESCE(NULLIF(description, ''), '(Missing description)') AS description,
    ROUND(SUM(line_amount_gbp), 2) AS gross_sales_gbp,
    SUM(quantity) AS units,
    COUNT(DISTINCT invoice_no) AS invoices
FROM order_lines
WHERE is_positive_sale_line = 1
GROUP BY stock_code, COALESCE(NULLIF(description, ''), '(Missing description)')
ORDER BY gross_sales_gbp DESC
LIMIT 20;

-- 6. Invoice-level average order value for identified-customer positive sales.
WITH invoice_sales AS (
    SELECT
        invoice_no,
        customer_id,
        SUM(line_amount_gbp) AS invoice_value_gbp
    FROM order_lines
    WHERE is_positive_sale_line = 1 AND customer_id IS NOT NULL
    GROUP BY invoice_no, customer_id
), ranked_invoices AS (
    SELECT
        invoice_value_gbp,
        ROW_NUMBER() OVER (ORDER BY invoice_value_gbp) AS invoice_rank,
        COUNT(*) OVER () AS invoice_count
    FROM invoice_sales
)
SELECT
    MAX(invoice_count) AS identified_customer_invoices,
    ROUND(AVG(invoice_value_gbp), 2) AS average_invoice_value_gbp,
    ROUND(AVG(CASE
        WHEN invoice_rank IN ((invoice_count + 1) / 2, (invoice_count + 2) / 2)
        THEN invoice_value_gbp
    END), 2) AS median_invoice_value_gbp
FROM ranked_invoices;

-- 7. Customer RFM features and quintile scores, computed from positive sales only.
WITH customer_metrics AS (
    SELECT
        customer_id,
        CAST(julianday('2011-12-10') - julianday(MAX(invoice_date)) AS INTEGER) AS recency_days,
        COUNT(DISTINCT invoice_no) AS frequency,
        SUM(line_amount_gbp) AS monetary_gbp
    FROM order_lines
    WHERE is_positive_sale_line = 1 AND customer_id IS NOT NULL
    GROUP BY customer_id
), scored AS (
    SELECT
        *,
        6 - NTILE(5) OVER (ORDER BY recency_days DESC) AS recency_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS frequency_score,
        NTILE(5) OVER (ORDER BY monetary_gbp ASC) AS monetary_score
    FROM customer_metrics
)
SELECT
    customer_id,
    recency_days,
    frequency,
    ROUND(monetary_gbp, 2) AS monetary_gbp,
    recency_score,
    frequency_score,
    monetary_score,
    recency_score + frequency_score + monetary_score AS rfm_score
FROM scored
ORDER BY rfm_score DESC, monetary_gbp DESC;

-- 8. Monthly retention by first-purchase cohort. Each customer is counted once per month.
WITH customer_months AS (
    SELECT DISTINCT
        customer_id,
        substr(invoice_date, 1, 7) AS order_month
    FROM order_lines
    WHERE is_positive_sale_line = 1 AND customer_id IS NOT NULL
), first_month AS (
    SELECT customer_id, MIN(order_month) AS cohort_month
    FROM customer_months
    GROUP BY customer_id
), cohort_activity AS (
    SELECT
        f.cohort_month,
        m.order_month,
        CAST(substr(m.order_month, 1, 4) AS INTEGER) * 12
          + CAST(substr(m.order_month, 6, 2) AS INTEGER)
          - CAST(substr(f.cohort_month, 1, 4) AS INTEGER) * 12
          - CAST(substr(f.cohort_month, 6, 2) AS INTEGER) AS cohort_index,
        COUNT(DISTINCT m.customer_id) AS active_customers
    FROM customer_months AS m
    JOIN first_month AS f USING (customer_id)
    GROUP BY f.cohort_month, m.order_month
), cohort_sizes AS (
    SELECT cohort_month, active_customers AS cohort_size
    FROM cohort_activity
    WHERE cohort_index = 0
)
SELECT
    a.cohort_month,
    a.order_month,
    a.cohort_index,
    a.active_customers,
    s.cohort_size,
    ROUND(100.0 * a.active_customers / s.cohort_size, 2) AS retention_percent
FROM cohort_activity AS a
JOIN cohort_sizes AS s USING (cohort_month)
ORDER BY a.cohort_month, a.cohort_index;
