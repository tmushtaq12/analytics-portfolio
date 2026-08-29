-- SQL Analysis Project
-- Example business questions and queries

-- 1. Total revenue by product
SELECT
    product_name,
    SUM(revenue) AS total_revenue
FROM sales_data
GROUP BY product_name
ORDER BY total_revenue DESC;

-- 2. Revenue by region
SELECT
    region,
    SUM(revenue) AS total_revenue
FROM sales_data
GROUP BY region
ORDER BY total_revenue DESC;

-- 3. Monthly sales trend
SELECT
    DATE_TRUNC('month', order_date) AS month,
    SUM(revenue) AS monthly_revenue
FROM sales_data
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- 4. Top customers by revenue
SELECT
    customer_id,
    SUM(revenue) AS total_revenue
FROM sales_data
GROUP BY customer_id
ORDER BY total_revenue DESC
LIMIT 10;

-- 5. Moving average of revenue
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(revenue) AS revenue
    FROM sales_data
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT
    month,
    revenue,
    AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_average
FROM monthly_revenue
ORDER BY month;
