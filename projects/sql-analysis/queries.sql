-- SQL Analysis Project
-- Retail sales analysis

-- 1. Total revenue by region
SELECT
    region,
    SUM(revenue) AS total_revenue
FROM retail_sales
GROUP BY region
ORDER BY total_revenue DESC;

-- 2. Top products by revenue
SELECT
    product,
    SUM(revenue) AS total_revenue,
    SUM(units_sold) AS total_units_sold
FROM retail_sales
GROUP BY product
ORDER BY total_revenue DESC
LIMIT 10;

-- 3. Monthly sales trend
SELECT
    DATE_TRUNC('month', order_date) AS month,
    SUM(revenue) AS monthly_revenue
FROM retail_sales
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- 4. Average discount by region
SELECT
    region,
    AVG(discount_percent) AS avg_discount_percent,
    SUM(revenue) AS total_revenue
FROM retail_sales
GROUP BY region
ORDER BY total_revenue DESC;

-- 5. Revenue by customer segment and channel
SELECT
    customer_segment,
    channel,
    SUM(revenue) AS total_revenue
FROM retail_sales
GROUP BY customer_segment, channel
ORDER BY total_revenue DESC;

-- 6. Top 10 orders by revenue
SELECT
    order_id,
    region,
    category,
    revenue
FROM retail_sales
ORDER BY revenue DESC
LIMIT 10;
