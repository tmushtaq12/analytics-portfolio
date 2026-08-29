import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
data_path = base_dir / 'data' / 'raw' / 'retail_sales.csv'
images_dir = base_dir / 'images'
images_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

df['order_date'] = pd.to_datetime(df['order_date'])
df['order_month'] = df['order_date'].dt.to_period('M').astype(str)
df = df.sort_values('order_date').reset_index(drop=True)

# Clean obvious issues
for col in ['region', 'channel', 'category', 'product', 'customer_segment']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# KPI calculations
monthly_revenue = df.groupby('order_month', as_index=False)['revenue'].sum().sort_values('order_month')
region_revenue = df.groupby('region', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
category_revenue = df.groupby('category', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
segment_revenue = df.groupby('customer_segment', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
channel_revenue = df.groupby('channel', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)

# Additional business metrics
avg_order_value = df['revenue'].mean()
total_revenue = df['revenue'].sum()
total_units = df['units_sold'].sum()

# Promo / pricing insight
avg_discount_by_region = df.groupby('region', as_index=False)['discount_percent'].mean().sort_values('discount_percent', ascending=False)

print('Retail Sales Analysis')
print('=' * 60)
print(f'Total revenue: ${total_revenue:,.0f}')
print(f'Average order value: ${avg_order_value:,.0f}')
print(f'Total units sold: {total_units:,.0f}')
print(f'Average discount rate: {df["discount_percent"].mean():.1f}%')
print()
print('Revenue by region:')
print(region_revenue.head().to_string(index=False))
print()
print('Revenue by category:')
print(category_revenue.head().to_string(index=False))
print()
print('Revenue by customer segment:')
print(segment_revenue.head().to_string(index=False))
print()
print('Revenue by channel:')
print(channel_revenue.head().to_string(index=False))

# Visuals
plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_revenue, x='order_month', y='revenue', marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(images_dir / 'monthly_revenue_trend.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=region_revenue.head(5), x='revenue', y='region', palette='viridis')
plt.title('Top 5 Revenue by Region')
plt.tight_layout()
plt.savefig(images_dir / 'top_region_revenue.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=category_revenue.head(5), x='revenue', y='category', palette='magma')
plt.title('Top 5 Revenue by Category')
plt.tight_layout()
plt.savefig(images_dir / 'top_category_revenue.png', dpi=150)
plt.close()

# Summary insight
best_region = region_revenue.iloc[0]['region']
best_category = category_revenue.iloc[0]['category']
best_segment = segment_revenue.iloc[0]['customer_segment']

print(f'\nKey insight 1: {best_region} generated the largest revenue contribution.')
print(f'Key insight 2: {best_category} was the highest-performing product category.')
print(f'Key insight 3: {best_segment} contributed the strongest segment-level revenue.')
print('\nRecommendation: focus inventory and marketing spend on the strongest region, category, and customer segment while continuing to monitor discounting impact on margin.')
print(f'Generated charts saved in: {images_dir}')
