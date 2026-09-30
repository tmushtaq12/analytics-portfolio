from __future__ import annotations

import argparse
import shutil
import sqlite3
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "uci_online_retail"
WORKBOOK_PATH = RAW_DIR / "Online Retail.xlsx"
DATA_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
DATABASE_PATH = BASE_DIR / "data" / "processed" / "online_retail.sqlite"
RESULTS_DIR = BASE_DIR / "projects" / "retail-analytics" / "results"
IMAGE_PATH = BASE_DIR / "images" / "retail_analytics_dashboard.png"

SOURCE_COLUMNS = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price_gbp",
    "CustomerID": "customer_id",
    "Country": "country",
}


def download_source_data() -> Path:
    """Download the UCI archive and extract its workbook."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / "online_retail.zip"
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "analytics-portfolio/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as archive_file:
        shutil.copyfileobj(response, archive_file)

    with zipfile.ZipFile(archive_path) as archive:
        workbook_member = next(
            name for name in archive.namelist() if name.lower().endswith(".xlsx")
        )
        with archive.open(workbook_member) as workbook_source, WORKBOOK_PATH.open("wb") as workbook_file:
            shutil.copyfileobj(workbook_source, workbook_file)
    return WORKBOOK_PATH


def load_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load and validate source fields, retaining questionable rows with flags."""
    workbook_path = data_path or (WORKBOOK_PATH if WORKBOOK_PATH.exists() else download_source_data())
    source = pd.read_excel(workbook_path, engine="openpyxl")
    missing_columns = set(SOURCE_COLUMNS) - set(source.columns)
    if missing_columns:
        raise ValueError(f"Workbook is missing required columns: {sorted(missing_columns)}")

    data = source.rename(columns=SOURCE_COLUMNS)
    data["invoice_no"] = data["invoice_no"].astype("string").str.strip()
    data["stock_code"] = data["stock_code"].astype("string").str.strip()
    data["description"] = data["description"].astype("string").str.strip()
    data["customer_id"] = pd.to_numeric(data["customer_id"], errors="coerce").astype("Int64").astype("string")
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["unit_price_gbp"] = pd.to_numeric(data["unit_price_gbp"], errors="coerce")
    data["line_amount_gbp"] = data["quantity"] * data["unit_price_gbp"]

    data["is_cancellation_invoice"] = data["invoice_no"].str.upper().str.startswith("C").fillna(False)
    data["is_exact_duplicate"] = data.duplicated(subset=list(SOURCE_COLUMNS.values()), keep=False)
    data["is_duplicate_excess"] = data.duplicated(subset=list(SOURCE_COLUMNS.values()), keep="first")
    data["is_credit_line"] = (
        (data["is_cancellation_invoice"] | data["quantity"].lt(0))
        & ~data["is_duplicate_excess"]
    )
    data["is_positive_sale_line"] = (
        ~data["is_cancellation_invoice"]
        & ~data["is_duplicate_excess"]
        & data["quantity"].gt(0)
        & data["unit_price_gbp"].gt(0)
        & data["line_amount_gbp"].notna()
    )
    return data


def score_quantile(values: pd.Series, labels: list[int]) -> pd.Series:
    """Assign stable quantile scores even when many customers share a value."""
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels).astype(int)


def build_analysis(data: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, int | float | str]]:
    """Calculate sales, market, customer, cohort, and data-quality views."""
    sales = data.loc[data["is_positive_sale_line"]].copy()
    identified_sales = sales.dropna(subset=["customer_id"]).copy()

    sales["month"] = sales["invoice_date"].dt.to_period("M").astype(str)
    monthly_sales = (
        sales.groupby("month", as_index=False)
        .agg(
            gross_sales_gbp=("line_amount_gbp", "sum"),
            invoices=("invoice_no", "nunique"),
            customers=("customer_id", "nunique"),
            units=("quantity", "sum"),
            sales_lines=("invoice_no", "size"),
        )
        .sort_values("month")
    )

    country_sales = (
        sales.groupby("country", as_index=False)
        .agg(
            gross_sales_gbp=("line_amount_gbp", "sum"),
            invoices=("invoice_no", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .sort_values("gross_sales_gbp", ascending=False)
    )

    product_sales = sales.assign(
        description=sales["description"].fillna("(Missing description)")
    ).groupby(["stock_code", "description"], as_index=False).agg(
        gross_sales_gbp=("line_amount_gbp", "sum"),
        units=("quantity", "sum"),
        invoices=("invoice_no", "nunique"),
    ).sort_values("gross_sales_gbp", ascending=False)

    credits = data.loc[data["is_credit_line"]].copy()
    credits["month"] = credits["invoice_date"].dt.to_period("M").astype(str)
    credit_activity = credits.groupby("month", as_index=False).agg(
        credit_lines=("invoice_no", "size"),
        signed_credit_value_gbp=("line_amount_gbp", "sum"),
        cancellation_invoice_lines=("is_cancellation_invoice", "sum"),
    ).sort_values("month")

    customer_activity = identified_sales.groupby("customer_id").agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        invoice_count=("invoice_no", "nunique"),
        monetary_gbp=("line_amount_gbp", "sum"),
    ).reset_index()
    reference_date = data["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    customer_activity["recency_days"] = (
        reference_date - customer_activity["last_purchase"].dt.normalize()
    ).dt.days
    customer_activity["recency_score"] = score_quantile(
        -customer_activity["recency_days"], [1, 2, 3, 4, 5]
    )
    customer_activity["frequency_score"] = score_quantile(
        customer_activity["invoice_count"], [1, 2, 3, 4, 5]
    )
    customer_activity["monetary_score"] = score_quantile(
        customer_activity["monetary_gbp"], [1, 2, 3, 4, 5]
    )
    customer_activity["rfm_score"] = customer_activity[
        ["recency_score", "frequency_score", "monetary_score"]
    ].sum(axis=1)
    customer_activity["segment"] = np.select(
        [
            customer_activity["rfm_score"].ge(13),
            customer_activity["frequency_score"].ge(4) & customer_activity["monetary_score"].ge(3),
            customer_activity["recency_score"].ge(4) & customer_activity["frequency_score"].le(2),
            customer_activity["recency_score"].le(2) & customer_activity["frequency_score"].ge(3),
        ],
        ["Champions", "Loyal", "Promising", "At risk"],
        default="Other",
    )
    customer_activity = customer_activity.sort_values("monetary_gbp", ascending=False)
    segment_summary = customer_activity.groupby("segment", as_index=False).agg(
        customers=("customer_id", "nunique"),
        gross_sales_gbp=("monetary_gbp", "sum"),
        average_invoice_count=("invoice_count", "mean"),
    ).sort_values("gross_sales_gbp", ascending=False)

    cohort_source = identified_sales.assign(
        order_month=identified_sales["invoice_date"].dt.to_period("M"),
    )
    cohort_source["cohort_month"] = cohort_source.groupby("customer_id")["invoice_date"].transform("min").dt.to_period("M")
    cohort_source["cohort_index"] = (
        (cohort_source["order_month"].dt.year - cohort_source["cohort_month"].dt.year) * 12
        + cohort_source["order_month"].dt.month
        - cohort_source["cohort_month"].dt.month
    )
    cohort_counts = cohort_source.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique()
    cohort_sizes = cohort_counts.groupby(level=0).first()
    cohort_retention = cohort_counts.div(cohort_sizes, level=0).unstack().sort_index()
    cohort_retention.index = cohort_retention.index.astype(str)

    duplicate_excess_rows = int(data["is_duplicate_excess"].sum())
    quality_summary = {
        "source_rows": len(data),
        "positive_sales_lines": int(data["is_positive_sale_line"].sum()),
        "gross_positive_sales_gbp": round(float(sales["line_amount_gbp"].sum()), 2),
        "valid_invoices": int(sales["invoice_no"].nunique()),
        "identified_customers": int(identified_sales["customer_id"].nunique()),
        "countries": int(data["country"].nunique()),
        "products": int(data["stock_code"].nunique()),
        "missing_customer_id_rows": int(data["customer_id"].isna().sum()),
        "missing_description_rows": int(data["description"].isna().sum()),
        "cancellation_invoice_rows": int(data["is_cancellation_invoice"].sum()),
        "negative_quantity_rows": int(data["quantity"].lt(0).sum()),
        "nonpositive_price_rows": int(data["unit_price_gbp"].le(0).sum()),
        "exact_duplicate_excess_rows": duplicate_excess_rows,
        "credit_line_value_gbp": round(
            float(data.loc[data["is_credit_line"], "line_amount_gbp"].sum()), 2
        ),
        "start_date": data["invoice_date"].min().strftime("%Y-%m-%d"),
        "end_date": data["invoice_date"].max().strftime("%Y-%m-%d"),
    }

    return {
        "monthly_sales": monthly_sales,
        "country_sales": country_sales,
        "product_sales": product_sales,
        "credit_activity": credit_activity,
        "customer_activity": customer_activity,
        "segment_summary": segment_summary,
        "cohort_retention": cohort_retention,
        "quality_summary": quality_summary,
    }


def save_dashboard(analysis: dict[str, pd.DataFrame | dict[str, int | float | str]]) -> None:
    """Create a four-panel dashboard from the already-defined project metrics."""
    monthly = analysis["monthly_sales"]
    countries = analysis["country_sales"]
    products = analysis["product_sales"]
    cohorts = analysis["cohort_retention"]
    assert isinstance(monthly, pd.DataFrame)
    assert isinstance(countries, pd.DataFrame)
    assert isinstance(products, pd.DataFrame)
    assert isinstance(cohorts, pd.DataFrame)

    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    figure.suptitle("Online Retail | Transaction Analysis", fontsize=19, fontweight="bold")

    axes[0, 0].plot(monthly["month"], monthly["gross_sales_gbp"], color="#087f8c", marker="o", linewidth=2)
    axes[0, 0].set_title("Recorded positive sales by month")
    axes[0, 0].set_ylabel("GBP")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(axis="y", alpha=0.25)

    top_countries = countries.head(10).sort_values("gross_sales_gbp")
    axes[0, 1].barh(top_countries["country"], top_countries["gross_sales_gbp"], color="#e09f3e")
    axes[0, 1].set_title("Top countries by recorded sales")
    axes[0, 1].set_xlabel("GBP")

    top_products = products.head(10).sort_values("gross_sales_gbp")
    product_labels = top_products["description"].str.slice(0, 34)
    axes[1, 0].barh(product_labels, top_products["gross_sales_gbp"], color="#457b9d")
    axes[1, 0].set_title("Top products by recorded sales")
    axes[1, 0].set_xlabel("GBP")

    cohort_values = np.ma.masked_invalid(cohorts.iloc[:, :12].to_numpy(dtype=float))
    heatmap = axes[1, 1].imshow(cohort_values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[1, 1].set_title("Customer retention by cohort")
    axes[1, 1].set_xlabel("Months since first purchase")
    axes[1, 1].set_ylabel("First-purchase month")
    axes[1, 1].set_xticks(range(cohort_values.shape[1]))
    axes[1, 1].set_yticks(range(len(cohorts.index)))
    axes[1, 1].set_yticklabels(cohorts.index, fontsize=7)
    figure.colorbar(heatmap, ax=axes[1, 1], label="Share of cohort purchasing")

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(IMAGE_PATH, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_outputs(data: pd.DataFrame, analysis: dict[str, pd.DataFrame | dict[str, int | float | str]]) -> None:
    """Write CSV artifacts and the SQLite table used by the SQL project."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    quality = analysis["quality_summary"]
    assert isinstance(quality, dict)
    pd.DataFrame([{"metric": key, "value": value} for key, value in quality.items()]).to_csv(
        RESULTS_DIR / "data_quality_summary.csv", index=False
    )

    result_files = {
        "monthly_sales": "monthly_sales.csv",
        "country_sales": "country_sales.csv",
        "product_sales": "product_sales.csv",
        "credit_activity": "credit_activity.csv",
        "customer_activity": "customer_rfm.csv",
        "segment_summary": "customer_segments.csv",
    }
    for name, filename in result_files.items():
        result = analysis[name]
        assert isinstance(result, pd.DataFrame)
        result.to_csv(RESULTS_DIR / filename, index=False)

    cohorts = analysis["cohort_retention"]
    assert isinstance(cohorts, pd.DataFrame)
    cohorts.to_csv(RESULTS_DIR / "cohort_retention.csv", index_label="cohort_month")

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        data.to_sql("order_lines", connection, if_exists="replace", index=False, chunksize=25_000)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_order_lines_invoice_date ON order_lines(invoice_date)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_order_lines_customer_id ON order_lines(customer_id)")

    save_dashboard(analysis)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the UCI Online Retail transactions.")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Use a local workbook instead of downloading the official UCI source.",
    )
    args = parser.parse_args()

    data = load_data(args.data_path)
    analysis = build_analysis(data)
    save_outputs(data, analysis)

    quality = analysis["quality_summary"]
    assert isinstance(quality, dict)
    print("UCI Online Retail analysis")
    print("=" * 60)
    for metric, value in quality.items():
        print(f"{metric}: {value}")
    country_sales = analysis["country_sales"]
    assert isinstance(country_sales, pd.DataFrame)
    print("\nTop 5 countries by recorded positive sales:")
    print(country_sales.head().to_string(index=False))
    print(f"\nDashboard: {IMAGE_PATH.relative_to(BASE_DIR)}")
    print(f"SQL database: {DATABASE_PATH.relative_to(BASE_DIR)}")
    print(f"CSV outputs: {RESULTS_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
