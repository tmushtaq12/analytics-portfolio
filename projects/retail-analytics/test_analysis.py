import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis import build_analysis, load_data


class RetailAnalysisTests(unittest.TestCase):
    def test_sales_population_quality_flags_and_cohort_outputs(self):
        rows = []
        for customer_number in range(1, 7):
            rows.append(
                {
                    "InvoiceNo": f"I{customer_number}",
                    "StockCode": f"P{customer_number}",
                    "Description": f"Product {customer_number}",
                    "Quantity": customer_number,
                    "InvoiceDate": f"2011-01-{customer_number:02d}",
                    "UnitPrice": 10.0,
                    "CustomerID": 1000 + customer_number,
                    "Country": "United Kingdom",
                }
            )

        rows.extend(
            [
                {
                    "InvoiceNo": "R2",
                    "StockCode": "P2B",
                    "Description": "Repeat order",
                    "Quantity": 2,
                    "InvoiceDate": "2011-02-02",
                    "UnitPrice": 10.0,
                    "CustomerID": 1002,
                    "Country": "United Kingdom",
                },
                {
                    "InvoiceNo": "R3",
                    "StockCode": "P3B",
                    "Description": "Repeat order",
                    "Quantity": 3,
                    "InvoiceDate": "2011-02-03",
                    "UnitPrice": 10.0,
                    "CustomerID": 1003,
                    "Country": "United Kingdom",
                },
                {
                    "InvoiceNo": "I1",
                    "StockCode": "P1",
                    "Description": "Product 1",
                    "Quantity": 1,
                    "InvoiceDate": "2011-01-01",
                    "UnitPrice": 10.0,
                    "CustomerID": 1001,
                    "Country": "United Kingdom",
                },
                {
                    "InvoiceNo": "I11",
                    "StockCode": "P11",
                    "Description": "Unknown customer",
                    "Quantity": 2,
                    "InvoiceDate": "2011-02-04",
                    "UnitPrice": 10.0,
                    "CustomerID": None,
                    "Country": "France",
                },
                {
                    "InvoiceNo": "C1",
                    "StockCode": "P1",
                    "Description": "Cancellation",
                    "Quantity": -1,
                    "InvoiceDate": "2011-02-05",
                    "UnitPrice": 10.0,
                    "CustomerID": 1001,
                    "Country": "United Kingdom",
                },
                {
                    "InvoiceNo": "I12",
                    "StockCode": "P12",
                    "Description": "Return",
                    "Quantity": -1,
                    "InvoiceDate": "2011-02-06",
                    "UnitPrice": 10.0,
                    "CustomerID": 1002,
                    "Country": "United Kingdom",
                },
                {
                    "InvoiceNo": "I13",
                    "StockCode": "P13",
                    "Description": "Zero price",
                    "Quantity": 1,
                    "InvoiceDate": "2011-02-07",
                    "UnitPrice": 0.0,
                    "CustomerID": 1003,
                    "Country": "United Kingdom",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_directory:
            workbook_path = Path(temp_directory) / "retail.xlsx"
            pd.DataFrame(rows).to_excel(workbook_path, index=False)
            data = load_data(workbook_path)

        analysis = build_analysis(data)
        quality = analysis["quality_summary"]

        self.assertEqual(quality["source_rows"], 13)
        self.assertEqual(quality["positive_sales_lines"], 9)
        self.assertEqual(quality["gross_positive_sales_gbp"], 280.0)
        self.assertEqual(quality["valid_invoices"], 9)
        self.assertEqual(quality["identified_customers"], 6)
        self.assertEqual(quality["missing_customer_id_rows"], 1)
        self.assertEqual(quality["cancellation_invoice_rows"], 1)
        self.assertEqual(quality["negative_quantity_rows"], 2)
        self.assertEqual(quality["nonpositive_price_rows"], 1)
        self.assertEqual(quality["exact_duplicate_excess_rows"], 1)

        segments = analysis["segment_summary"]
        cohorts = analysis["cohort_retention"]
        self.assertEqual(int(segments["customers"].sum()), 6)
        self.assertEqual(cohorts.loc["2011-01", 0], 1.0)
        self.assertAlmostEqual(cohorts.loc["2011-01", 1], 2 / 6)


if __name__ == "__main__":
    unittest.main()
