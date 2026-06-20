import sqlite3
from pathlib import Path

import pandas as pd

from app.engines.analytics_engine import AnalyticsEngine


def test_csv_db_ingestion_and_processing(tmp_path: Path):
    engine = AnalyticsEngine()
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"category": ["a", "a", "b"], "value": [10, 20, 30]}).to_csv(csv_path, index=False)

    df_csv = engine.ingest_csv(str(csv_path))
    assert len(df_csv) == 3

    db_path = tmp_path / "data.db"
    with sqlite3.connect(db_path) as conn:
        df_csv.to_sql("records", conn, index=False, if_exists="replace")
    df_db = engine.ingest_db(str(db_path), "SELECT * FROM records")
    assert len(df_db) == 3

    grouped = engine.process_pandas(df_db, {"groupby": {"by": "category", "agg": {"value": "sum"}}})
    assert set(grouped["category"]) == {"a", "b"}

    sql_processed = engine.process_sql(df_db, "SELECT category, AVG(value) as avg_value FROM dataset GROUP BY category")
    assert "avg_value" in sql_processed.columns


def test_report_visualization_anomaly_and_forecast(tmp_path: Path):
    engine = AnalyticsEngine()
    df = pd.DataFrame(
        {
            "day": [1, 2, 3, 4, 5, 6],
            "sales": [100, 110, 120, 130, 500, 140],
        }
    )

    anomalies = engine.detect_anomalies(df, "sales", z_threshold=1.6)
    assert any(a["value"] == 500.0 for a in anomalies)

    forecast = engine.forecast(df, "sales", periods=2)
    assert len(forecast) == 2

    chart_path = tmp_path / "chart.png"
    report = engine.auto_report_generate(
        df,
        numeric_column="sales",
        x_col="day",
        y_col="sales",
        chart_output_path=str(chart_path),
        forecast_periods=2,
    )
    assert Path(report.output_files[0]).exists()
    assert "Dataset has" in report.summary
