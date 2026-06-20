"""Data analysis and big data analysis engine with ingestion, SQL/Pandas processing, reports, visualization, anomaly detection, and forecasting."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class AnalysisReport:
    summary: str
    schema: dict[str, str]
    row_count: int
    anomalies: list[dict]
    forecast: list[float]
    output_files: list[str]


class AnalyticsEngine:
    def ingest_csv(self, csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path)

    def ingest_db(self, db_path: str, query: str) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)

    def process_pandas(self, df: pd.DataFrame, ops: dict | None = None) -> pd.DataFrame:
        ops = ops or {}
        out = df.copy()
        if "dropna" in ops:
            out = out.dropna(subset=ops["dropna"])
        if "filter" in ops:
            col = ops["filter"]["column"]
            min_val = ops["filter"].get("min")
            if min_val is not None:
                out = out[out[col] >= min_val]
        if "groupby" in ops:
            gb = ops["groupby"]
            out = out.groupby(gb["by"], as_index=False).agg(gb["agg"])
        return out

    def process_sql(self, df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
        with sqlite3.connect(":memory:") as conn:
            df.to_sql("dataset", conn, index=False, if_exists="replace")
            return pd.read_sql_query(sql_query, conn)

    def detect_anomalies(self, df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> list[dict]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return []
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0:
            return []
        z_scores = (series - mean) / std
        anomaly_idx = z_scores[abs(z_scores) >= z_threshold].index
        anomalies = []
        for idx in anomaly_idx:
            anomalies.append({"index": int(idx), "value": float(df.loc[idx, column]), "z_score": float(z_scores.loc[idx])})
        return anomalies

    def forecast(self, df: pd.DataFrame, column: str, periods: int = 3) -> list[float]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 2:
            return [float(series.iloc[-1])] * periods if len(series) == 1 else []
        delta = series.diff().dropna().mean()
        current = float(series.iloc[-1])
        forecast_values = []
        for _ in range(periods):
            current += float(delta)
            forecast_values.append(round(current, 4))
        return forecast_values

    def visualize(self, df: pd.DataFrame, x_col: str, y_col: str, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(8, 4))
            plt.plot(df[x_col], df[y_col], marker="o")
            plt.title(f"{y_col} over {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        except ModuleNotFoundError:
            # Fallback artifact for minimal environments without matplotlib installed.
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(f"Visualization unavailable. Missing matplotlib. Columns: {x_col}, {y_col}")
        return output_path

    def auto_report_generate(
        self,
        df: pd.DataFrame,
        numeric_column: str,
        x_col: str,
        y_col: str,
        chart_output_path: str,
        forecast_periods: int = 3,
    ) -> AnalysisReport:
        anomalies = self.detect_anomalies(df, numeric_column)
        forecast_values = self.forecast(df, numeric_column, periods=forecast_periods)
        chart_path = self.visualize(df, x_col=x_col, y_col=y_col, output_path=chart_output_path)

        summary = (
            f"Dataset has {len(df)} rows and {len(df.columns)} columns. "
            f"Detected {len(anomalies)} anomalies in '{numeric_column}'. "
            f"Forecasted next {forecast_periods} values: {forecast_values}."
        )

        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return AnalysisReport(
            summary=summary,
            schema=schema,
            row_count=len(df),
            anomalies=anomalies,
            forecast=forecast_values,
            output_files=[chart_path],
        )
