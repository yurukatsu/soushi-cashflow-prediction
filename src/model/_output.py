import tempfile

import mlflow
import plotly.graph_objects as go
import polars as pl


class FeatureImportance:
    col_feature = "feature"
    col_importance = "importance"

    def __init__(self, df: pl.DataFrame):
        if self.col_feature not in df.columns or self.col_importance not in df.columns:
            raise ValueError(
                "DataFrame must contain 'feature' and 'importance' columns."
            )

        self.df = df.select(
            pl.col("feature").cast(pl.Utf8), pl.col("importance").cast(pl.Float64)
        ).drop_nulls(["feature"])

    def get_figure(self) -> go.Figure:
        df = self.df.to_pandas().sort_values(self.col_importance, ascending=True)
        fig = go.Figure(
            go.Bar(
                x=df[self.col_importance],
                y=df[self.col_feature],
                orientation="h",
            )
        )
        fig.update_layout(
            title="Feature Importance",
            xaxis=dict(title="Importance"),
            yaxis=dict(title="Feature"),
        )
        return fig

    def save_figure(self, filepath: str):
        fig = self.get_figure()
        fig.write_html(filepath)

    def save_csv(self, filepath: str):
        self.df.write_csv(filepath)

    def log_to_mlflow(
        self, artifact_paths: dict[str | None, str | None] = {"html": None, "csv": None}
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            fig = self.get_figure()
            html_path = f"{temp_dir}/feature_importance.html"
            fig.write_html(html_path)
            mlflow.log_artifact(html_path, artifact_path=artifact_paths["html"])

            csv_path = f"{temp_dir}/feature_importance.csv"
            self.df.write_csv(csv_path)
            mlflow.log_artifact(csv_path, artifact_path=artifact_paths["csv"])
