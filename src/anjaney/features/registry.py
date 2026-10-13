"""Unified feature matrix builder with strict column partitioning."""

import pandas as pd

from anjaney.features.lexical import LexicalExtractor
from anjaney.features.text_embed import TextEmbeddingExtractor
from anjaney.features.timing import TimingFeatureExtractor


class FeatureRegistry:
    FEATURE_GROUPS = ["text_embed", "timing", "lexical"]

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.text_extractor = TextEmbeddingExtractor(model_name=embedding_model)
        self.timing_extractor = TimingFeatureExtractor()
        self.lexical_extractor = LexicalExtractor()

    def extract_features(
        self,
        df_windows: pd.DataFrame,
        groups: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract requested feature families and return a combined feature DataFrame."""
        selected_groups = groups or self.FEATURE_GROUPS
        for g in selected_groups:
            if g not in self.FEATURE_GROUPS:
                raise ValueError(
                    f"Unknown feature group '{g}'. Supported groups: {self.FEATURE_GROUPS}"
                )

        df_out = df_windows.copy()

        # 1. Text Embeddings
        if "text_embed" in selected_groups:
            texts = df_windows["text"].tolist()
            emb_matrix = self.text_extractor.encode_batch(texts)
            dim = emb_matrix.shape[1]
            for col_idx in range(dim):
                df_out[f"embed_{col_idx}"] = emb_matrix[:, col_idx]

        # 2. Timing Features
        if "timing" in selected_groups:
            timing_rows = [
                self.timing_extractor.extract_window_features(row)
                for row in df_windows.to_dict(orient="records")
            ]
            df_timing = pd.DataFrame(timing_rows)
            for col in df_timing.columns:
                df_out[col] = df_timing[col].values

        # 3. Lexical Features
        if "lexical" in selected_groups:
            lexical_rows = [
                self.lexical_extractor.extract_features(text)
                for text in df_windows["text"].tolist()
            ]
            df_lexical = pd.DataFrame(lexical_rows)
            for col in df_lexical.columns:
                df_out[col] = df_lexical[col].values

        return df_out

    @classmethod
    def get_feature_columns(
        cls, df: pd.DataFrame, groups: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Return the column names associated with each feature group."""
        selected_groups = groups or cls.FEATURE_GROUPS
        col_map: dict[str, list[str]] = {}

        if "text_embed" in selected_groups:
            col_map["text_embed"] = [c for c in df.columns if c.startswith("embed_")]
        if "timing" in selected_groups:
            col_map["timing"] = [c for c in df.columns if c.startswith("timing_")]
        if "lexical" in selected_groups:
            col_map["lexical"] = [c for c in df.columns if c.startswith("lex_")]

        return col_map
