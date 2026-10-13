"""Tests verifying feature extractors, caching behavior, and registry column partitioning."""

from pathlib import Path

import numpy as np
import pandas as pd

from anjaney.features.lexical import LexicalExtractor
from anjaney.features.registry import FeatureRegistry
from anjaney.features.text_embed import TextEmbeddingExtractor
from anjaney.features.timing import TimingFeatureExtractor


def test_lexical_extractor_features():
    extractor = LexicalExtractor()
    text = "I feel completely exhausted, and I cannot study anymore."
    features = extractor.extract_features(text)

    assert features["lex_word_count"] > 0
    assert features["lex_first_person_ratio"] > 0.0  # contains 'I'
    assert features["lex_absolutist_ratio"] > 0.0  # contains 'completely'
    assert features["lex_type_token_ratio"] > 0.0
    assert "lex_vader_compound" in features


def test_timing_feature_extraction():
    extractor = TimingFeatureExtractor()
    sample_window = {
        "turn_starts": [2.0, 8.0, 15.0],
        "turn_ends": [5.0, 11.0, 18.0],
        "prev_interviewer_ends": [1.0, 6.5, 14.0],
        "n_tokens": 25,
        "start_s": 2.0,
        "end_s": 18.0,
        "rel_position": 0.5,
    }
    feats = extractor.extract_window_features(sample_window)

    assert feats["timing_mean_turn_duration"] == 3.0
    # Latencies: (2.0-1.0)=1.0, (8.0-6.5)=1.5, (15.0-14.0)=1.0 -> mean 1.167
    assert 1.0 <= feats["timing_mean_latency"] <= 1.3
    assert feats["timing_speech_rate_wps"] > 0.0
    assert feats["timing_turns_per_minute"] > 0.0


def test_embedding_extractor_cache(tmp_path: Path):
    encoder = TextEmbeddingExtractor(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=tmp_path / "cache",
    )
    text = "This is a test check-in utterance."

    # First call: computes and caches
    emb1 = encoder.encode_text(text)
    assert emb1.shape == (384,)

    # Verify cache file exists
    cache_files = list((tmp_path / "cache").glob("*.npy"))
    assert len(cache_files) == 1

    # Second call: reads directly from cache
    emb2 = encoder.encode_text(text)
    assert np.allclose(emb1, emb2)


def test_registry_column_partitioning():
    df_dummy = pd.DataFrame(
        [
            {
                "participant_id": "P_01",
                "text": "Feeling good today.",
                "turn_starts": [0.0],
                "turn_ends": [2.0],
                "prev_interviewer_ends": [-1.0],
                "n_tokens": 3,
                "start_s": 0.0,
                "end_s": 2.0,
                "rel_position": 0.0,
            }
        ]
    )
    registry = FeatureRegistry()
    df_feats = registry.extract_features(df_dummy, groups=["timing", "lexical"])

    cols = FeatureRegistry.get_feature_columns(df_feats, groups=["timing", "lexical"])
    timing_cols = set(cols["timing"])
    lexical_cols = set(cols["lexical"])

    # Disjoint column partitions
    assert len(timing_cols.intersection(lexical_cols)) == 0
    assert len(timing_cols) > 0
    assert len(lexical_cols) > 0
