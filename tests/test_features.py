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


def test_acoustic_feature_extractor_vuv_and_scrub_masking():
    from anjaney.features.acoustic import AcousticFeatureExtractor, compute_window_acoustics

    # 10 frames total:
    # Frame 0-1: scrubbed (all zeros across all 74 cols)
    # Frame 2-4: unvoiced (VUV=0, F0=0, NAQ=0.99, MCEP_1=-10.0, F1=999.0)
    # Frame 5-9: voiced (VUV=1, F0=120.0, NAQ=0.15, QOQ=0.5, MCEP_1=-5.0, F1=500.0, F2=1500.0)
    n_frames = 10
    cov = np.zeros((n_frames, 74))
    form = np.zeros((n_frames, 5))

    # Frames 2-4 (unvoiced):
    cov[2:5, 0] = 0.0
    cov[2:5, 1] = 0.0
    cov[2:5, 2] = 0.99
    cov[2:5, 3] = 0.88
    cov[2:5, 11] = -10.0
    form[2:5, 0] = 999.0

    # Frames 5-9 (voiced):
    cov[5:10, 0] = 120.0
    cov[5:10, 1] = 1.0
    cov[5:10, 2] = 0.15
    cov[5:10, 3] = 0.50
    cov[5:10, 11] = -5.0
    form[5:10, 0] = 500.0
    form[5:10, 1] = 1500.0

    # Valid non-scrubbed frames: frames 2 to 9 (8 frames).
    # Voiced frames: frames 5 to 9 (5 frames).
    # Voiced fraction: 5 / 8 = 0.625
    feats = compute_window_acoustics(cov, form)

    assert np.isclose(feats["acoustic_voiced_fraction"], 5.0 / 8.0)
    assert np.isclose(feats["acoustic_f0_mean"], 120.0)
    assert np.isclose(feats["acoustic_naq_mean"], 0.15)
    assert np.isclose(feats["acoustic_qoq_mean"], 0.50)
    assert np.isclose(feats["acoustic_f1_mean"], 500.0)
    assert np.isclose(feats["acoustic_f2_mean"], 1500.0)
    assert feats["acoustic_f0_std"] == 0.0

    # Test registry extraction
    df_dummy = pd.DataFrame(
        [
            {
                "participant_id": "P_01",
                "window_idx": 0,
                "text": "test",
                "turn_starts": [0.0],
                "turn_ends": [1.0],
                "prev_interviewer_ends": [-1.0],
                "n_tokens": 1,
                "start_s": 0.0,
                "end_s": 1.0,
                "rel_position": 0.0,
            }
        ]
    )
    registry = FeatureRegistry()
    df_extracted = registry.extract_features(df_dummy, groups=["acoustic"])
    cols = FeatureRegistry.get_feature_columns(df_extracted, groups=["acoustic"])
    assert len(cols["acoustic"]) == 16
    for c in AcousticFeatureExtractor.FEATURE_COLUMNS:
        assert c in cols["acoustic"]
        assert c in df_extracted.columns


def test_acoustics_offline_only_never_in_bundle():
    """Verify acoustic features are strictly offline-only and never leaked to bundle.json or apps/web."""
    import json
    from pathlib import Path

    bundle_path = Path("data/processed/bundle/bundle.json")
    if bundle_path.exists():
        with open(bundle_path, encoding="utf-8") as f:
            bundle_data = json.load(f)
        feature_names = bundle_data.get("feature_names", [])
        acoustic_in_bundle = [f for f in feature_names if "acoustic" in f]
        assert len(acoustic_in_bundle) == 0, (
            f"Found acoustic features in bundle.json: {acoustic_in_bundle}"
        )

    # Also verify apps/web contains zero references to acoustic feature extraction
    web_dir = Path("apps/web")
    if web_dir.exists():
        for file in web_dir.rglob("*.ts*"):
            text = file.read_text(encoding="utf-8", errors="ignore")
            assert "acoustic_" not in text, f"Found acoustic feature reference in web code: {file}"
