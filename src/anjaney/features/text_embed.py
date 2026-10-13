"""Frozen sentence embedding extractor with persistent SHA-256 disk cache."""

import hashlib
from pathlib import Path

import numpy as np


class TextEmbeddingExtractor:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Path | str = "data/interim/embedding_cache",
        max_seq_length: int = 256,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_length = max_seq_length
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def _compute_cache_key(self, text: str) -> str:
        clean_text = text.strip()
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))
        hasher.update(b"::")
        hasher.update(clean_text.encode("utf-8"))
        return hasher.hexdigest()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text snippet, utilizing cache if available."""
        key = self._compute_cache_key(text)
        cache_file = self.cache_dir / f"{key}.npy"

        if cache_file.exists():
            return np.load(cache_file)

        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        embedding = embedding.astype(np.float32)

        np.save(cache_file, embedding)
        return embedding

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts with disk cache lookup."""
        embeddings: list[np.ndarray] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, text in enumerate(texts):
            key = self._compute_cache_key(text)
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                embeddings.append(np.load(cache_file))
            else:
                embeddings.append(None)  # type: ignore
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            model = self._get_model()
            batch_emb = model.encode(
                uncached_texts,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            for idx, text, emb in zip(uncached_indices, uncached_texts, batch_emb, strict=False):
                key = self._compute_cache_key(text)
                cache_file = self.cache_dir / f"{key}.npy"
                np.save(cache_file, emb)
                embeddings[idx] = emb

        return np.vstack(embeddings)
