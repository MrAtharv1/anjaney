"""Interpretable lexical statistics: affective sentiment, pronoun usage, and lexical richness."""

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FIRST_PERSON_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}

ABSOLUTIST_WORDS = {
    "always",
    "never",
    "completely",
    "totally",
    "absolutely",
    "all",
    "none",
    "every",
    "nothing",
    "everything",
    "constant",
    "definitely",
    "entirely",
}


class LexicalExtractor:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def extract_features(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        total_tokens = len(tokens)
        if total_tokens == 0:
            return {
                "lex_word_count": 0.0,
                "lex_first_person_ratio": 0.0,
                "lex_absolutist_ratio": 0.0,
                "lex_type_token_ratio": 0.0,
                "lex_vader_compound": 0.0,
                "lex_vader_neg": 0.0,
                "lex_vader_pos": 0.0,
            }

        # 1. Pronoun and absolutist frequencies
        n_first_person = sum(1 for t in tokens if t in FIRST_PERSON_PRONOUNS)
        n_absolutist = sum(1 for t in tokens if t in ABSOLUTIST_WORDS)

        # 2. Lexical diversity (Type-Token Ratio)
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / total_tokens

        # 3. Sentiment analysis
        scores = self.vader.polarity_scores(text)

        return {
            "lex_word_count": float(total_tokens),
            "lex_first_person_ratio": float(n_first_person / total_tokens),
            "lex_absolutist_ratio": float(n_absolutist / total_tokens),
            "lex_type_token_ratio": float(ttr),
            "lex_vader_compound": float(scores["compound"]),
            "lex_vader_neg": float(scores["neg"]),
            "lex_vader_pos": float(scores["pos"]),
        }
