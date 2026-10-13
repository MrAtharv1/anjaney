"""Deterministic crisis lexicon evaluator. Bypasses ML scoring on self-harm / suicide mentions."""

import re
from typing import Any

CRISIS_LEXICON = [
    # English
    r"\bsuicid\w*",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\bself[\s-]?harm\b",
    r"\bcut myself\b",
    r"\boverdose\b",
    r"\bno reason to live\b",
    # Hinglish transliterations
    r"\bjaan deni hai\b",
    r"\bkhatam karna chahta\b",
    r"\bmar jana chahta\b",
    r"\bmarne ka dil\b",
]


class CrisisSafetyGuard:
    def __init__(self, custom_phrases: list[str] | None = None):
        phrases = CRISIS_LEXICON + (custom_phrases or [])
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in phrases]

    @staticmethod
    def _normalize(text: str) -> str:
        # Collapse 3+ repeated characters down to 1 (e.g., 'dieeeee' -> 'die', 'deniiii' -> 'deni')
        collapsed = re.sub(r"(.)\1{2,}", r"\1", text.lower())
        return " ".join(collapsed.split())

    def check_crisis(self, text: str) -> tuple[bool, str | None]:
        clean = self._normalize(text)
        for pattern in self.compiled_patterns:
            match = pattern.search(clean)
            if match:
                return True, match.group(0)
        return False, None

    def create_crisis_response(self, triggered_phrase: str | None = None) -> dict[str, Any]:
        """Produce deterministic crisis intervention card."""
        return {
            "crisis_triggered": True,
            "bypassed_ml_model": True,
            "matched_keyword": triggered_phrase,
            "helplines": [
                {
                    "name": "Tele-MANAS (Govt of India)",
                    "number": "14416",
                    "toll_free": "1800-891-4416",
                    "hours": "24x7",
                },
                {
                    "name": "KIRAN Mental Health Helpline",
                    "number": "1800-599-0019",
                    "hours": "24x7",
                },
                {
                    "name": "Campus Wellness Center",
                    "contact": "wellness@campus.edu",
                    "location": "Student Services Block",
                },
            ],
            "message": "We care about your safety. Free, confidential support is available right now.",
        }
