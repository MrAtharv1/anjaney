# Datasheet: DAIC-WOZ & Synthetic Cohort

## Motivation & Data Collection
- **DAIC-WOZ Dataset:** Distributed by USC Institute for Creative Technologies for the AVEC challenges. Contains audio, video, and transcripts of semi-structured clinical interviews conducted by virtual interviewer Ellie.
- **Synthetic Cohort (ANJANEY Fallback):** 50 algorithmic sessions generated with correlated response latencies, linguistic markers, and simulated within-session fatigue to guarantee leak-proof development without violating DAIC EULA terms.

## Preprocessing & Acoustic Scrubbing
- **Annotations Stripped:** Acoustic markers (`<laughter>`, `<sigh>`, `<gasp>`, `<synch>`) and PII markers (`scrubbed_entry_*`) are purged by regex before windowing.
- **Participant-Only Text:** Interviewer utterances are excluded from all text embeddings to prevent conversational shortcut learning (Burdisso et al., 2024).