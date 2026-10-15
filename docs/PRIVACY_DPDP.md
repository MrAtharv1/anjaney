# Digital Personal Data Protection (DPDP) Act 2023 Alignment Matrix

| DPDP Principle | Architectural Implementation | Codebase Location |
| :--- | :--- | :--- |
| **Notice & Consent (Sec. 5 & 6)** | Granular consent flags (`share_aggregate`, `share_escalation`) required before accepting student data. | `apps/api/src/api/routes/api_v1.py:53` |
| **Purpose Limitation (Sec. 7)** | Zero free-text storage. Check-in transcripts are processed in-browser and never transmitted to campus servers. | `apps/web/src/lib/inference.ts:10` |
| **Data Minimization (Sec. 6)** | Database models lack free-text columns. Server logs only numerical quality scores and triage categories. | `apps/api/src/api/models/entities.py:27` |
| **Storage Limitation (Sec. 8)** | Device baseline state is encrypted at rest using WebCrypto AES-GCM; raw text is discarded immediately after scoring. | `apps/web/src/lib/crypto.ts:40` |
| **Right to Erasure (Sec. 12)** | `DELETE /api/v1/me/{pseudonymous_id}` removes all consent, aggregate, and escalation records for that student. | `apps/api/src/api/routes/api_v1.py:195` |
| **Protection Against Re-identification** | $k$-Anonymity suppression: Summary cells with fewer than 5 individuals ($k < 5$) are redacted. | `apps/api/src/api/routes/api_v1.py:128` |