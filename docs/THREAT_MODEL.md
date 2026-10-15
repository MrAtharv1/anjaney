# ANJANEY Threat Model

## Attack Vectors & Mitigations
1. **Compromised Campus Network / Man-in-the-Middle:**
   - *Risk:* Intercepting student check-in text.
   - *Mitigation:* Zero free-text leaves the client device. Inference runs locally in a browser Web Worker; network payloads carry only pseudonymous categories if opted in.
2. **Re-identification via Demographic Bucketing:**
   - *Risk:* Identifying high-risk students in small campus cohorts.
   - *Mitigation:* $k$-Anonymity suppression enforced at the database query level ($k \ge 5$).
3. **Database Compromise:**
   - *Risk:* Data leak from server breach.
   - *Mitigation:* Database tables store only pseudonymous hashes, numerical intervals, and timestamps.
4. **Local Device Inspection:**
   - *Risk:* Unauthorized physical access to student device.
   - *Mitigation:* IndexedDB stores only embedding centroids, encrypted with a non-extractable WebCrypto AES-GCM key.