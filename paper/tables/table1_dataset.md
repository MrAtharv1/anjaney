# Table 1: Dataset Partitioning, Demographics, and Window Counts

| Partition | Sessions ($N$) | Female ($N$, %) | Male ($N$, %) | PHQ-8 $\ge 10$ ($N$, %) | Windows ($N$) | Mean Windows / Part. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 107 | 63 (58.9%) | 44 (41.1%) | 30 (28.0%) | 3323 | 31.1 |
| **Dev (Disk)**$^*$ | 34 | 16 (47.1%) | 18 (52.9%) | 11 (32.4%) | 1251 | 36.8 |
| **Test** | 47 | 23 (48.9%) | 24 (51.1%) | 14 (29.8%) | 1739 | 37.0 |
| **Total** | 188 | 102 (54.3%) | 86 | 55 (29.3%) | 6313 | 33.6 |

\* *Note on Session 440: Official AVEC2017 protocol lists $N=35$ dev sessions. Session 440 server archive terminates mid-stream at byte 78,273,750 (inside `440_CLNF_hog.bin`) and lacks transcript, COVAREP, and FORMANT files. It is excluded from analysis per investigator directive, yielding $N=34$ dev sessions on disk.*
