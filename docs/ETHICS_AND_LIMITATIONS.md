# Ethics, Clinical Stance & Technical Limitations

## Ethical Framework
1. **Screening Aid, Never Diagnostic Tool:**
   - ANJANEY produces risk trends to help students reflect and to assist wellness staff with queue prioritization. It does not diagnose clinical depression.
2. **Domain Shift Caveat:**
   - Models trained on semi-structured interview transcripts (DAIC-WOZ) may exhibit distribution shift when evaluated on informal campus check-ins.
3. **False Negative Mitigation:**
   - The conformal layer optimizes for conservative coverage ($\alpha = 0.10$). Ambiguous scores default to `ESCALATE_TO_HUMAN` rather than false reassurance.

## Structural Constraints
- Single-session check-ins capture situational state, not long-term trait depression.
- Passive timing features require keyboard input; accessibility alternatives (voice transcription) require dedicated calibration.