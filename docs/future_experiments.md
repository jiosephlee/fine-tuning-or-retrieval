# Future Experiments

## Experiment 1: Split Aux View Transfer (Leakage-Proof Validation)

**Goal:** Validate that auxiliary views provide genuine learning benefits beyond leakage by showing cross-paper transfer.

**Design:** Train with aux views for only half the papers and test whether the other half (which never saw any aux views) also improve.

| Condition | Training data | Test on |
|-----------|--------------|---------|
| **Baseline** | Source only (all 6) | All 6 probes |
| **Full aux** | Source + aux views (all 6) | All 6 probes |
| **Split aux** | Source (all 6) + aux views (A, B, C only) | All 6 probes |

**Key comparison:** Split aux vs. Baseline on papers D, E, F. Any improvement there is pure transfer — no leakage possible.

**Bonus comparison:** Split aux vs. Full aux on A, B, C — tells you whether the model benefits from having aux views for *its own* paper vs. just having aux views in general.

**Design choices:**
- Which 3/3 split? Consider multiple random splits and average, or a principled split by topic similarity (put related papers on the same side to make transfer harder and the result more convincing).
- Run within ML papers first (most natural since papers share a domain). Cross-domain transfer (ML aux views helping medical probes) would be extraordinary evidence but probably too big a leap.
- Run for both fact and inference probes — if transfer shows up on fact probes too, that strengthens the story.

**Why this works:** If aux views for papers A, B, C improve probes on papers D, E, F — which never saw any aux views — leakage is impossible as an explanation. The improvement must come from transferable reasoning or understanding.
