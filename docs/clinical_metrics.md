# SynCVE Clinical Metrics — Definitions & Formulas

> All metrics are implemented in `src/backend/clinical_metrics.py` and consumed
> by the clinical-report endpoint. Each metric has a clear formula, a
> psychological literature anchor, and a defined output range so that
> downstream UI components and the PDF report can render them without
> guessing.

The system uses a 7-class emotion probability vector
`p_t = (p_angry, p_disgust, p_fear, p_happy, p_sad, p_surprise, p_neutral)`
at time `t`, with `Σ p_t = 1`. The valence map `v: Emotion → [-1, +1]` is
configured in `settings.yml::clinical.valence_map` and defaults to:

```yaml
happy:    +1.0
surprise: +0.5
neutral:   0.0
fear:     -0.7
sad:      -0.8
disgust:  -0.6
angry:    -0.9
```

These signed values are derived from a Russell circumplex + clinician
judgement; users can override per-cohort.

---

## 1. Valence at Time `t`

```
valence(t) = Σ_e v_e · p_e(t)
```

- Range: `[-1, +1]`
- Unit: dimensionless valence index.
- Interpretation: positive = pleasant affect; negative = unpleasant.

Used as the raw signal for **drift**, **blunting**, and **incongruence**.

---

## 2. Valence Drift

Long-horizon trend in the valence trace.

```
drift = OLS-slope( valence(t) over t ∈ [0, T] )
```

- Range: `[-1/T, +1/T]` (units: valence per frame).
- Reported in **valence per minute** for clinician readability:
  `drift_per_min = drift × fps × 60`.
- Bootstrap CI is computed via 200 resamples; reported as `drift ± 95% CI`.

A negative drift over a 30-minute interview suggests progressive
affective negativity, often relevant to depression / trauma contexts.

---

## 3. Affect Blunting Score (ABS)

Composite of variance contraction and range contraction over the session,
relative to a configured baseline (cohort or prior session).

```
σ_ratio   = σ_valence(session) / σ_valence(baseline)
range_ratio = range_valence(session) / range_valence(baseline)
ABS = 1 − clip(σ_ratio · range_ratio, 0, 1)
```

- Range: `[0, 1]` — higher = more blunted.
- Default baseline: cohort means computed from
  `eval/cohort_calibration.json` (a separate calibration run); when absent,
  defaults to `σ_baseline = 0.30, range_baseline = 1.20`.
- Asymmetric variant `ABS_per_emotion(e)` reports per-emotion contraction
  to distinguish flat affect (all suppressed) from MDD-typical pattern
  (joy suppressed, sadness intact).

Anchored in DSM-5 negative-symptom criteria (restricted affect) and
operationalised after Hamm et al. 2011 and Vijay et al. 2022.

---

## 4. Reactivity

Event count per minute.

```
reactivity = count(consensus_events) / session_duration_min
```

- Range: `[0, ∞)`.
- Cohort baseline (configurable): non-clinical adult, ≈ 2.1 events/min in a
  structured interview (placeholder until cohort calibration is run).
- Used jointly with ABS to distinguish **flat-but-stable** from
  **flat-and-reactive-but-suppressed** affective patterns.

---

## 5. Reaction Latency

For a clinician trigger word `w` at time `T_w` and the next consensus event
at `T_e`:

```
latency(w) = T_e − T_w   (only if 0 ≤ latency ≤ reaction_latency_max_sec)
```

- Unit: milliseconds.
- Range of clinical interest: `[50, 3000] ms`.
  - `[50, 200]` ms = canonical micro-expression window.
  - `(200, 1000]` ms = near-immediate reactive expression.
  - `(1000, 3000]` ms = delayed / processed reaction.
- Reported per trigger AND aggregated as `mean ± std` per session.

---

## 6. Suppression Index

Fraction of detected events that revert to neutral within
`refractory_frames`:

```
suppression_index = | events with t_neutral ≤ t_event + Δ | / | total events |
```

where `Δ = refractory_frames / fps` and `t_neutral` is the next frame whose
dominant emotion is neutral and `p_neutral > 0.5`.

- Range: `[0, 1]`.
- High suppression index in trauma interviews is a published correlate of
  active emotion regulation / masking (Gross J., *Handbook of Emotion
  Regulation*, 2014).

---

## 7. Affect Incongruence Index

Available **only when ASR with semantic valence labelling is provided**
(Phase 2; current pipeline emits a stub field). Defined as:

```
incongruence(window) = | valence_facial − valence_semantic | / 2
```

- Window width: `incongruence_window_sec` from settings.
- Range: `[0, 1]`.
- Interpretation: 0 = facial and verbal valence aligned; 1 = maximal mismatch.
- Reported as both per-window series and session-level mean.

Anchored in studies of nonverbal/verbal channel discrepancy in depression,
psychosis, and deceptive communication.

---

## 8. Event Confidence

Per-event confidence reported by `EventDetector`:

```
event_confidence =
    0.5 · (Δp_normalised)                    // magnitude evidence
  + 0.3 · (n_methods_agreeing / n_methods)   // method consensus
  + 0.2 · (1 − fused_entropy / log(7))       // fusion uncertainty
```

- Range: `[0, 1]`.
- Used to rank events in the Top-K list and to populate the sensitivity
  slider on the frontend.

---

## 9. Per-Detector Reliability (Axis 1C diagnostic)

For each ensemble backend `b` over a session:

```
reliability(b) = 1 − mean( H( p_b(t) ) / log(7) )
```

where `H(·)` is Shannon entropy. Reported in the report appendix to
diagnose which detector is contributing useful signal vs. noise. This is
the diagnostic that justifies the **uncertainty-aware fusion** replacing
the legacy 50/50 weighting.

---

## Implementation Notes

- All metrics handle the edge cases of `n < min_samples` (returned as
  `null` in the JSON, dashed in the PDF).
- Bootstrap CIs use a fixed seed for reproducibility (configurable).
- The valence map is the only psychologically anchored constant; everything
  else is derived. Per-cohort recalibration only requires updating
  `valence_map` and `*_baseline` in `settings.yml`.
- All numeric outputs are rounded to 4 decimals at the API boundary
  (`_to_json_safe` already in place in `session_manager.py`).
