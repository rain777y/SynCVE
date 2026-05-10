# SynCVE Clinical Use Cases

> Three concrete clinical-interview scenarios that motivate the event-level
> emotion analysis pipeline. Each use case binds an algorithmic capability
> (Axis 1A event detector + Axis 1C uncertainty-aware fusion) to a clinical
> outcome a clinician can act on. These are the scenarios the system is
> evaluated against in `eval/event_eval.py` and the ones the demo answers to
> the supervisor's prompt: *"When the doctor asks 'do you hate him?' — what
> can your algorithm let the doctor see that the naked eye cannot?"*

---

## Use Case 1 — Trauma-Interview Trigger-Word Micro-Expression Capture

### Scenario

A clinician conducts a trauma-history interview. Patients commonly suppress
explicit affective response to high-salience words (perpetrator name, place,
event), but residual leakage in the form of a 50–200 ms micro-expression is a
known clinical phenomenon (Ekman, *Telling Lies*, 1985; Yan et al.,
*PLoS ONE* 2014).

### Inputs

- Live or recorded video stream.
- Optional ASR transcript with word-level timestamps (Whisper / Deepgram).
- A clinician-curated **trigger lexicon** for the session
  (e.g. `["father", "house", "uncle"]`).

### Pipeline

1. Per-frame ensemble inference produces a 7-class emotion probability vector
   `p_t ∈ ℝ^7`.
2. EMA smoothing yields `p̂_t`.
3. `EventDetector` runs three change-point methods (sliding-window z-score,
   CUSUM, PELT-RBF on `p̂_t`) and emits *consensus events* with
   `(t_start, t_end, from_emo, to_emo, magnitude, confidence, methods_agreeing)`.
4. For each trigger word at time `T_w` in the ASR stream, a window
   `[T_w, T_w + reaction_latency_max_sec]` is scanned for events.
5. Matched (trigger, event) pairs are surfaced with **Reaction Latency** in
   milliseconds and the **Suppression Index** (whether the event was
   immediately followed by a regression to neutral within `refractory_frames`).

### Output (clinician-readable)

```text
[14:32:08]  trigger: "father"
            event:   neutral → disgust  (Δp = 0.31)
            latency: 184 ms
            confidence: 0.79  (3/3 methods agreed)
            suppression: yes — returned to neutral within 320 ms
```

### Clinical significance

- 184 ms latency falls within the canonical 50–200 ms micro-expression window.
- Suppression suggests **conscious affect masking** rather than spontaneous
  expression, which is clinically distinct.
- The event would have been undetectable by the clinician because of the
  short duration and the patient's overall composed affect.

### Literature anchor

- Ekman & Friesen, *Nonverbal Leakage and Clues to Deception*, 1969.
- Yan W., Wu Q., Liu Y., Wang S., Fu X., *CASME II*, PLoS ONE 2014.
- Yang Z. et al., *AI-based recognition of facial and micro-expressions for
  diagnosis of mental and neurological disorders*, BMC Psychiatry 2025.

---

## Use Case 2 — Depression Screening: Affect Blunting Quantification

### Scenario

In a structured depression screening interview (e.g. PHQ-9 administered as a
clinical interview, MINI-International Neuropsychiatric Interview), reduced
range and reactivity of affect is a core diagnostic feature
(*affective blunting / restricted affect*, DSM-5).
Clinicians estimate this qualitatively; the system provides a quantitative,
session-level score.

### Inputs

- Continuous video of the patient during the structured interview.
- Optional segmentation of question/answer turns from ASR.

### Pipeline

1. Per-frame emotion probability vector with **uncertainty-aware fusion**
   (Axis 1C): each detector backend's contribution is inversely weighted by
   the entropy of its softmax distribution, so under-confident backends do
   not pull the consensus toward chance.
2. `clinical_metrics.compute_affect_blunting`:

       ABS = 1 − (σ(valence_t) / σ_baseline) × (range(valence_t) / range_baseline)

   where `valence_t = Σ_e v_e · p_e` and `v_e ∈ [-1, +1]` is the
   psychologically grounded valence map (configured in `settings.yml`).
3. **Reactivity** is computed as the count of consensus events per minute.
4. **Expressive range** is `max(p_e) − min(p_e)` per emotion across the session.

### Output

```text
Affect Blunting Score:        0.74   (range 0–1, higher = more blunted)
Reactivity:                   0.4 events/min  (cohort baseline ≈ 2.1)
Expressive Range (joy):       0.18   (cohort baseline ≈ 0.62)
Expressive Range (sadness):   0.81
Dominant valence trajectory:  -0.31 → -0.42  over 22 minutes  (drifting negative)
```

### Clinical significance

- ABS ≥ 0.7 + reactivity < 1/min flags candidate cases for clinician review.
- Asymmetric range (suppressed joy with intact sadness) is the canonical
  pattern in MDD as opposed to flat affect of psychosis spectrum.

### Literature anchor

- Bedi G. et al., *Automated analysis of free speech predicts psychosis onset
  in high-risk youths*, npj Schizophrenia 2015.
- Vijay S. et al., *Computerized analysis of facial expressions in serious
  mental illness*, Schizophrenia Research 2022.
- Hamm J. et al., *Automated facial action coding system for dynamic analysis
  of facial expressions in psychiatric disorders*, J Neurosci Methods 2011.

---

## Use Case 3 — Long-Interview Clinician-Fatigue Assist

### Scenario

In hour-long clinical interviews, clinician attention to subtle nonverbal cues
declines. The system serves as a **second observer** that flags candidate
events for clinician review during a post-session debrief — never replacing
the clinician's judgement, only widening the recall surface.

### Inputs

- Full session video.
- Session-level configuration: clinician-set sensitivity (loose / strict).

### Pipeline

1. Run the entire pipeline post-hoc.
2. Emit a **ranked event list**: events sorted by `magnitude × confidence`
   with rare-emotion bonus (events transitioning into low-frequency emotions
   like *disgust* or *fear* are upweighted).
3. Generate **PDF/Markdown clinical report** containing:
   - Session timeline thumbnail with event markers.
   - Top-K event list with timestamps, screen-grabs, and (if ASR available)
     the contextual transcript window.
   - Clinical metric summary (ABS, reactivity, drift).
   - Limitations and ethical disclaimer.

### Output (excerpt from a real report)

```markdown
## Top 5 Flagged Events

| # | Time     | Transition           | Δp   | Conf. | Context (ASR ±2s)                     |
|---|----------|----------------------|------|-------|---------------------------------------|
| 1 | 18:42.3  | neutral → fear       | 0.42 | 0.88  | "...the day my brother left for..."   |
| 2 | 32:11.7  | sad → anger          | 0.36 | 0.81  | "...he never said sorry, never..."    |
| 3 | 47:08.2  | neutral → disgust    | 0.34 | 0.79  | "...when she touched my hand..."      |
| 4 | 51:22.9  | happy → neutral      | 0.29 | 0.74  | "...I told her I forgive him..."      |
| 5 | 58:04.5  | neutral → surprise   | 0.27 | 0.71  | "...I didn't know that..."            |
```

### Clinical significance

- The clinician reviews 5 short clips instead of re-watching 60 minutes.
- The recall metric on **events the clinician missed live** is the headline
  quantitative result of the project (Axis 3 evaluation).

### Literature anchor

- Krumhuber E., Kappas A., Manstead A., *Effects of dynamic aspects of facial
  expressions: A review*, Emotion Review 2013.
- D'Mello S., Graesser A., *AutoTutor and Affective AutoTutor: Learning by
  Talking with Cognitively and Emotionally Intelligent Computers That Talk
  Back*, ACM TiiS 2012.

---

## Cross-cutting Ethical & Safety Constraints

1. **Probabilistic disclaimer** is shown on every report: emotion predictions
   are inferred from facial expression patterns and do not represent
   definitive measures of internal emotional states.
2. **Calibration warning**: results are calibrated to the cohort distribution
   the system was tuned on (see `eval/cohort_calibration.md` once produced).
   Cross-cohort generalisation is an open research question.
3. **Consent and storage**: video frames are stored only for the duration of
   the session in Supabase storage with TTL eviction; raw frames are not
   retained beyond `_SESSION_TTL_SECONDS = 30 * 60` unless the session is
   explicitly archived by the clinician.
4. **No diagnostic claim**: the system supports clinician decision-making but
   does not produce a diagnosis. UI copy and the PDF disclaimer enforce this.
