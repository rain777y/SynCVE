# Human Baseline Protocol — Recall on Events Humans Miss

> The headline research result of SynCVE is **not** generic emotion-recognition
> accuracy (DeepFace caps around 52% on FER-class data — see
> `dev/reference/papers/PAPERS.md`). It is the system's **recall on
> affective events that an untrained human observer misses** in a clinical
> interview.
>
> This document specifies the protocol that turns that claim into a
> reproducible, defensible measurement.

---

## 1. Stimuli

A pool of ≥ 8 minutes of video is needed, sourced from one of:

- **Public micro-expression datasets** with frame-level annotation:
  CASME II (255 spontaneous micro-expressions), SAMM (159), SMIC (164).
  These provide gold-standard onset / apex / offset frames.
- **Public clinical interview corpora**: DAIC-WOZ (semi-structured
  depression interviews); annotation requires us to label trigger events
  ourselves but the linguistic context is rich.
- **Lab-recorded mock-interview**: 2 trained confederates roleplay a
  trauma-history interview with scripted trigger words; the patient is
  briefed to suppress reactions. Annotated by the experimenter directly.

For the FYP run, the order of preference is **CASME II → DAIC-WOZ →
self-recorded**. The self-recorded fallback ensures the protocol is
demonstrable even if dataset access requests do not return in time.

## 2. Annotators

- **N ≥ 5 untrained observers** (e.g., undergraduates from outside the
  CV / psychology programmes). Why ≥ 5: lets us compute Fleiss' κ with
  reasonable precision and average over individual variance.
- **1 expert annotator** (course supervisor or psychology-trained PhD).
  Acts as adjudicator for disagreements and as the **gold-standard
  baseline** the system is compared against.

Annotators must be naive to the system output; show *only* the video.

## 3. Annotation tool

Local web tool with:

1. Video player with frame-precise scrub.
2. **Event button**: clicking creates a marker at the current frame.
3. **Optional label**: brief free-text (e.g., "twitch around mouth", "eye
   widening").
4. CSV export per annotator with `frame_idx, t_sec, label`.

The tool is built on top of `TimelineView.jsx` with annotation mode
enabled (toggleable via a query param). Implementation lives in
`src/frontend/src/pages/Annotate.js`.

## 4. Procedure

1. Each video is shown to each annotator **once**, no rewind allowed —
   this approximates a clinician watching live. Total per-annotator
   commitment: ~12 min × video count.
2. Order is randomised per annotator to control for fatigue effects.
3. Between videos, annotators rest for ≥ 60 s.
4. The expert annotator runs the same procedure but with rewind
   allowed; produces the **gold-standard** event list.
5. The system runs the same videos offline and produces its event list
   under default settings *frozen at the start of the experiment*.

## 5. Measurements

For each video V and annotator a:

- `H_a(V)` = annotator a's event list.
- `H_consensus(V)` = events in `H_expert(V)` that ≥ 2 untrained annotators
  also flagged within `tolerance_frames`.
- `H_missed(V)` = events in `H_expert(V)` that **no** untrained
  annotator flagged within tolerance.
- `S(V)` = system's event list.

### Primary metric — recall on missed events

```
R_missed = | S(V) ∩ H_missed(V) |  /  | H_missed(V) |
```

This is the headline number. It directly answers the supervisor's
prompt: *what does the algorithm see that the human eye misses?*

### Secondary metrics

- **Standard event-level F1** of S(V) against `H_expert(V)` with
  `tolerance_frames` from `eval/event_eval.py`.
- **Inter-rater reliability** Fleiss' κ on the untrained annotators —
  bounds how trustworthy `H_consensus` is.
- **System–expert F1** against `H_expert(V)` to anchor the system at the
  expert level.
- **System–consensus F1** to ground the system against the easier
  "events humans agree on" target.

### Tolerance

- `tolerance_frames = 4` at 30 fps ≈ **133 ms** — within the canonical
  micro-expression duration window.
- All metrics are also reported at 8 frames (267 ms) and 12 frames
  (400 ms) to show the recall–tolerance curve.

## 6. Statistical reporting

- Bootstrap 95% CI on `R_missed` over videos (n ≥ 200 resamples).
- Per-video drill-down table: `(video_id, |H_expert|, |H_consensus|,
  |H_missed|, system_recall_on_missed, system_f1_vs_expert)`.
- Open-source `eval/reports/human_baseline.json` contains the raw
  per-video / per-annotator counts so the result is auditable.

## 7. Ethics & consent

- All recordings used in the FYP run **must** be either:
  (a) drawn from an IRB-approved public dataset whose terms permit this
      use, **or**
  (b) the lab-recorded mock-interview where confederates and the patient-
      role actor sign the consent form in `eval/forms/consent.md`.
- Annotator participation is voluntary, unpaid, and they can withdraw
  data at any point during the study. Their data is anonymised before
  storage (only annotator number is retained, no name / contact).
- The system's predictions are never shown to annotators during the
  procedure — preventing system-anchored bias.

## 8. Negative-result plan

If `R_missed` ≤ 0.10 across all videos, the project's narrative pivots
to **transparency** rather than **superhuman recall**: we report what
the system catches that humans don't *and* what it misses, framed as a
calibration study for clinician decision support. Even a null recall
result is a defensible scientific finding when accompanied by failure-
mode analysis (per-emotion breakdown, per-video difficulty, lighting /
angle confounds).

## 9. Output artefacts

After the protocol is run end-to-end, the following deliverables exist:

- `eval/data/raw/*.mp4` — videos (or pointers to dataset paths).
- `eval/data/annotations/<annotator>/*.csv` — per-annotator event lists.
- `eval/reports/human_baseline.json` — aggregated counts + statistics.
- `eval/reports/human_baseline.md` — human-readable summary report.
- A figure (matplotlib) for the paper showing `R_missed` per video with
  bootstrap CI bars.

These artefacts feed Section 4 (Evaluation) of the paper directly.
