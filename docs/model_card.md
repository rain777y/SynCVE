# Model Card: SynCVE Emotion Recognition Pipeline

Following the framework of Mitchell et al. (2019), "Model Cards for Model Reporting."

---

## Model Details

### Overview

SynCVE is a **pipeline wrapper** over the DeepFace library's emotion recognition model, not a custom-trained model. The base model is a VGG-Face CNN fine-tuned on FER2013 for 7-class emotion classification. SynCVE adds preprocessing, multi-detector ensemble fusion, and temporal post-processing around this fixed base model.

### Components

| Component | Implementation | Trainable? |
|-----------|---------------|------------|
| Face Detection | RetinaFace + MTCNN + CenterFace (ensemble) | No (pretrained) |
| Emotion Recognition | DeepFace VGG-Face CNN | No (pretrained, fine-tuned on FER2013) |
| Preprocessing | Bicubic super-resolution + CLAHE normalization | No (deterministic) |
| Post-processing | EMA smoothing + noise floor + transition detection | No (configurable parameters) |
| Report Generation | Google Gemini API (text + multimodal) | No (API-based) |

### Development Context

- **Project:** USM Final Year Project (FYP), 2025-2026
- **Developer:** USM undergraduate student
- **Purpose:** Academic research and educational demonstration
- **License:** Academic use
- **Date:** March 2026
- **Version:** 1.0

### Base Model Citation

```
Serengil, S. I., & Ozpinar, A. (2021). HyperExtended LightFace: A Facial
Attribute Analysis Framework. ICEET 2021, 1-4.
```

---

## Intended Use

### Primary Intended Uses

- **Educational tool:** Demonstrating real-time facial expression recognition concepts in academic settings.
- **Research platform:** Providing a reproducible baseline for studying pipeline optimization effects on FER accuracy.
- **FYP evaluation artifact:** Documenting the empirical contribution of system-level optimization over a base model.

### Primary Intended Users

- Academic researchers in affective computing and computer vision.
- Students learning about facial expression recognition systems.
- The FYP examiner evaluating this work.

### Out-of-Scope Uses

The following uses are **explicitly not supported** and are potentially harmful:

- **Employment screening:** Using emotion predictions to evaluate job candidates.
- **Law enforcement:** Using emotion detection for threat assessment, interrogation, or surveillance.
- **Clinical diagnosis:** Using emotion predictions as input to mental health diagnosis or treatment decisions.
- **Education monitoring:** Using emotion detection to grade, evaluate, or penalize students based on perceived engagement.
- **Access control:** Denying services or access based on detected emotional state.
- **Marketing manipulation:** Using detected emotions to manipulate purchasing decisions in real-time.

These uses are explicitly prohibited because:
1. The scientific basis for inferring internal emotional states from facial expressions is contested (see Ethical Considerations below).
2. The base model (trained on FER2013) has known demographic biases.
3. The EU AI Act (effective February 2025) prohibits emotion recognition in workplaces and schools.
4. Microsoft retired its Azure emotion detection API in 2023 for these reasons.

---

## Factors

### Relevant Factors

The following factors are known to affect model performance and should be considered when interpreting results:

#### Demographic Factors

- **Age:** The FER2013 training dataset is not demographically balanced. Performance may vary across age groups, with potential underperformance for children and elderly subjects who are underrepresented in training data.
- **Gender:** Studies have documented differential performance across gender, with some emotion categories (e.g., anger) showing gender-dependent accuracy variations.
- **Ethnicity/Race:** FER2013 has documented demographic skew. Buolamwini and Gebru (2018) demonstrated systematic performance disparities in facial analysis systems across skin tones. SynCVE inherits these biases from its base model.
- **Cultural background:** Emotional expression varies across cultures. The Ekman universality hypothesis (that basic emotions have universal facial expressions) is scientifically contested. SynCVE's 7-class model assumes universality, which may not hold for all users.

#### Environmental Factors

- **Lighting:** CLAHE normalization mitigates some lighting variation, but extreme backlighting, colored lighting, or very low light conditions will degrade performance.
- **Resolution:** Super-resolution upscaling helps with low-resolution faces (< 256px), but there is a lower bound below which no preprocessing can recover sufficient detail.
- **Pose/Angle:** Performance degrades with non-frontal face poses. The multi-detector ensemble partially mitigates this (MTCNN handles varied angles better than RetinaFace for some cases).
- **Occlusion:** Masks, glasses, hair, and hands covering the face will degrade detection and recognition accuracy.
- **Distance from camera:** Affects effective face resolution and detection confidence.

#### Technical Factors

- **Webcam quality:** Consumer webcams vary significantly in resolution, frame rate, dynamic range, and noise characteristics.
- **Anti-spoofing sensitivity:** The integrated anti-spoofing check may reject legitimate users in certain lighting conditions (false positives) or fail to detect sophisticated presentation attacks (false negatives).

---

## Metrics

### Quantitative Metrics

All metrics are computed on standard test splits with seed fixed at 42.

#### FER2013 Test Set

| Metric | Baseline (raw DeepFace) | SynCVE Full Pipeline |
|--------|------------------------|---------------------|
| Overall Accuracy | `[PENDING: load from eval/results/baseline_fer2013.json]` | `[PENDING: load from eval/results/full_pipeline_fer2013.json]` |
| Weighted F1 | `[PENDING: baseline_fer2013 → weighted_avg_f1]` | `[PENDING: full_pipeline_fer2013 → weighted_avg_f1]` |
| Macro F1 | `[PENDING: baseline_fer2013 → macro_avg_f1]` | `[PENDING: full_pipeline_fer2013 → macro_avg_f1]` |
| Macro AUC | `[PENDING: baseline_fer2013 → macro_auc]` | `[PENDING: full_pipeline_fer2013 → macro_auc]` |

#### RAF-DB Test Set

| Metric | Baseline (raw DeepFace) | SynCVE Full Pipeline |
|--------|------------------------|---------------------|
| Overall Accuracy | `[PENDING: load from eval/results/baseline_rafdb.json]` | `[PENDING: load from eval/results/full_pipeline_rafdb.json]` |
| Weighted F1 | `[PENDING: baseline_rafdb → weighted_avg_f1]` | `[PENDING: full_pipeline_rafdb → weighted_avg_f1]` |
| Macro F1 | `[PENDING: baseline_rafdb → macro_avg_f1]` | `[PENDING: full_pipeline_rafdb → macro_avg_f1]` |

#### Per-Class Performance (FER2013, Full Pipeline)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| angry | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| disgust | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| fear | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| happy | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| neutral | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| sad | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| surprise | `[PENDING]` | `[PENDING]` | `[PENDING]` |

#### Latency

| Configuration | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|--------------|-----------|-------------|----------|----------|
| Single detector (RetinaFace) | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| Full ensemble (3 detectors) | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

#### Real-Time Stability

| Metric | Without Post-Processing | With EMA + Noise Floor |
|--------|------------------------|----------------------|
| Frame-to-Frame Jitter (std dev) | `[PENDING]` | `[PENDING]` |
| False Transition Rate | `[PENDING]` | `[PENDING]` |
| Stability Score (0-1) | `[PENDING]` | `[PENDING]` |

### Decision Thresholds

- **Confidence threshold:** Default 10% (configurable). Predictions below this threshold are flagged as low-confidence.
- **Noise floor:** Emotion scores below 10% are suppressed in post-processing.
- **Transition threshold:** Dominant emotion changes require a 15% confidence delta to be registered as a transition.

---

## Evaluation Data

### FER2013

- **Source:** Goodfellow et al. (2015), Kaggle
- **Size:** 3,589 test images (from 35,887 total)
- **Resolution:** 48x48 grayscale
- **Labels:** 7 emotions, auto-labeled via FACS from Google Image Search
- **Known issues:** Noisy labels (~65-72% estimated human agreement), severe class imbalance (disgust ~1.6%), low resolution, limited demographic diversity documentation

### RAF-DB

- **Source:** Li, Deng, & Du (2017)
- **Size:** 3,068 test images (from ~30,000 total)
- **Resolution:** Variable (in-the-wild)
- **Labels:** 7 basic emotions, each labeled by ~40 crowdsourced annotators
- **Quality:** Higher diversity than FER2013 in age, gender, ethnicity, pose, lighting, and occlusion

---

## Training Data

**SynCVE does not train a model.** The base emotion recognition model (DeepFace's VGG-Face emotion module) was trained on FER2013 by the DeepFace authors. We document the training data characteristics for transparency:

- **Dataset:** FER2013 (35,887 images total: 28,709 training, 3,589 validation, 3,589 test)
- **Resolution:** 48x48 grayscale
- **Collection method:** Automated Google Image Search with 184 emotion-related keywords
- **Labeling:** Semi-automated with FACS-based annotation
- **Known biases:**
  - Limited demographic documentation (no published breakdown by age, gender, or ethnicity)
  - Class imbalance: happy (~25%) vs. disgust (~1.6%)
  - Western-centric image sourcing (English-language search queries)
  - Low resolution limits facial detail available for learning

The face detection models (RetinaFace, MTCNN, CenterFace) were each trained on their respective datasets (primarily WIDER FACE) by their original authors. These are used as-is without fine-tuning.

---

## Ethical Considerations

### Scientific Validity of Emotion Recognition

The fundamental premise of facial emotion recognition — that internal emotional states can be reliably inferred from facial expressions — is scientifically contested:

- **Barrett (2019)** and colleagues have challenged the universality hypothesis, arguing that facial expressions are culturally mediated and context-dependent.
- **AffectNet** annotator agreement is only ~60%, suggesting that even humans cannot reliably agree on emotion labels from facial images alone.
- **Microsoft (2023)** retired its Azure emotion detection API, explicitly citing that "facial expressions do not reliably correspond to internal emotional states."

**SynCVE's position:** We classify facial expressions, not emotions. The 7-class output represents the model's best estimate of which prototypical expression pattern is present. Users must understand that this is a statistical pattern match, not a measurement of internal feeling.

### Demographic Bias

- FER2013's training data has limited demographic diversity documentation.
- Buolamwini and Gebru (2018) demonstrated that commercial facial analysis systems show error rate disparities of up to 34.7% across intersections of gender and skin type.
- SynCVE inherits whatever biases exist in the DeepFace base model and the FER2013 training data.
- **We have not conducted a formal demographic bias audit.** This is a limitation acknowledged in the experiment report.

### Regulatory Context

- **EU AI Act (effective February 2025):** Prohibits emotion recognition in workplaces and educational institutions across Europe. Fines up to EUR 35M or 7% of global revenue.
- **Microsoft Azure retirement (2023):** Industry precedent for withdrawing emotion AI from commercial availability.
- **GDPR:** Emotion data constitutes sensitive biometric data requiring explicit consent under Article 9.

### Privacy

- SynCVE processes webcam frames in real-time. Frames are sent to a backend server for analysis.
- Session data (emotion scores, timestamps) is stored in Supabase for report generation.
- Keyframe images may be stored temporarily for visual report generation.
- A consent banner is displayed before webcam access, but the consent mechanism is basic and does not meet the standard required for GDPR-compliant informed consent in production deployment.

### Potential for Harm

Even in academic/educational contexts:
- Incorrect emotion labels could cause distress if users believe the system is "reading their mind."
- Persistent monitoring could create a surveillance dynamic, even in voluntary use.
- Demographic performance disparities mean some users will receive systematically less accurate results.

---

## Caveats and Recommendations

### What this system can do

- Classify static images or video frames into 7 prototypical facial expression categories.
- Provide probability distributions across emotion classes with confidence scores.
- Smooth real-time output for more stable display through temporal post-processing.
- Generate session-level aggregate reports with emotion timelines and trends.

### What this system cannot do

- Detect what someone is actually feeling (internal emotional state).
- Reliably classify expressions across all demographic groups with equal accuracy.
- Work well in very low light, extreme angles, or heavy occlusion.
- Replace human judgment in any consequential decision.
- Detect deception, intent, or complex emotional states beyond the 7-class taxonomy.

### Recommendations for responsible use

1. **Always display the disclaimer** that predictions represent facial expression patterns, not internal emotional states.
2. **Never use predictions as sole input** to any decision affecting a person's opportunities, access, or wellbeing.
3. **Be transparent** about the system's limitations, including demographic bias and scientific contestation.
4. **Obtain informed consent** before processing anyone's facial data, with a genuine option to decline.
5. **Minimize data retention.** Delete session data when no longer needed for the stated purpose.
6. **Do not deploy in prohibited contexts** (employment, education monitoring, law enforcement) per EU AI Act and ethical guidelines.
7. **Interpret per-class metrics honestly.** Disgust and fear classes have low support and unreliable metrics; do not report aggregate accuracy without this context.

---

## References

- Mitchell, M. et al. (2019). Model Cards for Model Reporting. *FAT* 2019*.
- Serengil, S. I., & Ozpinar, A. (2021). HyperExtended LightFace: A Facial Attribute Analysis Framework. *ICEET 2021*.
- Goodfellow, I. J. et al. (2015). Challenges in Representation Learning. *Neural Networks*, 64, 59-63.
- Buolamwini, J., & Gebru, T. (2018). Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification. *FAT* 2018*.
- Barrett, L. F. et al. (2019). Emotional Expressions Reconsidered. *Psychological Science in the Public Interest*, 20(1), 1-68.
- Bird & Bird (2024). What is an Emotion Recognition System under the EU's Artificial Intelligence Act?
- Microsoft (2023). Azure AI Face retirement notice.
- Li, S., Deng, W., & Du, J. (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild. *CVPR 2017*.
- Deng, J. et al. (2020). RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild. *CVPR 2020*.
- Zhang, K. et al. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. *IEEE Signal Processing Letters*, 23(10).
