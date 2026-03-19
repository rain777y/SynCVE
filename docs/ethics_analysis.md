# Ethics and Bias Analysis: SynCVE Emotion Recognition System

**USM Final Year Project 2025-2026**

---

## 1. Scientific Validity

### 1.1 The Ekman Universality Debate

SynCVE classifies facial expressions into seven categories (angry, disgust, fear, happy, neutral, sad, surprise) derived from Ekman and Friesen's (1978) Facial Action Coding System (FACS) and the universality hypothesis — the claim that basic emotions produce universal, cross-cultural facial expressions.

This hypothesis is scientifically contested:

- **Barrett et al. (2019)** conducted a comprehensive review and concluded that "the common view that emotional expressions are universal is not well supported by the evidence." Facial expressions vary with culture, context, and individual differences. The same facial configuration can signal different emotions in different cultural contexts, and the same emotion can produce different facial configurations.
- **Feldman Barrett (2017)** argues that emotions are constructed experiences, not hardwired biological categories, fundamentally challenging the theoretical basis of discrete emotion classification.
- **AffectNet annotator agreement** is only ~60% (Mollahosseini et al., 2019), indicating that even trained human annotators cannot reliably agree on emotion labels from facial images alone. If humans disagree on the ground truth, the ceiling for any automated system is inherently limited.

### 1.2 Cultural Mediation of Expression

The relationship between facial muscle movements and perceived emotion is culturally mediated:

- Display rules (Ekman, 1972) govern when and how emotions are expressed in social contexts.
- Some cultures actively suppress emotional expression in public (e.g., collectivist societies may mask negative emotions).
- Compound and mixed expressions (e.g., bittersweet, nervous excitement) do not fit cleanly into 7-class taxonomies.

**Implication for SynCVE:** The system classifies facial expression patterns, not internal emotional states. This distinction is critical and is reflected in our UI disclaimer: "Emotion predictions are probabilistic estimates based on facial patterns. They do not represent definitive measures of internal emotional states."

### 1.3 Microsoft's 2023 Retirement Decision

In June 2023, Microsoft retired its Azure Face API emotion detection capabilities, stating that "facial expressions do not reliably correspond to internal emotional states" and citing the potential for misuse in consequential settings. This decision from a major cloud provider validates the scientific concerns and sets an important industry precedent.

Microsoft specifically noted that:
- There is no consensus among researchers on the definition of "emotions."
- Labeling someone's emotional state based on facial appearance alone is problematic.
- The potential for misuse in hiring, insurance, and other high-stakes scenarios outweighs the benefits.

**Relevance:** SynCVE operates in the same technical domain. We must be transparent about inheriting these same fundamental limitations, even as a research tool.

---

## 2. Data Bias

### 2.1 FER2013 Limitations

The base model used by SynCVE (DeepFace's VGG-Face emotion module) was trained on FER2013 (Goodfellow et al., 2013). FER2013 has several documented limitations:

**Collection methodology.** Images were gathered through Google Image Search using 184 emotion-related keywords. This introduces multiple biases:
- **Linguistic bias:** English-language keywords yield images predominantly from English-speaking contexts.
- **Search engine bias:** Google's image ranking algorithms may surface certain demographics more than others.
- **Labeling bias:** Semi-automated FACS-based labeling produces noisy annotations with estimated human agreement of only 65-72%.

**Class imbalance.** The distribution is severely skewed:
- Happy: ~25% of samples (overrepresented)
- Disgust: ~1.6% of samples (severely underrepresented)
- This means the model has seen approximately 15x more "happy" training examples than "disgust" examples.

**Resolution.** At 48x48 grayscale, FER2013 images contain limited facial detail. Fine-grained muscle movements (Action Units in FACS terminology) are difficult to discern at this resolution, forcing the model to rely on coarse features.

**Demographic documentation.** FER2013 does not publish a demographic breakdown of its images. We do not know the age, gender, or ethnicity distribution of the training data. This makes it impossible to quantify representation gaps.

### 2.2 Demographic Performance Disparities

Buolamwini and Gebru (2018) demonstrated in their landmark "Gender Shades" study that commercial facial analysis systems show error rate disparities of up to 34.7% across intersections of gender and skin type. Specifically:
- Lighter-skinned males had the lowest error rates.
- Darker-skinned females had the highest error rates.
- The gap was consistent across all three commercial systems tested (Microsoft, IBM, Face++).

While their study focused on gender classification rather than emotion recognition, the underlying issue — biased training data leading to differential performance — applies directly to FER systems. Subsequent research has confirmed that emotion recognition systems show performance disparities across racial groups, with some emotions (particularly anger and fear) being disproportionately misattributed to certain demographic groups.

**SynCVE's position:** We inherit whatever biases exist in the DeepFace base model and FER2013 training data. We have not conducted a formal demographic bias audit due to the lack of demographic labels in our evaluation datasets. This is an acknowledged limitation.

### 2.3 The Broader Dataset Landscape

| Dataset | Size | Resolution | Demographic Diversity | Annotation Quality |
|---------|------|------------|----------------------|-------------------|
| FER2013 | 35,887 | 48x48 grayscale | Undocumented | ~65-72% agreement |
| AffectNet | 450K+ | Variable | Undocumented | ~60% agreement |
| RAF-DB | ~30,000 | Variable (in-the-wild) | Better diversity | ~40 annotators per image |

No widely-used FER benchmark provides comprehensive demographic metadata. This is a systemic problem in the field, not specific to SynCVE.

---

## 3. Regulatory Context

### 3.1 EU AI Act (Effective February 2025)

The EU AI Act is the first binding legislation in the world to specifically regulate emotion recognition technology:

**Article 5 — Prohibited AI Practices:**
- Emotion recognition systems are **prohibited** in workplaces and educational institutions.
- Fines of up to EUR 35 million or 7% of total worldwide annual turnover for violations.

**Key distinction:** The EU AI Act distinguishes between:
- **Emotion inference** (claiming to detect internal emotional states): regulated/prohibited in many contexts.
- **Expression detection** (detecting facial muscle configurations): less restricted, but still subject to transparency and consent requirements.

SynCVE operates in the expression detection space but must be careful not to overclaim. Calling our output "emotion" rather than "expression classification" could place us in the more restricted category.

**Implications for SynCVE:**
- The system must not be deployed for employee monitoring or student engagement tracking in the EU.
- Even for research use, GDPR consent requirements apply when processing facial data of EU subjects.
- Documentation (this document) must clearly state limitations and prohibited uses.

### 3.2 Industry Precedent

Beyond Microsoft's retirement, the broader industry trend is toward caution:
- **IBM** exited the facial recognition market entirely in 2020.
- **Google** offers only 4 emotion categories with categorical likelihood scores (conservative approach).
- **Hume AI** has developed a detailed ethical framework emphasizing empathy-focused design.
- **MorphCast** adopted privacy-first architecture with all processing in the browser (no server-side facial data).

### 3.3 Other Regulatory Developments

- **CCPA/CPRA (California):** Biometric information is classified as sensitive personal information requiring explicit consent.
- **GDPR (EU):** Emotion data is biometric/sensitive data under Article 9, requiring explicit consent.
- **Colorado AI Act:** Addresses algorithmic discrimination in high-risk AI systems.
- Multiple US cities and states have enacted or proposed facial recognition restrictions.

---

## 4. Consent and Privacy

### 4.1 Current Implementation

SynCVE implements a consent mechanism in the frontend:

**Consent banner** displayed before webcam access:
- Informs the user that the application will access the webcam.
- States that facial expressions will be analyzed using machine learning.
- Notes that session data will be stored for report generation.
- Includes the disclaimer: "Emotion predictions are probabilistic estimates based on facial patterns. They do not represent definitive measures of internal emotional states."
- Requires explicit click-through ("I Understand, Continue") before proceeding.
- Users can stop detection at any time.

### 4.2 Gaps in Current Implementation

The current consent mechanism has several limitations compared to GDPR-compliant informed consent:

1. **No granular consent options.** Users cannot consent to detection but decline storage, or consent to analysis but decline report generation. It is all-or-nothing.
2. **No data retention policy.** The consent banner does not specify how long session data will be retained or when it will be deleted.
3. **No withdrawal mechanism.** While users can stop detection, there is no in-app mechanism to request deletion of previously collected session data.
4. **No age verification.** The system does not verify that users are of legal age to consent to biometric data processing.
5. **No data export.** Users cannot download their own session data (GDPR right of access).
6. **Limited information.** The consent text does not detail what specific data is collected, where it is stored (Supabase), or who has access.

### 4.3 Data Flow

For transparency, the data flow in SynCVE is:

1. **Client-side:** Webcam frames are captured by the browser and sent as base64 images to the backend API.
2. **Server-side processing:** Frames are preprocessed, analyzed by DeepFace, and post-processed. Results are stored in Supabase.
3. **Storage:** Emotion scores, timestamps, session metadata, and (in "full" mode) keyframe images are stored in Supabase cloud storage.
4. **Report generation:** Aggregated metrics may be sent to Google Gemini API for text summary generation.
5. **Third-party data sharing:** Frame data is processed locally (no frames sent to third-party APIs), but session summaries are sent to Google Gemini for report generation.

---

## 5. Harm Mitigation

### 5.1 Technical Mitigations

**Disclaimers and framing.** The system UI consistently refers to "facial expression analysis" and includes the disclaimer that predictions do not represent internal emotional states. The consent banner is mandatory and cannot be bypassed.

**Confidence thresholds.** Predictions below 10% confidence are flagged as low-confidence. The noise floor filter suppresses scores below 10%, preventing the system from presenting uncertain predictions as definitive.

**Anti-spoofing.** Integrated anti-spoofing detection rejects presentation attacks (photos of faces on screens, printed images), preventing the system from being used to classify non-live subjects without their knowledge.

**No persistent surveillance.** The system operates in session-based mode. Users explicitly start and stop detection sessions. There is no background or continuous monitoring.

### 5.2 Research Positioning

SynCVE is positioned as a research and educational tool, not a commercial product or decision-support system:

- All documentation explicitly states that the system is for academic use only.
- The model card lists prohibited uses (employment screening, law enforcement, clinical diagnosis, education monitoring).
- Evaluation metrics are reported honestly, including per-class breakdowns that expose weaknesses (disgust class, demographic considerations).

### 5.3 Limitations of Current Mitigations

- **Disclaimers are not enforcement.** A text disclaimer cannot prevent downstream misuse by someone who copies or adapts the code.
- **Anti-spoofing is imperfect.** The integrated liveness detection has both false positives (rejecting legitimate users) and false negatives (failing to detect sophisticated attacks).
- **Research framing does not prevent harm.** Even in academic contexts, incorrectly labeled emotions can cause distress or reinforce stereotypes.
- **Open-source risk.** If SynCVE is released publicly, the code could be adapted for prohibited uses without the ethical guardrails.

---

## 6. Responsible Use Guidelines

### Guideline 1: Never treat predictions as ground truth

Emotion predictions are probabilistic estimates based on facial expression patterns. They have a measurable error rate (see model card metrics). Always communicate uncertainty to end users. Never present a prediction as "this person is feeling X."

### Guideline 2: Do not use in consequential decisions

Do not use SynCVE output as input to decisions affecting a person's employment, education, healthcare, legal status, insurance, or access to services. The scientific basis for such use is insufficient, and it is prohibited under the EU AI Act in many jurisdictions.

### Guideline 3: Obtain informed, voluntary consent

Before processing anyone's facial data:
- Explain what data is collected and how it is used.
- Provide a genuine option to decline without penalty.
- Offer the ability to stop at any time and request deletion of collected data.
- Do not process facial data of minors without parental consent.

### Guideline 4: Be transparent about limitations

When presenting or discussing SynCVE:
- Acknowledge that FER2013 has known limitations (noisy labels, class imbalance, limited demographic documentation).
- State that demographic performance disparities likely exist but have not been formally audited.
- Note that the universality hypothesis underlying 7-class emotion classification is scientifically contested.
- Report per-class metrics, not just aggregate accuracy.

### Guideline 5: Minimize data collection and retention

- Collect only the data necessary for the stated purpose.
- Delete session data when the research purpose has been fulfilled.
- Do not retain raw facial images longer than necessary for report generation.
- If deploying beyond the FYP context, implement a data retention policy with automatic deletion.

### Guideline 6: Monitor for misuse

If the system is made available to others:
- Include the model card and this ethics analysis as part of the distribution.
- Add terms of use that prohibit deployment in prohibited contexts.
- Monitor usage patterns for signs of misuse (e.g., bulk processing, integration into decision-making systems).
- Be prepared to restrict access if misuse is identified.

---

## 7. References

Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional Expressions Reconsidered: Challenges to Inferring Emotion from Human Facial Movements. *Psychological Science in the Public Interest*, 20(1), 1-68.

Bird & Bird (2024). What is an Emotion Recognition System under the EU's Artificial Intelligence Act? Legal briefing. https://www.twobirds.com/en/insights/2024/global/what-is-an-emotion-recognition-system-under-the-eus-artificial-intelligence-act-part-1

Buolamwini, J., & Gebru, T. (2018). Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification. *Proceedings of the 1st Conference on Fairness, Accountability and Transparency (FAT* 2018)*, PMLR 81:77-91.

Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Consulting Psychologists Press.

Goodfellow, I. J. et al. (2015). Challenges in Representation Learning: A Report on Three Machine Learning Contests. *Neural Networks*, 64, 59-63.

Li, S., Deng, W., & Du, J. (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild. *CVPR 2017*.

Microsoft (2023). Azure AI Face retirement notice. https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-face-detection

Mitchell, M. et al. (2019). Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* 2019)*.

Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019). AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild. *IEEE Transactions on Affective Computing*, 10(1), 18-31.

Serengil, S. I., & Ozpinar, A. (2021). HyperExtended LightFace: A Facial Attribute Analysis Framework. *2021 International Conference on Engineering and Emerging Technologies (ICEET)*, 1-4.

(2023). Ethical Considerations in Emotion Recognition Technologies: A Review of the Literature. *AI and Ethics*, Springer. https://link.springer.com/article/10.1007/s43681-023-00307-3

(2024). Not in My Face: Challenges and Ethical Considerations in Automatic Face Emotion Recognition Technology. *Machine Learning and Knowledge Extraction*, 6(4), 109.

(2024). The Price of Emotion: Privacy, Manipulation, and Bias in Emotional AI. *Business Law Today*, American Bar Association.

(2025). Ethical Considerations in Emotion Recognition Research. *Societies*, 7(2), 43.

EU AI Act, Article 5 — Prohibited AI Practices. https://artificialintelligenceact.eu/article/5/
