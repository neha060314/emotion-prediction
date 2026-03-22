# ERROR ANALYSIS
## ARVYAX Emotional State Prediction

**Total misclassified (on training data, in-sample):** 103 / 1200 (8.6%)

> Note: These errors are from applying the final model (trained on full data) back to training data.
> The cross-validation F1 of ~0.63 reflects true generalization error.
> All 10 cases are real entries from the training dataset — no values fabricated.

---

## Case 1 — ID: 496

| Field | Value |
|---|---|
| **Text** | `"felt good for a moment"` |
| **Actual** | calm |
| **Predicted** | neutral |
| **Confidence** | 0.263 (top2: neutral=0.26 vs calm=0.26) |
| **Stress** | 1 | **Energy** | 2 | **Face** | tired_face | **Reflection** | clear |

**Why it failed:**
The text is only 5 words with a transient emotional qualifier ("for a moment"). This phrase is inherently ambiguous — it implies the feeling didn't persist, which could map to calm, neutral, or even mixed. TF-IDF cannot distinguish between "felt good for a moment" (calm with a qualifier) and genuinely neutral or mixed states.

**Missing signal:** Duration of the feeling. "For a moment" suggests it was fleeting — the model lacks temporal signal about emotional persistence.

**How to improve:** Add phrase-level sentiment duration features; train a sequence model (e.g., simple LSTM) that captures modifiers like "for a moment", "briefly", "not quite".

---

## Case 2 — ID: 508

| Field | Value |
|---|---|
| **Text** | `"At first felt good for a moment."` |
| **Actual** | restless |
| **Predicted** | calm |
| **Confidence** | 0.271 (top2: calm=0.27 vs restless=0.25) |
| **Stress** | 5 | **Energy** | 4 | **Face** | unknown | **Reflection** | conflicted |

**Why it failed:**
The text starts with "felt good" — a positive signal — while the actual label is **restless** with stress=5. The surface-level text is misleading; the person likely wrote about a brief positive phase before the actual emotional state took hold. The model picks up on the positive text cue ("good") and predicts calm.

**Missing signal:** The full arc of the journal entry. "At first" signals contrast; what follows is missing from the short text. Conflicted reflection quality was not weighted heavily enough.

**How to improve:** Add lexical contrast detection ("at first", "but then", "however") as engineered features. Give more weight to `reflection_quality == conflicted` as a prior for restless/mixed prediction.

---

## Case 3 — ID: 510

| Field | Value |
|---|---|
| **Text** | `"got distracted again"` |
| **Actual** | calm |
| **Predicted** | mixed |
| **Confidence** | 0.234 (top2: mixed=0.23 vs calm=0.23) |
| **Stress** | 5 | **Energy** | 1 | **Face** | calm_face | **Reflection** | conflicted |

**Why it failed:**
A strongly conflicted signal set. The metadata says stress=5, energy=1, reflection=conflicted — all pointing toward overwhelmed or mixed. But the face hint shows calm_face and the actual label is calm. The model reasonably predicts mixed because the metadata overwhelms the face-emotion signal. This may be a **noisy label** — a calm face with stress=5 and energy=1 is physiologically inconsistent.

**Missing signal:** The model has no way to know which signal (face or metadata) is more reliable. Ground truth labeling may be subjective here.

**How to improve:** Use label smoothing in training. Add uncertainty-aware training. Flag face-emotion vs stress contradiction as a meta-feature.

---

## Case 4 — ID: 511

| Field | Value |
|---|---|
| **Text** | `"honestly not much change"` |
| **Actual** | calm |
| **Predicted** | focused |
| **Confidence** | 0.244 (top2: focused=0.24 vs calm=0.22) |
| **Stress** | 4 | **Energy** | 2 | **Face** | tired_face | **Reflection** | conflicted |

**Why it failed:**
"Not much change" is a near-zero-signal phrase. "Honestly" is common across all emotional states in the training corpus. The model defaults to focused because energy=2 + stress=4 + no strong text signal creates ambiguity. The actual calm label with these physiological signals is counterintuitive — likely a noisy label.

**Missing signal:** Vocabulary is too generic. TF-IDF bigrams like "not much" and "much change" did not appear frequently enough in calm training examples.

**How to improve:** Add a "generic phrase" flag when TF-IDF vector L2-norm is below threshold → increases uncertain_flag and reduces confidence penalty in evaluation.

---

## Case 5 — ID: 512

| Field | Value |
|---|---|
| **Text** | `"kept thinking about work"` |
| **Actual** | calm |
| **Predicted** | restless |
| **Confidence** | 0.231 (top2: restless=0.23 vs calm=0.20) |
| **Stress** | 5 | **Energy** | 1 | **Face** | calm_face | **Reflection** | conflicted |

**Why it failed:**
"kept thinking about work" is semantically restless — it implies rumination. Yet the label is calm, possibly because the person resolved to accept it, or the journaling itself was calming. Stress=5 + energy=1 compounds the restless prediction. This is a genuine ambiguity where the surface semantics (restless) and final state (calm) diverge — the journal captures the thought, not the outcome.

**Missing signal:** Narrative resolution. The journal text stops mid-thought; there is no resolution phrase ("but then I let it go", "and that was okay").

**How to improve:** Fine-tune on longer texts that include resolution phrases. Detect absent resolution as a restless/mixed prior.

---

## Case 6 — ID: 519

| Field | Value |
|---|---|
| **Text** | `"for some reason okay session."` |
| **Actual** | restless |
| **Predicted** | calm |
| **Confidence** | 0.312 (top2: calm=0.31 vs restless=0.30) |
| **Stress** | 3 | **Energy** | 2 | **Face** | calm_face | **Reflection** | conflicted |

**Why it failed:**
"Okay session" is a near-neutral or mildly positive phrase. The hedge "for some reason" suggests uncertainty, but TF-IDF does not learn hedge-word semantics well. With calm_face and moderate stress, the model picks calm. The actual restless label with these signals is unexpected — possibly restless at the session start with mild resolution.

**Missing signal:** The phrase "for some reason" is a cognitive hedge indicating the speaker themselves don't fully understand their state. This is a marker of restless/mixed, not calm.

**How to improve:** Add hedge-word lexicon as binary feature ("for some reason", "not sure why", "oddly", "surprisingly"). These correlate with mixed/restless states.

---

## Case 7 — ID: 520

| Field | Value |
|---|---|
| **Text** | `"still a bit off tbh"` |
| **Actual** | focused |
| **Predicted** | restless |
| **Confidence** | 0.280 (top2: restless=0.28 vs focused=0.22) |
| **Stress** | 2 | **Energy** | 5 | **Face** | tired_face | **Reflection** | vague |

**Why it failed:**
"A bit off" semantically signals being not-quite-right — closer to restless or mixed. The actual label is focused, possibly because the person was able to work despite feeling off. Energy=5 supports focused but tired_face contradicts it. "tbh" (to be honest) is informal filler that adds no signal.

**Missing signal:** The text contradicts the label. "Still a bit off" does not semantically map to focused. This is likely a **label quality issue** — a focused person might describe their session as focused even when the pre-session state was "a bit off".

**How to improve:** Separate pre-session state from during-session state in the labeling schema. The label seems to capture session outcome (focused) while the text captures pre-session feeling (off).

---

## Case 8 — ID: 524

| Field | Value |
|---|---|
| **Text** | `"felt heavy"` |
| **Actual** | calm |
| **Predicted** | overwhelmed |
| **Confidence** | 0.245 (top2: overwhelmed=0.25 vs focused=0.18) |
| **Stress** | 5 | **Energy** | 2 | **Face** | tense_face | **Reflection** | vague |

**Why it failed:**
"Felt heavy" + stress=5 + energy=2 + tense_face is a near-perfect overwhelmed signal pattern. The model's overwhelmed prediction is arguably more plausible than the calm label. This is a strong candidate for a **mislabeled training example**.

**Missing signal:** None — the model's prediction is defensible. The ground truth label (calm) may be incorrect or may reflect a post-reflection state ("felt heavy but then calmed down") that is not present in the text.

**How to improve:** Apply label noise correction (e.g., confident learning) to identify and relabel suspicious samples where all signals contradict the label.

---

## Case 9 — ID: 530

| Field | Value |
|---|---|
| **Text** | `"it was fine"` |
| **Actual** | calm |
| **Predicted** | focused |
| **Confidence** | 0.249 (top2: focused=0.25 vs calm=0.19) |
| **Stress** | 2 | **Energy** | 4 | **Face** | none | **Reflection** | clear |

**Why it failed:**
"It was fine" is a maximally generic 3-word statement that appears across all emotional states in the corpus. Energy=4 + stress=2 creates a physiologically focused/active signature, which is why the model predicts focused. Calm vs focused with energy=4 is genuinely ambiguous.

**Missing signal:** Three-word texts have almost no discriminative TF-IDF signal. The model relies entirely on metadata, which in this case points to focused rather than calm.

**How to improve:** For very short texts (< 5 tokens after stopword removal), automatically raise uncertain_flag=1 and use a fallback rule based purely on metadata + face emotion.

---

## Case 10 — ID: 532

| Field | Value |
|---|---|
| **Text** | `"honestly not much change"` |
| **Actual** | calm |
| **Predicted** | overwhelmed |
| **Confidence** | 0.296 (top2: overwhelmed=0.30 vs calm=0.20) |
| **Stress** | 4 | **Energy** | 4 | **Face** | none | **Reflection** | vague |

**Why it failed:**
Same text as Case 4 ("honestly not much change") but different ID and different metadata — here stress=4 + vague reflection pushes the model toward overwhelmed. The actual label is again calm, which is very difficult to explain given these physiological signals.

**Missing signal:** This is almost certainly a noisy or ambiguous label. Stress=4 + energy=4 + vague reflection + generic text → calm is a difficult combination to learn reliably.

**How to improve:** De-duplicate or cluster semantically identical journal texts in the training set. Investigate if the same short text appears with multiple labels — this is a sign of label noise that degrades model training.

---

## Systemic Patterns

| Pattern | Count | Root Cause |
|---|---|---|
| Short text (< 6 words) | 7/10 | TF-IDF cannot extract signal; model relies on metadata |
| Conflicted reflection quality | 5/10 | Reflection label itself signals ambiguity |
| High stress + calm/neutral label | 4/10 | Physiological signals contradict emotional label |
| Face hint contradiction | 3/10 | Face emotion and text diverge; one is unreliable |
| Likely noisy labels | 3/10 | Cases 3, 8, 10 where model prediction seems more plausible |

## Recommended Improvements

1. **Short-text fallback:** If TF-IDF vector norm < threshold, override with metadata-only classifier
2. **Hedge-word features:** Manually engineer "for some reason", "a bit off", "not sure" as features
3. **Confident learning:** Apply label noise detection to remove/relabel ~3-5% of likely mislabeled samples
4. **Longer journals:** Collect entries with resolution phrases ("...but then", "eventually I felt")
5. **Reflection quality as prior:** When reflection_quality == conflicted, increase uncertain_flag probability
