# ARVYAX Emotional Intelligence System

---

## Overview

A production-quality, fully local ML pipeline that:
1. **Understands** emotional state and intensity from reflective journal text + biometric signals
2. **Decides** what activity to recommend and when to do it
3. **Guides** with a short, human-like supportive message

No external APIs used. Runs entirely on-device.

---

## Approach

### Explicit Assumptions
| # | Assumption | Rationale |
|---|---|---|
| 1 | `intensity` treated as **classification** (not regression) | Values are discrete integers 1–5; fractional predictions (e.g. 2.7) have no semantic meaning |
| 2 | `face_emotion_hint` NaN → `'unknown'` category | 10%+ missing; informative missingness; dedicated OHE column |
| 3 | `previous_day_mood` NaN → `'unknown'` category | Same reasoning |
| 4 | `sleep_hours` NaN → training median imputation | Avoids data leakage; 0 NaN in test set |
| 5 | Confidence threshold = 0.45 | Random baseline for 6-class = 0.167; 0.45 catches near-tie cases reliably |
| 6 | Decision rules are domain-derived, not learned | Ensures explainability, determinism, and safety |

---

## Feature Engineering

### Text Features
- **TF-IDF** with 500 features, bigrams (1–2), min_df=2, sublinear TF scaling
- Captures vocabulary of emotional language: "drained", "calmer", "couldn't relax", etc.
- Text contributes ~95% of total XGBoost feature importance

### Metadata Features
- **Categorical** (OHE, handle_unknown='ignore'): `ambience_type`, `time_of_day`, `previous_day_mood`, `face_emotion_hint`, `reflection_quality`
- **Numeric** (StandardScaler): `duration_min`, `sleep_hours`, `energy_level`, `stress_level`
- Top metadata signals: `stress_level`, `energy_level`, `face_emotion_hint`, `reflection_quality`

---

## Model Choices

### Emotional State (6-class classification)
| Model | F1 (CV) | Notes |
|---|---|---|
| Logistic Regression (text only) | ~0.598 | Baseline, interpretable |
| XGBoost (text + metadata) | ~0.633 | Best performer, chosen for inference |

### Intensity (5-class ordinal classification)
| Model | F1 (CV) | MAE |
|---|---|---|
| Logistic Regression (text only) | ~0.230 | ~1.45 |
| XGBoost (text + metadata) | ~0.222 | ~1.52 |

**Why classification for intensity?** Values are always whole integers 1–5. Regression would yield meaningless fractional values (e.g. 3.2). Classification preserves the discrete nature and provides per-class probabilities useful for confidence estimation.

**Note on intensity MAE:** Both models show high MAE (~1.4–1.5), indicating intensity is weakly correlated with the available features. This is expected — intensity is subjective and likely requires richer sensor data or longer text for strong prediction.

---

## Ablation Study Results

| Metric | Text-Only (LogReg) | Text + Metadata (XGB) | Delta |
|---|---|---|---|
| State F1 | 0.5978 | 0.6325 | +0.0347 |
| State Acc | 0.5975 | 0.6333 | +0.0358 |

**Key finding:** Adding physiological metadata (stress, energy, sleep, face emotion) meaningfully improves state classification. Metadata alone cannot classify well, but it resolves ambiguous text — e.g., "felt okay" is neutral/calm from text alone, but if stress=5 and energy=1, the model correctly skews toward overwhelmed or restless.

---

## Decision Engine

A fully explicit, deterministic rule-based system. No ML. Pure Python conditionals.

### Activity Selection Rules

```
STATE        | INTENSITY | STRESS | ENERGY | ACTIVITY
─────────────────────────────────────────────────────
overwhelmed  | 4-5       | 4-5    | any    | box_breathing
overwhelmed  | 4-5       | 1-3    | any    | grounding
overwhelmed  | 1-3       | any    | any    | journaling
restless     | 4-5       | 4-5    | any    | box_breathing
restless     | 4-5       | 1-3    | 3-5    | movement
restless     | 4-5       | 1-3    | 1-2    | yoga
restless     | 1-3       | any    | any    | journaling
focused      | any       | 1-3    | 3-5    | deep_work
focused      | any       | 4-5    | any    | light_planning
focused      | any       | any    | 1-2    | rest
calm         | any       | 4-5    | any    | grounding
calm         | 4-5       | 1-4    | any    | sound_therapy
calm         | 1-3       | 1-3    | any    | journaling
neutral      | any       | 4-5    | any    | pause
neutral      | any       | any    | 1-2    | rest
neutral      | 1-3       | 1-3    | 3-5    | light_planning
mixed        | 4-5       | any    | any    | grounding
mixed        | 1-3       | 1-3    | 3-5    | journaling
mixed        | any       | any    | any    | pause
```

### Timing Rules

```
TIME_OF_DAY    | ACTIVITY TYPE        | TIMING
──────────────────────────────────────────────────────
morning/early  | calming/active       | now / within_15_min
afternoon      | calming              | now
afternoon      | rest/leisure         | later_today
evening        | breathing/grounding  | now
evening        | rest/relaxation      | tonight
night          | box_breathing        | now
night          | rest/sound           | tonight
night          | planning/work        | tomorrow_morning
```

---

## Uncertainty Modeling

- **Confidence score**: `max(predict_proba)` — max class probability from XGBoost
- **Threshold**: 0.45
  - Random 6-class baseline = 0.167
  - Reliable prediction = max_prob > 0.5
  - 0.45 flags near-tie ambiguous cases without being overly conservative
- **uncertain_flag = 1** when confidence < 0.45

In the test set: **41/120 predictions (34%)** flagged as uncertain. These are cases where:
- Text is short or ambiguous
- Stress and emotional language contradict each other
- The top two classes have near-equal probabilities

---

## How to Run

### Requirements
```bash
pip install pandas openpyxl scikit-learn xgboost scipy
```

### Run Pipeline
```bash
python arvyax_pipeline.py
```

This will:
1. Load and analyze both datasets
2. Preprocess and build feature matrices
3. Train models with 5-fold CV ablation
4. Show feature importance and error analysis
5. Generate `predictions.csv`
6. Save `models.pkl` for reuse

### Output Files
| File | Description |
|---|---|
| `predictions.csv` | 120 test predictions with all required columns |
| `models.pkl` | Serialized models + transformers |
| `README.md` | This file |
| `ERROR_ANALYSIS.md` | Detailed error breakdown |
| `EDGE_PLAN.md` | Mobile/edge deployment plan |

---

## Output Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Test sample ID |
| `predicted_state` | str | One of: calm, restless, neutral, focused, mixed, overwhelmed |
| `predicted_intensity` | int | 1–5 |
| `confidence` | float | 0–1, max class probability |
| `uncertain_flag` | int | 1 if confidence < 0.45 |
| `what_to_do` | str | Recommended activity |
| `when_to_do` | str | Timing recommendation |
| `supportive_message` | str | Human-like guidance message |
