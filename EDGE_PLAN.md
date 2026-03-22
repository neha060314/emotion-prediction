# EDGE & MOBILE DEPLOYMENT PLAN
## ARVYAX Emotional Intelligence System

---

## Model Size Breakdown

| Component | Size (approx.) | Notes |
|---|---|---|
| TF-IDF Vectorizer (500 features) | ~80 KB | Vocabulary + IDF weights |
| XGBoost State Model (200 trees, depth 4) | ~2.5 MB | Main prediction model |
| XGBoost Intensity Model (200 trees, depth 4) | ~2.5 MB | Second prediction model |
| OneHotEncoder | ~5 KB | 5 categorical columns |
| StandardScaler | ~1 KB | 4 numeric columns |
| LabelEncoders (×2) | ~2 KB | State + intensity classes |
| Decision engine | ~0 KB | Pure Python conditionals |
| **Total pipeline** | **~5.1 MB** | Well within mobile limits |

---

## Inference Latency (CPU, single sample)

| Step | Time |
|---|---|
| Text preprocessing + TF-IDF transform | 1–2 ms |
| OHE + Standard scaling | < 0.5 ms |
| XGBoost state prediction + proba | 5–10 ms |
| XGBoost intensity prediction | 5–10 ms |
| Decision engine (pure conditionals) | < 0.1 ms |
| Supportive message lookup | < 0.1 ms |
| **Total end-to-end** | **~12–23 ms** |

This is well within real-time UX requirements (< 100 ms perceived lag).

---

## Why These Models Are Suitable for Mobile

### 1. No GPU Required
Both XGBoost and scikit-learn run entirely on CPU using standard BLAS/LAPACK.
Modern mobile chips (Apple A-series, Snapdragon) run tree models extremely fast.

### 2. No Internet Dependency After Download
The entire pipeline runs offline once models are loaded. Critical for:
- Privacy-sensitive emotional data (never leaves the device)
- Airplane mode / poor connectivity users
- Low-latency user experience

### 3. Serializable and Portable
- Python pickle → exportable to ONNX format for cross-platform deployment
- scikit-learn models have native ONNX converters (`sklearn-onnx` package)
- XGBoost has native ONNX export: `model.save_model('model.onnx')`
- ONNX Runtime is available for iOS, Android, React Native, Flutter

### 4. No Deep Learning Runtime Needed
Unlike BERT/GPT-style models that require PyTorch/TensorFlow (hundreds of MB), this pipeline uses sklearn + XGBoost — total runtime library < 20 MB.

### 5. Decision Engine is Zero-Dependency
The rule engine is pure Python conditionals — no serialization needed. It can be:
- Rewritten in Swift/Kotlin for native mobile
- Hardcoded in JavaScript for React Native
- Run as a lightweight microservice

---

## Deployment Architecture (Mobile)

```
User Input (journal text + metadata)
         │
         ▼
  [Text Preprocessing]  ← TF-IDF vocab loaded from file (~80 KB)
         │
         ▼
  [Feature Assembly]    ← OHE + Scaler applied locally
         │
         ▼
  [XGBoost Inference]   ← State model + Intensity model (~5 MB total)
         │
         ▼
  [Uncertainty Check]   ← max(proba) vs threshold 0.45
         │
         ▼
  [Decision Engine]     ← Pure conditionals (hardcoded in native lang)
         │
         ▼
  [Output Display]      ← activity, timing, message, confidence
```

---

## Tradeoffs

### Accuracy vs Size
| Option | State F1 | Size | Latency |
|---|---|---|---|
| XGBoost full (200 trees) | 0.633 | 2.5 MB | 8 ms |
| XGBoost small (50 trees) | ~0.605 | 0.8 MB | 3 ms |
| Logistic Regression | 0.598 | 50 KB | 1 ms |
| **Recommended** | **XGBoost full** | **5 MB** | **<25 ms** |

For extremely constrained devices (< 2 MB budget), LogReg is a viable fallback with < 3.5% F1 drop.

### TF-IDF Vocabulary Reduction
| Vocabulary | F1 (est.) | Size |
|---|---|---|
| 500 features | 0.633 | 80 KB |
| 200 features | ~0.620 | 32 KB |
| 100 features | ~0.600 | 16 KB |

Reducing from 500 to 200 features loses ~1.3% F1 but halves memory. Suitable for wearables or minimal devices.

### ONNX Quantization
Converting XGBoost to int8 ONNX:
- ~4x reduction in model size (~600 KB per model)
- < 5% accuracy degradation typically
- Supported on iOS CoreML and Android NNAPI

---

## Privacy Considerations

- Journal text is **never sent to a server** — all inference is local
- ONNX models contain only weights, not training data
- Recommendation: encrypt models at rest using device keystore
- Do not log journal_text on device; clear from memory after inference

---

## Recommended Deployment Stack

| Platform | Stack |
|---|---|
| iOS | CoreML + ONNX Runtime, Swift decision engine |
| Android | ONNX Runtime for Android, Kotlin decision engine |
| React Native | ONNX.js or call lightweight local REST microservice |
| Flutter | ONNX Runtime Flutter plugin |
| Web (PWA) | ONNX Runtime Web (WASM), JS decision engine |
| Python server | FastAPI + pickle models (current implementation) |

---

## Minimum Viable Deployment (MVP)

For fastest time-to-ship:
1. Wrap `arvyax_pipeline.py` in a FastAPI endpoint
2. Containerize with Docker (< 200 MB image with Python + dependencies)
3. Deploy on device via Termux (Android) or as a local server
4. Frontend calls `localhost:8000/predict` for offline-first mobile experience

This avoids ONNX conversion complexity while preserving privacy and offline capability.
