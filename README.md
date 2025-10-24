# SynthSecure Pay — Real‑Time Fraud Detection with Explainable UI

An end‑to‑end, production‑style demo that takes you from data to model to API to a delightful UI that explains each decision. The app predicts the fraud risk of a transaction, shows why, and gives analysts the controls and context to act quickly and confidently.

---

## 1) Problem statement, goal, and objectives

Modern payment systems fight a constantly evolving adversary. The cost of missing fraud is high, but so is the cost of false positives that block good users. Teams need a system that is both accurate and explainable—fast enough for real‑time use, yet transparent enough to build trust with analysts and business stakeholders.

Goal

- Deliver an interactive fraud detection product that is fast, explainable, and easy to operate locally—covering modeling, serving, UI, and documentation.

Objectives

- Data & modeling
  - Explore tabular fraud data; optionally augment rare fraud cases using a lightweight GAN to reduce class imbalance.
  - Train a strong baseline classifier (XGBoost) with a reproducible pipeline; optionally compare with Random Forest.
  - Export and version model artifacts (e.g., `xg_model.pkl`) for consistent inference.
- Serving
  - Provide a clean Flask API with prediction, probabilities, and per‑input explanations.
  - Expose baseline statistics and global feature importances for explainability dashboards.
- Frontend
  - Build a modern React UI that is accessible and fast.
  - Show a risk gauge, live threshold control, QR auto‑fill, “Top Signals,” history, analytics, and alert sounds for accessibility.
  - Support Google Sign‑In (Firebase) with a safe local demo fallback.

Why explainability matters

- Analysts can triage faster when the UI states not only “what” (Fraud/Not Fraud) but also “why.”
- Transparency increases organizational trust and accelerates adoption.

---

## 2) Tech stack

Frontend

- React 18 + Vite 5 + TypeScript 5 — fast HMR, type safety, and small bundles.
- Tailwind CSS 3 — consistent styling and dark mode utilities.
- react‑router‑dom 6 — routing for Dashboard, History, Analytics, and Login.
- axios — robust HTTP client with interceptors support.
- jsQR — QR decoding for form auto‑fill (file upload or camera preview in future).
- Optional: framer‑motion for subtle micro‑interactions.

Backend (API)

- Python 3, Flask 3, flask‑cors — thin, predictable web layer.
- pandas, numpy, scikit‑learn, xgboost — data prep and ML.
- joblib/pickle — portable artifacts.

Modeling / R&D

- TensorFlow/Keras — lightweight tabular GAN for synthetic fraud samples.
- Matplotlib, Seaborn — quick diagnostics and charts.

Project structure (high‑level)

- `AI_model_Py_Scripts/` — Notebooks and scripts for dataset prep, GAN augmentation, and training (saves `xg_model.pkl`).
- `AI_model_server_Flask/` — Flask server exposing `/predict`, `/metrics/baseline`, `/metrics/feature-importance`.
- `synth_frontend/` — Vite/React/TS app: Dashboard, History, Analytics, Login.

---

## 3) Key features

Real‑time prediction & risk score

- Frontend calls `/predict`; the UI displays a 0–100 risk derived from the model’s positive‑class probability when available.

Decision threshold control (live)

- Analyst sets a threshold (10–90%) and the decision chip updates instantly. Stored in `localStorage`.

Explain the decision (Top Signals)

- What it is: For each numeric input feature, the server computes a z‑score against the training baseline: how many standard deviations the current value is away from the typical training value. We then surface the most unusual features (largest |z|) as “Top Signals.” The UI shows the actual value, μ (mean), σ (std), the direction (higher/lower than typical), and a severity style based on |z|.
- Why it helps: Unusual inputs often explain why risk moved up (or down). This is model‑agnostic and fast, so analysts always get an immediate reason—even if the underlying model changes.
- How to read it:
  - Positive z (e.g., +2.8) → Value is higher than the training average; negative z (e.g., −2.1) → lower than average.
  - Severity cues: |z| ≥ 3 “Strong” (red), 2–3 “Moderate” (amber), 1–2 “Mild” (indigo). Low |z| means “close to typical.”
  - Example: amount = 9500 with μ = 1200 and σ = 1800 → z ≈ +4.6 (very high amount); hour_of_day = 02 with μ = 14 and σ = 5 → z ≈ −2.4 (unusual time). Together these are compelling fraud signals.
- What appears in the UI:
  - A summary (“Why this decision?”) with decision vs threshold and overall signals strength.
  - A ranked list of top features with bars proportional to |z| and inline μ/σ/Δ annotations.
  - Plain‑language hints (e.g., “higher than typical”) to make the explanation friendly for non‑technical users.
- Guardrails and limitations:
  - Baseline‑driven: uses training μ and σ. If your data distribution shifts, z‑scores can drift—monitor with the Analytics page and retrain when needed.
  - σ ≈ 0: for near‑constant features, z is suppressed to avoid division by very small σ.
  - Categorical/text features: only numeric features are directly scored. Derived numeric encodings can be included if present in the training pipeline.
  - Not causal attribution: z indicates “unusualness,” not feature weight. The true model may rely on different signals. For global weightings, see the “Model Insights” (feature importance) panel; for model‑specific local attributions, SHAP/TreeSHAP can be added later.
  - Correlated features: multiple correlated fields can all look “unusual.” Consider them together rather than in isolation.
  - Thresholding: The final decision is risk vs analyst‑chosen threshold; “Top Signals” explain why the risk was high/low, not the threshold itself.

Model insights (global)

- `/metrics/feature-importance` summarizes which features matter globally via normalized importances.

Baseline metrics

- `/metrics/baseline` provides μ and σ for numeric columns—used for explanations and drift intuition.

QR auto‑fill (accessibility & speed)

- Upload a QR image to auto‑populate form fields. Robust label normalization maps common variations.

Alert sounds on fraud (illiterate‑friendly)

- 4‑second alert plays when the decision is Fraud. Toggle on/off in the UI; persisted in `localStorage`.

History page

- Dedicated page listing recent transactions (last 20 by default) with risk color bands and timestamps.

Analytics page

- Visualizations powered by Recharts: decision breakdown (pie), risk over time (line), risk distribution (histogram), top recipients (bars), and average amount by decision.

Authentication with safe fallback

- Google Sign‑In via Firebase if configured; otherwise a clear “demo mode” mock user keeps the app usable.

Dark mode & polished UI

- Glass surfaces, gradient accents, and accessible contrast.

---

## 4) System overview and architecture

High level flow

1. User enters transaction details (or uploads a QR) in the Dashboard.
2. Frontend calls Flask `/predict` with features (array or dict).
3. Server reindexes features to training order, applies preprocessing/model, and returns `{ prediction, probability?, explanations }`.
4. UI renders the risk gauge, decision, and explanations; optionally plays an alert sound.
5. Transaction is stored locally (for demo) and visible in History and Analytics.

Contracts (inputs/outputs)

- Input features: either an ordered array or an object `{ col: value }`. Dicts are reindexed on the server.
- Output decision: prediction label and positive‑class probability when available; derived risk = `probability * 100`.
- Explanations: list of `{ feature, value, z }` computed against baseline stats.

Edge cases we handle

- Missing probability shapes — UI extracts positive probability robustly from several formats.
- Dict vs array inputs — server reindexes safely to training column order.
- Double preprocessing — server avoids applying preprocessors twice.

---

## 5) API reference (selected)

POST `/predict`

- Request body
  - `{ features: Record<string, any> }` or `{ features: any[] }`
- Response
  - `{ prediction: number|string|array, probability?: number|number[]|number[][], explanations?: Array<{ feature: string; value: number; z: number }>} `

GET `/metrics/baseline`

- `{ columns: string[], stats: { [feature: string]: { mean: number; std: number } } }`

GET `/metrics/feature-importance`

- `{ items: Array<{ feature: string; importance: number }> } // importance normalized to sum≈1`

Z‑score formula

- $z = \frac{x - \mu}{\sigma}$ where $\mu$ is training mean and $\sigma$ is training std for the feature.

---

## 6) Challenges and solutions

Input schema alignment (model ↔ API ↔ UI)

- Problem: Training column order and API payload didn’t always match, causing 500s.
- Fix: Server introspects the pipeline, extracts classifier core, applies preprocessing once, and reindexes dict inputs to the training order.

Probability extraction from heterogeneous shapes

- Problem: Different models return various probability shapes.
- Fix: UI defensively extracts the positive class probability from `[ [neg, pos] ]`, `[pos]`, or single values.

Fast explanations without heavy dependencies

- Problem: SHAP is powerful but heavier to compute and wire universally.
- Fix: Use baseline z‑scores (μ, σ) per feature—fast, model‑agnostic, and still informative.

QR label mapping & robustness

- Problem: Field names in QR payloads vary widely.
- Fix: Normalize keys (case/spacing/punctuation) and map consistently; show inline success/error messages.

Auth differences across machines

- Problem: No Firebase env → silent mock user.
- Fix: Explicit “Demo mode” badge and initials‑based avatar fallback.

Usability polish (threshold & visibility)

- Problem: Threshold needed to impact decision in real time; some text had low contrast.
- Fix: Live recomputation via `onInput` and better Tailwind classes.

---

## 7) Outcomes and learning

End‑to‑end path from data to UI

- One repo, reproducible steps: generate/augment data → train XGBoost → serve via Flask → interact in a polished UI.

Practical explainability

- Baseline z‑scores and global importances are quick to compute and sufficiently helpful for analysts in many cases.

Operational discipline

- Schema consistency across CSV → preprocessor → model → API prevents the majority of runtime issues.

Accessible UX matters

- Alert sounds, dark mode, and QR auto‑fill significantly improve adoption in real workflows.

---

## 8) Future scope

Explainability upgrades

- SHAP/TreeSHAP for tree models with caching; waterfall plots and PDP/ICE on the Analytics page.
- Counterfactual suggestions: minimal changes to flip a decision.

Monitoring & drift

- Population vs training drift (PSI/KS), calibration curves, ROC/PR, confusion matrix over labeled recent data.

Productization

- Role‑based access control, audit trails, webhooks (Slack/Teams), CSV batch scoring, export to data lakes.

MLOps

- Model registry and versioning, canary/A‑B rollout, scheduled retraining with validation gates.

Accessibility

- Voice prompts (TTS) in multiple languages; larger “quick‑action” UI for kiosk/field use; vibration/notification for PWA.

---

## 9) How to run (Windows)

Prereqs

- Python 3.10+ for the API, Node.js 18+ for the frontend.

Backend (Flask)

1. Open a terminal in `AI_model_server_Flask/`.
2. Create/activate a venv (optional) and install:
   - `pip install -r requirements.txt`
3. Ensure a trained model artifact exists (e.g., `xg_model.pkl`). If not, train via scripts/notebooks in `AI_model_Py_Scripts/`.
4. Run the server:
   - `python app.py`
5. The API defaults to `http://127.0.0.1:5000`.

Frontend (Vite + React)

1. Open a terminal in `synth_frontend/`.
2. Install deps: `npm install`.
3. Configure `.env` (optional): `VITE_API_BASE_URL` (defaults to localhost:5000). If using Firebase auth, add `VITE_FIREBASE_*` keys.
4. Start dev server: `npm run dev` and open the printed local URL.

Notes

- For production build, run `npm run build` and `npm run preview`.
- If the browser blocks the alert sound, interact with the page (e.g., click/submit) once, then retry.

---

## 10) Frontend pages and components

Pages

- Dashboard — Form, risk gauge, decision chip, threshold slider, explanations, model insights.
- History — Recent transactions with risk color chips and timestamps; refresh from localStorage.
- Analytics — Recharts visualizations (pie/line/bar) for decisions, risk distribution, top recipients, and average amounts.
- Login — Google Sign‑In or demo‑mode fallback.

Selected components

- `FraudForm`, `RiskGauge`, `MetricCard`, `TransactionsTable`, `Navbar` (dark‑mode toggle, active route highlighting, avatar fallback).

---

## 11) Troubleshooting

- 500 on `/predict` — Check feature names/order and whether the server is applying preprocessing twice.
- No sound on fraud — Browsers may block autoplay until user interaction. Toggle the sound or click once then retry.
- Firebase popup missing — Without Firebase env vars, the app enters demo mode by design (badge shown on Login).
- Charts empty — Submit a few transactions; the demo uses localStorage for analytics/history data.

---

## Appendix — Notable endpoints and environment

API endpoints (Flask)

- `POST /predict` → `{ prediction, probability?, explanations? }`
- `GET /metrics/baseline` → `{ columns, stats }`
- `GET /metrics/feature-importance` → `{ items }`

Environment

- Frontend `.env`:
  - `VITE_API_BASE_URL` (default `http://127.0.0.1:5000`)
  - Optional Firebase keys: `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, `VITE_FIREBASE_APP_ID`

---
