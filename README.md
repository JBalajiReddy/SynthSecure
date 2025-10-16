# SynthSecure Pay — Real‑Time Fraud Detection with Explainable UI

## 1) Problem statement, goal, and objectives

Online payment systems face evolving fraud tactics and a tight trade‑off between catching fraud and minimizing false positives. Analysts need both high detection performance and clear explanations for why transactions are flagged. Engineering teams need an end‑to‑end path from data to model to a reliable API and an intuitive user interface.

Goal:

- Build an end‑to‑end, interactive fraud detection product that is fast, explainable, and easy to operate locally.

Objectives:

- Data and modeling:
  - Explore fraud data and optionally augment scarce fraud cases using a lightweight GAN to generate synthetic samples.
  - Train a strong baseline classifier (XGBoost) and/or Random Forest.
  - Export a portable model artifact for inference.
- Serving:
  - Provide a clean Flask API for prediction, probability, and simple per‑input explanations.
  - Expose baseline statistics and global feature importances to power an explainable UI.
- Frontend:
  - Build a modern React UI that is “beautiful, fast, and informative.”
  - Show a risk gauge, decision threshold control, recent transactions, and “Top Signals” that explain each decision.
  - Provide QR‑based auto‑fill to reduce manual errors and speed up analyst workflows.
  - Support Google Sign‑In (Firebase) with a safe local demo fallback.

---

## 2) Tech stack

- Frontend:

  - React 18, Vite 5, TypeScript 5
  - Tailwind CSS 3, minimal framer‑motion
  - react‑router‑dom 6
  - axios for API calls
  - jsQR for QR code decoding (file upload)
  - Optional Firebase Auth (Google Sign‑In)

- Backend (API):

  - Python, Flask 3, flask‑cors
  - pandas, numpy, scikit‑learn, xgboost
  - joblib/pickle for model artifacts

- Modeling / R&D:
  - TensorFlow/Keras for a simple tabular GAN (generator/discriminator) to synthesize additional training samples
  - Matplotlib, Seaborn for quick diagnostics

Project structure (high‑level):

- `AI_model_Py_Scripts/` — Notebooks and scripts for data prep, GAN augmentation, and model training (XGBoost and RF). Saves model artifacts like `xg_model.pkl`.
- `AI_model_server_Flask/` — Flask server exposing `/predict`, `/metrics/baseline`, and `/metrics/feature-importance`.
- `synth_frontend/` — React + Vite app with Dashboard, QR‑enabled form, explanations, and authentication.

---

## 3) Key features

- Real‑time prediction and risk score

  - Frontend calls Flask `/predict` and renders a 0–100 risk score. When the model exposes `predict_proba`, the UI derives risk from the positive‑class probability.

- Decision threshold control (live)

  - Analyst can set a threshold slider (e.g., 60%). Decision chip updates immediately: “Fraud” when risk ≥ threshold. Threshold persists in localStorage.

- Explain the decision (Top Signals)

  - The backend returns simple, fast explanations: per‑feature z‑scores vs. training baseline mean/σ (μ, σ). The UI highlights features with high |z| (e.g., ≥ 2) and shows actual value alongside μ, σ.

- Global model insights

  - `/metrics/feature-importance` provides top feature importances from the trained model. The UI displays a compact bar list to communicate what generally matters for the model.

- Baseline metrics

  - `/metrics/baseline` exposes μ and σ for numeric columns based on the training CSV, letting the UI annotate explanations and provide drift intuition.

- QR auto‑fill of transaction fields

  - Upload a QR code image; the app decodes a JSON payload and maps keys to feature inputs (with robust normalization). Inline messages replace disruptive alerts.

- Authentication with safe fallback

  - If Firebase env vars are provided, the app uses Google Sign‑In. If not, it clearly enters “Demo mode” and creates a local mock user so the app remains usable.

- Recent transactions and UX polish
  - Stores the last 20 checks in localStorage with their risk and decisions. Dark mode toggle, glass UI accents, and accessible contrast.

---

## 4) Challenges and solutions

- Input schema alignment (model vs. API vs. UI)

  - Issue: Early 500s on `/predict` were caused by mismatched feature order/columns and double‑preprocessing.
  - Solution: The server introspects the model pipeline and preprocessor. If present, it extracts the classifier core and runs the DataFrame through the preprocessor once (avoids double transform). When the client sends a dict, the server reindexes columns to the training order.

- Probability extraction from heterogeneous shapes

  - Issue: Models return probabilities in different shapes (e.g., `[ [neg, pos] ]`, or single value).
  - Solution: Frontend extracts the positive‑class probability defensively with several fallbacks, then derives the percent risk.

- Clear, fast explanations without heavy compute

  - Issue: SHAP is powerful but can be heavy/slow and not always easy to wire for every model.
  - Solution: Use baseline z‑scores (μ, σ) per feature to show “how unusual” the current input is. It’s fast, model‑agnostic, and still actionable for analysts.

- QR label mapping and robustness

  - Issue: QR payload keys vary in casing/spacing; some include annotations.
  - Solution: The UI normalizes keys (case/spacing/punctuation), strips annotations, and maps to input fields reliably. Inline success/error messages improve UX.

- Authentication differences across machines

  - Issue: On machines without Firebase `.env`, Google popup doesn’t appear; app silently used local mock user.
  - Solution: Add a visible “Demo mode” badge on Login when Firebase isn’t configured and an initials‑based avatar fallback in Navbar when `photoURL` is absent or fails to load.

- Usability polish (threshold and visibility)
  - Issues: Users wanted the threshold slider to affect decisions immediately and better text contrast in forms.
  - Solutions: Live recomputation via `onInput`/`onChange`; Tailwind classes adjusted for form text and placeholders.

---

## 5) Outcomes and learning

- End‑to‑end path from data to UI

  - You can generate/augment data, train an XGBoost model, serve it from Flask, and interact with it in a polished UI—within one repository.

- Explainability that’s fast enough for product

  - Baseline z‑scores for per‑transaction signals and global importances are “good enough” to build trust quickly, and they require minimal compute and operational overhead.

- Schema discipline is essential

  - Keeping feature ordering and column names consistent across CSV → preprocessor → model → API removes many runtime surprises.

- Pragmatic auth strategy
  - Having a demo‑mode fallback keeps the app usable in workshops and on fresh machines, while still supporting Google Sign‑In where configured.

---

## 6) Future scope

- Better explanations

  - Optional SHAP/TreeSHAP for tree models with caching; per‑transaction feature contribution bars.

- Data/Model ops

  - Drift dashboards over time, model versioning, feature store, and scheduled retraining.

- Productization

  - Role‑based access, audit logs, export/ingestion of cases, alerting/webhooks (Slack, email), bulk scoring.

- UX enhancements

  - Live camera QR scanning, richer gauge with threshold marker, micro‑interactions for confidence.

- Packaging and CI/CD
  - Docker images for API/frontend, GitHub Actions for build/test/deploy, environment‑specific configs.

---

## Appendix — Notable components and endpoints

- Frontend

  - Pages: `Dashboard`, `Login`
  - Components: `FraudForm`, `RiskGauge`, `TransactionsTable`, `MetricCard`, `Navbar`
  - Services: `api.ts` (predict, baseline, feature importance), `firebase.ts` (optional auth)

- API endpoints (Flask)

  - `POST /predict` → `{ prediction, probability?, explanations? }`
    - Accepts `features` as either an ordered array or a key/value dict; dict is reindexed to match training columns.
    - `explanations` is a list of top z‑score signals: `{ feature, value, z }`.
  - `GET /metrics/baseline` → `{ columns, stats: { col: { mean, std }, ... } }`
  - `GET /metrics/feature-importance` → `{ items: [ { feature, importance }, ... ] }`

- Environment
  - Frontend `.env`:
    - `VITE_API_BASE_URL` (default http://127.0.0.1:5000)
    - Optional Firebase keys: `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, `VITE_FIREBASE_APP_ID`

---
