# SynthSecure Pay: Real-Time, Explainable Fraud Detection

SynthSecure is a full-stack, end-to-end system for real-time financial fraud detection that combines a high-performance machine learning model with a modern, interactive user interface.

This project addresses the critical shortcomings of traditional, rule-based fraud detection systems, which are often too rigid and slow to adapt to the evolving tactics of fraudsters. SynthSecure provides an intelligent, adaptable, and transparent solution designed for operational efficiency.

The project's name, *SynthSecure*, reflects its sophisticated, data-centric approach to security.
"Synth" alludes to the optional use of Generative Adversarial Networks (GANs) to create high-quality *synthetic* data, a cutting-edge technique used to train a more robust model, especially when real fraud data is scarce.
"Secure" represents the project's primary mission: to provide a powerful tool for securing financial transactions.

-----

## Application Showcase

The user interface is not merely a display; it is an operational tool designed to empower fraud analysts.
It translates complex model outputs into actionable controls, bridging the gap between abstract predictions and real-world decision-making.

| Main Dashboard | Analytics Dashboard | Transaction History |
| :---: | :---: | :---: |
|  |  |  |
| *"The Main Dashboard provides a consolidated view for real-time analysis, featuring the transaction input form, a dynamic risk gauge, and the ""Top Signals"" explainability panel."* | *"The Analytics Dashboard offers a high-level overview with data visualizations that summarize historical transaction trends, such as the breakdown of fraud vs. non-fraud decisions."* | *"The History Page maintains a persistent, client-side log of the most recent transactions, allowing analysts to easily review past predictions and their corresponding details."* |

-----

## Core System Features

  * **Real-Time Risk Assessment:** The system leverages a pre-trained XGBoost model served via a low-latency API to provide instantaneous fraud probability scores for each transaction submitted.
  * **"Top Signals" Explainability:** To combat the "black box" nature of complex models, SynthSecure provides immediate, human-readable explanations for its predictions. It highlights the transaction features that are most statistically anomalous compared to a baseline of normal behavior.
  * **Operator-Controlled Decision Threshold:** An interactive slider allows users to dynamically tune the model's sensitivity in real-time. This critical feature empowers analysts to manage the operational trade-off between catching more fraud (higher recall) and reducing false alarms (higher precision).
  * **QR Code Data Entry:** To improve speed and reduce manual input errors, the interface can use the device's camera to scan a QR code, which automatically populates the transaction form fields.
  * **Persistent Transaction History:** The application automatically saves the last 20 transactions in the browser's local storage. This allows for quick review and analysis of recent activity without needing a backend database.
  * **Integrated Analytics Dashboard:** A dedicated analytics page provides visual summaries of the transaction history, using charts to illustrate trends and the overall distribution of fraud and non-fraud classifications.
  * **Audible High-Risk Alerts:** An optional sound alert can be enabled to provide an immediate, non-visual notification whenever a transaction is classified as fraudulent based on the user-defined threshold.

-----

## The Challenge of Modern Fraud and the SynthSecure Solution

The global shift to digital finance has brought convenience but has also opened the door to sophisticated fraudulent activities.
Financial institutions have traditionally relied on static, rule-based systems (e.g., IF-THEN rules) to detect fraud.
While simple to understand, these legacy systems are fundamentally flawed in the modern landscape: they are rigid, reactive, and struggle to identify novel fraud patterns.
This leads to a high volume of both *false positives* (legitimate transactions being incorrectly flagged) and *false negatives* (fraudulent transactions going unnoticed), resulting in financial loss and diminished customer trust.

SynthSecure was engineered to overcome these challenges by being built upon three foundational pillars:

  * **Accuracy:** Utilizing a state-of-the-art Extreme Gradient Boosting (XGBoost) machine learning model, the system can identify complex, non-linear patterns indicative of fraud with high precision and recall.
  * **Speed:** The system is designed for real-time processing, with a lightweight backend API that delivers predictions and explanations with minimal latency, enabling immediate decision-making in a live transaction environment.
  * **Transparency:** A core objective is to solve the "black box" problem inherent in many ML systems. Through its "Top Signals" feature, SynthSecure provides clear, actionable insights into *why* a transaction is deemed risky, fostering trust and enabling analysts to take informed action.

-----

## System Architecture and Design

SynthSecure is built on a modern, *decoupled, three-tier architecture*.
This design promotes modularity, independent scalability, and maintainability by separating the system's core responsibilities into distinct layers.

1.  **Layer 1: Presentation Layer (Client):** A responsive Single-Page Application (SPA) built with *React 18*, *TypeScript*, and *Vite*. This layer runs entirely in the user's web browser and is responsible for rendering the UI, managing user interactions, and handling all application state. For this prototype, user settings and transaction history are persisted client-side using the browser's localStorage API. Communication with the backend is handled via asynchronous calls to the REST API.
2.  **Layer 2: Application Layer (Server):** A lightweight microservice built with *Python* and the *Flask* framework. Its sole responsibility is to expose the machine learning model's functionality through a well-defined REST API. Upon startup, it loads the trained model and baseline statistics into memory to ensure low-latency inference for every prediction request.
3.  **Layer 3: Model Layer (Artifacts):** This is not a running service but a collection of static files generated during the offline model training phase. It consists of the serialized, pre-trained XGBoost model (xg\_model.pkl) and a JSON file containing baseline feature statistics (baseline\_stats.json) required for the explainability calculations.

This decoupled architecture allows the frontend and backend to be developed, deployed, and scaled independently.
For example, the static frontend can be hosted on a global CDN for fast delivery, while the backend API can be scaled up or down based on prediction load.

-----

## The Machine Learning Engine

### Predictive Model: Extreme Gradient Boosting (XGBoost)

The core of the detection engine is an *XGBoost (Extreme Gradient Boosting)* classifier.
This model was deliberately chosen as it is widely regarded as a state-of-the-art algorithm for classification tasks on structured or tabular data, the kind typically found in financial transactions.
Empirical evidence consistently shows that well-tuned XGBoost models outperform other methods like Random Forests and even complex Deep Neural Networks for this type of problem. Key advantages include its built-in L1/L2 regularization to prevent overfitting and its efficient, native handling of class imbalance through the scale\_pos\_weight hyperparameter, which is critical in fraud detection where fraudulent transactions are rare.

### Real-Time Explainability: "Top Signals"

A significant challenge with powerful models like XGBoost is their "black box" nature, making it difficult to understand their reasoning.
SynthSecure addresses this with a novel, low-latency explainability feature called *"Top Signals"*.
This method was chosen as a pragmatic solution that prioritizes real-time performance, a critical non-functional requirement for the API (\<500ms response time), over the deeper but more computationally expensive explanations offered by methods like SHAP (which are planned for future integration).

The "Top Signals" feature works by calculating the z-score for each feature of an incoming transaction:

$$z = \frac{(x - \mu)}{\sigma}$$

Where:

  * $x$ is the value of a feature for the current transaction.
  * $\mu$ (mu) is the mean of that feature, pre-calculated from a baseline of normal (non-fraudulent) transactions.
  * $\sigma$ (sigma) is the standard deviation of that feature from the same baseline.

The features with the highest absolute z-scores are presented to the user as the "Top Signals."
This immediately highlights which aspects of a transaction are the most statistically unusual or anomalous, providing an intuitive and model-agnostic rationale for the risk score.

### Data Augmentation with Generative Adversarial Networks (GANs)

A fundamental problem in fraud detection is severe class imbalance—fraudulent transactions are extremely rare.
A model trained on such imbalanced data can easily become biased towards the majority class.
To mitigate this, the modeling pipeline includes an optional step that uses *Generative Adversarial Networks (GANs)* to create high-quality, realistic synthetic data for the minority (fraud) class.
This data augmentation technique helps create a more balanced dataset for training, enabling the XGBoost classifier to learn the patterns of fraudulent behavior more effectively and build a more robust decision boundary.

-----

## Quantitative Performance Analysis

The final XGBoost model was evaluated on a held-out test set that maintained the original, imbalanced class distribution.

**Confusion Matrix:**

  * **True Positives (TP):** 2421 (Fraud correctly identified)
  * **True Negatives (TN):** 2423 (Legitimate correctly identified)
  * **False Positives (FP):** 777 (Legitimate incorrectly flagged as fraud)
  * **False Negatives (FN):** 779 (Fraud missed by the model)

**ROC AUC Score:** The model achieved an Area Under the ROC Curve (AUC) score of **0.879**, indicating a strong ability to discriminate between fraudulent and legitimate transactions across all decision thresholds.

| Metric | Formula | Calculated Value | Interpretation |
| :--- | :--- | :--- | :--- |
| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ | 75.69% | The overall percentage of correctly classified transactions. |
| Precision | $TP/(TP+FP)$ | 75.70% | "When the model predicts a transaction is fraudulent, it is correct 75.70% of the time." |
| Recall | $TP/(TP+FN)$ | 75.66% | The model successfully identifies 75.66% of all actual fraudulent transactions. |
| F1-Score | $2 \times \frac{(Precision \times Recall)}{(Precision + Recall)}$ | 75.68% | "The harmonic mean of Precision and Recall, providing a single score that balances both concerns." |

-----

## Technology Stack

### Frontend

  * **React 18:** A JavaScript library for building user interfaces.
  * **TypeScript:** A statically typed superset of JavaScript for enhanced code quality and developer experience.
  * **Vite:** A next-generation frontend tooling for fast development and optimized builds.
  * **Tailwind CSS:** A utility-first CSS framework for rapid UI development.
  * **Recharts:** A composable charting library for creating data visualizations.
  * **jsQR:** A JavaScript library for decoding QR codes from a camera stream.

### Backend

  * **Python:** A versatile programming language for web development and data science.
  * **Flask:** A lightweight and flexible web framework for building the REST API.
  * **Pandas:** A data manipulation and analysis library.
  * **Scikit-learn:** A machine learning library for data preprocessing.
  * **XGBoost:** An optimized gradient boosting library for model training and inference.

### Data Science & Modeling

  * **Jupyter Notebooks:** An interactive environment for data exploration, model experimentation, and analysis.
  * **TensorFlow/Keras:** A deep learning framework used for implementing and training GANs for data augmentation.

-----

## API Reference

The backend exposes a RESTful API to facilitate communication with the client.

| Endpoint | Method | Request Payload (JSON) | Success Response (200 OK) (JSON) | Error Response (400/500) (JSON) |
| :--- | :--- | :--- | :--- | :--- |
| `/predict` | POST | \`{ "features": { "feature1": val1,... } }\` | \`{ "prediction": 0|1, "probability": float, "explanations": [...] }\` | \`{ "error": "Error message" }\` |
| `/metrics/baseline` | GET | \`None\` | \`{ "stats": { "feature1": { "mean": float, "std": float },... } }\` | \`{ "error": "Could not load baseline" }\` |
| `/metrics/feature-importance` | GET | \`None\` | \`{ "importances": [{ "feature": string, "importance": float }] }\` | \`{ "error": "Could not load importances" }\` |

-----

## Local Development and Setup

### Prerequisites

  * Node.js v18+
  * Python 3.9+
  * pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JBalajiReddy/SynthSecure.git
    cd synthsecure
    ```
2.  **Install backend dependencies:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
3.  **Install frontend dependencies:**
    ```bash
    cd ../frontend
    npm install
    ```

### Execution

1.  **Start the Backend API:** In one terminal, from the \`/backend\` directory, run:
    ```bash
    flask run
    ```
    The API will be available at \`http://127.0.0.1:5000\`.
2.  **Start the Frontend Application:** In a second terminal, from the \`/frontend\` directory, run:
    ```bash
    npm run dev
    ```
    The application will be available at \`http://localhost:5173\`.

-----

## Codebase Overview

The project is organized into distinct modules for a clean separation of concerns.

```plaintext
/synthsecure/
├── /backend/           # Flask API and ML model artifacts
│   ├── /artifacts/     # Serialized model (xg_model.pkl) and baseline stats
│   └── app.py          # Main Flask application with API routes
├── /frontend/          # React/TypeScript client application
│   └── /src/
│       ├── /components/ # Reusable UI components (e.g., RiskGauge, FraudForm)
│       ├── /pages/      # Main application pages (Dashboard, History, Analytics)
│       └── /hooks/      # Custom React hooks (e.g., useLocalStorage)
└── /notebooks/         # Jupyter notebooks for data exploration and model training
```

-----

## Future Enhancements

The current prototype provides a solid foundation, but several enhancements are planned to increase its robustness and prepare it for a production environment.

  * **Enhanced Explainability with SHAP:** Integrate SHAP (SHapley Additive exPlanations) to provide more granular, attribution-based explanations. This would involve a new API endpoint to compute SHAP values and a new UI component to visualize them, offering deeper insights into the model's internal logic.
  * **Data and Concept Drift Monitoring:** Implement a monitoring module to track the statistical properties of incoming transaction data over time. If significant drift from the training data is detected, the system could alert operators that the model may need retraining to maintain performance.
  * **Adversarial Robustness:** Harden the model against adversarial attacks, where malicious actors make subtle changes to input data to evade detection. This would involve incorporating adversarial training techniques into the model development pipeline.
  * **Production-Ready Architecture:** Transition the system to a production-grade architecture by:
      * Replacing localStorage with a robust, centralized database (e.g., PostgreSQL).
      * Implementing user authentication and Role-Based Access Control (RBAC).
      * Containerizing the frontend and backend applications with Docker and managing deployment with an orchestrator like Kubernetes to facilitate scalability and reliability.
