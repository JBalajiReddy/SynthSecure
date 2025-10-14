# SynthSecurePay Frontend

A modern React + Vite + Tailwind UI for real-time fraud checks against your Flask backend.

## Features

- Google Sign-In via Firebase (optional; local demo auth fallback)
- Transaction form with 20 fraud features
- Real-time risk gauge and decision
- Recent transactions table (localStorage fallback)
- Responsive, animated UI

## Prerequisites

- Node.js 18+ and npm
- Running backend at http://127.0.0.1:5000 (or set VITE_API_BASE_URL)

## Quickstart (Windows PowerShell)

```powershell
cd e:\mini-project\synth_frontend
npm install
copy .env.example .env.local
# Optionally edit .env.local to configure backend URL and Firebase
npm run dev
```

Open http://localhost:3000

## Build

```powershell
npm run build; npm run preview
```

## Env Variables

- VITE_API_BASE_URL: Flask base URL (default http://127.0.0.1:5000)
- VITE*FIREBASE*\*: Firebase config for Google Sign-In (optional)

## Backend Contract

POST /predict
Body: { "features": number[] }
Response: { "prediction": string | number | (string[]|number[]) }

If only a label is returned, UI maps Fraud -> high risk, Not Fraud -> low risk.
