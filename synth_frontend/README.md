# SynthSecurePay Frontend

A modern React + Vite + Tailwind UI for real-time fraud checks against your Flask backend.

## Features

- Google Sign-In via Firebase (optional; local demo auth fallback)
- Transaction form with 20 fraud features
- Real-time risk gauge and decision
- Recent transactions table (localStorage fallback)
- Responsive, animated UI
- QR code upload to auto-fill fraud feature form

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

## QR Code Auto-Fill

You can encode a JSON payload in a QR code image and upload it via the "Upload QR" button to populate the form.

Accepted JSON shapes:

1. Flat (any matching labels):

```json
{
  "recipient": "bob@bank",
  "amount": 725,
  "Transaction Frequency": 4,
  "Recipient Blacklist Status": 0,
  "Device Fingerprinting": 1,
  "VPN or Proxy Usage": 0,
  "Behavioral Biometrics": 0.42,
  "Time Since Last Transaction": 12,
  "Social Trust Score": 50,
  "Account Age": 6,
  "High-Risk Transaction Times": 0,
  "Past Fraudulent Behavior Flags": 0,
  "Location-Inconsistent Transactions": 0,
  "Normalized Transaction Amount": 0.33,
  "Transaction Context Anomalies": 0.1,
  "Fraud Complaints Count": 0,
  "Merchant Category Mismatch": 0,
  "User Daily Limit Exceeded": 0,
  "Recent High-Value Transaction Flags": 1,
  "Recipient Verification Status": "verified",
  "Geo-Location Flags": "normal"
}
```

2. Nested features object:

```json
{
  "recipient": "bob@bank",
  "amount": 725,
  "verification": "verified",
  "geoFlag": "high-risk",
  "features": {
    "Transaction Frequency": 4,
    "Recipient Blacklist Status": 0,
    "Device Fingerprinting": 1,
    "VPN or Proxy Usage": 0,
    "Behavioral Biometrics": 0.42,
    "Time Since Last Transaction": 12,
    "Social Trust Score": 50,
    "Account Age": 6,
    "High-Risk Transaction Times": 0,
    "Past Fraudulent Behavior Flags": 0,
    "Location-Inconsistent Transactions": 0,
    "Normalized Transaction Amount": 0.33,
    "Transaction Context Anomalies": 0.1,
    "Fraud Complaints Count": 0,
    "Merchant Category Mismatch": 0,
    "User Daily Limit Exceeded": 0,
    "Recent High-Value Transaction Flags": 1
  }
}
```

Generate QR (example with a CLI):

```bash
echo '{"recipient":"bob@bank","amount":725,"Transaction Frequency":4}' | qrencode -o sample.png
```

Upload `sample.png` via the UI.
