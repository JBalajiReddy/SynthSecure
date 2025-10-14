import { useState } from "react";

type Props = {
  onSubmit: (payload: {
    recipient: string;
    amount: number;
    features: any; // named object sent to backend
  }) => void | Promise<void>;
  loading?: boolean;
};

// We retain numeric helpers for convenience, but will submit a named object aligned
// to the backend dataset schema.
const featureLabels = [
  "Transaction Frequency",
  "Recipient Blacklist Status (0/1)",
  "Device Fingerprinting (0/1)",
  "VPN or Proxy Usage (0/1)",
  "Behavioral Biometrics (0..100)",
  "Time Since Last Transaction",
  "Social Trust Score",
  "Account Age",
  "High-Risk Transaction Times (0/1)",
  "Past Fraudulent Behavior Flags (0/1)",
  "Location-Inconsistent Transactions (0/1)",
  "Normalized Transaction Amount (0..1)",
  "Transaction Context Anomalies (0..1)",
  "Fraud Complaints Count",
  "Merchant Category Mismatch (0/1)",
  "User Daily Limit Exceeded (0/1)",
  "Recent High-Value Transaction Flags (0/1)",
];

export function FraudForm({ onSubmit, loading }: Props) {
  const [recipient, setRecipient] = useState("alice@bank");
  const [amount, setAmount] = useState<number>(499);
  const [verification, setVerification] = useState<
    "verified" | "recently_registered"
  >("verified");
  const [geoFlag, setGeoFlag] = useState<"normal" | "high-risk">("normal");
  const [features, setFeatures] = useState<number[]>(Array(17).fill(0));

  const updateFeature = (idx: number, val: number) => {
    setFeatures((f) => {
      const next = [...f];
      next[idx] = val;
      return next;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Build a named payload aligned with the CSV columns used by the model
    const [
      txFrequency,
      recipientBlacklist,
      deviceFingerprint,
      vpnProxy,
      behavioral,
      timeSinceLast,
      socialTrust,
      accountAge,
      offHours,
      pastFraud,
      locationInconsistent,
      normalizedAmount,
      contextAnomaly,
      complaintsCount,
      merchantMismatch,
      dailyLimitExceeded,
      recentHighValue,
    ] = features;

    const named = {
      "Transaction Amount": amount,
      "Transaction Frequency": Number(txFrequency),
      "Recipient Verification Status": verification,
      "Recipient Blacklist Status": Number(recipientBlacklist),
      "Device Fingerprinting": Number(deviceFingerprint),
      "VPN or Proxy Usage": Number(vpnProxy),
      "Geo-Location Flags": geoFlag,
      "Behavioral Biometrics": Number(behavioral),
      "Time Since Last Transaction": Number(timeSinceLast),
      "Social Trust Score": Number(socialTrust),
      "Account Age": Number(accountAge),
      "High-Risk Transaction Times": Number(offHours),
      "Past Fraudulent Behavior Flags": Number(pastFraud),
      "Location-Inconsistent Transactions": Number(locationInconsistent),
      "Normalized Transaction Amount": Number(normalizedAmount),
      "Transaction Context Anomalies": Number(contextAnomaly),
      "Fraud Complaints Count": Number(complaintsCount),
      "Merchant Category Mismatch": Number(merchantMismatch),
      "User Daily Limit Exceeded": Number(dailyLimitExceeded),
      "Recent High-Value Transaction Flags": Number(recentHighValue),
    };

    onSubmit({ recipient, amount, features: named });
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="glass rounded-xl p-4 text-black dark:text-gray-100"
    >
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-black dark:text-gray-200">
            Recipient UPI
          </label>
          <input
            className="input mt-1"
            value={recipient}
            onChange={(e) => setRecipient(e.target.value)}
            placeholder="user@bank"
          />
        </div>
        <div>
          <label className="text-sm text-black dark:text-gray-200">
            Amount (₹)
          </label>
          <input
            type="number"
            className="input mt-1"
            value={amount}
            onChange={(e) => setAmount(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="mt-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <div>
          <label className="text-sm text-black dark:text-gray-200">
            3. Recipient Verification Status
          </label>
          <select
            className="input mt-1"
            value={verification}
            onChange={(e) => setVerification(e.target.value as any)}
          >
            <option value="verified">verified</option>
            <option value="recently_registered">recently_registered</option>
          </select>
        </div>
        <div>
          <label className="text-sm text-black dark:text-gray-200">
            7. Geo-Location Flags
          </label>
          <select
            className="input mt-1"
            value={geoFlag}
            onChange={(e) => setGeoFlag(e.target.value as any)}
          >
            <option value="normal">normal</option>
            <option value="high-risk">high-risk</option>
          </select>
        </div>

        {featureLabels.map((label, idx) => (
          <div key={idx}>
            <label className="text-sm text-black dark:text-gray-200">
              {label}
            </label>
            <input
              type="number"
              step="0.01"
              className="input mt-1"
              value={features[idx]}
              onChange={(e) => updateFeature(idx, Number(e.target.value))}
            />
          </div>
        ))}
      </div>

      <div className="mt-4 flex gap-3">
        <button disabled={loading} className="btn-primary">
          Verify Fraud Status
        </button>
        <button
          type="button"
          className="px-4 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-700"
          onClick={() => setFeatures(Array(17).fill(0))}
        >
          Reset
        </button>
      </div>
    </form>
  );
}
