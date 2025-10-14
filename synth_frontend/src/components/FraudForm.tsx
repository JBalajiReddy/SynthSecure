import { useEffect, useRef, useState } from "react";
import jsQR from "jsqr";

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
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [qrMessage, setQrMessage] = useState<string | null>(null);
  const [qrError, setQrError] = useState<string | null>(null);
  const [msgTimeout, setMsgTimeout] = useState<number | null>(null);

  const normalizeKey = (k: string) =>
    k
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  const featureLabelToIndex: Record<string, number> = featureLabels.reduce(
    (acc, label, idx) => {
      const base = label.replace(/\s*\(.*?\)\s*/g, ""); // remove annotations like (0/1)
      acc[normalizeKey(label)] = idx;
      acc[normalizeKey(base)] = idx;
      // Also map dataset canonical forms if they differ slightly
      // Example: "Past Fraudulent Behavior Flags" vs label variant
      return acc;
    },
    {} as Record<string, number>
  );

  const triggerFile = () => fileInputRef.current?.click();

  const handleQRFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const code = jsQR(imageData.data, imageData.width, imageData.height);
        if (code) {
          try {
            const parsed = JSON.parse(code.data);
            applyQRData(parsed);
            setQrError(null);
            setQrMessage("QR data applied successfully.");
          } catch (err) {
            setQrMessage(null);
            setQrError("Invalid JSON inside QR code.");
          }
        } else {
          setQrMessage(null);
          setQrError("No QR code detected in the image.");
        }
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  const applyQRData = (data: any) => {
    // Accept two formats:
    // 1. Direct mapping of form fields e.g. { recipient: "...", amount: 123, "Transaction Frequency": 5 }
    // 2. { features: { label: value, ... }, recipient, amount }
    if (data.recipient) setRecipient(String(data.recipient));
    if (data.amount != null) setAmount(Number(data.amount));
    if (data.verification) setVerification(data.verification);
    if (data.geoFlag) setGeoFlag(data.geoFlag);

    const featObj =
      data.features && typeof data.features === "object" ? data.features : data;
    const newFeats = [...features];
    Object.keys(featObj).forEach((k) => {
      const normalized = normalizeKey(k);
      if (featureLabelToIndex.hasOwnProperty(normalized)) {
        const idx = featureLabelToIndex[normalized];
        const v = featObj[k];
        const num = typeof v === "number" ? v : parseFloat(v);
        if (!Number.isNaN(num)) newFeats[idx] = num;
      }
    });
    setFeatures(newFeats);
  };

  // Auto-clear QR messages after 5s
  useEffect(() => {
    if (msgTimeout) {
      window.clearTimeout(msgTimeout);
      setMsgTimeout(null);
    }
    if (qrMessage || qrError) {
      const id = window.setTimeout(() => {
        setQrMessage(null);
        setQrError(null);
      }, 5000);
      setMsgTimeout(id);
    }
  }, [qrMessage, qrError]);

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
        <button
          type="button"
          className="px-4 py-2 text-sm rounded-lg border border-brand-500 text-brand-700 hover:bg-brand-50 dark:border-brand-400 dark:text-brand-300"
          onClick={triggerFile}
        >
          Upload QR
        </button>
      </div>
      {(qrMessage || qrError) && (
        <div className="mt-3 text-sm">
          {qrMessage && (
            <div className="rounded-md bg-emerald-50 text-emerald-700 px-3 py-2 border border-emerald-200">
              {qrMessage}
            </div>
          )}
          {qrError && (
            <div className="rounded-md bg-rose-50 text-rose-700 px-3 py-2 border border-rose-200">
              {qrError}
            </div>
          )}
        </div>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleQRFile(file);
          e.target.value = "";
        }}
      />
      <canvas ref={canvasRef} className="hidden" />
    </form>
  );
}
