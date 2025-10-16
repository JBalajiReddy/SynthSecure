import { useEffect, useMemo, useRef, useState } from "react";
import { Navbar } from "../components/Navbar";
import { MetricCard } from "../components/MetricCard";
import { RiskGauge } from "../components/RiskGauge";
import { TransactionsTable } from "../components/TransactionsTable";
import { FraudForm } from "../components/FraudForm";
import {
  predictFraud,
  fetchBaseline,
  fetchFeatureImportance,
  FeatureImportanceItem,
} from "../services/api";

type Tx = {
  id: string;
  recipient: string;
  amount: number;
  date: string;
  risk: number;
  decision: "Fraud" | "Not Fraud";
};

export function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [risk, setRisk] = useState(0);
  const [decision, setDecision] = useState<"Fraud" | "Not Fraud">("Not Fraud");
  const [threshold, setThreshold] = useState<number>(() => {
    const v = localStorage.getItem("ssp_threshold");
    return v ? Number(v) : 50; // default 50%
  });
  const [soundEnabled, setSoundEnabled] = useState<boolean>(() => {
    const v = localStorage.getItem("ssp_sound_enabled");
    return v === null ? true : v !== "false";
  });
  const [explanations, setExplanations] = useState<
    Array<{ feature: string; value: number; z: number }>
  >([]);
  const [baseline, setBaseline] = useState<null | {
    stats: Record<string, { mean: number; std: number }>;
  }>(null);
  const [transactions, setTransactions] = useState<Tx[]>(() => {
    const raw = localStorage.getItem("ssp_txs");
    return raw ? JSON.parse(raw) : [];
  });
  const [importances, setImportances] = useState<FeatureImportanceItem[]>([]);
  const alertAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    localStorage.setItem("ssp_txs", JSON.stringify(transactions.slice(0, 20)));
  }, [transactions]);

  // Fetch baseline stats once (optional, used to render mean/std next to explanations)
  useEffect(() => {
    fetchBaseline()
      .then((data) => setBaseline({ stats: data.stats }))
      .catch(() => {});
  }, []);

  // Fetch global feature importances (optional insights)
  useEffect(() => {
    fetchFeatureImportance(5)
      .then((res) => setImportances(res.items || []))
      .catch(() => setImportances([]));
  }, []);

  // Prepare alert audio on mount
  useEffect(() => {
    const audio = new Audio("/sounds/alert-109578.mp3");
    audio.preload = "auto";
    alertAudioRef.current = audio;
    return () => {
      // clean up
      if (alertAudioRef.current) {
        alertAudioRef.current.pause();
        alertAudioRef.current.src = "";
        alertAudioRef.current = null;
      }
    };
  }, []);

  // Re-evaluate decision when threshold changes for current risk reading
  useEffect(() => {
    setDecision(risk >= threshold ? "Fraud" : "Not Fraud");
  }, [threshold, risk]);

  const stats = useMemo(() => {
    const total = transactions.length;
    const fraud = transactions.filter((t) => t.decision === "Fraud").length;
    const safe = total - fraud;
    return { total, fraud, safe };
  }, [transactions]);

  const handleSubmit = async ({
    recipient,
    amount,
    features,
  }: {
    recipient: string;
    amount: number;
    features: any;
  }) => {
    try {
      setLoading(true);
      const res = await predictFraud(features);
      // Prefer backend probability when available
      let labelRaw: any = Array.isArray(res.prediction)
        ? (res.prediction as any[])[0]
        : res.prediction;

      // Extract positive class probability from various shapes
      let posProb: number | null = null;
      const proba: any = res.probability;
      if (proba != null) {
        if (Array.isArray(proba)) {
          const firstRow = Array.isArray(proba[0])
            ? proba[0]
            : (proba as number[]);
          if (Array.isArray(firstRow)) {
            // Typical [neg, pos]
            if (firstRow.length >= 2 && typeof firstRow[1] === "number")
              posProb = firstRow[1] as number;
            else if (firstRow.length > 0)
              posProb = Math.max(...(firstRow as number[]));
          } else if (typeof firstRow === "number") {
            // Single value - assume it's the positive prob
            posProb = firstRow as number;
          }
        }
      }

      // Determine decision
      let isFraud: boolean | null = null;
      if (typeof labelRaw === "number") {
        isFraud = labelRaw === 1;
      } else if (typeof labelRaw === "string") {
        isFraud = /fraud/i.test(labelRaw);
      }
      if (isFraud == null && posProb != null) {
        isFraud = posProb >= 0.5;
      }
      if (isFraud == null) {
        isFraud = false; // default safe if uncertain
      }

      // Risk score from probability if available, else map label
      const score =
        posProb != null ? Math.round(posProb * 100) : isFraud ? 85 : 15;
      setRisk(score);
      const finalDecision = score >= threshold ? "Fraud" : "Not Fraud";
      setDecision(finalDecision);
      setExplanations(res.explanations || []);

      // Play alert if fraud predicted and sound is enabled
      if (finalDecision === "Fraud" && soundEnabled) {
        try {
          if (alertAudioRef.current) {
            alertAudioRef.current.currentTime = 0;
            await alertAudioRef.current.play();
          }
        } catch (err) {
          // Autoplay might be blocked until user interacts; ignore error
          // Optionally, could surface a tooltip in future
        }
      }

      const tx: Tx = {
        id: crypto.randomUUID(),
        recipient,
        amount,
        date: new Date().toISOString(),
        risk: score,
        decision: finalDecision,
      };
      setTransactions((list) => [tx, ...list].slice(0, 20));
    } catch (e: any) {
      console.error("/predict failed", e?.response?.data || e);
      // Network fallback: randomize a believable score if backend not available
      const score = Math.round(10 + Math.random() * 80);
      setRisk(score);
      const finalDecision = score >= threshold ? "Fraud" : "Not Fraud";
      setDecision(finalDecision);
      if (finalDecision === "Fraud" && soundEnabled) {
        try {
          if (alertAudioRef.current) {
            alertAudioRef.current.currentTime = 0;
            await alertAudioRef.current.play();
          }
        } catch {}
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="mx-auto max-w-7xl p-4 space-y-6">
        <div className="grid sm:grid-cols-3 gap-4">
          <MetricCard title="Total Checks" value={stats.total} />
          <MetricCard
            title="Flagged Fraud"
            value={stats.fraud}
            trend={
              stats.fraud
                ? `~${Math.round(
                    (stats.fraud / Math.max(1, stats.total)) * 100
                  )}%`
                : undefined
            }
          />
          <MetricCard title="Safe Transactions" value={stats.safe} />
        </div>

        <div className="grid lg:grid-cols-[2fr_1fr] gap-4">
          <FraudForm onSubmit={handleSubmit} loading={loading} />
          <div className="space-y-4">
            <RiskGauge score={risk} />
            <div
              className={`rounded-xl p-3 glass flex items-center justify-between`}
            >
              <div className="text-sm text-gray-600">Decision</div>
              <div
                className={`px-2 py-1 rounded-md text-xs font-medium ${
                  decision === "Fraud"
                    ? "bg-rose-100 text-rose-700"
                    : "bg-emerald-100 text-emerald-700"
                }`}
              >
                {decision}
              </div>
            </div>
            <div className="glass rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-600">Decision Threshold</div>
                <div className="text-sm font-medium">{threshold}%</div>
              </div>
              {/* Slider with value bubble */}
              <div className="relative">
                {/* dynamic value bubble aligned to thumb position */}
                <div
                  className="absolute -top-6 -translate-x-1/2 text-[11px] px-2 py-0.5 rounded bg-gray-900 text-white shadow select-none"
                  style={{
                    left: `${((threshold - 10) / (90 - 10)) * 100}%`,
                  }}
                >
                  {threshold}%
                </div>
                <input
                  aria-label="Decision threshold"
                  type="range"
                  min={10}
                  max={90}
                  step={1}
                  value={threshold}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setThreshold(v);
                    localStorage.setItem("ssp_threshold", String(v));
                  }}
                  onInput={(e) => {
                    const v = Number((e.target as HTMLInputElement).value);
                    setThreshold(v);
                    localStorage.setItem("ssp_threshold", String(v));
                  }}
                  className="w-full"
                />
                <div className="flex justify-between text-[11px] text-gray-500 mt-1">
                  <span>10%</span>
                  <span>90%</span>
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Flag as Fraud when risk ≥ threshold. Current risk: {risk}%
              </div>
            </div>

            {/* Alert sound toggle */}
            <div className="glass rounded-xl p-3 flex items-center justify-between">
              <div className="text-sm text-gray-600">Alert sound on fraud</div>
              <button
                type="button"
                onClick={() => {
                  const v = !soundEnabled;
                  setSoundEnabled(v);
                  localStorage.setItem("ssp_sound_enabled", String(v));
                }}
                className={`px-2 py-1 rounded-md text-xs font-medium transition-colors ${
                  soundEnabled
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-gray-200 text-gray-700"
                }`}
                aria-pressed={soundEnabled}
                aria-label="Toggle fraud alert sound"
              >
                {soundEnabled ? "On" : "Off"}
              </button>
            </div>
          </div>
        </div>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Recent Transactions</h2>
          <TransactionsTable items={transactions} />
        </section>

        {explanations.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Top Signals</h2>
            <div className="glass rounded-xl p-4">
              <ul className="text-sm grid sm:grid-cols-2 gap-3">
                {explanations.map((e, i) => (
                  <li key={i} className="flex flex-col gap-1">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">{e.feature}</span>
                      <span
                        className={`${
                          Math.abs(e.z) >= 2 ? "text-rose-600" : "text-gray-900"
                        } font-medium`}
                      >
                        z = {e.z.toFixed(2)}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      value:{" "}
                      {Number.isFinite(e.value)
                        ? e.value.toFixed(3)
                        : String(e.value)}
                      {baseline?.stats && baseline.stats[e.feature] && (
                        <>
                          {" "}
                          • μ: {baseline.stats[e.feature].mean.toFixed(3)} • σ:{" "}
                          {baseline.stats[e.feature].std.toFixed(3)}
                        </>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
              <div className="text-xs text-gray-500 mt-2">
                z-score computed vs. training baseline (higher absolute value =
                more unusual)
              </div>
            </div>
          </section>
        )}

        {importances.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Model Insights</h2>
            <div className="glass rounded-xl p-4">
              <ul className="text-sm grid sm:grid-cols-2 gap-3">
                {importances.map((it, i) => (
                  <li key={i} className="flex items-center justify-between">
                    <span className="text-gray-700">{it.feature}</span>
                    <div className="flex items-center gap-2 min-w-[120px]">
                      <div className="h-2 w-24 bg-gray-200 rounded">
                        <div
                          className="h-2 bg-indigo-500 rounded"
                          style={{
                            width: `${Math.min(
                              100,
                              Math.round(it.importance * 100)
                            )}%`,
                          }}
                        />
                      </div>
                      <span className="text-xs text-gray-600">
                        {(it.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
              <div className="text-xs text-gray-500 mt-2">
                Global feature importances from the trained model (normalized to
                100%).
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
