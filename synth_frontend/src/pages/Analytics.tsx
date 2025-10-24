import { useEffect, useMemo, useState } from "react";
import { Navbar } from "../components/Navbar";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip as ReTooltip,
  Legend,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";

type Tx = {
  id: string;
  recipient: string;
  amount: number;
  date: string;
  risk: number;
  decision: "Fraud" | "Not Fraud";
};

const COLORS = ["#ef4444", "#10b981", "#6366f1", "#f59e0b", "#06b6d4"]; // rose, emerald, indigo, amber, cyan

export function Analytics() {
  const [transactions, setTransactions] = useState<Tx[]>([]);

  useEffect(() => {
    const raw = localStorage.getItem("ssp_txs");
    const list: Tx[] = raw ? JSON.parse(raw) : [];
    setTransactions(list);
  }, []);

  const { pieData, riskSeries, riskHistogram, topRecipients, avgAmount } =
    useMemo(() => {
      const total = transactions.length;

      // Pie breakdown by decision
      const fraudCount = transactions.filter(
        (t) => t.decision === "Fraud"
      ).length;
      const safeCount = total - fraudCount;
      const pieData = [
        { name: "Fraud", value: fraudCount },
        { name: "Not Fraud", value: safeCount },
      ];

      // Risk over time (sorted by date asc)
      const sorted = [...transactions].sort(
        (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
      );
      const riskSeries = sorted.map((t, idx) => ({
        idx,
        time: new Date(t.date).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
        risk: t.risk,
        decision: t.decision,
      }));

      // Risk histogram (0-100 buckets of width 10)
      const buckets: { range: string; count: number }[] = [];
      for (let start = 0; start < 100; start += 10) {
        buckets.push({ range: `${start}-${start + 9}`, count: 0 });
      }
      for (const t of transactions) {
        const idx = Math.min(9, Math.floor(t.risk / 10));
        buckets[idx].count += 1;
      }
      const riskHistogram = buckets;

      // Top recipients by count
      const byRecipient: Record<
        string,
        { count: number; fraud: number; totalAmount: number }
      > = {};
      for (const t of transactions) {
        const rec = (byRecipient[t.recipient] ||= {
          count: 0,
          fraud: 0,
          totalAmount: 0,
        });
        rec.count += 1;
        rec.totalAmount += t.amount;
        if (t.decision === "Fraud") rec.fraud += 1;
      }
      const topRecipients = Object.entries(byRecipient)
        .map(([recipient, v]) => ({ recipient, ...v }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5);

      // Avg amount by decision
      const sum = { Fraud: 0, Safe: 0, fN: 0, sN: 0 };
      for (const t of transactions) {
        if (t.decision === "Fraud") {
          sum.Fraud += t.amount;
          sum.fN += 1;
        } else {
          sum.Safe += t.amount;
          sum.sN += 1;
        }
      }
      const avgAmount = [
        { name: "Fraud", value: sum.fN ? sum.Fraud / sum.fN : 0 },
        { name: "Not Fraud", value: sum.sN ? sum.Safe / sum.sN : 0 },
      ];

      return { pieData, riskSeries, riskHistogram, topRecipients, avgAmount };
    }, [transactions]);

  const hasData = transactions.length > 0;

  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="mx-auto max-w-7xl p-4 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Analytics</h1>
          <button
            type="button"
            onClick={() => {
              const raw = localStorage.getItem("ssp_txs");
              const list: Tx[] = raw ? JSON.parse(raw) : [];
              setTransactions(list);
            }}
            className="text-sm px-3 py-1 rounded-md border border-gray-200 dark:border-gray-700 bg-white/70 dark:bg-gray-800/70 text-gray-700 dark:text-gray-200 hover:bg-white dark:hover:bg-gray-800 transition-colors"
          >
            Refresh
          </button>
        </div>

        {!hasData && (
          <div className="glass rounded-xl p-6 text-gray-600">
            No data yet. Submit a few predictions to populate analytics.
          </div>
        )}

        {hasData && (
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Pie breakdown */}
            <div className="glass rounded-xl p-4">
              <h2 className="text-sm font-medium text-gray-700 mb-3">
                Decision breakdown
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                    >
                      {pieData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={index === 0 ? COLORS[0] : COLORS[1]}
                        />
                      ))}
                    </Pie>
                    <Legend />
                    <ReTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Risk over time */}
            <div className="glass rounded-xl p-4">
              <h2 className="text-sm font-medium text-gray-700 mb-3">
                Risk over time
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={riskSeries} margin={{ left: 8, right: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                    <ReTooltip />
                    <Line
                      type="monotone"
                      dataKey="risk"
                      stroke="#6366f1"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Risk histogram */}
            <div className="glass rounded-xl p-4">
              <h2 className="text-sm font-medium text-gray-700 mb-3">
                Risk distribution
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riskHistogram} margin={{ left: 8, right: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" tick={{ fontSize: 12 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <ReTooltip />
                    <Bar dataKey="count" fill="#06b6d4" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Top recipients */}
            <div className="glass rounded-xl p-4">
              <h2 className="text-sm font-medium text-gray-700 mb-3">
                Top recipients (by checks)
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topRecipients} margin={{ left: 8, right: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="recipient"
                      tick={{ fontSize: 12 }}
                      hide={topRecipients.length > 4}
                    />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <ReTooltip />
                    <Bar dataKey="count" name="Total" fill="#6366f1" />
                    <Bar dataKey="fraud" name="Fraud" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Avg amount by decision */}
            <div className="glass rounded-xl p-4">
              <h2 className="text-sm font-medium text-gray-700 mb-3">
                Average amount by decision
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={avgAmount} margin={{ left: 8, right: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <ReTooltip />
                    <Bar dataKey="value" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
