import { useEffect, useState } from "react";
import { Navbar } from "../components/Navbar";
import { TransactionsTable } from "../components/TransactionsTable";

type Tx = {
  id: string;
  recipient: string;
  amount: number;
  date: string;
  risk: number;
  decision: "Fraud" | "Not Fraud";
};

export function History() {
  const [transactions, setTransactions] = useState<Tx[]>([]);

  useEffect(() => {
    const raw = localStorage.getItem("ssp_txs");
    const list: Tx[] = raw ? JSON.parse(raw) : [];
    setTransactions(list);
  }, []);

  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="mx-auto max-w-7xl p-4 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Transaction History</h1>
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

        <TransactionsTable items={transactions} />

        {transactions.length === 0 && (
          <div className="text-sm text-gray-600">No transactions yet.</div>
        )}
      </main>
    </div>
  );
}
