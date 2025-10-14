type Tx = {
  id: string;
  recipient: string;
  amount: number;
  date: string;
  risk: number;
  decision: "Fraud" | "Not Fraud";
};

export function TransactionsTable({ items }: { items: Tx[] }) {
  return (
    <div className="glass rounded-xl overflow-hidden text-black dark:text-gray-100">
      <table className="min-w-full">
        <thead className="bg-gray-50/60 dark:bg-gray-800/60">
          <tr>
            <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider px-4 py-3">
              Recipient
            </th>
            <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider px-4 py-3">
              Amount
            </th>
            <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider px-4 py-3">
              Date
            </th>
            <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider px-4 py-3">
              Risk
            </th>
            <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider px-4 py-3">
              Decision
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((tx) => (
            <tr
              key={tx.id}
              className="border-t border-gray-100 dark:border-gray-800"
            >
              <td className="px-4 py-3">{tx.recipient}</td>
              <td className="px-4 py-3">₹{tx.amount.toFixed(2)}</td>
              <td className="px-4 py-3 text-gray-500 text-sm">
                {new Date(tx.date).toLocaleString()}
              </td>
              <td className="px-4 py-3">
                <span
                  className={`px-2 py-1 text-xs rounded-full ${
                    tx.risk < 40
                      ? "bg-emerald-100 text-emerald-700"
                      : tx.risk < 70
                      ? "bg-amber-100 text-amber-700"
                      : "bg-rose-100 text-rose-700"
                  }`}
                >
                  {tx.risk}%
                </span>
              </td>
              <td className="px-4 py-3">{tx.decision}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
