type Props = {
  title: string;
  value: string | number;
  trend?: string;
};

export function MetricCard({ title, value, trend }: Props) {
  return (
    <div className="glass rounded-xl p-4 text-black dark:text-gray-100">
      <div className="text-sm text-gray-500 mb-1">{title}</div>
      <div className="text-2xl font-semibold">{value}</div>
      {trend && <div className="text-xs text-emerald-600 mt-1">{trend}</div>}
    </div>
  );
}
