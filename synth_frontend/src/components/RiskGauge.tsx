type Props = {
  score: number; // 0..100
};

export function RiskGauge({ score }: Props) {
  const clamped = Math.max(0, Math.min(100, score));
  const deg = (clamped / 100) * 180 - 90;
  const color =
    clamped < 40
      ? "text-emerald-500"
      : clamped < 70
      ? "text-amber-500"
      : "text-rose-500";
  return (
    <div className="glass rounded-xl p-4 text-black dark:text-gray-100">
      <div className="text-sm text-gray-500 mb-2">Fraud Risk</div>
      <div className="relative h-28">
        <div className="absolute inset-0 flex items-end justify-center">
          <div className={`text-3xl font-semibold ${color}`}>{clamped}%</div>
        </div>
        <div className="absolute inset-0">
          <svg viewBox="0 0 100 50" className="w-full h-full">
            <path
              d="M5,50 A45,45 0 0,1 95,50"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="10"
            />
            <path
              d="M5,50 A45,45 0 0,1 95,50"
              fill="none"
              stroke="url(#grad)"
              strokeWidth="10"
              strokeDasharray="141.37"
              strokeDashoffset={`${141.37 - (141.37 * clamped) / 100}`}
            />
            <defs>
              <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="50%" stopColor="#f59e0b" />
                <stop offset="100%" stopColor="#ef4444" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <div
          className="absolute left-1/2 bottom-0 origin-bottom -translate-x-1/2"
          style={{
            transform: `translateX(-50%) rotate(${deg}deg)`,
            transformOrigin: "bottom center",
          }}
        >
          <div className="w-0 h-0 border-l-8 border-r-8 border-b-[26px] border-l-transparent border-r-transparent border-b-gray-800 dark:border-b-white" />
        </div>
      </div>
    </div>
  );
}
