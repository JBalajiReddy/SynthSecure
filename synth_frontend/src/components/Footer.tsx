export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="mt-10 border-t border-gray-100 dark:border-gray-800 bg-white/60 dark:bg-gray-900/60 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="mx-auto max-w-7xl px-4 py-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-gray-700 dark:text-gray-300">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="font-semibold bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
                SynthSecure
              </span>
              <span className="hidden sm:inline text-[11px] text-gray-500">
                © {year}
              </span>
            </div>
          </div>

          <p className="text-center sm:text-right text-sm italic">
            “Security through clarity—detect, explain, and act.”
          </p>
        </div>
      </div>
    </footer>
  );
}
