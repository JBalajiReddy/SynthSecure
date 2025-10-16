import { useEffect, useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function Navbar() {
  const { user, logout } = useAuth();
  const [avatarError, setAvatarError] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("ssp_theme");
    if (saved === "light" || saved === "dark") return saved;
    const prefersDark =
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    return prefersDark ? "dark" : "light";
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
    localStorage.setItem("ssp_theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  // Helpers for fallback avatar
  const getInitials = (name?: string | null, email?: string | null) => {
    const basis =
      (name && name.trim()) || (email && email.split("@")[0]) || "U";
    const parts = basis.split(/\s+/).filter(Boolean);
    const initials =
      parts.length >= 2 ? parts[0][0] + parts[1][0] : basis.slice(0, 2);
    return initials.toUpperCase();
  };
  const colorFromString = (s: string) => {
    // simple hash -> hue
    let hash = 0;
    for (let i = 0; i < s.length; i++)
      hash = s.charCodeAt(i) + ((hash << 5) - hash);
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue} 65% 45%)`;
  };
  return (
    <header className="sticky top-0 z-30 border-b border-gray-100 dark:border-gray-800 bg-white/60 dark:bg-gray-900/60 backdrop-blur supports-[backdrop-filter]:bg-white/60 shadow-sm">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
        {/* Brand */}
        <Link to="/" className="group flex items-center gap-3">
          {/* <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-indigo-500 to-fuchsia-600 p-[2px] shadow-sm">
            <div className="h-full w-full rounded-md bg-white/70 dark:bg-gray-900/70 flex items-center justify-center overflow-hidden transition-transform group-hover:scale-[1.02]">
              <img
                src="/logo/SynthSecure.svg"
                alt="SynthSecure logo"
                className="h-6 w-6 select-none pointer-events-none"
                draggable="false"
              />
            </div>
          </div> */}
          <span className="hidden sm:inline-block font-semibold bg-gradient-to-r from-indigo-600 to-fuchsia-600 bg-clip-text text-transparent">
            SynthSecure Pay
          </span>
        </Link>

        {/* Right actions */}
        <div className="flex items-center gap-3">
          {/* Nav links */}
          <nav className="hidden sm:flex items-center gap-2 mr-2">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `text-sm px-2 py-1 rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500/40 dark:focus:ring-fuchsia-400/40 ${
                  isActive
                    ? "bg-gradient-to-r from-indigo-600/10 to-fuchsia-600/10 text-indigo-700 dark:text-fuchsia-200 border border-indigo-500/30 dark:border-fuchsia-400/30 shadow-sm"
                    : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                }`
              }
            >
              Home
            </NavLink>
            <NavLink
              to="/history"
              className={({ isActive }) =>
                `text-sm px-2 py-1 rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500/40 dark:focus:ring-fuchsia-400/40 ${
                  isActive
                    ? "bg-gradient-to-r from-indigo-600/10 to-fuchsia-600/10 text-indigo-700 dark:text-fuchsia-200 border border-indigo-500/30 dark:border-fuchsia-400/30 shadow-sm"
                    : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                }`
              }
            >
              History
            </NavLink>
          </nav>
          {/* Theme toggle */}
          <button
            type="button"
            onClick={toggleTheme}
            title={
              theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
            }
            className="h-9 w-9 inline-flex items-center justify-center rounded-md border border-gray-200 dark:border-gray-700 bg-white/70 dark:bg-gray-800/70 hover:bg-white dark:hover:bg-gray-800 transition-colors"
          >
            <span className="text-lg" aria-hidden>
              {theme === "dark" ? "🌙" : "☀️"}
            </span>
            <span className="sr-only">Toggle theme</span>
          </button>
          {user && (
            <>
              {user.photoURL && !avatarError ? (
                <img
                  src={user.photoURL}
                  alt="avatar"
                  className="h-8 w-8 rounded-full ring-2 ring-white/70 dark:ring-gray-900/70 object-cover"
                  onError={() => setAvatarError(true)}
                />
              ) : (
                <div
                  className="h-8 w-8 rounded-full grid place-items-center text-white text-xs font-semibold select-none"
                  style={{
                    background: colorFromString(
                      (user.displayName || user.email || "User") as string
                    ),
                  }}
                  aria-label="user initials avatar"
                >
                  {getInitials(user.displayName, user.email)}
                </div>
              )}
              <span className="hidden sm:block text-sm text-gray-600 dark:text-gray-300">
                {user.displayName || user.email || "User"}
              </span>
              <button
                className="btn-primary bg-gray-900 hover:bg-black dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-white border border-transparent hover:border-gray-800 dark:hover:border-gray-200 transition-colors"
                onClick={logout}
              >
                Logout
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
