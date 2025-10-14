import { useAuth } from "../context/AuthContext";

export function Navbar() {
  const { user, logout } = useAuth();
  return (
    <header className="sticky top-0 z-30 bg-white/60 dark:bg-gray-900/60 backdrop-blur supports-[backdrop-filter]:bg-white/60 border-b border-gray-100 dark:border-gray-800">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-brand-500 to-brand-700"></div>
          <span className="font-semibold">SynthSecurePay</span>
        </div>
        <div className="flex items-center gap-3">
          {user && (
            <>
              {user.photoURL ? (
                <img
                  src={user.photoURL}
                  alt="avatar"
                  className="h-8 w-8 rounded-full"
                />
              ) : (
                <div className="h-8 w-8 rounded-full bg-gray-200" />
              )}
              <span className="hidden sm:block text-sm text-gray-600 dark:text-gray-300">
                {user.displayName || user.email || "User"}
              </span>
              <button
                className="btn-primary bg-gray-900 hover:bg-black dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-white"
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
