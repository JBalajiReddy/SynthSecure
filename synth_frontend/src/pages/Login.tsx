import { useAuth } from "../context/AuthContext";
import { Navigate } from "react-router-dom";

export function Login() {
  const { login, user } = useAuth();

  if (user) {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="relative min-h-screen grid place-items-center p-6 overflow-hidden">
      {/* background accents */}
      <div className="pointer-events-none absolute -top-16 -left-16 h-72 w-72 rounded-full bg-gradient-to-tr from-indigo-500 to-fuchsia-600 opacity-25 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 -right-24 h-96 w-96 rounded-full bg-gradient-to-tr from-fuchsia-600 to-indigo-500 opacity-20 blur-3xl" />

      <div className="relative glass rounded-2xl p-8 w-full max-w-md text-center shadow-lg">
        <h1 className="text-2xl font-semibold">
          <span className="bg-gradient-to-r from-indigo-600 to-fuchsia-600 bg-clip-text text-transparent">
            Welcome to SynthSecure Pay
          </span>
        </h1>
        <p className="text-gray-600 dark:text-gray-300 mt-2">
          Secure. Fast. AI-powered fraud detection.
        </p>

        <button
          className="mt-7 w-full inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 bg-gray-900 text-white hover:bg-black dark:bg-white dark:text-gray-900 dark:hover:bg-gray-100 transition-colors shadow"
          onClick={login}
        >
          <span className="text-base">Sign-in with Google</span>
        </button>
        {/* <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">
          If Firebase isn’t configured, we’ll log you in locally for demo.
        </p> */}
      </div>
    </div>
  );
}
