import { useAuth } from "../context/AuthContext";
import { Navigate } from "react-router-dom";

export function Login() {
  const { login, user } = useAuth();

  if (user) {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="min-h-screen grid place-items-center p-6">
      <div className="glass rounded-2xl p-8 w-full max-w-md text-center">
        <div className="mx-auto h-12 w-12 rounded-xl bg-gradient-to-tr from-brand-500 to-brand-700 mb-4" />
        <h1 className="text-2xl font-semibold">Welcome to SynthSecurePay</h1>
        <p className="text-gray-600 mt-1">
          Secure. Fast. AI-powered fraud detection.
        </p>
        <button className="btn-primary mt-6 w-full" onClick={login}>
          Sign in with Google (or continue)
        </button>
        <p className="text-xs text-gray-500 mt-3">
          If Firebase isn’t configured, we’ll log you in locally for demo.
        </p>
      </div>
    </div>
  );
}
