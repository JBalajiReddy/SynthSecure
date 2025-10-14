import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  auth,
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithPopup,
  signOut,
  firebaseAvailable,
  type User,
} from "../services/firebase";

type AuthUser =
  | Pick<User, "uid" | "displayName" | "email" | "photoURL">
  | {
      uid: string;
      displayName?: string | null;
      email?: string | null;
      photoURL?: string | null;
    };

type AuthContextType = {
  user: AuthUser | null;
  loading: boolean;
  login: () => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem("ssp_local_user");
    if (!firebaseAvailable() && stored) {
      setUser(JSON.parse(stored));
      setLoading(false);
      return;
    }

    if (firebaseAvailable() && auth) {
      const unsub = onAuthStateChanged(auth, (u) => {
        if (u) {
          const lite: AuthUser = {
            uid: u.uid,
            displayName: u.displayName,
            email: u.email,
            photoURL: u.photoURL,
          };
          setUser(lite);
        } else {
          setUser(null);
        }
        setLoading(false);
      });
      return () => unsub();
    }

    setLoading(false);
  }, []);

  const login = async () => {
    if (firebaseAvailable() && auth) {
      const provider = new GoogleAuthProvider();
      await signInWithPopup(auth, provider);
      return;
    }
    // Fallback: create a local mock user
    const mock = {
      uid: "local-user",
      displayName: "Local User",
      email: "local@example.com",
      photoURL: "",
    };
    localStorage.setItem("ssp_local_user", JSON.stringify(mock));
    setUser(mock);
  };

  const logout = async () => {
    if (firebaseAvailable() && auth) {
      await signOut(auth);
    }
    localStorage.removeItem("ssp_local_user");
    setUser(null);
  };

  const value = useMemo(
    () => ({ user, loading, login, logout }),
    [user, loading]
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
