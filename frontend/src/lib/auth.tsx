'use client';

import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import * as api from './api';
import type { User } from './types';

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (data: api.RegisterInput) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  login: async () => {},
  register: async () => {},
  logout: () => {},
});

export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const logout = useCallback(() => {
    api.clearToken();
    setUser(null);
    router.replace('/login');
  }, [router]);

  // Bootstrap from a stored token; register the global 401 handler.
  useEffect(() => {
    api.setUnauthorizedHandler(() => {
      setUser(null);
      router.replace('/login');
    });
    const token = api.getToken();
    if (!token) {
      setLoading(false);
      return () => api.setUnauthorizedHandler(null);
    }
    api
      .me()
      .then((u) => setUser(u))
      .catch(() => api.clearToken())
      .finally(() => setLoading(false));
    return () => api.setUnauthorizedHandler(null);
  }, [router]);

  const login = useCallback(async (username: string, password: string) => {
    const res = await api.login(username, password);
    api.setToken(res.access_token);
    setUser(res.user);
  }, []);

  // Registration creates the account but does NOT log the user in — the sign-up
  // view then switches to the sign-in screen.
  const register = useCallback(async (data: api.RegisterInput) => {
    await api.register(data);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}