'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Alert from '@mui/material/Alert';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';
import { useAuth } from '@/lib/auth';

export default function LoginPage() {
  const { user, loading, login, register } = useAuth();
  const router = useRouter();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Already authenticated → leave the login screen.
  useEffect(() => {
    if (!loading && user) router.replace('/chat');
  }, [loading, user, router]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === 'login') await login(username.trim(), password);
      else await register(username.trim(), password);
      router.replace('/chat');
    } catch (err: any) {
      setError(err?.message || 'Échec de l’authentification.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'grid',
        placeItems: 'center',
        bgcolor: 'background.default',
        p: 2,
      }}
    >
      <Paper sx={{ p: 4, width: '100%', maxWidth: 400, borderRadius: 3 }} elevation={0}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, mb: 0.5 }}>
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: 1.5,
              bgcolor: 'primary.main',
              color: 'primary.contrastText',
              display: 'grid',
              placeItems: 'center',
            }}
          >
            <AutoAwesomeRoundedIcon fontSize="small" />
          </Box>
          <Typography variant="h6" sx={{ fontWeight: 800 }}>
            RAG&nbsp;<Box component="span" sx={{ color: 'primary.main' }}>Local</Box>
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Connectez-vous pour accéder à votre base de connaissance privée.
        </Typography>

        <Tabs
          value={mode}
          onChange={(_, v) => {
            setMode(v);
            setError(null);
          }}
          sx={{ mb: 2 }}
        >
          <Tab value="login" label="Connexion" />
          <Tab value="register" label="Créer un compte" />
        </Tabs>

        <form onSubmit={submit}>
          <TextField
            fullWidth
            label="Nom d’utilisateur"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            type="password"
            label="Mot de passe"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            sx={{ mb: 2 }}
          />
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          <Button
            fullWidth
            type="submit"
            variant="contained"
            size="large"
            disabled={busy || !username.trim() || !password}
          >
            {mode === 'login' ? 'Se connecter' : 'Créer le compte'}
          </Button>
        </form>
      </Paper>
    </Box>
  );
}
