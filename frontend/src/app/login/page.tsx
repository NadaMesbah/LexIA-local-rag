'use client';

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Link from '@mui/material/Link';
import Alert from '@mui/material/Alert';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import InputAdornment from '@mui/material/InputAdornment';
import CircularProgress from '@mui/material/CircularProgress';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';
import VisibilityRoundedIcon from '@mui/icons-material/VisibilityRounded';
import VisibilityOffRoundedIcon from '@mui/icons-material/VisibilityOffRounded';
import VerifiedUserRoundedIcon from '@mui/icons-material/VerifiedUserRounded';
import FolderSpecialRoundedIcon from '@mui/icons-material/FolderSpecialRounded';
import ChatRoundedIcon from '@mui/icons-material/ChatRounded';
import { useAuth } from '@/lib/auth';

type Mode = 'login' | 'register';

const EMPTY = { firstName: '', lastName: '', username: '', email: '', password: '', confirm: '' };

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export default function LoginPage() {
  const { user, loading, login, register } = useAuth();
  const router = useRouter();

  const [mode, setMode] = useState<Mode>('login');
  const [form, setForm] = useState({ ...EMPTY });
  const [showPwd, setShowPwd] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Already authenticated → leave the auth screen.
  useEffect(() => {
    if (!loading && user) router.replace('/chat');
  }, [loading, user, router]);

  const set = (k: keyof typeof EMPTY) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const switchMode = (m: Mode) => {
    setMode(m);
    setError(null);
    setShowPwd(false);
    setShowConfirm(false);
    if (m === 'register') setSuccess(null);
  };

  const canSubmit = useMemo(() => {
    if (mode === 'login') return EMAIL_RE.test(form.email.trim()) && form.password.length > 0;
    return (
      form.firstName.trim() &&
      form.lastName.trim() &&
      form.username.trim().length >= 3 &&
      EMAIL_RE.test(form.email.trim()) &&
      form.password.length >= 8 &&
      form.confirm.length > 0
    );
  }, [mode, form]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (mode === 'register' && form.password !== form.confirm) {
      setError('Les mots de passe ne correspondent pas.');
      return;
    }

    setBusy(true);
    try {
      if (mode === 'login') {
        await login(form.email.trim(), form.password);
        router.replace('/chat');
      } else {
        await register({
          first_name: form.firstName.trim(),
          last_name: form.lastName.trim(),
          username: form.username.trim(),
          email: form.email.trim(),
          password: form.password,
        });
        // Account created — send the user to the sign-in screen.
        setForm({ ...EMPTY });
        switchMode('login');
        setSuccess('Compte créé avec succès. Connectez-vous pour continuer.');
      }
    } catch (err: any) {
      setError(err?.message || 'Une erreur est survenue.');
    } finally {
      setBusy(false);
    }
  };

  const pwdAdornment = (visible: boolean, toggle: () => void) => ({
    endAdornment: (
      <InputAdornment position="end">
        <IconButton onClick={toggle} edge="end" size="small">
          {visible ? <VisibilityRoundedIcon fontSize="small" /> : <VisibilityOffRoundedIcon fontSize="small" />}
        </IconButton>
      </InputAdornment>
    ),
  });

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', bgcolor: 'background.default' }}>
      {/* Left brand / marketing panel */}
      <Box
        sx={{
          display: { xs: 'none', md: 'flex' },
          flexDirection: 'column',
          justifyContent: 'center',
          width: '46%',
          p: 8,
          color: 'common.white',
          background: 'linear-gradient(135deg, #016F70 0%, #018687 55%, #01B7B7 100%)',
        }}
      >
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 5 }}>
          <Box
            sx={{
              width: 44,
              height: 44,
              borderRadius: 2,
              bgcolor: 'rgba(255,255,255,0.16)',
              display: 'grid',
              placeItems: 'center',
            }}
          >
            <AutoAwesomeRoundedIcon />
          </Box>
          <Typography variant="h5" sx={{ fontWeight: 800 }}>
            LexIA
          </Typography>
        </Stack>

        <Typography variant="h3" sx={{ fontWeight: 800, lineHeight: 1.2, mb: 2 }}>
          Votre base de connaissance,
          <br />
          privée et locale.
        </Typography>

        <Stack spacing={2.5} sx={{ maxWidth: 440 }}>
          {[
            { icon: <ChatRoundedIcon />, t: 'Chat intelligent', d: 'Réponses sourcées sur vos documents.' },
            { icon: <FolderSpecialRoundedIcon />, t: 'Documents versionnés', d: 'Historique, restauration et corbeille.' },
            { icon: <VerifiedUserRoundedIcon />, t: 'Accès contrôlé', d: 'Espace partagé + collections personnelles.' },
          ].map((f) => (
            <Stack key={f.t} direction="row" spacing={2} alignItems="flex-start">
              <Box
                sx={{
                  width: 38,
                  height: 38,
                  flexShrink: 0,
                  borderRadius: 1.5,
                  bgcolor: 'rgba(255,255,255,0.16)',
                  display: 'grid',
                  placeItems: 'center',
                }}
              >
                {f.icon}
              </Box>
              <Box>
                <Typography sx={{ fontWeight: 700 }}>{f.t}</Typography>
                <Typography variant="body2" sx={{ opacity: 0.85 }}>
                  {f.d}
                </Typography>
              </Box>
            </Stack>
          ))}
        </Stack>
      </Box>

      {/* Right form panel */}
      <Box
        sx={{
          flexGrow: 1,
          display: 'grid',
          placeItems: 'center',
          p: { xs: 2, sm: 4 },
        }}
      >
        <Paper elevation={0} sx={{ p: { xs: 3, sm: 5 }, width: '100%', maxWidth: 440, borderRadius: 3 }}>
          {/* Mobile brand */}
          <Stack
            direction="row"
            spacing={1.25}
            alignItems="center"
            sx={{ mb: 3, display: { xs: 'flex', md: 'none' } }}
          >
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
              Lex<Box component="span" sx={{ color: 'primary.main' }}>IA</Box>
            </Typography>
          </Stack>

          <Typography variant="h4" sx={{ fontWeight: 800, mb: 0.5 }}>
            {mode === 'login' ? 'Bon retour' : 'Créer un compte'}
          </Typography>
          <Stack direction="row" spacing={0.5} sx={{ mb: 4 }}>
            <Typography variant="body2" color="text.secondary">
              {mode === 'login' ? 'Pas encore de compte ?' : 'Vous avez déjà un compte ?'}
            </Typography>
            <Link
              component="button"
              type="button"
              variant="subtitle2"
              onClick={() => switchMode(mode === 'login' ? 'register' : 'login')}
              sx={{ fontWeight: 700 }}
            >
              {mode === 'login' ? 'Inscrivez-vous' : 'Se connecter'}
            </Link>
          </Stack>

          {success && (
            <Alert severity="success" sx={{ mb: 3 }}>
              {success}
            </Alert>
          )}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          <form onSubmit={submit}>
            <Stack spacing={2.5}>
              {mode === 'register' && (
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                  <TextField
                    fullWidth
                    label="Prénom"
                    value={form.firstName}
                    onChange={set('firstName')}
                    autoComplete="given-name"
                  />
                  <TextField
                    fullWidth
                    label="Nom"
                    value={form.lastName}
                    onChange={set('lastName')}
                    autoComplete="family-name"
                  />
                </Stack>
              )}

              {mode === 'register' && (
                <TextField
                  fullWidth
                  label="Nom d'utilisateur"
                  value={form.username}
                  onChange={set('username')}
                  autoComplete="username"
                  helperText="3 caractères minimum"
                />
              )}

              <TextField
                fullWidth
                type="email"
                label="Adresse e-mail"
                value={form.email}
                onChange={set('email')}
                autoComplete="email"
                error={!!form.email && !EMAIL_RE.test(form.email.trim())}
                helperText={
                  !!form.email && !EMAIL_RE.test(form.email.trim())
                    ? 'Adresse e-mail invalide.'
                    : mode === 'login'
                      ? 'Connectez-vous avec votre e-mail.'
                      : undefined
                }
              />

              <TextField
                fullWidth
                label="Mot de passe"
                type={showPwd ? 'text' : 'password'}
                value={form.password}
                onChange={set('password')}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                helperText={mode === 'register' ? '8 caractères minimum' : undefined}
                InputProps={pwdAdornment(showPwd, () => setShowPwd((v) => !v))}
              />

              {mode === 'register' && (
                <TextField
                  fullWidth
                  label="Confirmer le mot de passe"
                  type={showConfirm ? 'text' : 'password'}
                  value={form.confirm}
                  onChange={set('confirm')}
                  autoComplete="new-password"
                  error={!!form.confirm && form.confirm !== form.password}
                  helperText={
                    !!form.confirm && form.confirm !== form.password
                      ? 'Les mots de passe ne correspondent pas.'
                      : undefined
                  }
                  InputProps={pwdAdornment(showConfirm, () => setShowConfirm((v) => !v))}
                />
              )}

              <Button
                fullWidth
                type="submit"
                variant="contained"
                size="large"
                disabled={busy || !canSubmit}
                startIcon={busy ? <CircularProgress size={18} color="inherit" /> : undefined}
              >
                {mode === 'login' ? 'Se connecter' : 'Créer le compte'}
              </Button>
            </Stack>
          </form>
        </Paper>
      </Box>
    </Box>
  );
}