'use client';

import { useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import Avatar from '@mui/material/Avatar';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import CircularProgress from '@mui/material/CircularProgress';
import ChatBubbleOutlineRoundedIcon from '@mui/icons-material/ChatBubbleOutlineRounded';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';
import LogoutRoundedIcon from '@mui/icons-material/LogoutRounded';
import { useAuth } from '@/lib/auth';

const NAV_WIDTH = 260;

const NAV = [
  { label: 'Chat', href: '/chat', icon: <ChatBubbleOutlineRoundedIcon /> },
  { label: 'Documents', href: '/documents', icon: <FolderOutlinedIcon /> },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, logout } = useAuth();

  const isLogin = pathname === '/login';

  // Redirect unauthenticated users to the login screen.
  useEffect(() => {
    if (!loading && !user && !isLogin) router.replace('/login');
  }, [loading, user, isLogin, router]);

  // The login page renders without the shell chrome.
  if (isLogin) return <>{children}</>;

  // Bootstrapping the session, or about to redirect.
  if (loading || !user) {
    return (
      <Box sx={{ height: '100vh', display: 'grid', placeItems: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: NAV_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: NAV_WIDTH,
            boxSizing: 'border-box',
            borderRight: '1px dashed rgba(145,158,171,0.2)',
            bgcolor: 'background.paper',
          },
        }}
      >
        {/* Brand */}
        <Box sx={{ px: 2.5, py: 3, display: 'flex', alignItems: 'center', gap: 1.25 }}>
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

        <Typography
          variant="caption"
          sx={{ px: 2.5, color: 'text.disabled', fontWeight: 700, letterSpacing: 0.6 }}
        >
          GÉNÉRAL
        </Typography>

        <List sx={{ px: 1.5, pt: 1 }}>
          {NAV.map((item) => {
            const active = pathname === item.href || pathname?.startsWith(item.href + '/');
            return (
              <ListItemButton
                key={item.href}
                onClick={() => router.push(item.href)}
                selected={active}
                sx={{
                  borderRadius: 1.5,
                  mb: 0.5,
                  color: active ? 'primary.main' : 'text.secondary',
                  '&.Mui-selected': {
                    bgcolor: 'rgba(0,167,111,0.08)',
                    '&:hover': { bgcolor: 'rgba(0,167,111,0.16)' },
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 36, color: 'inherit' }}>{item.icon}</ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{ fontWeight: active ? 700 : 600, fontSize: 14 }}
                />
              </ListItemButton>
            );
          })}
        </List>

        <Box sx={{ flexGrow: 1 }} />

        {/* Account card + logout */}
        <Box sx={{ p: 2 }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.25,
              p: 1.25,
              borderRadius: 2,
              bgcolor: 'rgba(145,158,171,0.08)',
            }}
          >
            <Avatar sx={{ width: 34, height: 34, bgcolor: 'primary.main', fontSize: 15, fontWeight: 700 }}>
              {user.username.slice(0, 1).toUpperCase()}
            </Avatar>
            <Box sx={{ minWidth: 0, flexGrow: 1 }}>
              <Typography variant="body2" noWrap sx={{ fontWeight: 700 }}>
                {user.username}
              </Typography>
              <Chip
                label={user.role === 'admin' ? 'Administrateur' : 'Utilisateur'}
                size="small"
                color={user.role === 'admin' ? 'primary' : 'default'}
                sx={{ height: 18, fontSize: 10, fontWeight: 700 }}
              />
            </Box>
            <Tooltip title="Se déconnecter">
              <IconButton size="small" onClick={logout} sx={{ '&:hover': { color: 'error.main' } }}>
                <LogoutRoundedIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, minWidth: 0, height: '100%', overflow: 'hidden' }}>
        {children}
      </Box>
    </Box>
  );
}
