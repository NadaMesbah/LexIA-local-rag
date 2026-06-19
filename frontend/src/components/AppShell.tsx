'use client';

import { usePathname, useRouter } from 'next/navigation';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import ChatBubbleOutlineRoundedIcon from '@mui/icons-material/ChatBubbleOutlineRounded';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';

const NAV_WIDTH = 260;

const NAV = [
  { label: 'Chat', href: '/chat', icon: <ChatBubbleOutlineRoundedIcon /> },
  { label: 'Documents', href: '/documents', icon: <FolderOutlinedIcon /> },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

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
        <Box sx={{ p: 2.5 }}>
          <Box
            sx={{
              p: 2,
              borderRadius: 2,
              bgcolor: 'rgba(0,167,111,0.08)',
              textAlign: 'center',
            }}
          >
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
              100% local & privé
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.disabled' }}>
              Vos documents ne quittent pas la machine
            </Typography>
          </Box>
        </Box>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, minWidth: 0, height: '100%', overflow: 'hidden' }}>
        {children}
      </Box>
    </Box>
  );
}
