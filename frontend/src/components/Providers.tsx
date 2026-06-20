'use client';

import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from '@/theme';
import { AuthProvider } from '@/lib/auth';
import { ChatStreamProvider } from '@/lib/chatStream';

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <ChatStreamProvider>{children}</ChatStreamProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}