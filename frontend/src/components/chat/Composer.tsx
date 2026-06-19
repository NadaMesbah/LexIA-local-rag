'use client';

import { useState } from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import StopRoundedIcon from '@mui/icons-material/StopRounded';

export default function Composer({
  streaming,
  onSend,
  onStop,
}: {
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [value, setValue] = useState('');

  const submit = () => {
    const t = value.trim();
    if (!t || streaming) return;
    setValue('');
    onSend(t);
  };

  return (
    <Box sx={{ px: { xs: 2, md: 6 }, pb: 2.5, pt: 1 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 1,
          p: 1,
          pl: 2,
          borderRadius: 3,
          border: '1px solid rgba(145,158,171,0.32)',
          bgcolor: 'background.paper',
          maxWidth: 900,
          mx: 'auto',
          '&:focus-within': { borderColor: 'primary.main' },
        }}
      >
        <TextField
          fullWidth
          multiline
          maxRows={8}
          variant="standard"
          placeholder="Posez votre question…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          InputProps={{ disableUnderline: true }}
          sx={{ '& textarea': { fontSize: 14, lineHeight: 1.6 } }}
        />
        {streaming ? (
          <IconButton color="error" onClick={onStop} sx={{ bgcolor: 'rgba(255,86,48,0.12)' }}>
            <StopRoundedIcon />
          </IconButton>
        ) : (
          <IconButton
            color="primary"
            onClick={submit}
            disabled={!value.trim()}
            sx={{ bgcolor: 'primary.main', color: '#fff', '&:hover': { bgcolor: 'primary.dark' }, '&.Mui-disabled': { bgcolor: 'rgba(145,158,171,0.24)', color: '#fff' } }}
          >
            <SendRoundedIcon />
          </IconButton>
        )}
      </Box>
      <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', mt: 1, color: 'text.disabled' }}>
        Entrée pour envoyer · Maj+Entrée pour un saut de ligne
      </Typography>
    </Box>
  );
}
