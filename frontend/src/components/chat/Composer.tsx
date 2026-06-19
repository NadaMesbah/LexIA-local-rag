'use client';

import { useRef, useState } from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import CircularProgress from '@mui/material/CircularProgress';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import StopRoundedIcon from '@mui/icons-material/StopRounded';
import AttachFileRoundedIcon from '@mui/icons-material/AttachFileRounded';

const ACCEPT = '.pdf,.txt,.docx,.md,.csv,.xlsx';

export default function Composer({
  streaming,
  uploading,
  onSend,
  onStop,
  onAttach,
}: {
  streaming: boolean;
  uploading: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onAttach: (files: FileList) => void;
}) {
  const [value, setValue] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

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
          display: 'flex', alignItems: 'flex-end', gap: 1, p: 1, pl: 1,
          borderRadius: 3, border: '1px solid rgba(145,158,171,0.32)', bgcolor: 'background.paper',
          maxWidth: 900, mx: 'auto', '&:focus-within': { borderColor: 'primary.main' },
        }}
      >
        <input ref={fileRef} type="file" hidden multiple accept={ACCEPT}
          onChange={(e) => { if (e.target.files?.length) onAttach(e.target.files); if (fileRef.current) fileRef.current.value = ''; }} />
        <Tooltip title="Joindre des documents">
          <span>
            <IconButton onClick={() => fileRef.current?.click()} disabled={uploading} sx={{ color: 'text.secondary' }}>
              {uploading ? <CircularProgress size={20} /> : <AttachFileRoundedIcon />}
            </IconButton>
          </span>
        </Tooltip>

        <TextField
          fullWidth multiline maxRows={8} variant="standard" placeholder="Posez votre question…"
          value={value} onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
          InputProps={{ disableUnderline: true }}
          sx={{ '& textarea': { fontSize: 14, lineHeight: 1.6, py: 1 } }}
        />

        {streaming ? (
          <IconButton color="error" onClick={onStop} sx={{ bgcolor: 'rgba(255,86,48,0.12)' }}>
            <StopRoundedIcon />
          </IconButton>
        ) : (
          <IconButton color="primary" onClick={submit} disabled={!value.trim()}
            sx={{ bgcolor: 'primary.main', color: '#fff', '&:hover': { bgcolor: 'primary.dark' },
              '&.Mui-disabled': { bgcolor: 'rgba(145,158,171,0.24)', color: '#fff' } }}>
            <SendRoundedIcon />
          </IconButton>
        )}
      </Box>
      <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', mt: 1, color: 'text.disabled' }}>
        {uploading ? 'Indexation du document en cours…' : 'Entrée pour envoyer · Shift+Entrée pour un saut de ligne · 📎 pour joindre un document'}
      </Typography>
    </Box>
  );
}