'use client';

import { useRef, useState } from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import StopRoundedIcon from '@mui/icons-material/StopRounded';
import AttachFileRoundedIcon from '@mui/icons-material/AttachFileRounded';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import ErrorOutlineRoundedIcon from '@mui/icons-material/ErrorOutlineRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import FilterAltRoundedIcon from '@mui/icons-material/FilterAltRounded';
import type { IndexingStatus, PendingAttachment } from '@/lib/types';

const ACCEPT = '.pdf,.txt,.docx,.md,.csv,.xlsx';

export type { PendingAttachment };

export default function Composer({
  streaming,
  uploading,
  attachments = [],
  strict = false,
  onStrictChange,
  scopeDocs = [],
  onClearScope,
  onSend,
  onStop,
  onAttach,
  onRemoveAttachment,
}: {
  streaming: boolean;
  uploading: boolean;
  attachments?: PendingAttachment[];
  strict?: boolean;
  onStrictChange?: (v: boolean) => void;
  scopeDocs?: { id: string; filename: string }[];
  onClearScope?: () => void;
  onSend: (text: string) => void;
  onStop: () => void;
  onAttach: (files: FileList) => void;
  onRemoveAttachment?: (id: string) => void;
}) {
  const [value, setValue] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    const t = value.trim();
    if (!t || streaming) return;
    setValue('');
    onSend(t);
  };

  const statusIcon = (s: IndexingStatus) => {
    if (s === 'indexed') return <CheckCircleRoundedIcon sx={{ color: 'success.main' }} />;
    if (s === 'failed') return <ErrorOutlineRoundedIcon sx={{ color: 'error.main' }} />;
    return <CircularProgress size={14} />;
  };

  return (
    <Box sx={{ px: { xs: 2, md: 6 }, pb: 2.5, pt: 1 }}>
      <Box sx={{ maxWidth: 900, mx: 'auto', mb: 1, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
        {scopeDocs.length > 0 && (
          <Tooltip title="Recherche limitée à ce(s) document(s) pour toute la conversation. Cliquez sur ✕ pour revenir à toute la base.">
            <Chip
              icon={<FilterAltRoundedIcon />}
              label={scopeDocs.length === 1 ? `Portée : ${scopeDocs[0].filename}` : `Portée : ${scopeDocs.length} documents`}
              color="secondary"
              variant="filled"
              onDelete={onClearScope}
              size="small"
            />
          </Tooltip>
        )}
        <Tooltip title="Mode RAG strict : répondre uniquement à partir des documents (refuse si l'information n'y figure pas).">
          <Chip
            icon={<FactCheckRoundedIcon />}
            label="Documents uniquement"
            color={strict ? 'primary' : 'default'}
            variant={strict ? 'filled' : 'outlined'}
            onClick={() => onStrictChange?.(!strict)}
            size="small"
            sx={{ cursor: 'pointer' }}
          />
        </Tooltip>
      </Box>
      {attachments.length > 0 && (
        <Box sx={{ maxWidth: 900, mx: 'auto', mb: 1, display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {attachments.map((a) => (
            <Chip
              key={a.id}
              icon={statusIcon(a.status)}
              label={
                a.status === 'indexed'
                  ? a.filename
                  : a.status === 'failed'
                    ? `${a.filename} (échec)`
                    : `${a.filename} · indexation…`
              }
              onDelete={onRemoveAttachment ? () => onRemoveAttachment(a.id) : undefined}
              size="small"
              variant="outlined"
              sx={{ maxWidth: 280, borderColor: 'rgba(1,134,135,0.4)' }}
            />
          ))}
        </Box>
      )}
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
          fullWidth multiline maxRows={8} variant="standard" placeholder="Posez votre question à LexIA…"
          value={value} onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
          InputProps={{ disableUnderline: true }}
          inputProps={{ dir: 'auto' }}
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
        {uploading ? 'Indexation du document en cours…' : <>LexIA is AI and can make mistakes. Please double-check responses.</>}
      </Typography>
    </Box>
  );
}