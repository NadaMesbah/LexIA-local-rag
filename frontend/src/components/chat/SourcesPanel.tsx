'use client';

import { useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import CircularProgress from '@mui/material/CircularProgress';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined';
import type { DocumentItem, Scope, Source } from '@/lib/types';

function SourceCard({ s }: { s: Source }) {
  const [open, setOpen] = useState(false);
  return (
    <Box onClick={() => setOpen((o) => !o)} sx={{
      p: 1.25, mb: 1, borderRadius: 1.5, border: '1px solid rgba(145,158,171,0.2)',
      cursor: 'pointer', '&:hover': { borderColor: 'primary.light' },
    }}>
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 0.5 }}>
        <DescriptionOutlinedIcon fontSize="small" sx={{ color: 'primary.main' }} />
        <Typography variant="caption" sx={{ fontWeight: 700, flexGrow: 1, wordBreak: 'break-all' }}>
          {s.filename}{s.version ? ` · v${s.version}` : ''}
        </Typography>
        {s.page && s.page !== '?' && <Chip label={`p.${s.page}`} size="small" sx={{ height: 18, fontSize: 10 }} />}
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{
        display: '-webkit-box', WebkitLineClamp: open ? 'unset' : 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
      }}>
        {s.text}
      </Typography>
    </Box>
  );
}

export default function SourcesPanel({
  documents, lastSources, scope, onScopeChange, onUploaded, uploadFn,
}: {
  documents: DocumentItem[];
  lastSources: Source[];
  scope: Scope;
  onScopeChange: (s: Scope) => void;
  onUploaded: () => void;
  uploadFn: (file: File) => Promise<unknown>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    try {
      for (const f of Array.from(files)) await uploadFn(f);
      onUploaded();
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  return (
    <Box sx={{
      width: 320, flexShrink: 0, borderLeft: '1px dashed rgba(145,158,171,0.2)',
      bgcolor: 'background.paper', height: '100%', overflowY: 'auto', p: 2.5,
    }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5 }}>Sources & portée</Typography>

      <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 700 }}>PORTÉE</Typography>
      <FormControl fullWidth size="small" sx={{ mt: 0.75, mb: 2 }}>
        <Select
          value={scope.type === 'document' ? scope.id : 'all'}
          onChange={(e) => {
            const v = e.target.value;
            if (v === 'all') onScopeChange({ type: 'all' });
            else {
              const d = documents.find((x) => x.id === v);
              if (d) onScopeChange({ type: 'document', id: d.id, name: d.name });
            }
          }}
        >
          <MenuItem value="all">Tous les documents</MenuItem>
          {documents.map((d) => (
            <MenuItem key={d.id} value={d.id}>{d.name}</MenuItem>
          ))}
        </Select>
      </FormControl>

      <input ref={inputRef} type="file" hidden multiple accept=".pdf,.txt,.docx,.md,.csv,.xlsx"
        onChange={(e) => handleFiles(e.target.files)} />
      <Button fullWidth variant="outlined" color="inherit"
        startIcon={busy ? <CircularProgress size={16} /> : <UploadFileRoundedIcon />}
        disabled={busy} onClick={() => inputRef.current?.click()} sx={{ mb: 2.5 }}>
        {busy ? 'Envoi…' : 'Ajouter un document'}
      </Button>

      <Divider sx={{ mb: 2 }} />

      <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 700 }}>
        DOCUMENTS INDEXÉS ({documents.length})
      </Typography>
      <Box sx={{ mt: 1, mb: 2.5 }}>
        {documents.length === 0 && (
          <Typography variant="caption" color="text.disabled">
            Aucun document. Ajoutez-en pour des réponses sourcées.
          </Typography>
        )}
        {documents.map((d) => (
          <Box key={d.id} sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 0.5 }}>
            <DescriptionOutlinedIcon fontSize="small" sx={{ color: 'text.disabled' }} />
            <Typography variant="caption" noWrap sx={{ flexGrow: 1 }} title={d.name}>{d.name}</Typography>
            {d.latest && d.latest.indexing_status !== 'indexed' ? (
              <Chip label={d.latest.indexing_status === 'failed' ? 'échec' : 'indexation…'} size="small"
                color={d.latest.indexing_status === 'failed' ? 'error' : 'info'} variant="outlined"
                sx={{ height: 18, fontSize: 10 }} />
            ) : (
              <Typography variant="caption" color="text.disabled">v{d.current_version}</Typography>
            )}
          </Box>
        ))}
      </Box>

      <Divider sx={{ mb: 2 }} />

      <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 700 }}>
        SOURCES DE LA DERNIÈRE RÉPONSE
      </Typography>
      <Box sx={{ mt: 1 }}>
        {lastSources.length === 0 ? (
          <Typography variant="caption" color="text.disabled">La dernière réponse n&apos;utilise aucun document.</Typography>
        ) : (
          lastSources.map((s, i) => <SourceCard key={i} s={s} />)
        )}
      </Box>
    </Box>
  );
}
