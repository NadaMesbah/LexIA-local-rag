'use client';

import { useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Collapse from '@mui/material/Collapse';
import CircularProgress from '@mui/material/CircularProgress';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined';
import LayersRoundedIcon from '@mui/icons-material/LayersRounded';
import type { DocumentItem, Source } from '@/lib/types';

function SourceCard({ s }: { s: Source }) {
  const [open, setOpen] = useState(false);
  return (
    <Box
      onClick={() => setOpen((o) => !o)}
      sx={{
        p: 1.25,
        mb: 1,
        borderRadius: 1.5,
        border: '1px solid rgba(145,158,171,0.2)',
        cursor: 'pointer',
        '&:hover': { borderColor: 'primary.light' },
      }}
    >
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 0.5 }}>
        <DescriptionOutlinedIcon fontSize="small" sx={{ color: 'primary.main' }} />
        <Typography variant="caption" sx={{ fontWeight: 700, flexGrow: 1, wordBreak: 'break-all' }}>
          {s.filename}
        </Typography>
        {s.page && s.page !== '?' && (
          <Chip label={`p.${s.page}`} size="small" sx={{ height: 18, fontSize: 10 }} />
        )}
      </Box>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{
          display: '-webkit-box',
          WebkitLineClamp: open ? 'unset' : 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}
      >
        {s.text}
      </Typography>
    </Box>
  );
}

export default function SourcesPanel({
  documents,
  lastSources,
  onUploaded,
  uploadFn,
}: {
  documents: DocumentItem[];
  lastSources: Source[];
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
    <Box
      sx={{
        width: 320,
        flexShrink: 0,
        borderLeft: '1px dashed rgba(145,158,171,0.2)',
        bgcolor: 'background.paper',
        height: '100%',
        overflowY: 'auto',
        p: 2.5,
      }}
    >
      <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5 }}>
        Sources & portée
      </Typography>

      <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 700 }}>
        PORTÉE
      </Typography>
      <Box sx={{ mt: 0.75, mb: 2 }}>
        <Chip icon={<LayersRoundedIcon />} label="Tous les documents" color="primary" variant="outlined" size="small" />
        <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.disabled' }}>
          La portée par document/collection arrivera bientôt.
        </Typography>
      </Box>

      <input
        ref={inputRef}
        type="file"
        hidden
        multiple
        accept=".pdf,.txt,.docx,.md,.csv,.xlsx"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <Button
        fullWidth
        variant="outlined"
        color="inherit"
        startIcon={busy ? <CircularProgress size={16} /> : <UploadFileRoundedIcon />}
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        sx={{ mb: 2.5 }}
      >
        {busy ? 'Indexation…' : 'Ajouter un document'}
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
          <Box key={d.name} sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 0.5 }}>
            <DescriptionOutlinedIcon fontSize="small" sx={{ color: 'text.disabled' }} />
            <Typography variant="caption" noWrap sx={{ flexGrow: 1 }} title={d.name}>
              {d.name}
            </Typography>
            <Typography variant="caption" color="text.disabled">
              {d.chunks}
            </Typography>
          </Box>
        ))}
      </Box>

      <Divider sx={{ mb: 2 }} />

      <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 700 }}>
        SOURCES DE LA DERNIÈRE RÉPONSE
      </Typography>
      <Box sx={{ mt: 1 }}>
        {lastSources.length === 0 ? (
          <Typography variant="caption" color="text.disabled">
            La dernière réponse n'utilise aucun document.
          </Typography>
        ) : (
          lastSources.map((s, i) => <SourceCard key={i} s={s} />)
        )}
      </Box>
    </Box>
  );
}
