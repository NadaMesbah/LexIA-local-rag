'use client';

import { useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import EditRoundedIcon from '@mui/icons-material/EditRounded';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import ChevronLeftRoundedIcon from '@mui/icons-material/ChevronLeftRounded';
import ChevronRightRoundedIcon from '@mui/icons-material/ChevronRightRounded';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined';
import AttachFileRoundedIcon from '@mui/icons-material/AttachFileRounded';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import ErrorOutlineRoundedIcon from '@mui/icons-material/ErrorOutlineRounded';
import CircularProgress from '@mui/material/CircularProgress';
import Markdown from '@/components/Markdown';
import type { IndexingStatus, Message, Source } from '@/lib/types';

function AttachmentChip({ id, filename, status }: { id: string; filename: string; status?: IndexingStatus | null }) {
  // Visualise the attached document's real indexing status.
  let icon = <AttachFileRoundedIcon />;
  let label = filename;
  if (status === 'indexed') icon = <CheckCircleRoundedIcon sx={{ color: 'success.main' }} />;
  else if (status === 'failed') { icon = <ErrorOutlineRoundedIcon sx={{ color: 'error.main' }} />; label = `${filename} (échec)`; }
  else if (status === 'pending' || status === 'processing') { icon = <CircularProgress size={13} />; label = `${filename} · indexation…`; }
  return (
    <Chip
      icon={icon}
      label={label}
      size="small"
      sx={{
        maxWidth: 260,
        bgcolor: 'rgba(0,167,111,0.08)',
        border: '1px solid rgba(0,167,111,0.24)',
        '& .MuiChip-label': { fontWeight: 600 },
      }}
    />
  );
}

function SourceRow({ s }: { s: Source }) {
  const [open, setOpen] = useState(false);
  return (
    <Box onClick={() => setOpen((o) => !o)} sx={{
      p: 1.25, borderRadius: 1.5, border: '1px solid rgba(145,158,171,0.2)', cursor: 'pointer',
      width: '100%', '&:hover': { borderColor: 'primary.light' },
    }}>
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 0.5 }}>
        <DescriptionOutlinedIcon fontSize="small" sx={{ color: 'primary.main' }} />
        <Typography variant="caption" sx={{ fontWeight: 700, flexGrow: 1, wordBreak: 'break-all' }}>
          {s.filename}{s.version ? ` · v${s.version}` : ''}
        </Typography>
        {s.page && s.page !== '?' && <Chip label={`p.${s.page}`} size="small" sx={{ height: 18, fontSize: 10 }} />}
      </Box>
      <Typography dir="auto" variant="caption" color="text.secondary" sx={{
        display: '-webkit-box', WebkitLineClamp: open ? 'unset' : 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
      }}>
        {s.text}
      </Typography>
    </Box>
  );
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <Tooltip title={done ? 'Copié' : 'Copier'}>
      <IconButton
        size="small"
        onClick={() => {
          const ok = () => {
            setDone(true);
            setTimeout(() => setDone(false), 1400);
          };
          if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(ok).catch(() => {});
          else ok();
        }}
        sx={{ color: done ? 'success.main' : 'text.disabled' }}
      >
        {done ? <CheckRoundedIcon fontSize="small" /> : <ContentCopyRoundedIcon fontSize="small" />}
      </IconButton>
    </Tooltip>
  );
}

function VersionNav({ m, onSwitch }: { m: Message; onSwitch: (id: string) => void }) {
  if (!m.version_count || m.version_count <= 1) return null;
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25, color: 'text.disabled' }}>
      <IconButton
        size="small"
        disabled={m.version_index <= 1}
        onClick={() => onSwitch(m.sibling_ids[m.version_index - 2])}
      >
        <ChevronLeftRoundedIcon fontSize="small" />
      </IconButton>
      <Typography variant="caption" sx={{ fontVariantNumeric: 'tabular-nums' }}>
        {m.version_index}/{m.version_count}
      </Typography>
      <IconButton
        size="small"
        disabled={m.version_index >= m.version_count}
        onClick={() => onSwitch(m.sibling_ids[m.version_index])}
      >
        <ChevronRightRoundedIcon fontSize="small" />
      </IconButton>
    </Box>
  );
}

export default function MessageItem({
  m,
  showGroundingFlag,
  streaming = false,
  statusOf,
  onSwitchVersion,
  onEdit,
  onRegenerate,
}: {
  m: Message;
  showGroundingFlag: boolean;
  streaming?: boolean;
  statusOf?: (id: string) => IndexingStatus | undefined;
  onSwitchVersion: (id: string) => void;
  onEdit: (id: string, text: string) => void;
  onRegenerate: (userMessageId: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(m.content);
  const [showSources, setShowSources] = useState(false);
  const isUser = m.role === 'user';

  if (isUser) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 0.5 }}>
        {!editing && m.attachments && m.attachments.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, justifyContent: 'flex-end', maxWidth: '78%' }}>
            {m.attachments.map((a) => (
              <AttachmentChip
                key={a.id}
                id={a.id}
                filename={a.filename}
                status={(statusOf && statusOf(a.id)) ?? a.indexing_status}
              />
            ))}
          </Box>
        )}
        {editing ? (
          <Box sx={{ width: '100%', maxWidth: 720 }}>
            <TextField
              fullWidth
              multiline
              minRows={2}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoFocus
              size="small"
              inputProps={{ dir: 'auto' }}
            />
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end', mt: 1 }}>
              <Button size="small" color="inherit" onClick={() => { setEditing(false); setDraft(m.content); }}>
                Annuler
              </Button>
              <Button
                size="small"
                variant="contained"
                onClick={() => { setEditing(false); onEdit(m.id, draft.trim()); }}
              >
                Enregistrer
              </Button>
            </Box>
          </Box>
        ) : (
          <Paper
            sx={{
              px: 2,
              py: 1.25,
              maxWidth: '78%',
              bgcolor: 'rgba(0,167,111,0.12)',
              border: '1px solid rgba(0,167,111,0.24)',
              boxShadow: 'none',
              borderRadius: 2,
              borderBottomRightRadius: 4,
            }}
          >
            <Typography dir="auto" variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
              {m.content}
            </Typography>
          </Paper>
        )}
        {!editing && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
            <VersionNav m={m} onSwitch={onSwitchVersion} />
            <CopyButton text={m.content} />
            {/* Editing is hidden while a reply is generating: editing then would
                race the in-flight answer and target an id that isn't persisted yet. */}
            {!streaming && (
              <Tooltip title="Modifier">
                <IconButton size="small" onClick={() => { setDraft(m.content); setEditing(true); }} sx={{ color: 'text.disabled' }}>
                  <EditRoundedIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        )}
      </Box>
    );
  }

  // assistant
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 0.5 }}>
      <Typography variant="caption" sx={{ color: 'warning.dark', fontWeight: 700, letterSpacing: 0.5 }}>
        ASSISTANT
      </Typography>
      <Paper
        sx={{
          px: 2,
          py: 1.5,
          width: '100%',
          border: '1px solid rgba(145,158,171,0.2)',
          boxShadow: 'none',
          borderRadius: 2,
          borderBottomLeftRadius: 4,
        }}
      >
        <Markdown>{m.content}</Markdown>
        {showGroundingFlag && (!m.sources || m.sources.length === 0) && (
          <Chip
            icon={<InfoOutlinedIcon />}
            label="Réponse générale (hors documents)"
            size="small"
            variant="outlined"
            sx={{ mt: 1.25, color: 'text.secondary', borderColor: 'rgba(145,158,171,0.4)' }}
          />
        )}
      </Paper>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
        <VersionNav m={m} onSwitch={onSwitchVersion} />
        <CopyButton text={m.content} />
        {m.parent_id && !streaming && (
          <Tooltip title="Régénérer">
            <IconButton size="small" onClick={() => onRegenerate(m.parent_id as string)} sx={{ color: 'text.disabled' }}>
              <RefreshRoundedIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
        {m.sources && m.sources.length > 0 && (
          <Button size="small" variant="text" startIcon={<DescriptionOutlinedIcon fontSize="small" />}
            onClick={() => setShowSources((v) => !v)}
            sx={{ color: 'text.secondary', textTransform: 'none', minWidth: 0 }}>
            {showSources ? 'Masquer les sources' : `Sources (${m.sources.length})`}
          </Button>
        )}
      </Box>
      {m.sources && m.sources.length > 0 && (
        <Collapse in={showSources} sx={{ width: '100%' }}>
          <Box sx={{ mt: 0.5, display: 'flex', flexDirection: 'column', gap: 0.75 }}>
            {m.sources.map((s, i) => <SourceRow key={i} s={s} />)}
          </Box>
        </Collapse>
      )}
    </Box>
  );
}