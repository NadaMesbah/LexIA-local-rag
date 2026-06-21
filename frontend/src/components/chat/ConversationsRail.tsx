'use client';

import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import ChatBubbleOutlineRoundedIcon from '@mui/icons-material/ChatBubbleOutlineRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import type { SessionSummary } from '@/lib/types';

export default function ConversationsRail({
  sessions,
  currentId,
  busyIds = [],
  onNew,
  onSelect,
  onDelete,
}: {
  sessions: SessionSummary[];
  currentId: string | null;
  busyIds?: string[];
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <Box
      sx={{
        width: 268,
        flexShrink: 0,
        borderRight: '1px dashed rgba(145,158,171,0.2)',
        bgcolor: 'background.paper',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      <Box sx={{ p: 2 }}>
        <Button
          fullWidth
          variant="contained"
          startIcon={<AddRoundedIcon />}
          onClick={onNew}
          sx={{ py: 1 }}
        >
          Nouvelle conversation
        </Button>
      </Box>

      <Typography
        variant="caption"
        sx={{ px: 2, pb: 0.5, color: 'text.disabled', fontWeight: 700, letterSpacing: 0.6 }}
      >
        CONVERSATIONS
      </Typography>

      <List sx={{ px: 1, flexGrow: 1, overflowY: 'auto' }}>
        {sessions.length === 0 && (
          <Typography variant="body2" sx={{ px: 1.5, py: 2, color: 'text.disabled' }}>
            Aucune conversation. Créez-en une pour commencer.
          </Typography>
        )}
        {sessions.map((s) => {
          const active = s.id === currentId;
          return (
            <ListItemButton
              key={s.id}
              selected={active}
              onClick={() => onSelect(s.id)}
              sx={{
                borderRadius: 1.5,
                mb: 0.25,
                pr: 1,
                '&.Mui-selected': { bgcolor: 'rgba(0,167,111,0.10)' },
                '&:hover .del': { opacity: 1 },
              }}
            >
              <ChatBubbleOutlineRoundedIcon
                fontSize="small"
                sx={{ mr: 1, color: active ? 'primary.main' : 'text.disabled' }}
              />
              <Typography
                dir="auto"
                variant="body2"
                noWrap
                sx={{ flexGrow: 1, fontWeight: active ? 700 : 500, color: active ? 'text.primary' : 'text.secondary' }}
              >
                {s.title}
              </Typography>
              {busyIds.includes(s.id) && (
                <CircularProgress size={14} thickness={6} sx={{ mr: 0.5, color: 'primary.main' }} />
              )}
              <IconButton
                className="del"
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s.id);
                }}
                sx={{ opacity: 0, color: 'text.disabled', '&:hover': { color: 'error.main' } }}
              >
                <DeleteOutlineRoundedIcon fontSize="small" />
              </IconButton>
            </ListItemButton>
          );
        })}
      </List>
    </Box>
  );
}