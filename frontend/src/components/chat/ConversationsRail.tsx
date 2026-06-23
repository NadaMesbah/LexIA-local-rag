'use client';

import { useEffect, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import InputBase from '@mui/material/InputBase';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import EditRoundedIcon from '@mui/icons-material/EditRounded';
import type { SessionSummary } from '@/lib/types';

export default function ConversationsRail({
  sessions,
  currentId,
  busyIds = [],
  onNew,
  onSelect,
  onDelete,
  onRename,
}: {
  sessions: SessionSummary[];
  currentId: string | null;
  busyIds?: string[];
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename?: (id: string, title: string) => void;
}) {
  // Menu state
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null);

  // Inline rename state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus the input when editingId changes
  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  const openMenu = (e: React.MouseEvent<HTMLElement>, id: string) => {
    e.stopPropagation();
    e.preventDefault();
    setMenuAnchor(e.currentTarget);
    setMenuSessionId(id);
  };

  const closeMenu = () => {
    setMenuAnchor(null);
    setMenuSessionId(null);
  };

  const startRename = () => {
    const id = menuSessionId;
    closeMenu();
    if (!id) return;
    const session = sessions.find((s) => s.id === id);
    setEditValue(session?.title || '');
    // Use requestAnimationFrame to ensure the menu is fully closed before switching to edit mode
    requestAnimationFrame(() => {
      setEditingId(id);
    });
  };

  const confirmDelete = () => {
    const id = menuSessionId;
    closeMenu();
    if (id) onDelete(id);
  };

  const submitRename = () => {
    if (!editingId) return;
    const trimmed = editValue.trim();
    const original = sessions.find((s) => s.id === editingId)?.title || '';
    if (trimmed && trimmed !== original && onRename) {
      onRename(editingId, trimmed);
    }
    setEditingId(null);
    setEditValue('');
  };

  const cancelRename = () => {
    setEditingId(null);
    setEditValue('');
  };

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
          const isEditing = editingId === s.id;

          return (
            <ListItemButton
              key={s.id}
              selected={active}
              onClick={() => {
                if (!isEditing) onSelect(s.id);
              }}
              sx={{
                borderRadius: 1.5,
                mb: 0.25,
                pr: 1,
                '&.Mui-selected': { bgcolor: 'rgba(1,134,135,0.10)' },
                '&:hover .more-btn': { opacity: 1 },
              }}
            >
              {isEditing ? (
                <InputBase
                  inputRef={inputRef}
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      submitRename();
                    } else if (e.key === 'Escape') {
                      e.preventDefault();
                      cancelRename();
                    }
                  }}
                  onBlur={submitRename}
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  sx={{
                    flexGrow: 1,
                    fontSize: '0.875rem',
                    fontWeight: 500,
                    px: 0.5,
                    py: 0.25,
                    borderRadius: 0.5,
                    border: '1px solid',
                    borderColor: 'primary.main',
                  }}
                />
              ) : (
                <Typography
                  dir="auto"
                  variant="body2"
                  noWrap
                  sx={{
                    flexGrow: 1,
                    fontWeight: active ? 700 : 500,
                    color: active ? 'text.primary' : 'text.secondary',
                  }}
                >
                  {s.title}
                </Typography>
              )}

              {busyIds.includes(s.id) && (
                <CircularProgress size={14} thickness={6} sx={{ mr: 0.5, color: 'primary.main' }} />
              )}

              {!isEditing && (
                <IconButton
                  className="more-btn"
                  size="small"
                  onClick={(e) => openMenu(e, s.id)}
                  onMouseDown={(e) => e.stopPropagation()}
                  sx={{ opacity: 0, color: 'text.disabled', '&:hover': { color: 'text.primary' } }}
                >
                  <MoreVertIcon fontSize="small" />
                </IconButton>
              )}
            </ListItemButton>
          );
        })}
      </List>

      {/* Context menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={closeMenu}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{ paper: { sx: { minWidth: 160, boxShadow: 3 } } }}
      >
        <MenuItem onClick={startRename}>
          <ListItemIcon><EditRoundedIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Renommer</ListItemText>
        </MenuItem>
        <MenuItem onClick={confirmDelete} sx={{ color: 'error.main' }}>
          <ListItemIcon><DeleteOutlineRoundedIcon fontSize="small" sx={{ color: 'error.main' }} /></ListItemIcon>
          <ListItemText>Supprimer</ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );
}
