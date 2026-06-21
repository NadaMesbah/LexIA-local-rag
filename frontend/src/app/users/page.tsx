'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import MenuItem from '@mui/material/MenuItem';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import Avatar from '@mui/material/Avatar';
import { DataGrid, type GridColDef } from '@mui/x-data-grid';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import PersonAddRoundedIcon from '@mui/icons-material/PersonAddRounded';
import EditRoundedIcon from '@mui/icons-material/EditRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import * as api from '@/lib/api';
import { useAuth } from '@/lib/auth';
import type { ManagedUser, Role } from '@/lib/types';

interface FormState {
  first_name: string;
  last_name: string;
  username: string;
  email: string;
  password: string;
  role: Role;
}

const EMPTY_FORM: FormState = {
  first_name: '',
  last_name: '',
  username: '',
  email: '',
  password: '',
  role: 'user',
};

export default function UsersPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const isAdmin = user?.role === 'admin';

  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState('');
  const [toast, setToast] = useState<{ msg: string; sev: 'success' | 'error' } | null>(null);
  // Dialog state: `null` = closed; otherwise create (no id) or edit (with id).
  const [editor, setEditor] = useState<{ target: ManagedUser | null } | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<ManagedUser | null>(null);

  // Guard: only admins may see this page.
  useEffect(() => {
    if (!loading && user && !isAdmin) router.replace('/chat');
  }, [loading, user, isAdmin, router]);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const { users } = await api.listUsers();
      setUsers(users);
    } catch (e: any) {
      setToast({ msg: `Échec du chargement : ${e?.message}`, sev: 'error' });
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (isAdmin) refresh();
  }, [isAdmin, refresh]);

  const openCreate = useCallback(() => {
    setForm(EMPTY_FORM);
    setEditor({ target: null });
  }, []);

  const openEdit = useCallback((u: ManagedUser) => {
    setForm({
      first_name: u.first_name || '',
      last_name: u.last_name || '',
      username: u.username || '',
      email: u.email || '',
      password: '',
      role: u.role,
    });
    setEditor({ target: u });
  }, []);

  const save = useCallback(async () => {
    if (!editor) return;
    setSaving(true);
    try {
      if (editor.target) {
        // Edit: only send the password when the admin typed a new one.
        await api.updateUser(editor.target.id, {
          first_name: form.first_name.trim(),
          last_name: form.last_name.trim(),
          username: form.username.trim(),
          email: form.email.trim(),
          role: form.role,
          ...(form.password ? { password: form.password } : {}),
        });
        setToast({ msg: `« ${form.username} » mis à jour.`, sev: 'success' });
      } else {
        await api.createUser({
          first_name: form.first_name.trim(),
          last_name: form.last_name.trim(),
          username: form.username.trim(),
          email: form.email.trim(),
          password: form.password,
          role: form.role,
        });
        setToast({ msg: `Utilisateur « ${form.username} » créé.`, sev: 'success' });
      }
      setEditor(null);
      refresh();
    } catch (e: any) {
      setToast({ msg: `Échec : ${e?.message}`, sev: 'error' });
    } finally {
      setSaving(false);
    }
  }, [editor, form, refresh]);

  const doDelete = useCallback(async () => {
    if (!confirmDelete) return;
    try {
      await api.deleteUser(confirmDelete.id);
      setToast({ msg: `« ${confirmDelete.username} » supprimé.`, sev: 'success' });
      refresh();
    } catch (e: any) {
      setToast({ msg: `Échec : ${e?.message}`, sev: 'error' });
    }
    setConfirmDelete(null);
  }, [confirmDelete, refresh]);

  const rows = useMemo(
    () =>
      users
        .filter((u) => {
          const q = query.toLowerCase();
          return (
            u.username.toLowerCase().includes(q) ||
            u.email.toLowerCase().includes(q) ||
            `${u.first_name} ${u.last_name}`.toLowerCase().includes(q)
          );
        })
        .map((u) => ({
          id: u.id,
          user: u,
          name: [u.first_name, u.last_name].filter(Boolean).join(' ') || u.username,
          username: u.username,
          email: u.email,
          role: u.role,
          created: u.created_at ? new Date(u.created_at * 1000) : null,
        })),
    [users, query]
  );

  const columns: GridColDef[] = [
    {
      field: 'name',
      headerName: 'Nom',
      flex: 1,
      minWidth: 200,
      renderCell: (p) => (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Avatar sx={{ width: 30, height: 30, bgcolor: 'primary.main', fontSize: 13, fontWeight: 700 }}>
            {(p.row.name || '?').slice(0, 1).toUpperCase()}
          </Avatar>
          <Typography variant="body2" noWrap sx={{ fontWeight: 600 }}>{p.row.name}</Typography>
        </Box>
      ),
    },
    { field: 'username', headerName: "Nom d'utilisateur", width: 170 },
    { field: 'email', headerName: 'Email', flex: 1, minWidth: 220 },
    {
      field: 'role',
      headerName: 'Rôle',
      width: 150,
      renderCell: (p) => (
        <Chip
          label={p.row.role === 'admin' ? 'Administrateur' : 'Utilisateur'}
          size="small"
          color={p.row.role === 'admin' ? 'primary' : 'default'}
          variant={p.row.role === 'admin' ? 'filled' : 'outlined'}
          sx={{ height: 22, fontWeight: 700 }}
        />
      ),
    },
    {
      field: 'created',
      headerName: 'Créé le',
      width: 170,
      valueFormatter: (value) => (value ? new Date(value as Date).toLocaleString() : '—'),
    },
    {
      field: 'actions',
      headerName: '',
      width: 110,
      sortable: false,
      filterable: false,
      align: 'right',
      renderCell: (p) => {
        const isSelf = p.row.id === user?.id;
        return (
          <Box>
            <Tooltip title="Modifier">
              <IconButton size="small" onClick={() => openEdit(p.row.user)}
                sx={{ '&:hover': { color: 'primary.main' } }}>
                <EditRoundedIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title={isSelf ? 'Vous ne pouvez pas supprimer votre propre compte' : 'Supprimer'}>
              <span>
                <IconButton size="small" disabled={isSelf}
                  onClick={() => setConfirmDelete(p.row.user)}
                  sx={{ '&:hover': { color: 'error.main' } }}>
                  <DeleteOutlineRoundedIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        );
      },
    },
  ];

  if (loading || !user) {
    return (
      <Box sx={{ height: '100%', display: 'grid', placeItems: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }
  if (!isAdmin) return null;

  const isEdit = !!editor?.target;
  const formValid =
    form.first_name.trim() &&
    form.last_name.trim() &&
    form.username.trim().length >= 3 &&
    /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(form.email.trim()) &&
    (isEdit ? form.password === '' || form.password.length >= 8 : form.password.length >= 8);

  return (
    <Box sx={{ height: '100%', overflowY: 'auto', p: { xs: 2, md: 4 } }}>
      <Box sx={{ maxWidth: 1100, mx: 'auto' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="h4" sx={{ fontWeight: 800 }}>Utilisateurs</Typography>
          <Button variant="contained" startIcon={<PersonAddRoundedIcon />} onClick={openCreate}>
            Ajouter un utilisateur
          </Button>
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Gérez les comptes : créez, modifiez le rôle ou réinitialisez le mot de passe, et supprimez
          des utilisateurs.
        </Typography>

        <Card sx={{ p: 0 }}>
          <Box sx={{ p: 2 }}>
            <TextField size="small" placeholder="Rechercher un utilisateur…" value={query}
              onChange={(e) => setQuery(e.target.value)} sx={{ width: { xs: '100%', sm: 320 } }}
              InputProps={{ startAdornment: (<InputAdornment position="start"><SearchRoundedIcon fontSize="small" /></InputAdornment>) }} />
          </Box>
          <DataGrid
            autoHeight rows={rows} columns={columns} loading={busy} disableRowSelectionOnClick
            pageSizeOptions={[10, 25, 50]} initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
            sx={{ border: 0, '& .MuiDataGrid-columnHeaders': { bgcolor: 'grey.100' },
              '& .MuiDataGrid-cell:focus, & .MuiDataGrid-cell:focus-within': { outline: 'none' } }}
            localeText={{ noRowsLabel: 'Aucun utilisateur.' }}
          />
        </Card>
      </Box>

      {/* Create / edit dialog */}
      <Dialog open={!!editor} onClose={() => !saving && setEditor(null)} fullWidth maxWidth="sm"
        PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle>{isEdit ? `Modifier — ${editor?.target?.username}` : 'Nouvel utilisateur'}</DialogTitle>
        <DialogContent dividers>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2, pt: 0.5 }}>
            <TextField label="Prénom" value={form.first_name}
              onChange={(e) => setForm({ ...form, first_name: e.target.value })} fullWidth />
            <TextField label="Nom" value={form.last_name}
              onChange={(e) => setForm({ ...form, last_name: e.target.value })} fullWidth />
            <TextField label="Nom d'utilisateur" value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })} fullWidth
              helperText="Au moins 3 caractères" />
            <TextField label="Email" type="email" value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })} fullWidth />
            <TextField
              label={isEdit ? 'Nouveau mot de passe' : 'Mot de passe'}
              type="password"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              fullWidth
              helperText={isEdit ? 'Laisser vide pour ne pas changer · sinon 8+ caractères' : 'Au moins 8 caractères'}
            />
            <TextField select label="Rôle" value={form.role}
              onChange={(e) => setForm({ ...form, role: e.target.value as Role })} fullWidth>
              <MenuItem value="user">Utilisateur</MenuItem>
              <MenuItem value="admin">Administrateur</MenuItem>
            </TextField>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button color="inherit" onClick={() => setEditor(null)} disabled={saving}>Annuler</Button>
          <Button variant="contained" onClick={save} disabled={!formValid || saving}>
            {saving ? 'Enregistrement…' : isEdit ? 'Enregistrer' : 'Créer'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete confirm */}
      <Dialog open={!!confirmDelete} onClose={() => setConfirmDelete(null)} PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle>Supprimer cet utilisateur ?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {`« ${confirmDelete?.username} » sera définitivement supprimé, ainsi que ses conversations
            et ses documents personnels. Cette action est irréversible.`}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button color="inherit" onClick={() => setConfirmDelete(null)}>Annuler</Button>
          <Button color="error" variant="contained" onClick={doDelete}>Supprimer</Button>
        </DialogActions>
      </Dialog>

      <Snackbar open={!!toast} autoHideDuration={4000} onClose={() => setToast(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        {toast ? <Alert severity={toast.sev} variant="filled" onClose={() => setToast(null)}>{toast.msg}</Alert> : undefined}
      </Snackbar>
    </Box>
  );
}
