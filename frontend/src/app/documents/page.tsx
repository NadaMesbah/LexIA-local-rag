'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import { DataGrid, type GridColDef } from '@mui/x-data-grid';
import CloudUploadRoundedIcon from '@mui/icons-material/CloudUploadRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import DeleteForeverRoundedIcon from '@mui/icons-material/DeleteForeverRounded';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import RestoreRoundedIcon from '@mui/icons-material/RestoreRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import * as api from '@/lib/api';
import type { DocumentItem, DocumentVersion, IndexingStatus } from '@/lib/types';

const EXT_COLOR: Record<string, 'error' | 'primary' | 'success' | 'warning' | 'default'> = {
  pdf: 'error', txt: 'primary', docx: 'success', doc: 'success', md: 'warning', csv: 'default', xlsx: 'default',
};
const ext = (name: string) => name.split('.').pop()?.toLowerCase() || 'file';

const STATUS: Record<IndexingStatus, { label: string; color: 'default' | 'info' | 'success' | 'error' | 'warning' }> = {
  pending: { label: 'En attente', color: 'warning' },
  processing: { label: 'Indexation…', color: 'info' },
  indexed: { label: 'Indexé', color: 'success' },
  failed: { label: 'Échec', color: 'error' },
};

function StatusChip({ s }: { s?: IndexingStatus }) {
  const meta = STATUS[s || 'pending'];
  const spinning = s === 'pending' || s === 'processing';
  return (
    <Chip
      size="small"
      color={meta.color}
      variant="outlined"
      icon={spinning ? <CircularProgress size={12} thickness={6} /> : undefined}
      label={meta.label}
      sx={{ height: 24, fontWeight: 700 }}
    />
  );
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [query, setQuery] = useState('');
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; sev: 'success' | 'error' } | null>(null);
  const [confirm, setConfirm] = useState<{ doc: DocumentItem; mode: 'soft' | 'purge' } | null>(null);
  const [historyDoc, setHistoryDoc] = useState<DocumentItem | null>(null);
  const [versions, setVersions] = useState<DocumentVersion[]>([]);
  const versionInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const { documents } = await api.listDocuments();
      setDocs(documents);
    } catch {/* ignore */}
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while any document is still indexing.
  const anyPending = useMemo(
    () => docs.some((d) => d.latest && (d.latest.indexing_status === 'pending' || d.latest.indexing_status === 'processing')),
    [docs]
  );
  useEffect(() => {
    if (!anyPending) return;
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, [anyPending, refresh]);

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setUploading(true);
      for (const f of files) {
        try {
          await api.uploadDocument(f);
          setToast({ msg: `« ${f.name} » envoyé — indexation en cours.`, sev: 'success' });
        } catch (e: any) {
          setToast({ msg: `Échec : ${e?.message || f.name}`, sev: 'error' });
        }
      }
      setUploading(false);
      refresh();
    },
    [refresh]
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop: handleUpload,
    noClick: true,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt', '.md'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
  });

  const openHistory = useCallback(async (doc: DocumentItem) => {
    setHistoryDoc(doc);
    setVersions([]);
    try {
      const { versions } = await api.listVersions(doc.id);
      setVersions(versions);
    } catch {/* ignore */}
  }, []);

  const doRestore = useCallback(
    async (doc: DocumentItem, version: number) => {
      try {
        await api.restoreVersion(doc.id, version);
        setToast({ msg: `Version ${version} restaurée comme actuelle.`, sev: 'success' });
        await refresh();
        const { versions } = await api.listVersions(doc.id);
        setVersions(versions);
      } catch (e: any) {
        setToast({ msg: `Échec : ${e?.message}`, sev: 'error' });
      }
    },
    [refresh]
  );

  const doDelete = useCallback(async () => {
    if (!confirm) return;
    try {
      if (confirm.mode === 'soft') await api.softDeleteDocument(confirm.doc.id);
      else await api.purgeDocument(confirm.doc.id);
      setToast({ msg: `« ${confirm.doc.name} » supprimé.`, sev: 'success' });
      refresh();
    } catch (e: any) {
      setToast({ msg: `Échec : ${e?.message}`, sev: 'error' });
    }
    setConfirm(null);
  }, [confirm, refresh]);

  const onPickNewVersion = useCallback(
    async (files: FileList | null) => {
      if (!files?.length || !historyDoc) return;
      try {
        await api.uploadNewVersion(historyDoc.id, files[0]);
        setToast({ msg: `Nouvelle version envoyée pour « ${historyDoc.name} ».`, sev: 'success' });
        await refresh();
        const { versions } = await api.listVersions(historyDoc.id);
        setVersions(versions);
      } catch (e: any) {
        setToast({ msg: `Échec : ${e?.message}`, sev: 'error' });
      }
      if (versionInputRef.current) versionInputRef.current.value = '';
    },
    [historyDoc, refresh]
  );

  const rows = useMemo(
    () =>
      docs
        .filter((d) => d.name.toLowerCase().includes(query.toLowerCase()))
        .map((d) => ({
          id: d.id,
          doc: d,
          name: d.name,
          version: d.current_version,
          versionCount: d.version_count,
          status: d.latest?.indexing_status || 'pending',
          chunks: d.latest?.chunks ?? 0,
          updated: d.updated_at ? new Date(d.updated_at * 1000) : null,
        })),
    [docs, query]
  );

  const columns: GridColDef[] = [
    {
      field: 'name',
      headerName: 'Nom',
      flex: 1,
      minWidth: 240,
      renderCell: (p) => (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip label={ext(p.row.name).toUpperCase()} size="small" color={EXT_COLOR[ext(p.row.name)] || 'default'}
            variant="outlined" sx={{ height: 22, fontSize: 11, fontWeight: 700 }} />
          <Typography variant="body2" noWrap title={p.row.name}>{p.row.name}</Typography>
        </Box>
      ),
    },
    {
      field: 'version',
      headerName: 'Version',
      width: 110,
      renderCell: (p) => (
        <Tooltip title={`${p.row.versionCount} version(s)`}>
          <Chip label={`v${p.row.version}`} size="small" sx={{ height: 22, fontWeight: 700 }} />
        </Tooltip>
      ),
    },
    { field: 'status', headerName: 'Statut', width: 150, renderCell: (p) => <StatusChip s={p.row.status} /> },
    { field: 'chunks', headerName: 'Chunks', width: 100, type: 'number' },
    {
      field: 'updated', headerName: 'Mis à jour', width: 170,
      valueFormatter: (value) => (value ? new Date(value as Date).toLocaleString() : '—'),
    },
    {
      field: 'actions', headerName: '', width: 130, sortable: false, filterable: false, align: 'right',
      renderCell: (p) => (
        <Box>
          <Tooltip title="Versions / restaurer">
            <IconButton size="small" onClick={() => openHistory(p.row.doc)}><HistoryRoundedIcon fontSize="small" /></IconButton>
          </Tooltip>
          <Tooltip title="Supprimer (corbeille)">
            <IconButton size="small" onClick={() => setConfirm({ doc: p.row.doc, mode: 'soft' })}
              sx={{ '&:hover': { color: 'warning.main' } }}><DeleteOutlineRoundedIcon fontSize="small" /></IconButton>
          </Tooltip>
          <Tooltip title="Supprimer définitivement">
            <IconButton size="small" onClick={() => setConfirm({ doc: p.row.doc, mode: 'purge' })}
              sx={{ '&:hover': { color: 'error.main' } }}><DeleteForeverRoundedIcon fontSize="small" /></IconButton>
          </Tooltip>
        </Box>
      ),
    },
  ];

  return (
    <Box sx={{ height: '100%', overflowY: 'auto', p: { xs: 2, md: 4 } }}>
      <Box sx={{ maxWidth: 1100, mx: 'auto' }}>
        <Typography variant="h4" sx={{ fontWeight: 800, mb: 0.5 }}>Documents</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Importez, versionnez, restaurez et supprimez les documents de la base. Le chat utilise
          toujours la dernière version indexée.
        </Typography>

        <Box {...getRootProps()} sx={{
          p: 4, mb: 3, borderRadius: 2, border: '1.5px dashed',
          borderColor: isDragActive ? 'primary.main' : 'rgba(145,158,171,0.32)',
          bgcolor: isDragActive ? 'rgba(0,167,111,0.08)' : 'background.paper', textAlign: 'center', transition: 'all .2s',
        }}>
          <input {...getInputProps()} />
          <CloudUploadRoundedIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>Glissez vos fichiers ici</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>PDF · TXT · DOCX · MD · CSV · XLSX</Typography>
          <Button variant="contained" onClick={open} disabled={uploading}>Parcourir les fichiers</Button>
          {uploading && <LinearProgress sx={{ mt: 2, borderRadius: 1 }} />}
        </Box>

        <Card sx={{ p: 0 }}>
          <Box sx={{ p: 2 }}>
            <TextField size="small" placeholder="Rechercher un document…" value={query}
              onChange={(e) => setQuery(e.target.value)} sx={{ width: { xs: '100%', sm: 320 } }}
              InputProps={{ startAdornment: (<InputAdornment position="start"><SearchRoundedIcon fontSize="small" /></InputAdornment>) }} />
          </Box>
          <DataGrid
            autoHeight rows={rows} columns={columns} disableRowSelectionOnClick
            pageSizeOptions={[10, 25, 50]} initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
            sx={{ border: 0, '& .MuiDataGrid-columnHeaders': { bgcolor: 'grey.100' },
              '& .MuiDataGrid-cell:focus, & .MuiDataGrid-cell:focus-within': { outline: 'none' } }}
            localeText={{ noRowsLabel: 'Aucun document indexé.' }}
          />
        </Card>
      </Box>

      {/* Version history */}
      <Dialog open={!!historyDoc} onClose={() => setHistoryDoc(null)} fullWidth maxWidth="sm" PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <HistoryRoundedIcon /> Versions — {historyDoc?.name}
        </DialogTitle>
        <DialogContent dividers>
          <input ref={versionInputRef} type="file" hidden accept=".pdf,.txt,.docx,.md,.csv,.xlsx"
            onChange={(e) => onPickNewVersion(e.target.files)} />
          <Button startIcon={<UploadFileRoundedIcon />} variant="outlined" size="small" sx={{ mb: 1 }}
            onClick={() => versionInputRef.current?.click()}>
            Téléverser une nouvelle version
          </Button>
          <List dense>
            {versions.map((v) => (
              <ListItem key={v.id} secondaryAction={
                v.is_latest ? (
                  <Chip label="Actuelle" color="primary" size="small" sx={{ fontWeight: 700 }} />
                ) : v.indexing_status === 'indexed' ? (
                  <Tooltip title="Restaurer cette version">
                    <IconButton edge="end" onClick={() => historyDoc && doRestore(historyDoc, v.version)}>
                      <RestoreRoundedIcon />
                    </IconButton>
                  </Tooltip>
                ) : null
              }>
                <ListItemText
                  primary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 700 }}>v{v.version}</Typography>
                    <StatusChip s={v.indexing_status} />
                  </Box>}
                  secondary={`${v.chunks} chunks · ${new Date(v.uploaded_at * 1000).toLocaleString()}${v.error ? ' · ' + v.error : ''}`}
                />
              </ListItem>
            ))}
            {versions.length === 0 && <Typography variant="body2" color="text.disabled">Chargement…</Typography>}
          </List>
        </DialogContent>
        <DialogActions><Button onClick={() => setHistoryDoc(null)}>Fermer</Button></DialogActions>
      </Dialog>

      {/* Delete confirm */}
      <Dialog open={!!confirm} onClose={() => setConfirm(null)} PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle>{confirm?.mode === 'purge' ? 'Supprimer définitivement ?' : 'Mettre à la corbeille ?'}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {confirm?.mode === 'purge'
              ? `« ${confirm?.doc.name} » et tous ses vecteurs seront définitivement supprimés. Irréversible.`
              : `« ${confirm?.doc.name} » sera retiré du chat et de la liste active (récupérable, vecteurs conservés).`}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button color="inherit" onClick={() => setConfirm(null)}>Annuler</Button>
          <Button color="error" variant="contained" onClick={doDelete}>
            {confirm?.mode === 'purge' ? 'Supprimer définitivement' : 'Supprimer'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar open={!!toast} autoHideDuration={4000} onClose={() => setToast(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        {toast ? <Alert severity={toast.sev} variant="filled" onClose={() => setToast(null)}>{toast.msg}</Alert> : undefined}
      </Snackbar>
    </Box>
  );
}
