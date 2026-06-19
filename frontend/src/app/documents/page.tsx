'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
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
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import { DataGrid, type GridColDef } from '@mui/x-data-grid';
import CloudUploadRoundedIcon from '@mui/icons-material/CloudUploadRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import * as api from '@/lib/api';
import type { DocumentItem } from '@/lib/types';

const EXT_COLOR: Record<string, 'error' | 'primary' | 'success' | 'warning' | 'default'> = {
  pdf: 'error',
  txt: 'primary',
  docx: 'success',
  doc: 'success',
  md: 'warning',
  csv: 'default',
  xlsx: 'default',
};

function ext(name: string) {
  return name.split('.').pop()?.toLowerCase() || 'file';
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [query, setQuery] = useState('');
  const [uploading, setUploading] = useState(false);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [toast, setToast] = useState<{ msg: string; sev: 'success' | 'error' } | null>(null);

  const refresh = useCallback(async () => {
    try {
      const { documents } = await api.listDocuments();
      setDocs(documents);
    } catch {/* ignore */}
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setUploading(true);
      for (const f of files) {
        try {
          const res = await api.uploadDocument(f);
          setToast({ msg: `« ${res.filename} » indexé (${res.chunks_added} chunks).`, sev: 'success' });
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

  const doDelete = useCallback(
    async (name: string) => {
      try {
        await api.deleteDocument(name);
        setToast({ msg: `« ${name} » supprimé.`, sev: 'success' });
        refresh();
      } catch (e: any) {
        setToast({ msg: `Échec de la suppression : ${e?.message}`, sev: 'error' });
      }
      setConfirm(null);
    },
    [refresh]
  );

  const rows = useMemo(
    () =>
      docs
        .filter((d) => d.name.toLowerCase().includes(query.toLowerCase()))
        .map((d) => ({
          id: d.name,
          name: d.name,
          type: ext(d.name).toUpperCase(),
          chunks: d.chunks,
          status: 'Indexé',
          updated: d.created_at ? new Date(d.created_at * 1000) : null,
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
          <Chip
            label={ext(p.row.name).toUpperCase()}
            size="small"
            color={EXT_COLOR[ext(p.row.name)] || 'default'}
            variant="outlined"
            sx={{ height: 22, fontSize: 11, fontWeight: 700 }}
          />
          <Typography variant="body2" noWrap title={p.row.name}>
            {p.row.name}
          </Typography>
        </Box>
      ),
    },
    { field: 'chunks', headerName: 'Chunks', width: 110, type: 'number' },
    {
      field: 'status',
      headerName: 'Statut',
      width: 130,
      renderCell: () => (
        <Chip label="Indexé" size="small" color="success" variant="outlined" sx={{ height: 22, fontWeight: 700 }} />
      ),
    },
    {
      field: 'updated',
      headerName: 'Ajouté le',
      width: 180,
      valueFormatter: (value) => (value ? new Date(value as Date).toLocaleString() : '—'),
    },
    {
      field: 'actions',
      headerName: '',
      width: 70,
      sortable: false,
      filterable: false,
      align: 'right',
      renderCell: (p) => (
        <Tooltip title="Supprimer">
          <IconButton size="small" onClick={() => setConfirm(p.row.name)} sx={{ color: 'text.disabled', '&:hover': { color: 'error.main' } }}>
            <DeleteOutlineRoundedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      ),
    },
  ];

  return (
    <Box sx={{ height: '100%', overflowY: 'auto', p: { xs: 2, md: 4 } }}>
      <Box sx={{ maxWidth: 1100, mx: 'auto' }}>
        <Typography variant="h4" sx={{ fontWeight: 800, mb: 0.5 }}>
          Documents
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Gérez la base de connaissance : importez, consultez et supprimez les documents indexés.
        </Typography>

        {/* Dropzone */}
        <Box
          {...getRootProps()}
          sx={{
            p: 4,
            mb: 3,
            borderRadius: 2,
            border: '1.5px dashed',
            borderColor: isDragActive ? 'primary.main' : 'rgba(145,158,171,0.32)',
            bgcolor: isDragActive ? 'rgba(0,167,111,0.08)' : 'background.paper',
            textAlign: 'center',
            transition: 'all .2s',
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadRoundedIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
            Glissez vos fichiers ici
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            PDF · TXT · DOCX · MD · CSV · XLSX
          </Typography>
          <Button variant="contained" onClick={open} disabled={uploading}>
            Parcourir les fichiers
          </Button>
          {uploading && <LinearProgress sx={{ mt: 2, borderRadius: 1 }} />}
        </Box>

        <Card sx={{ p: 0 }}>
          <Box sx={{ p: 2 }}>
            <TextField
              size="small"
              placeholder="Rechercher un document…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              sx={{ width: { xs: '100%', sm: 320 } }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchRoundedIcon fontSize="small" />
                  </InputAdornment>
                ),
              }}
            />
          </Box>
          <DataGrid
            autoHeight
            rows={rows}
            columns={columns}
            disableRowSelectionOnClick
            pageSizeOptions={[10, 25, 50]}
            initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
            sx={{
              border: 0,
              '& .MuiDataGrid-columnHeaders': { bgcolor: 'grey.100' },
              '& .MuiDataGrid-cell:focus, & .MuiDataGrid-cell:focus-within': { outline: 'none' },
            }}
            localeText={{ noRowsLabel: 'Aucun document indexé.' }}
          />
        </Card>
      </Box>

      <Dialog open={!!confirm} onClose={() => setConfirm(null)} PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle>Supprimer le document ?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            « {confirm} » sera retiré de la base et ses vecteurs seront supprimés. Cette action est irréversible.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button color="inherit" onClick={() => setConfirm(null)}>Annuler</Button>
          <Button color="error" variant="contained" onClick={() => confirm && doDelete(confirm)}>
            Supprimer
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={!!toast}
        autoHideDuration={4000}
        onClose={() => setToast(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {toast ? (
          <Alert severity={toast.sev} variant="filled" onClose={() => setToast(null)}>
            {toast.msg}
          </Alert>
        ) : undefined}
      </Snackbar>
    </Box>
  );
}