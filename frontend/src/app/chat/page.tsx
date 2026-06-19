'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';

import ConversationsRail from '@/components/chat/ConversationsRail';
import MessageItem from '@/components/chat/MessageItem';
import Composer from '@/components/chat/Composer';
import type { Message, SessionSummary, Status, StreamEvent, Source } from '@/lib/types';
import * as api from '@/lib/api';

interface StreamState {
  active: boolean;
  sessionId: string | null;
  answer: string;
  sources: Source[];
  thinking: boolean;
  error: string | null;
}

const EMPTY_STREAM: StreamState = {
  active: false, sessionId: null, answer: '', sources: [], thinking: false, error: null,
};

export default function ChatPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; sev: 'success' | 'error' } | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const currentIdRef = useRef<string | null>(null);
  useEffect(() => { currentIdRef.current = currentId; }, [currentId]);

  const refreshSessions = useCallback(async () => {
    try {
      const { sessions } = await api.listSessions();
      setSessions(sessions);
    } catch {/* ignore */}
  }, []);

  const refreshStatus = useCallback(() => {
    api.getStatus().then(setStatus).catch(() => {});
  }, []);

  useEffect(() => {
    refreshStatus();
    refreshSessions();
  }, [refreshStatus, refreshSessions]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, stream.answer, stream.active]);

  const openSession = useCallback(async (id: string) => {
    setCurrentId(id);
    try {
      const detail = await api.getSession(id);
      setMessages(detail.messages || []);
    } catch {
      setMessages([]);
    }
  }, []);

  const newConversation = useCallback(() => {
    setCurrentId(null);
    setMessages([]);
  }, []);

  const reloadCurrent = useCallback(async (id: string) => {
    try {
      const detail = await api.getSession(id);
      setMessages(detail.messages || []);
    } catch {/* ignore */}
  }, []);

  const runStream = useCallback(
    async (url: string, body: Record<string, unknown> | undefined, sid: string, opts: { firstTurn?: boolean } = {}) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setStream({ ...EMPTY_STREAM, active: true, thinking: true, sessionId: sid });
      let acc = '';
      let errored = false;
      let errMsg = '';
      try {
        await api.streamNdjson(
          url, body,
          (e: StreamEvent) => {
            if (e.type === 'token') { acc += e.data; setStream((s) => ({ ...s, thinking: false, answer: acc })); }
            else if (e.type === 'sources') { setStream((s) => ({ ...s, sources: e.data })); }
            else if (e.type === 'error') { errored = true; errMsg = e.data; setStream((s) => ({ ...s, error: e.data })); }
          },
          controller.signal
        );
      } catch (err: any) {
        if (err?.name !== 'AbortError') { errored = true; errMsg = err?.message || 'Erreur'; setStream((s) => ({ ...s, error: errMsg })); }
      } finally {
        abortRef.current = null;
        const ok = !errored && acc.trim().length > 0;
        if (opts.firstTurn && ok) await api.autotitleSession(sid).catch(() => {});
        await refreshSessions();
        if (currentIdRef.current === sid) {
          if (ok) {
            reloadCurrent(sid);
          } else {
            const text = errored ? `⚠️ ${errMsg || 'Erreur'}` : (acc.trim() || '⚠️ Aucune réponse générée.');
            setMessages((m) => [...m, { id: `tmp-a-${Date.now()}`, parent_id: null, role: 'assistant', content: text, version_index: 1, version_count: 1, sibling_ids: [] }]);
          }
        }
        setStream(EMPTY_STREAM);
      }
    },
    [refreshSessions, reloadCurrent]
  );

  const handleSend = useCallback(
    async (text: string) => {
      if (stream.active) return;
      let sid = currentId;
      const firstTurn = !sid || messages.length === 0;
      if (!sid) {
        try {
          const s = await api.createSession();
          sid = s.id;
          setCurrentId(sid);
          await refreshSessions();
        } catch { return; }
      }
      setMessages((m) => [...m, { id: `tmp-${Date.now()}`, parent_id: null, role: 'user', content: text, version_index: 1, version_count: 1, sibling_ids: [] }]);
      await runStream(api.chatUrl(sid!), { message: text }, sid!, { firstTurn });
    },
    [currentId, messages.length, stream.active, runStream, refreshSessions]
  );

  const handleEdit = useCallback(
    async (id: string, text: string) => {
      if (!currentId || !text || stream.active) return;
      setMessages((prev) => {
        const i = prev.findIndex((m) => m.id === id);
        if (i < 0) return prev;
        const copy = prev.slice(0, i + 1);
        copy[i] = { ...copy[i], content: text };
        return copy;
      });
      await runStream(api.editUrl(currentId, id), { message: text }, currentId, {});
    },
    [currentId, stream.active, runStream]
  );

  const handleRegenerate = useCallback(
    async (userMessageId: string) => {
      if (!currentId || stream.active) return;
      setMessages((prev) => {
        const i = prev.findIndex((m) => m.id === userMessageId);
        return i < 0 ? prev : prev.slice(0, i + 1);
      });
      await runStream(api.regenerateUrl(currentId, userMessageId), {}, currentId, {});
    },
    [currentId, stream.active, runStream]
  );

  const handleSwitchVersion = useCallback(
    async (messageId: string) => {
      if (!currentId || stream.active) return;
      try {
        await api.selectVersion(currentId, messageId);
        await reloadCurrent(currentId);
      } catch {/* ignore */}
    },
    [currentId, stream.active, reloadCurrent]
  );

  const handleStop = useCallback(() => abortRef.current?.abort(), []);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await api.deleteSession(id);
        if (currentId === id) newConversation();
        await refreshSessions();
      } catch {/* ignore */}
      setConfirmId(null);
    },
    [currentId, newConversation, refreshSessions]
  );

  const handleAttach = useCallback(async (files: FileList) => {
    setUploading(true);
    let n = 0;
    for (const f of Array.from(files)) {
      try { await api.uploadDocument(f); n += 1; }
      catch (e: any) { setToast({ msg: `Échec : ${e?.message || f.name}`, sev: 'error' }); }
    }
    setUploading(false);
    if (n) setToast({ msg: `${n} document(s) envoyé(s) — indexation en cours.`, sev: 'success' });
    refreshStatus();
  }, [refreshStatus]);

  const liveLastUser = messages.length > 0 && messages[messages.length - 1].role === 'user';
  const showGroundingFlag = (status?.total_documents || 0) > 0;

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      <ConversationsRail
        sessions={sessions}
        currentId={currentId}
        onNew={newConversation}
        onSelect={openSession}
        onDelete={(id) => setConfirmId(id)}
      />

      <Box sx={{ flexGrow: 1, minWidth: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box sx={{ px: 3, py: 1.75, borderBottom: '1px dashed rgba(145,158,171,0.2)', bgcolor: 'background.paper' }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }} noWrap>
            {sessions.find((s) => s.id === currentId)?.title || 'Nouvelle conversation'}
          </Typography>
        </Box>

        <Box ref={scrollRef} sx={{ flexGrow: 1, overflowY: 'auto', px: { xs: 2, md: 6 }, py: 3 }}>
          <Box sx={{ maxWidth: 860, mx: 'auto', display: 'flex', flexDirection: 'column', gap: 2.5 }}>
            {messages.length === 0 && !stream.active && (
              <Box sx={{ textAlign: 'center', py: 8, color: 'text.secondary' }}>
                <Box sx={{ width: 64, height: 64, mx: 'auto', mb: 2, borderRadius: 2.5, bgcolor: 'rgba(0,167,111,0.12)', color: 'primary.main', display: 'grid', placeItems: 'center' }}>
                  <AutoAwesomeRoundedIcon fontSize="large" />
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 800, color: 'text.primary' }}>
                  Interrogez votre base de connaissance
                </Typography>
                <Typography variant="body2" sx={{ mt: 1, maxWidth: 460, mx: 'auto' }}>
                  Posez une question, ou joignez un document avec 📎 et l&apos;assistant répondra
                  en s&apos;appuyant dessus — avec sources.
                </Typography>
              </Box>
            )}

            {messages.map((m) => (
              <MessageItem
                key={m.id}
                m={m}
                showGroundingFlag={showGroundingFlag}
                onSwitchVersion={handleSwitchVersion}
                onEdit={handleEdit}
                onRegenerate={handleRegenerate}
              />
            ))}

            {stream.active && stream.sessionId === currentId && (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 0.5 }}>
                <Typography variant="caption" sx={{ color: 'warning.dark', fontWeight: 700 }}>ASSISTANT</Typography>
                <Paper sx={{ px: 2, py: 1.5, width: '100%', border: '1px solid rgba(145,158,171,0.2)', boxShadow: 'none', borderRadius: 2 }}>
                  {stream.thinking && !stream.answer ? (
                    <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>Recherche dans la base…</Typography>
                  ) : stream.error ? (
                    <Typography variant="body2" color="error">⚠️ {stream.error}</Typography>
                  ) : (
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
                      {stream.answer}
                      <Box component="span" sx={{ animation: 'blink 1s steps(2) infinite', '@keyframes blink': { '50%': { opacity: 0 } } }}>▋</Box>
                    </Typography>
                  )}
                </Paper>
              </Box>
            )}

            {liveLastUser && !stream.active && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, color: 'text.secondary' }}>
                <Typography variant="body2">La réponse a été interrompue.</Typography>
                <Button size="small" variant="outlined" color="inherit" startIcon={<RefreshRoundedIcon />}
                  onClick={() => handleRegenerate(messages[messages.length - 1].id)}>
                  Régénérer
                </Button>
              </Box>
            )}
          </Box>
        </Box>

        <Composer streaming={stream.active} uploading={uploading} onSend={handleSend} onStop={handleStop} onAttach={handleAttach} />
      </Box>

      <Dialog open={!!confirmId} onClose={() => setConfirmId(null)} PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle>Supprimer la conversation ?</DialogTitle>
        <DialogContent><DialogContentText>Cette action est irréversible.</DialogContentText></DialogContent>
        <DialogActions>
          <Button color="inherit" onClick={() => setConfirmId(null)}>Annuler</Button>
          <Button color="error" variant="contained" onClick={() => confirmId && handleDelete(confirmId)}>Supprimer</Button>
        </DialogActions>
      </Dialog>

      <Snackbar open={!!toast} autoHideDuration={4000} onClose={() => setToast(null)} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        {toast ? <Alert severity={toast.sev} variant="filled" onClose={() => setToast(null)}>{toast.msg}</Alert> : undefined}
      </Snackbar>
    </Box>
  );
}
