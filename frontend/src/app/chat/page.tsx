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
  answer: string;
  sources: Source[];
  thinking: boolean;
  error: string | null;
}

const EMPTY_STREAM: StreamState = { active: false, answer: '', sources: [], thinking: false, error: null };

export default function ChatPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  // One in-flight generation per conversation → parallel chats across sessions.
  const [streams, setStreams] = useState<Record<string, StreamState>>({});
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; sev: 'success' | 'error' } | null>(null);

  const controllersRef = useRef<Record<string, AbortController>>({});
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const currentIdRef = useRef<string | null>(null);
  const sessionsRef = useRef<SessionSummary[]>([]);
  useEffect(() => { currentIdRef.current = currentId; }, [currentId]);
  useEffect(() => { sessionsRef.current = sessions; }, [sessions]);

  const setStreamFor = useCallback((sid: string, patch: Partial<StreamState>) => {
    setStreams((prev) => ({ ...prev, [sid]: { ...(prev[sid] || EMPTY_STREAM), ...patch } }));
  }, []);
  const clearStreamFor = useCallback((sid: string) => {
    setStreams((prev) => { const n = { ...prev }; delete n[sid]; return n; });
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const { sessions } = await api.listSessions();
      setSessions(sessions);
    } catch {/* ignore */}
  }, []);
  const refreshStatus = useCallback(() => { api.getStatus().then(setStatus).catch(() => {}); }, []);

  useEffect(() => { refreshStatus(); refreshSessions(); }, [refreshStatus, refreshSessions]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streams, currentId]);

  const openSession = useCallback(async (id: string) => {
    setCurrentId(id);
    try {
      const detail = await api.getSession(id);
      setMessages(detail.messages || []);
    } catch {
      setMessages([]);
    }
  }, []);

  const newConversation = useCallback(() => { setCurrentId(null); setMessages([]); }, []);

  const reloadCurrent = useCallback(async (id: string) => {
    try {
      const detail = await api.getSession(id);
      setMessages(detail.messages || []);
    } catch {/* ignore */}
  }, []);

  // Shared streaming runner, keyed by session so multiple can run at once.
  const runStream = useCallback(
    async (url: string, body: Record<string, unknown> | undefined, sid: string, opts: { firstTurn?: boolean } = {}) => {
      const controller = new AbortController();
      controllersRef.current[sid] = controller;
      setStreamFor(sid, { active: true, thinking: true, answer: '', sources: [], error: null });
      let acc = '';
      let errored = false;
      let errMsg = '';
      let doneTitle = '';
      try {
        await api.streamNdjson(
          url, body,
          (e: StreamEvent) => {
            if (e.type === 'token') { acc += e.data; setStreamFor(sid, { thinking: false, answer: acc }); }
            else if (e.type === 'sources') { setStreamFor(sid, { sources: e.data }); }
            else if (e.type === 'error') { errored = true; errMsg = e.data; setStreamFor(sid, { error: e.data }); }
            else if (e.type === 'done') { doneTitle = e.title; }
          },
          controller.signal
        );
      } catch (err: any) {
        if (err?.name !== 'AbortError') { errored = true; errMsg = err?.message || 'Erreur'; setStreamFor(sid, { error: errMsg }); }
      } finally {
        delete controllersRef.current[sid];
        const ok = !errored && acc.trim().length > 0;
        if (opts.firstTurn && ok) await api.autotitleSession(sid).catch(() => {});
        await refreshSessions();
        const viewing = currentIdRef.current === sid;
        if (viewing && ok) {
          await reloadCurrent(sid);
        } else if (viewing && !ok) {
          const text = errored ? `⚠️ ${errMsg || 'Erreur'}` : (acc.trim() || '⚠️ Aucune réponse générée.');
          setMessages((m) => [...m, { id: `tmp-a-${Date.now()}`, parent_id: null, role: 'assistant', content: text, version_index: 1, version_count: 1, sibling_ids: [] }]);
        } else if (!viewing && ok) {
          const title = doneTitle || sessionsRef.current.find((s) => s.id === sid)?.title || 'une conversation';
          setToast({ msg: `Réponse prête : « ${title} »`, sev: 'success' });
        }
        clearStreamFor(sid);
      }
    },
    [setStreamFor, clearStreamFor, refreshSessions, reloadCurrent]
  );

  const handleSend = useCallback(
    async (text: string) => {
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
      if (streams[sid]?.active) return; // this conversation is already generating
      setMessages((m) => [...m, { id: `tmp-${Date.now()}`, parent_id: null, role: 'user', content: text, version_index: 1, version_count: 1, sibling_ids: [] }]);
      void runStream(api.chatUrl(sid), { message: text }, sid, { firstTurn });
    },
    [currentId, messages.length, streams, runStream, refreshSessions]
  );

  const handleEdit = useCallback(
    async (id: string, text: string) => {
      if (!currentId || !text || streams[currentId]?.active) return;
      setMessages((prev) => {
        const i = prev.findIndex((m) => m.id === id);
        if (i < 0) return prev;
        const copy = prev.slice(0, i + 1);
        copy[i] = { ...copy[i], content: text };
        return copy;
      });
      void runStream(api.editUrl(currentId, id), { message: text }, currentId, {});
    },
    [currentId, streams, runStream]
  );

  const handleRegenerate = useCallback(
    async (userMessageId: string) => {
      if (!currentId || streams[currentId]?.active) return;
      setMessages((prev) => {
        const i = prev.findIndex((m) => m.id === userMessageId);
        return i < 0 ? prev : prev.slice(0, i + 1);
      });
      void runStream(api.regenerateUrl(currentId, userMessageId), {}, currentId, {});
    },
    [currentId, streams, runStream]
  );

  const handleSwitchVersion = useCallback(
    async (messageId: string) => {
      if (!currentId || streams[currentId]?.active) return;
      try {
        await api.selectVersion(currentId, messageId);
        await reloadCurrent(currentId);
      } catch {/* ignore */}
    },
    [currentId, streams, reloadCurrent]
  );

  const handleStop = useCallback(() => {
    if (currentId) controllersRef.current[currentId]?.abort();
  }, [currentId]);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        controllersRef.current[id]?.abort();
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
      try { await api.uploadDocument(f, 'personal'); n += 1; }
      catch (e: any) { setToast({ msg: `Échec : ${e?.message || f.name}`, sev: 'error' }); }
    }
    setUploading(false);
    if (n) setToast({ msg: `${n} document(s) envoyé(s) — indexation en cours.`, sev: 'success' });
    refreshStatus();
  }, [refreshStatus]);

  const cur = currentId ? streams[currentId] : undefined;
  const liveLastUser = messages.length > 0 && messages[messages.length - 1].role === 'user';
  const showGroundingFlag = (status?.total_documents || 0) > 0;
  const busySessions = Object.entries(streams).filter(([, s]) => s.active).map(([id]) => id);

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      <ConversationsRail
        sessions={sessions}
        currentId={currentId}
        busyIds={busySessions}
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
            {messages.length === 0 && !cur?.active && (
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

            {cur?.active && (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 0.5 }}>
                <Typography variant="caption" sx={{ color: 'warning.dark', fontWeight: 700 }}>ASSISTANT</Typography>
                <Paper sx={{ px: 2, py: 1.5, width: '100%', border: '1px solid rgba(145,158,171,0.2)', boxShadow: 'none', borderRadius: 2 }}>
                  {cur.thinking && !cur.answer ? (
                    <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>Recherche dans la base…</Typography>
                  ) : cur.error ? (
                    <Typography variant="body2" color="error">⚠️ {cur.error}</Typography>
                  ) : (
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
                      {cur.answer}
                      <Box component="span" sx={{ animation: 'blink 1s steps(2) infinite', '@keyframes blink': { '50%': { opacity: 0 } } }}>▋</Box>
                    </Typography>
                  )}
                </Paper>
              </Box>
            )}

            {liveLastUser && !cur?.active && (
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

        <Composer streaming={!!cur?.active} uploading={uploading} onSend={handleSend} onStop={handleStop} onAttach={handleAttach} />
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
