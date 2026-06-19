'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import ViewSidebarRoundedIcon from '@mui/icons-material/ViewSidebarRounded';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';

import ConversationsRail from '@/components/chat/ConversationsRail';
import MessageItem from '@/components/chat/MessageItem';
import SourcesPanel from '@/components/chat/SourcesPanel';
import Composer from '@/components/chat/Composer';
import type { DocumentItem, Message, SessionSummary, Source, Status, StreamEvent } from '@/lib/types';
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
  active: false,
  sessionId: null,
  answer: '',
  sources: [],
  thinking: false,
  error: null,
};

export default function ChatPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM);
  const [sourcesOpen, setSourcesOpen] = useState(true);
  const [confirmId, setConfirmId] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const refreshSessions = useCallback(async () => {
    try {
      const { sessions } = await api.listSessions();
      setSessions(sessions);
    } catch {/* ignore */}
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      const { documents } = await api.listDocuments();
      setDocuments(documents);
    } catch {/* ignore */}
  }, []);

  useEffect(() => {
    api.getStatus().then(setStatus).catch(() => {});
    refreshSessions();
    refreshDocuments();
  }, [refreshSessions, refreshDocuments]);

  // Autoscroll on new content.
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

  // Core streaming runner shared by send / edit / regenerate.
  const runStream = useCallback(
    async (
      url: string,
      body: Record<string, unknown> | undefined,
      sid: string,
      opts: { firstTurn?: boolean } = {}
    ) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setStream({ ...EMPTY_STREAM, active: true, thinking: true, sessionId: sid });
      let acc = '';
      let srcs: Source[] = [];
      try {
        await api.streamNdjson(
          url,
          body,
          (e: StreamEvent) => {
            if (e.type === 'token') {
              acc += e.data;
              setStream((s) => ({ ...s, thinking: false, answer: acc }));
            } else if (e.type === 'sources') {
              srcs = e.data;
              setStream((s) => ({ ...s, sources: srcs }));
            } else if (e.type === 'error') {
              setStream((s) => ({ ...s, error: e.data }));
            }
          },
          controller.signal
        );
      } catch (err: any) {
        if (err?.name !== 'AbortError') setStream((s) => ({ ...s, error: err?.message || 'Erreur' }));
      } finally {
        abortRef.current = null;
        setStream((s) => ({ ...s, active: false }));
        if (opts.firstTurn) await api.autotitleSession(sid).catch(() => {});
        await refreshSessions();
        // Reload the canonical tree (ids, version arrows) for the viewed session.
        setCurrentId((cur) => {
          if (cur === sid) reloadCurrent(sid);
          return cur;
        });
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
        } catch {
          return;
        }
      }
      // optimistic user bubble
      setMessages((m) => [
        ...m,
        { id: `tmp-${Date.now()}`, parent_id: null, role: 'user', content: text, version_index: 1, version_count: 1, sibling_ids: [] },
      ]);
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

  const lastSources = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') return messages[i].sources || [];
    }
    return [];
  }, [messages]);

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

      {/* Center column */}
      <Box sx={{ flexGrow: 1, minWidth: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Header */}
        <Box
          sx={{
            px: 3,
            py: 1.5,
            borderBottom: '1px dashed rgba(145,158,171,0.2)',
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            bgcolor: 'background.paper',
          }}
        >
          <Typography variant="subtitle1" sx={{ fontWeight: 700, flexGrow: 1 }} noWrap>
            {sessions.find((s) => s.id === currentId)?.title || 'Nouvelle conversation'}
          </Typography>
          <Tooltip title={sourcesOpen ? 'Masquer les sources' : 'Afficher les sources'}>
            <IconButton onClick={() => setSourcesOpen((o) => !o)} color={sourcesOpen ? 'primary' : 'default'}>
              <ViewSidebarRoundedIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Messages */}
        <Box ref={scrollRef} sx={{ flexGrow: 1, overflowY: 'auto', px: { xs: 2, md: 6 }, py: 3 }}>
          <Box sx={{ maxWidth: 860, mx: 'auto', display: 'flex', flexDirection: 'column', gap: 2.5 }}>
            {messages.length === 0 && !stream.active && (
              <Box sx={{ textAlign: 'center', py: 8, color: 'text.secondary' }}>
                <Box
                  sx={{
                    width: 64,
                    height: 64,
                    mx: 'auto',
                    mb: 2,
                    borderRadius: 2.5,
                    bgcolor: 'rgba(0,167,111,0.12)',
                    color: 'primary.main',
                    display: 'grid',
                    placeItems: 'center',
                  }}
                >
                  <AutoAwesomeRoundedIcon fontSize="large" />
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 800, color: 'text.primary' }}>
                  Interrogez votre base de connaissance
                </Typography>
                <Typography variant="body2" sx={{ mt: 1, maxWidth: 460, mx: 'auto' }}>
                  Posez une question générale, ou ajoutez des documents et l&apos;assistant répondra
                  en s&apos;appuyant dessus — avec citations.
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

            {/* Live streaming bubble */}
            {stream.active && stream.sessionId === currentId && (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 0.5 }}>
                <Typography variant="caption" sx={{ color: 'warning.dark', fontWeight: 700 }}>
                  ASSISTANT
                </Typography>
                <Paper sx={{ px: 2, py: 1.5, width: '100%', border: '1px solid rgba(145,158,171,0.2)', boxShadow: 'none', borderRadius: 2 }}>
                  {stream.thinking && !stream.answer ? (
                    <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
                      Recherche dans la base…
                    </Typography>
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

            {/* Interrupted-answer recovery */}
            {liveLastUser && !stream.active && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, color: 'text.secondary' }}>
                <Typography variant="body2">La réponse a été interrompue.</Typography>
                <Button
                  size="small"
                  variant="outlined"
                  color="inherit"
                  startIcon={<RefreshRoundedIcon />}
                  onClick={() => handleRegenerate(messages[messages.length - 1].id)}
                >
                  Régénérer
                </Button>
              </Box>
            )}
          </Box>
        </Box>

        <Composer streaming={stream.active} onSend={handleSend} onStop={handleStop} />
      </Box>

      {sourcesOpen && (
        <SourcesPanel
          documents={documents}
          lastSources={lastSources}
          uploadFn={api.uploadDocument}
          onUploaded={() => {
            refreshDocuments();
            api.getStatus().then(setStatus).catch(() => {});
          }}
        />
      )}

      <Dialog open={!!confirmId} onClose={() => setConfirmId(null)} PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle>Supprimer la conversation ?</DialogTitle>
        <DialogContent>
          <DialogContentText>Cette action est irréversible.</DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button color="inherit" onClick={() => setConfirmId(null)}>Annuler</Button>
          <Button color="error" variant="contained" onClick={() => confirmId && handleDelete(confirmId)}>
            Supprimer
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
