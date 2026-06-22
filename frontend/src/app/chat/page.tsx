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
import AttachFileRoundedIcon from '@mui/icons-material/AttachFileRounded';

import ConversationsRail from '@/components/chat/ConversationsRail';
import MessageItem from '@/components/chat/MessageItem';
import Composer from '@/components/chat/Composer';
import type { Message, SessionSummary, Status, IndexingStatus, PendingAttachment } from '@/lib/types';
import * as api from '@/lib/api';
import { useChatStream } from '@/lib/chatStream';
import { useAttachments } from '@/lib/attachments';

// Per-conversation UI prefs (strict mode + active document scope) persisted across
// page navigation / reload. Keyed by conversation id (and '__draft__' before first send).
const STRICT_LS_KEY = 'rag.strictByConv';
const SCOPE_LS_KEY = 'rag.scopeByConv';

function loadConvMap<T extends object>(key: string): T {
  if (typeof window === 'undefined') return {} as T;
  try { return (JSON.parse(window.localStorage.getItem(key) || '{}') as T) || ({} as T); }
  catch { return {} as T; }
}
function saveConvMap(key: string, val: unknown): void {
  if (typeof window === 'undefined') return;
  try { window.localStorage.setItem(key, JSON.stringify(val)); } catch { /* ignore quota */ }
}

export default function ChatPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  // Strict RAG ("Documents uniquement") AND active document scope are stored PER
  // conversation, persisted in localStorage so they survive navigating to /documents
  // or /users and back (the chat page unmounts). '__draft__' holds the choice for a
  // not-yet-created conversation (migrated to its id on the first send).
  const [strictByConv, setStrictByConv] = useState<Record<string, boolean>>(() => loadConvMap(STRICT_LS_KEY));
  const [scopeByConv, setScopeByConv] = useState<Record<string, { id: string; filename: string }[]>>(() => loadConvMap(SCOPE_LS_KEY));
  useEffect(() => { saveConvMap(STRICT_LS_KEY, strictByConv); }, [strictByConv]);
  useEffect(() => { saveConvMap(SCOPE_LS_KEY, scopeByConv); }, [scopeByConv]);
  const docsOnly = strictByConv[currentId ?? '__draft__'] ?? false;
  const setDocsOnly = useCallback(
    (v: boolean) => setStrictByConv((m) => ({ ...m, [currentId ?? '__draft__']: v })),
    [currentId],
  );
  const [toast, setToast] = useState<{ msg: string; sev: 'success' | 'error' | 'info'; sid?: string } | null>(null);

  // Streaming lives ABOVE this page (in a layout-level provider), so a running
  // generation survives navigating to /documents and back — no false "réponse
  // interrompue", while parallel generations per conversation still work.
  const { streams, busyIds, isActive, startStream, abort, completion } = useChatStream();
  // Attachment indexing state ALSO lives above the page (same pattern): draft
  // attachments and their real (backend-polled) status survive navigation.
  const { pending: pendingAttachments, addAttachments, removeAttachment, clearPending, watch, statusOf } =
    useAttachments();

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const currentIdRef = useRef<string | null>(null);
  const sessionsRef = useRef<SessionSummary[]>([]);
  const lastCompletionSeq = useRef(0);
  useEffect(() => { currentIdRef.current = currentId; }, [currentId]);
  useEffect(() => { sessionsRef.current = sessions; }, [sessions]);

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

  // After (re)loading a conversation, keep following any attachment that is still
  // indexing so its chip updates to "done" live, and survives navigation.
  const watchMsgAttachments = useCallback((msgs: Message[]) => {
    const ids = msgs.flatMap((m) =>
      (m.attachments || [])
        .filter((a) => a.indexing_status === 'pending' || a.indexing_status === 'processing')
        .map((a) => a.id)
    );
    if (ids.length) watch(ids);
  }, [watch]);

  const reloadCurrent = useCallback(async (id: string) => {
    try {
      const detail = await api.getSession(id);
      setMessages(detail.messages || []);
      watchMsgAttachments(detail.messages || []);
    } catch {/* ignore */}
  }, [watchMsgAttachments]);

  const openSession = useCallback(async (id: string) => {
    setCurrentId(id);
    try {
      const detail = await api.getSession(id);
      setMessages(detail.messages || []);
      watchMsgAttachments(detail.messages || []);
    } catch {
      setMessages([]);
    }
  }, [watchMsgAttachments]);

  const newConversation = useCallback(() => { setCurrentId(null); setMessages([]); }, []);

  // React to a generation finishing (it may have completed while this page was
  // unmounted, e.g. the user was on /documents). Sync the view to the server.
  useEffect(() => {
    if (!completion || completion.seq === lastCompletionSeq.current) return;
    lastCompletionSeq.current = completion.seq;
    const { sid, ok, errored, errMsg, title } = completion;

    if (title) {
      setSessions((prev) => prev.map((s) => (s.id === sid ? { ...s, title } : s)));
    }
    void refreshSessions();

    const viewing = currentIdRef.current === sid;
    if (viewing) {
      // Always re-sync from the server, even on stop/error. The optimistic user
      // message uses a temporary id; without reloading it would keep that id and
      // a subsequent edit/regenerate would fail ("Editable user message not
      // found"). The backend persists the user message (and any partial answer)
      // regardless of how the stream ended.
      void reloadCurrent(sid);
      if (!ok && errored) {
        setToast({ msg: `⚠️ ${errMsg || 'Erreur'}`, sev: 'error' });
      }
    } else if (ok && !isActive(sid)) {
      // A generation we navigated away from has finished. Show a notification that
      // names the real conversation and lets the user jump straight to it. Resolve
      // the title from the (just-finished) auto-title, the session list, or a fresh
      // fetch — never the meaningless "une conversation" placeholder.
      void (async () => {
        let t = title || sessionsRef.current.find((s) => s.id === sid)?.title || '';
        if (!t) {
          try { t = (await api.getSession(sid)).title; } catch {/* ignore */}
        }
        if (isActive(sid)) return; // a new generation started in the meantime
        setToast({
          msg: t ? `Réponse prête dans « ${t} »` : 'Une réponse est prête dans une autre conversation',
          sev: 'success',
          sid,
        });
      })();
    }
  }, [completion, refreshSessions, reloadCurrent, isActive]);

  const handleSend = useCallback(
    async (text: string) => {
      let sid = currentId;
      const firstTurn = !sid || messages.length === 0;
      // Strict mode chosen for THIS conversation (or the draft, before it exists).
      const effStrict = strictByConv[currentId ?? '__draft__'] ?? false;
      if (!sid) {
        try {
          const s = await api.createSession();
          sid = s.id;
          setCurrentId(sid);
          // Carry the draft strict choice onto the freshly-created conversation.
          setStrictByConv((m) => ({ ...m, [sid as string]: effStrict }));
          await refreshSessions();
        } catch { return; }
      }
      if (isActive(sid)) return; // this conversation is already generating

      const atts = pendingAttachments.map((a) => ({ id: a.id, filename: a.filename }));
      // Sticky scope: newly attached docs become the conversation's active scope;
      // otherwise reuse the scope already active for this conversation (follow-ups).
      const sticky = scopeByConv[sid as string] || [];
      const scopeDocs = atts.length ? atts : sticky;
      const scope = scopeDocs.length ? { type: 'document', ids: scopeDocs.map((a) => a.id) } : undefined;
      const notReady = pendingAttachments.some((a) => a.status !== 'indexed' && a.status !== 'failed');
      if (atts.length) setScopeByConv((m) => ({ ...m, [sid as string]: atts }));

      setMessages((m) => [
        ...m,
        {
          id: `tmp-${Date.now()}`, parent_id: null, role: 'user', content: text,
          attachments: atts.length ? atts : undefined,
          version_index: 1, version_count: 1, sibling_ids: [],
        },
      ]);
      clearPending(); // consumed by this turn (status keeps updating via the provider)

      if (notReady) {
        setToast({ msg: 'Indexation du document en cours — la réponse démarrera dès qu\'il sera prêt.', sev: 'info' });
      }

      startStream({
        url: api.chatUrl(sid),
        body: { message: text, scope, attachments: atts.length ? atts : undefined, strict: effStrict || undefined },
        sid,
        firstTurn,
      });
    },
    [currentId, messages.length, pendingAttachments, isActive, startStream, refreshSessions, clearPending, strictByConv, scopeByConv]
  );

  // Edit and regenerate must reuse the SAME effective strict value and active
  // document scope as a normal send (handleSend), read from the per-conversation
  // state. Without this, editing/regenerating silently dropped "Documents
  // uniquement" and the sticky document scope, falling back to server defaults.
  const handleEdit = useCallback(
    async (id: string, text: string) => {
      if (!currentId || !text || isActive(currentId)) return;
      const effStrict = strictByConv[currentId] ?? false;
      const sticky = scopeByConv[currentId] || [];
      const scope = sticky.length ? { type: 'document', ids: sticky.map((a) => a.id) } : undefined;
      setMessages((prev) => {
        const i = prev.findIndex((m) => m.id === id);
        if (i < 0) return prev;
        const copy = prev.slice(0, i + 1);
        copy[i] = { ...copy[i], content: text };
        return copy;
      });
      startStream({
        url: api.editUrl(currentId, id),
        body: { message: text, scope, strict: effStrict || undefined },
        sid: currentId,
      });
    },
    [currentId, isActive, startStream, strictByConv, scopeByConv]
  );

  const handleRegenerate = useCallback(
    async (userMessageId: string) => {
      if (!currentId || isActive(currentId)) return;
      const effStrict = strictByConv[currentId] ?? false;
      const sticky = scopeByConv[currentId] || [];
      const scope = sticky.length ? { type: 'document', ids: sticky.map((a) => a.id) } : undefined;
      setMessages((prev) => {
        const i = prev.findIndex((m) => m.id === userMessageId);
        return i < 0 ? prev : prev.slice(0, i + 1);
      });
      startStream({
        url: api.regenerateUrl(currentId, userMessageId),
        body: { scope, strict: effStrict || undefined },
        sid: currentId,
      });
    },
    [currentId, isActive, startStream, strictByConv, scopeByConv]
  );

  const handleSwitchVersion = useCallback(
    async (messageId: string) => {
      if (!currentId || isActive(currentId)) return;
      try {
        await api.selectVersion(currentId, messageId);
        await reloadCurrent(currentId);
      } catch {/* ignore */}
    },
    [currentId, isActive, reloadCurrent]
  );

  const handleStop = useCallback(() => { abort(currentId); }, [abort, currentId]);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        abort(id);
        await api.deleteSession(id);
        if (currentId === id) newConversation();
        await refreshSessions();
      } catch {/* ignore */}
      setConfirmId(null);
    },
    [abort, currentId, newConversation, refreshSessions]
  );

  const handleAttach = useCallback(async (files: FileList) => {
    setUploading(true);
    const added: PendingAttachment[] = [];
    for (const f of Array.from(files)) {
      try {
        const r = await api.uploadDocument(f, 'personal');
        added.push({ id: r.document_id, filename: r.filename, status: (r.indexing_status as IndexingStatus) || 'pending' });
      } catch (e: any) {
        setToast({ msg: `Échec : ${e?.message || f.name}`, sev: 'error' });
      }
    }
    setUploading(false);
    if (added.length) {
      // The provider tracks these and polls the backend for their real status
      // (survives navigation); no page-local timer that dies on unmount.
      addAttachments(added);
      setToast({ msg: `${added.length} document(s) joint(s) — ils seront prioritaires pour votre prochaine question.`, sev: 'success' });
    }
    refreshStatus();
  }, [addAttachments, refreshStatus]);

  const cur = currentId ? streams[currentId] : undefined;
  const liveLastUser = messages.length > 0 && messages[messages.length - 1].role === 'user';
  const showGroundingFlag = (status?.total_documents || 0) > 0;

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      <ConversationsRail
        sessions={sessions}
        currentId={currentId}
        busyIds={busyIds}
        onNew={newConversation}
        onSelect={openSession}
        onDelete={(id) => setConfirmId(id)}
        onRename={async (id, title) => {
          try {
            await api.renameSession(id, title);
            setSessions((prev) => prev.map((s) => s.id === id ? { ...s, title } : s));
          } catch { /* silent */ }
        }}
      />

      <Box sx={{ flexGrow: 1, minWidth: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box sx={{ px: 3, py: 1.75, borderBottom: '1px dashed rgba(145,158,171,0.2)', bgcolor: 'background.paper' }}>
          <Typography dir="auto" variant="subtitle1" sx={{ fontWeight: 700 }} noWrap>
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
                  Posez une question, ou joignez un document avec <AttachFileRoundedIcon sx={{ fontSize: 16, verticalAlign: 'middle', mx: 0.3 }} /> et l&apos;assistant répondra
                  en s&apos;appuyant dessus — avec sources.
                </Typography>
              </Box>
            )}

            {messages.map((m) => (
              <MessageItem
                key={m.id}
                m={m}
                showGroundingFlag={showGroundingFlag}
                streaming={!!cur?.active}
                statusOf={statusOf}
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
                    <Typography dir="auto" variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
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

        <Composer
          streaming={!!cur?.active}
          uploading={uploading}
          attachments={pendingAttachments}
          strict={docsOnly}
          onStrictChange={setDocsOnly}
          scopeDocs={currentId ? scopeByConv[currentId] || [] : []}
          onClearScope={() => currentId && setScopeByConv((m) => ({ ...m, [currentId]: [] }))}
          onSend={handleSend}
          onStop={handleStop}
          onAttach={handleAttach}
          onRemoveAttachment={removeAttachment}
        />
      </Box>

      <Dialog open={!!confirmId} onClose={() => setConfirmId(null)} PaperProps={{ sx: { borderRadius: 2 } }}>
        <DialogTitle>Supprimer la conversation ?</DialogTitle>
        <DialogContent><DialogContentText>Cette action est irréversible.</DialogContentText></DialogContent>
        <DialogActions>
          <Button color="inherit" onClick={() => setConfirmId(null)}>Annuler</Button>
          <Button color="error" variant="contained" onClick={() => confirmId && handleDelete(confirmId)}>Supprimer</Button>
        </DialogActions>
      </Dialog>

      <Snackbar open={!!toast} autoHideDuration={5000} onClose={() => setToast(null)} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        {toast ? (
          <Alert
            severity={toast.sev}
            variant="filled"
            onClose={() => setToast(null)}
            onClick={toast.sid ? () => { openSession(toast.sid as string); setToast(null); } : undefined}
            sx={toast.sid ? { cursor: 'pointer' } : undefined}
          >
            {toast.msg}{toast.sid ? ' — cliquer pour ouvrir' : ''}
          </Alert>
        ) : undefined}
      </Snackbar>
    </Box>
  );
}