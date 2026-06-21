'use client';

import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import type { DocumentItem, IndexingStatus, PendingAttachment } from './types';
import * as api from './api';

// Attachment indexing state lives ABOVE the chat page (mounted in the layout),
// exactly like the streaming state. This is the same fix as the "réponse
// interrompue" bug: the source of truth is the backend, queried fresh, and the
// in-progress state survives navigating to /documents and back instead of being
// trapped in (and wiped with) the ChatPage component.

const TERMINAL: IndexingStatus[] = ['indexed', 'failed'];
const isTerminal = (s?: IndexingStatus | null) => !!s && TERMINAL.includes(s);

interface AttachmentsContextValue {
  pending: PendingAttachment[];                 // composer draft (next message)
  addAttachments: (atts: PendingAttachment[]) => void;
  removeAttachment: (id: string) => void;
  clearPending: () => void;
  watch: (ids: string[]) => void;               // track docs (e.g. sent-message attachments) for live status
  statusOf: (id: string) => IndexingStatus | undefined;
}

const AttachmentsContext = createContext<AttachmentsContextValue | null>(null);

export function AttachmentsProvider({ children }: { children: React.ReactNode }) {
  const [pending, setPending] = useState<PendingAttachment[]>([]);
  const [statuses, setStatuses] = useState<Record<string, IndexingStatus>>({});

  // Ids whose indexing we still need to follow, and a guard so only one polling
  // loop runs at a time.
  const watchRef = useRef<Set<string>>(new Set());
  const pollingRef = useRef(false);

  const poll = useCallback(async () => {
    if (pollingRef.current) return;
    pollingRef.current = true;
    try {
      // Loop until every watched document reaches a terminal state. Reads
      // watchRef each pass so ids added mid-flight are picked up.
      while (watchRef.current.size > 0) {
        let docs: DocumentItem[] = [];
        try {
          docs = (await api.listDocuments()).documents;
        } catch {
          // Network/backend hiccup: wait and retry without dropping the watch set.
          await new Promise((r) => setTimeout(r, 2500));
          continue;
        }
        const byId = new Map(docs.map((d) => [d.id, d]));
        const updates: Record<string, IndexingStatus> = {};
        for (const id of [...watchRef.current]) {
          const found = byId.get(id);
          // A status we can read; if the doc isn't listed at all treat as failed
          // (deleted/inaccessible) so we stop watching instead of looping forever.
          const st: IndexingStatus =
            (found?.latest?.indexing_status as IndexingStatus) || (byId.size ? 'failed' : 'pending');
          updates[id] = st;
          if (isTerminal(st)) watchRef.current.delete(id);
        }
        setStatuses((prev) => ({ ...prev, ...updates }));
        setPending((prev) => prev.map((a) => (updates[a.id] ? { ...a, status: updates[a.id] } : a)));
        if (watchRef.current.size === 0) break;
        await new Promise((r) => setTimeout(r, 1800));
      }
    } finally {
      pollingRef.current = false;
    }
  }, []);

  const watch = useCallback((ids: string[]) => {
    let added = false;
    ids.forEach((id) => {
      if (id && !watchRef.current.has(id)) { watchRef.current.add(id); added = true; }
    });
    if (added) void poll();
  }, [poll]);

  const addAttachments = useCallback((atts: PendingAttachment[]) => {
    setPending((prev) => {
      const have = new Set(prev.map((a) => a.id));
      return [...prev, ...atts.filter((a) => !have.has(a.id))];
    });
    setStatuses((prev) => {
      const next = { ...prev };
      atts.forEach((a) => { next[a.id] = a.status; });
      return next;
    });
    watch(atts.map((a) => a.id));
  }, [watch]);

  const removeAttachment = useCallback((id: string) => {
    setPending((prev) => prev.filter((a) => a.id !== id));
  }, []);

  // Clear the draft list (e.g. after sending). Watching/statuses are kept so the
  // just-sent message's chip keeps updating until indexing finishes.
  const clearPending = useCallback(() => setPending([]), []);

  const statusOf = useCallback((id: string) => statuses[id], [statuses]);

  const value = useMemo<AttachmentsContextValue>(
    () => ({ pending, addAttachments, removeAttachment, clearPending, watch, statusOf }),
    [pending, addAttachments, removeAttachment, clearPending, watch, statusOf]
  );

  return <AttachmentsContext.Provider value={value}>{children}</AttachmentsContext.Provider>;
}

export function useAttachments(): AttachmentsContextValue {
  const ctx = useContext(AttachmentsContext);
  if (!ctx) throw new Error('useAttachments must be used within an AttachmentsProvider');
  return ctx;
}