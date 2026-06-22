'use client';

import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import type { Source, StreamEvent } from './types';
import * as api from './api';

// One in-flight generation per conversation. This state lives ABOVE the page
// (mounted in the layout via Providers), so a generation survives navigating
// between /chat and /documents — fixing the false "réponse interrompue" that
// happened when the page unmounted and its local state was destroyed.

export interface StreamState {
  active: boolean;
  answer: string;
  sources: Source[];
  thinking: boolean;
  error: string | null;
}

export const EMPTY_STREAM: StreamState = {
  active: false,
  answer: '',
  sources: [],
  thinking: false,
  error: null,
};

// Emitted once when a generation finishes, so the page can sync its view
// (reload messages, refresh the session list, apply the auto-title, toast).
export interface Completion {
  seq: number;
  sid: string;
  ok: boolean;
  errored: boolean;
  errMsg: string;
  answer: string;
  title: string;
}

export interface StartStreamArgs {
  url: string;
  body: Record<string, unknown> | undefined;
  sid: string;
  firstTurn?: boolean;
}

interface ChatStreamContextValue {
  streams: Record<string, StreamState>;
  busyIds: string[];
  isActive: (sid: string | null | undefined) => boolean;
  startStream: (args: StartStreamArgs) => void;
  abort: (sid: string | null | undefined) => void;
  completion: Completion | null;
}

const ChatStreamContext = createContext<ChatStreamContextValue | null>(null);

export function ChatStreamProvider({ children }: { children: React.ReactNode }) {
  const [streams, setStreams] = useState<Record<string, StreamState>>({});
  const [completion, setCompletion] = useState<Completion | null>(null);

  const controllersRef = useRef<Record<string, AbortController>>({});
  // Mirror of `streams` for synchronous reads (guards, no stale closures).
  const streamsRef = useRef<Record<string, StreamState>>({});
  const seqRef = useRef(0);

  const setStreamFor = useCallback((sid: string, patch: Partial<StreamState>) => {
    // Update the ref synchronously (so guards/`isActive` read fresh state with no
    // render-timing gap), then schedule the React state update from it.
    const next = {
      ...streamsRef.current,
      [sid]: { ...(streamsRef.current[sid] || EMPTY_STREAM), ...patch },
    };
    streamsRef.current = next;
    setStreams(next);
  }, []);

  const clearStreamFor = useCallback((sid: string) => {
    const next = { ...streamsRef.current };
    delete next[sid];
    streamsRef.current = next;
    setStreams(next);
  }, []);

  const isActive = useCallback(
    (sid: string | null | undefined) => !!(sid && streamsRef.current[sid]?.active),
    []
  );

  const abort = useCallback((sid: string | null | undefined) => {
    if (sid) controllersRef.current[sid]?.abort();
  }, []);

  const startStream = useCallback(
    ({ url, body, sid, firstTurn }: StartStreamArgs) => {
      // Never run two generations for the same conversation at once.
      if (streamsRef.current[sid]?.active) return;

      const controller = new AbortController();
      controllersRef.current[sid] = controller;
      setStreamFor(sid, { active: true, thinking: true, answer: '', sources: [], error: null });

      let acc = '';
      let errored = false;
      let errMsg = '';

      void (async () => {
        try {
          await api.streamNdjson(
            url,
            body,
            (e: StreamEvent) => {
              if (e.type === 'token') {
                acc += e.data;
                setStreamFor(sid, { thinking: false, answer: acc });
              } else if (e.type === 'sources') {
                setStreamFor(sid, { sources: e.data });
              } else if (e.type === 'error') {
                errored = true;
                errMsg = e.data;
                setStreamFor(sid, { error: e.data });
              }
            },
            controller.signal
          );
        } catch (err: any) {
          if (err?.name !== 'AbortError') {
            errored = true;
            errMsg = err?.message || 'Erreur';
            setStreamFor(sid, { error: errMsg });
          }
        } finally {
          delete controllersRef.current[sid];
          const ok = !errored && acc.trim().length > 0;

          // Finalize the stream IMMEDIATELY: stop the loading bubble and let the page
          // re-sync from the server. The auto-title (a separate LLM call that can take
          // seconds on a local model) must NOT block this — doing so left the
          // streaming bubble stuck on screen and made the answer appear to repeat.
          seqRef.current += 1;
          setCompletion({ seq: seqRef.current, sid, ok, errored, errMsg, answer: acc, title: '' });
          clearStreamFor(sid);

          // Auto-title asynchronously, AFTER the stream is finalized. When it returns,
          // emit a second completion carrying only the title so the page applies it.
          if (firstTurn && ok) {
            try {
              const r = await api.autotitleSession(sid);
              const title = r?.title || '';
              if (title) {
                seqRef.current += 1;
                setCompletion({ seq: seqRef.current, sid, ok: true, errored: false, errMsg: '', answer: acc, title });
              }
            } catch (e) {
              console.warn('[chat] auto-title failed for session', sid, e);
            }
          }
        }
      })();
    },
    [setStreamFor, clearStreamFor]
  );

  const busyIds = useMemo(
    () => Object.entries(streams).filter(([, s]) => s.active).map(([id]) => id),
    [streams]
  );

  const value = useMemo<ChatStreamContextValue>(
    () => ({ streams, busyIds, isActive, startStream, abort, completion }),
    [streams, busyIds, isActive, startStream, abort, completion]
  );

  return <ChatStreamContext.Provider value={value}>{children}</ChatStreamContext.Provider>;
}

export function useChatStream(): ChatStreamContextValue {
  const ctx = useContext(ChatStreamContext);
  if (!ctx) throw new Error('useChatStream must be used within a ChatStreamProvider');
  return ctx;
}