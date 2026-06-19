import type {
  DocumentItem,
  DocumentVersion,
  SessionDetail,
  SessionSummary,
  Status,
  StreamEvent,
} from './types';

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') || 'http://localhost:8000';

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ─── Status ────────────────────────────────────────────────────────────────
export const getStatus = () => fetch(`${API_BASE}/`).then((r) => asJson<Status>(r));

// ─── Sessions ──────────────────────────────────────────────────────────────
export const listSessions = () =>
  fetch(`${API_BASE}/sessions`).then((r) => asJson<{ sessions: SessionSummary[] }>(r));

export const createSession = (title?: string) =>
  fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: title ?? null }),
  }).then((r) => asJson<SessionSummary>(r));

export const getSession = (id: string) =>
  fetch(`${API_BASE}/sessions/${id}`).then((r) => asJson<SessionDetail>(r));

export const renameSession = (id: string, title: string) =>
  fetch(`${API_BASE}/sessions/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  }).then((r) => asJson(r));

export const deleteSession = (id: string) =>
  fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE' }).then((r) => asJson(r));

export const autotitleSession = (id: string) =>
  fetch(`${API_BASE}/sessions/${id}/autotitle`, { method: 'POST' }).then((r) =>
    asJson<{ title: string }>(r)
  );

export const selectVersion = (sessionId: string, messageId: string) =>
  fetch(`${API_BASE}/sessions/${sessionId}/messages/${messageId}/select`, {
    method: 'POST',
  }).then((r) => asJson(r));

// ─── Documents (versioned) ───────────────────────────────────────────────────
export const listDocuments = () =>
  fetch(`${API_BASE}/documents`).then((r) => asJson<{ documents: DocumentItem[] }>(r));

export const listTrash = () =>
  fetch(`${API_BASE}/documents/trash`).then((r) => asJson<{ documents: DocumentItem[] }>(r));

export const undeleteDocument = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}/undelete`, { method: 'POST' }).then((r) => asJson(r));

export const listVersions = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}/versions`).then((r) =>
    asJson<{ versions: DocumentVersion[] }>(r)
  );

export const restoreVersion = (documentId: string, version: number) =>
  fetch(`${API_BASE}/documents/${documentId}/versions/${version}/restore`, {
    method: 'POST',
  }).then((r) => asJson(r));

export const softDeleteDocument = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}`, { method: 'DELETE' }).then((r) => asJson(r));

export const purgeDocument = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}/purge`, { method: 'DELETE' }).then((r) => asJson(r));

// Upload a brand-new document (same filename → a new version of it).
export const uploadDocument = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return fetch(`${API_BASE}/documents`, { method: 'POST', body: form }).then((r) =>
    asJson<{ document_id: string; version: number; indexing_status: string; filename: string }>(r)
  );
};

// Upload a new version of an existing document.
export const uploadNewVersion = (documentId: string, file: File) => {
  const form = new FormData();
  form.append('file', file);
  return fetch(`${API_BASE}/documents/${documentId}/versions`, {
    method: 'POST',
    body: form,
  }).then((r) => asJson<{ version: number; indexing_status: string }>(r));
};

// ─── Streaming (NDJSON) ──────────────────────────────────────────────────────
// Consumes the chat/edit/regenerate stream and invokes `onEvent` per line.
export async function streamNdjson(
  url: string,
  body: Record<string, unknown> | undefined,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json())?.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        onEvent(JSON.parse(line) as StreamEvent);
      } catch {
        /* skip malformed line */
      }
    }
  }
}

export const chatUrl = (sessionId: string) => `${API_BASE}/sessions/${sessionId}/chat`;
export const editUrl = (sessionId: string, messageId: string) =>
  `${API_BASE}/sessions/${sessionId}/messages/${messageId}/edit`;
export const regenerateUrl = (sessionId: string, messageId: string) =>
  `${API_BASE}/sessions/${sessionId}/messages/${messageId}/regenerate`;