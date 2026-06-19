import type {
  AuthResponse,
  DocumentItem,
  DocumentVersion,
  SessionDetail,
  SessionSummary,
  Status,
  StreamEvent,
  User,
} from './types';

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') || 'http://localhost:8000';

// ─── Auth token storage + 401 handling ──────────────────────────────────────
const TOKEN_KEY = 'rag_token';

export const getToken = (): string | null =>
  typeof window !== 'undefined' ? window.localStorage.getItem(TOKEN_KEY) : null;
export const setToken = (t: string) => {
  if (typeof window !== 'undefined') window.localStorage.setItem(TOKEN_KEY, t);
};
export const clearToken = () => {
  if (typeof window !== 'undefined') window.localStorage.removeItem(TOKEN_KEY);
};

let unauthorizedHandler: (() => void) | null = null;
export const setUnauthorizedHandler = (fn: (() => void) | null) => {
  unauthorizedHandler = fn;
};

// Called when a protected request returns 401: drop the token and bounce to login.
function handleUnauthorized() {
  clearToken();
  if (unauthorizedHandler) {
    unauthorizedHandler();
  } else if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
    window.location.href = '/login';
  }
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const t = getToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

async function asJson<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    handleUnauthorized();
    throw new Error('Session expirée — veuillez vous reconnecter.');
  }
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

// Parse a public auth response WITHOUT triggering the 401 redirect (so a wrong
// password shows an inline error on the login screen instead of bouncing).
async function asAuth(res: Response): Promise<AuthResponse> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json())?.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<AuthResponse>;
}

// ─── Auth ────────────────────────────────────────────────────────────────────
export const login = (username: string, password: string) =>
  fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  }).then(asAuth);

export interface RegisterInput {
  first_name: string;
  last_name: string;
  username: string;
  password: string;
}

// Registration creates the account but does NOT log in — the UI sends the user
// to the sign-in screen afterwards.
export const register = async (data: RegisterInput): Promise<{ ok: boolean; username: string }> => {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json())?.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
};

export const me = () =>
  fetch(`${API_BASE}/auth/me`, { headers: authHeaders() }).then((r) => asJson<User>(r));

// ─── Status ────────────────────────────────────────────────────────────────
export const getStatus = () =>
  fetch(`${API_BASE}/`).then((r) => asJson<Status>(r));

// ─── Sessions ──────────────────────────────────────────────────────────────
export const listSessions = () =>
  fetch(`${API_BASE}/sessions`, { headers: authHeaders() }).then((r) =>
    asJson<{ sessions: SessionSummary[] }>(r)
  );

export const createSession = (title?: string) =>
  fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title: title ?? null }),
  }).then((r) => asJson<SessionSummary>(r));

export const getSession = (id: string) =>
  fetch(`${API_BASE}/sessions/${id}`, { headers: authHeaders() }).then((r) =>
    asJson<SessionDetail>(r)
  );

export const renameSession = (id: string, title: string) =>
  fetch(`${API_BASE}/sessions/${id}`, {
    method: 'PATCH',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title }),
  }).then((r) => asJson(r));

export const deleteSession = (id: string) =>
  fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE', headers: authHeaders() }).then((r) =>
    asJson(r)
  );

export const autotitleSession = (id: string) =>
  fetch(`${API_BASE}/sessions/${id}/autotitle`, { method: 'POST', headers: authHeaders() }).then(
    (r) => asJson<{ title: string }>(r)
  );

export const selectVersion = (sessionId: string, messageId: string) =>
  fetch(`${API_BASE}/sessions/${sessionId}/messages/${messageId}/select`, {
    method: 'POST',
    headers: authHeaders(),
  }).then((r) => asJson(r));

// ─── Documents (versioned) ───────────────────────────────────────────────────
export const listDocuments = () =>
  fetch(`${API_BASE}/documents`, { headers: authHeaders() }).then((r) =>
    asJson<{ documents: DocumentItem[] }>(r)
  );

export const listTrash = () =>
  fetch(`${API_BASE}/documents/trash`, { headers: authHeaders() }).then((r) =>
    asJson<{ documents: DocumentItem[] }>(r)
  );

export const undeleteDocument = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}/undelete`, {
    method: 'POST',
    headers: authHeaders(),
  }).then((r) => asJson(r));

export const listVersions = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}/versions`, { headers: authHeaders() }).then((r) =>
    asJson<{ versions: DocumentVersion[] }>(r)
  );

export const restoreVersion = (documentId: string, version: number) =>
  fetch(`${API_BASE}/documents/${documentId}/versions/${version}/restore`, {
    method: 'POST',
    headers: authHeaders(),
  }).then((r) => asJson(r));

export const softDeleteDocument = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}`, { method: 'DELETE', headers: authHeaders() }).then(
    (r) => asJson(r)
  );

export const purgeDocument = (documentId: string) =>
  fetch(`${API_BASE}/documents/${documentId}/purge`, {
    method: 'DELETE',
    headers: authHeaders(),
  }).then((r) => asJson(r));

// Upload a brand-new document. `scope` is honoured for admins (org | personal);
// regular users always create personal documents server-side.
export const uploadDocument = (file: File, scope?: 'org' | 'personal') => {
  const form = new FormData();
  form.append('file', file);
  if (scope) form.append('scope', scope);
  return fetch(`${API_BASE}/documents`, {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  }).then((r) =>
    asJson<{
      document_id: string;
      version: number;
      scope: string;
      indexing_status: string;
      filename: string;
    }>(r)
  );
};

// Upload a new version of an existing document.
export const uploadNewVersion = (documentId: string, file: File) => {
  const form = new FormData();
  form.append('file', file);
  return fetch(`${API_BASE}/documents/${documentId}/versions`, {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  }).then((r) => asJson<{ version: number; indexing_status: string }>(r));
};

// Fetch a protected file as a blob and trigger a browser download. Used for
// both the latest version and a specific historical version.
async function downloadBlob(url: string, filename: string): Promise<void> {
  const res = await fetch(url, { headers: authHeaders() });
  if (res.status === 401) {
    handleUnauthorized();
    throw new Error('Session expirée — veuillez vous reconnecter.');
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json())?.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const blob = await res.blob();
  const objectUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(objectUrl);
}

export const downloadDocument = (documentId: string, filename: string) =>
  downloadBlob(`${API_BASE}/documents/${documentId}/download`, filename);

export const downloadVersion = (documentId: string, version: number, filename: string) =>
  downloadBlob(`${API_BASE}/documents/${documentId}/versions/${version}/download`, filename);

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
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body ?? {}),
    signal,
  });
  if (res.status === 401) {
    handleUnauthorized();
    throw new Error('Session expirée — veuillez vous reconnecter.');
  }
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