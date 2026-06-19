export interface SessionSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface Source {
  text: string;
  score: number;
  filename: string;
  page: string | number;
  document_id?: string | null;
  version?: number | null;
}

export interface Message {
  id: string;
  parent_id: string | null;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  version_index: number;
  version_count: number;
  sibling_ids: string[];
}

export interface SessionDetail extends SessionSummary {
  messages: Message[];
}

export type IndexingStatus = 'pending' | 'processing' | 'indexed' | 'failed';

export interface DocumentVersion {
  id: string;
  document_id: string;
  version: number;
  filename: string;
  indexing_status: IndexingStatus;
  is_latest: number; // 0 | 1
  chunks: number;
  error: string | null;
  uploaded_at: number;
}

export interface DocumentItem {
  id: string;
  name: string;
  current_version: number;
  status: 'active' | 'deleted';
  created_at: number;
  updated_at: number;
  version_count: number;
  latest: DocumentVersion | null;
}

export type Scope = { type: 'all' } | { type: 'document'; id: string; name: string };

export interface Status {
  status: string;
  doc_count: number;
  collection_name: string;
  llm_model: string;
  embed_model: string;
  reranker: string | null;
  total_documents: number;
}

// One event from the NDJSON chat stream.
export type StreamEvent =
  | { type: 'token'; data: string }
  | { type: 'sources'; data: Source[] }
  | { type: 'done'; session_id: string; title: string }
  | { type: 'error'; data: string };