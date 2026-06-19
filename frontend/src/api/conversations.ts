// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import client from './client';

export interface ConversationMessage {
  message_id: string;
  session_id: string;
  sequence: number;
  role: 'system' | 'user' | 'assistant' | 'tool' | string;
  content: string;
  created_at: number;
  metadata: Record<string, unknown>;
}

export interface ConversationSession {
  schema_version: string;
  session_id: string;
  surface: string;
  title?: string | null;
  model_id?: string | null;
  peer_id?: string | null;
  source?: string | null;
  status: string;
  created_at: number;
  updated_at: number;
  metadata: Record<string, unknown>;
  messages?: ConversationMessage[];
}

export interface ConversationListResponse {
  ok: boolean;
  schema_version: string;
  count: number;
  items: ConversationSession[];
}

export const listConversations = (params?: {
  surface?: string;
  peer_id?: string;
  limit?: number;
  include_messages?: boolean;
}) =>
  client
    .get<ConversationListResponse>('/conversations', { params })
    .then((r) => r.data);

export const createConversation = (body: {
  session_id?: string;
  surface: string;
  title?: string;
  model_id?: string;
  peer_id?: string;
  source?: string;
  status?: string;
  metadata?: Record<string, unknown>;
}) =>
  client
    .post<{ ok: boolean; session: ConversationSession }>('/conversations', body)
    .then((r) => r.data.session);

export const getConversation = (sessionId: string) =>
  client
    .get<{ ok: boolean; session: ConversationSession }>(`/conversations/${encodeURIComponent(sessionId)}`)
    .then((r) => r.data.session);

export const updateConversation = (
  sessionId: string,
  body: {
    title?: string;
    model_id?: string;
    peer_id?: string;
    source?: string;
    status?: string;
    metadata?: Record<string, unknown>;
    merge_metadata?: boolean;
  },
) =>
  client
    .patch<{ ok: boolean; session: ConversationSession }>(
      `/conversations/${encodeURIComponent(sessionId)}`,
      body,
    )
    .then((r) => r.data.session);

export const replaceConversationMessages = (
  sessionId: string,
  messages: Array<{
    id?: string;
    message_id?: string;
    sequence?: number;
    role: string;
    content: string;
    created_at?: number;
    metadata?: Record<string, unknown>;
  }>,
) =>
  client
    .put<{ ok: boolean; count: number; messages: ConversationMessage[] }>(
      `/conversations/${encodeURIComponent(sessionId)}/messages`,
      { messages },
    )
    .then((r) => r.data);

export const deleteConversation = (sessionId: string) =>
  client
    .delete<{ ok: boolean; session_id: string }>(`/conversations/${encodeURIComponent(sessionId)}`)
    .then((r) => r.data);
