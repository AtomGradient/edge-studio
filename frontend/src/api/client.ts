// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import axios from 'axios';
import { useToastStore } from '@/stores/toastStore';

const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const { addToast } = useToastStore.getState();

    if (!error.response) {
      addToast('Cannot connect to backend. Is the server running?', 'error');
      return Promise.reject(error);
    }

    const status = error.response.status;

    if (status >= 500) {
      const detail = error.response.data?.detail;
      addToast(detail || 'An unexpected server error occurred.', 'error');
    }

    // 4xx errors propagate to page handlers (contextual)
    return Promise.reject(error);
  },
);

export default client;
