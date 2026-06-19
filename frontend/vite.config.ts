// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// Edge Studio backend listens on 18842 (avoids common 8000), overridable via VLM_PORT env var
const backendPort = process.env.VLM_PORT || '18842'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      'assert': 'assert',
      'buffer': 'buffer',
      'buffer/': 'buffer',
      'stream': 'stream-browserify',
    },
  },
  define: {
    global: 'globalThis',
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js', 'react-plotly.js'],
        },
      },
    },
  },
  server: {
    port: 5173,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${backendPort}`,
        // Upgrade matters for /api/mesh/events/stream (WebSocket)
        ws: true,
        // Bypass system proxy for local backend connection
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            // Force direct connection, skip any proxy agent
            ;(proxyReq as { agent?: unknown }).agent = undefined
          })
        },
      },
      '/ws': {
        target: `ws://127.0.0.1:${backendPort}`,
        ws: true,
      },
    },
  },
})
