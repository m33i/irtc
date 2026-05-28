import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    CESIUM_BASE_URL: JSON.stringify('/cesium'),
  },
  optimizeDeps: {
    // Cesium is too large for Vite to pre-bundle — serve it as-is
    exclude: ['cesium', 'resium'],
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  build: {
    // Silence the chunk-size warning; Cesium is inherently large
    chunkSizeWarningLimit: 6000,
  },
})
