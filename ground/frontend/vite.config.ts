import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// Vite configuration
// - dev server listens on :5173; all /api/* /admin/* /static/* are proxied to Django :8501
// - build output lands in ../django_phm/static/phm_site/dist/ (served by Django)
// - base defaults to '/' (dev); change base if deployed under a sub-path in production
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true, // fail hard if the port is taken (no silent port swap)
    open: false,      // do not auto-open a browser (the launch script opens it)
    proxy: {
      // Proxy every Django endpoint so the frontend can use relative paths
      '/api': {
        target: 'http://127.0.0.1:8501',
        changeOrigin: true,
      },
      '/admin': {
        target: 'http://127.0.0.1:8501',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://127.0.0.1:8501',
        changeOrigin: true,
      },
      '/media': {
        target: 'http://127.0.0.1:8501',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Output into the Django static dir so collectstatic / runserver can serve it directly
    outDir: resolve(__dirname, '../django_phm/static/phm_site/dist'),
    emptyOutDir: true,
    // Production build tuning
    chunkSizeWarningLimit: 1500, // ECharts is large; allow up to 1.5MB chunks
    rollupOptions: {
      output: {
        manualChunks: {
          // Split large deps to leverage browser caching
          'echarts': ['echarts', 'vue-echarts'],
          'element-plus': ['element-plus', '@element-plus/icons-vue'],
          'vendor': ['vue', 'vue-router', 'pinia', 'axios', 'dayjs'],
        },
      },
    },
  },
  // TypeScript type-checking
  // Note: the `build` script runs vue-tsc + vite build (strict type check);
  // `build:no-check` skips it for fast dev-time builds.
})
