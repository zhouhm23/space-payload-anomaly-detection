import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// Vite 配置
// - dev server 监听 :5173，所有 /api/* /admin/* /static/* 代理到 Django :8501
// - build 产物落到 ../django_phm/static/phm_site/dist/（Django 一并 serve）
// - base 默认 '/'（开发）；生产部署若挂子路径需改 base
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
    strictPort: true, // 端口被占直接报错（避免静默换端口）
    open: false,      // 不自动开浏览器（启动脚本自己开）
    proxy: {
      // 所有 Django 端点都代理过去，前端代码用相对路径即可
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
    // 产物输出到 Django 静态目录，由 collectstatic / runserver 直接 serve
    outDir: resolve(__dirname, '../django_phm/static/phm_site/dist'),
    emptyOutDir: true,
    // 生产构建优化
    chunkSizeWarningLimit: 1500, // ECharts 整包大，允许 1.5MB chunk
    rollupOptions: {
      output: {
        manualChunks: {
          // 拆分大依赖，利用浏览器缓存
          'echarts': ['echarts', 'vue-echarts'],
          'element-plus': ['element-plus', '@element-plus/icons-vue'],
          'vendor': ['vue', 'vue-router', 'pinia', 'axios', 'dayjs'],
        },
      },
    },
  },
  // TypeScript 类型检查配置
  // 注意：build 脚本走 vue-tsc + vite build（类型检查严格）
  // build:no-check 跳过类型检查（开发期快速构建）
})
