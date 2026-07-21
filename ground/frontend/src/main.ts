import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'

// 全局样式
import './assets/main.css'

const app = createApp(App)
app.use(createPinia())
app.mount('#app')

// 控制台输出，方便确认前端启动
console.log('[PHM] Vue3 monitor frontend started (v1.1-1a骨架)')
