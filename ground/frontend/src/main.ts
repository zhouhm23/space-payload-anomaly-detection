import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'

// Global styles
import './assets/main.css'

const app = createApp(App)
app.use(createPinia())
app.mount('#app')

// Console log to confirm the frontend has booted
console.log('[PHM] Vue3 monitor frontend started (v1.1-1a骨架)')
