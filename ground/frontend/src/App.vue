<script setup lang="ts">
/**
 * PHM monitor dashboard root component
 *
 * Layout:
 * - Top 60px: `HeaderBar` (system name / health / link / latency / UTC)
 * - Body flex:
 *   - Left 240px: `DeviceTree`
 *   - Center flex:1: `CenterCharts` (4:1:2 three-zone layout)
 *   - Right 340px: `RightDetail` (upper and lower sections)
 *
 * Startup state machine:
 * - An overlay is displayed until the Container is ready, after which the dashboard is revealed
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useSystemStore } from '@/stores/system'
import { api } from '@/api'
import HeaderBar from '@/components/HeaderBar.vue'
import DeviceTree from '@/components/DeviceTree.vue'
import CenterCharts from '@/components/CenterCharts.vue'
import RightDetail from '@/components/RightDetail.vue'
import ResizableSplitter from '@/components/ResizableSplitter.vue'

const store = useSystemStore()
const theme = ref<Record<string, any>>({})

// Left and right panel widths (default 240/340, matching historical fixed values; persisted to localStorage via `ResizableSplitter` after user drag)
const DEFAULT_LEFT = 240
const DEFAULT_RIGHT = 340
const leftWidth = ref(DEFAULT_LEFT)
const rightWidth = ref(DEFAULT_RIGHT)

let startupTimer: ReturnType<typeof setInterval> | null = null
let systemInfoTimer: ReturnType<typeof setInterval> | null = null
let deviceTreeTimer: ReturnType<typeof setInterval> | null = null
let rulTimer: ReturnType<typeof setInterval> | null = null

// Startup status polling (1 s interval until ready or failed)
async function pollStartup() {
  await store.checkStartup()
  if (store.startupState === 'ready' || store.startupState === 'failed') {
    if (startupTimer) {
      clearInterval(startupTimer)
      startupTimer = null
    }
    // When ready, immediately fetch business data once, start periodic polling, and start channel carousel
    if (store.startupState === 'ready') {
      store.refreshSystemInfo()
      store.refreshDeviceTree()
      store.refreshRul()
      systemInfoTimer = setInterval(() => store.refreshSystemInfo(), 3000)
      deviceTreeTimer = setInterval(() => store.refreshDeviceTree(), 5000)
      rulTimer = setInterval(() => store.refreshRul(), 5000)
      // Channel carousel: default 15 s, interval read from `theme.carousel.intervalMs`
      const interval = theme.value?.carousel?.intervalMs || 15000
      store.startCarousel(interval)
    }
  }
}

onMounted(() => {
  // Fetch theme (includes carousel interval and other config)
  api.theme().then((t) => (theme.value = t)).catch(() => {})
  pollStartup()
  startupTimer = setInterval(pollStartup, 1000)
})

onUnmounted(() => {
  if (startupTimer) clearInterval(startupTimer)
  if (systemInfoTimer) clearInterval(systemInfoTimer)
  if (deviceTreeTimer) clearInterval(deviceTreeTimer)
  if (rulTimer) clearInterval(rulTimer)
  store.stopCarousel()
})

// Whether to show the startup overlay
const showStartupOverlay = computed(() => store.startupState !== 'ready')
</script>

<template>
  <div class="app-shell">
    <HeaderBar />

    <main class="app-main">
      <aside class="panel-left" :style="{ width: leftWidth + 'px' }">
        <DeviceTree />
      </aside>

      <ResizableSplitter
        storage-key="phm.layout.left"
        :default-size="DEFAULT_LEFT"
        :min="160"
        :max="480"
        @resize="(v) => (leftWidth = v)"
      />

      <section class="panel-center">
        <CenterCharts />
      </section>

      <ResizableSplitter
        storage-key="phm.layout.right"
        :default-size="DEFAULT_RIGHT"
        :min="220"
        :max="560"
        @resize="(v) => (rightWidth = v)"
      />

      <aside class="panel-right" :style="{ width: rightWidth + 'px' }">
        <RightDetail />
      </aside>
    </main>

    <!-- Startup status overlay -->
    <div v-if="showStartupOverlay" class="startup-overlay">
      <div class="startup-card" :class="store.startupState">
        <div class="startup-icon">
          <span v-if="store.startupState === 'initializing' || store.startupState === 'idle'">⏳</span>
          <span v-else-if="store.startupState === 'failed'">❌</span>
          <span v-else-if="store.startupState === 'disconnected'">🔌</span>
        </div>
        <div class="startup-title">
          <template v-if="store.startupState === 'initializing' || store.startupState === 'idle'">
            正在初始化 PHM 系统...
          </template>
          <template v-else-if="store.startupState === 'failed'">
            PHM 初始化失败
          </template>
          <template v-else-if="store.startupState === 'disconnected'">
            无法连接后端
          </template>
        </div>
        <div class="startup-detail">
          <template v-if="store.startupState === 'disconnected'">
            请确认 Django 已启动（端口 8501）
          </template>
          <template v-else-if="store.startupState === 'failed'">
            {{ store.startupError || '未知错误' }}
          </template>
          <template v-else>
            加载模型中（TSPulse + TTM-R3 + RUL），预计 10 秒
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.app-main {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.panel-left {
  flex-shrink: 0;
  border-right: 1px solid #2a3050;
  overflow: hidden;
}

.panel-center {
  flex: 1;
  min-width: 0;
}

.panel-right {
  flex-shrink: 0;
  border-left: 1px solid #2a3050;
  overflow: hidden;
}

/* Startup overlay */
.startup-overlay {
  position: fixed;
  inset: 0;
  background: rgba(10, 14, 39, 0.92);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.startup-card {
  max-width: 480px;
  padding: 40px;
  background: #1a1f3a;
  border: 1px solid #2a3050;
  border-radius: 8px;
  text-align: center;
  border-left: 4px solid #e6a23c;
}

.startup-card.failed,
.startup-card.disconnected {
  border-left-color: #f56c6c;
}

.startup-icon {
  font-size: 48px;
  margin-bottom: 16px;
  animation: pulse 1.5s infinite;
}

.startup-card.failed .startup-icon,
.startup-card.disconnected .startup-icon {
  animation: none;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.startup-title {
  font-size: 18px;
  color: #e0e6ed;
  margin-bottom: 8px;
  font-weight: 500;
}

.startup-detail {
  font-size: 13px;
  color: #7a85a8;
  line-height: 1.6;
}

.startup-card.failed .startup-detail {
  color: #f56c6c;
  font-family: 'Consolas', monospace;
  font-size: 12px;
  word-break: break-all;
}
</style>
