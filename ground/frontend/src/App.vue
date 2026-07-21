<script setup lang="ts">
/**
 * PHM 监控大屏根组件
 *
 * 布局：
 * - 顶部 60px：HeaderBar（系统名/健康度/链路/延迟/UTC）
 * - 主体 flex：
 *   - 左 240px：DeviceTree
 *   - 中 flex:1：CenterCharts（4:1:2 三区）
 *   - 右 340px：RightDetail（上下两部分）
 *
 * 启动状态机：
 * - 显示一个遮罩，等 Container 就绪后再展示大屏
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useSystemStore } from '@/stores/system'
import { api } from '@/api'
import HeaderBar from '@/components/HeaderBar.vue'
import DeviceTree from '@/components/DeviceTree.vue'
import CenterCharts from '@/components/CenterCharts.vue'
import RightDetail from '@/components/RightDetail.vue'

const store = useSystemStore()
const theme = ref<Record<string, any>>({})

let startupTimer: ReturnType<typeof setInterval> | null = null
let systemInfoTimer: ReturnType<typeof setInterval> | null = null
let deviceTreeTimer: ReturnType<typeof setInterval> | null = null
let rulTimer: ReturnType<typeof setInterval> | null = null

// 启动状态轮询（1s 一次，直到 ready/failed）
async function pollStartup() {
  await store.checkStartup()
  if (store.startupState === 'ready' || store.startupState === 'failed') {
    if (startupTimer) {
      clearInterval(startupTimer)
      startupTimer = null
    }
    // ready 后立即拉一次业务数据 + 启动周期轮询 + 启动通道轮播
    if (store.startupState === 'ready') {
      store.refreshSystemInfo()
      store.refreshDeviceTree()
      store.refreshRul()
      systemInfoTimer = setInterval(() => store.refreshSystemInfo(), 3000)
      deviceTreeTimer = setInterval(() => store.refreshDeviceTree(), 5000)
      rulTimer = setInterval(() => store.refreshRul(), 5000)
      // 通道轮播：默认 15 秒，间隔从 theme.carousel.intervalMs 读
      const interval = theme.value?.carousel?.intervalMs || 15000
      store.startCarousel(interval)
    }
  }
}

onMounted(() => {
  // 拉主题（含轮播间隔等配置）
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

// 是否显示启动遮罩
const showStartupOverlay = computed(() => store.startupState !== 'ready')
</script>

<template>
  <div class="app-shell">
    <HeaderBar />

    <main class="app-main">
      <aside class="panel-left">
        <DeviceTree />
      </aside>

      <section class="panel-center">
        <CenterCharts />
      </section>

      <aside class="panel-right">
        <RightDetail />
      </aside>
    </main>

    <!-- 启动状态遮罩 -->
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
  width: 240px;
  flex-shrink: 0;
  border-right: 1px solid #2a3050;
}

.panel-center {
  flex: 1;
  min-width: 0;
}

.panel-right {
  width: 340px;
  flex-shrink: 0;
  border-left: 1px solid #2a3050;
}

/* 启动遮罩 */
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
