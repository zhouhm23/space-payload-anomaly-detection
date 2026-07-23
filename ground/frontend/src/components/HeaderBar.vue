<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useSystemStore } from '@/stores/system'
import { usePoll } from '@/composables/usePoll'
import { api } from '@/api'

const store = useSystemStore()

// Front-end theme (colours + display text)
const theme = ref<Record<string, any>>({ systemTitle: '空间有效载荷天地协同健康管理系统' })
api.theme().then((t) => (theme.value = t)).catch(() => {})

// Poll system info (3 s)
const { start, stop } = usePoll(() => store.refreshSystemInfo(), 3000, { immediate: true, autoStart: true })

// UTC clock local update (1 s, no backend call to avoid top-bar stutter)
const utcNow = ref<string>('')
let clockTimer: ReturnType<typeof setInterval> | null = null
function updateClock() {
  const d = new Date()
  utcNow.value = d.toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC')
}

// Computed properties
const systemTitle = computed(() => theme.value.systemTitle || '空间有效载荷天地协同健康管理系统')
const linkStatus = computed(() => store.systemInfo?.link_status || 'initializing')
const linkLatency = computed(() => store.systemInfo?.link_latency_ms)

// Link status text + colour class
const linkLabel = computed(() => {
  switch (linkStatus.value) {
    case 'online': return '天地链路 正常'
    case 'degraded': return '天地链路 降级'
    case 'offline': return '天地链路 中断'
    case 'waiting': return '天地链路 等待数据'
    case 'initializing': return '天地链路 初始化中'
    default: return '天地链路 未知'
  }
})

const linkDotClass = computed(() => {
  switch (linkStatus.value) {
    case 'online': return 'dot-online'
    case 'degraded': return 'dot-degraded'
    case 'offline': return 'dot-offline'
    default: return 'dot-waiting'
  }
})

// System overall health (fetched from `deviceTree`)
const systemHealth = computed(() => {
  const h = store.deviceTree?.health?.system
  if (typeof h !== 'number') return null
  return Math.round(h * 100)
})

const healthColor = computed(() => {
  if (systemHealth.value === null) return '#7a85a8'
  if (systemHealth.value >= 80) return '#67c23a'
  if (systemHealth.value >= 60) return '#e6a23c'
  return '#f56c6c'
})

onMounted(() => {
  updateClock()
  clockTimer = setInterval(updateClock, 1000)
})
onUnmounted(() => {
  if (clockTimer) clearInterval(clockTimer)
})
</script>

<template>
  <div class="header-bar">
    <!-- Left: system title -->
    <div class="header-left">
      <div class="system-logo">🛰️</div>
      <div class="system-title">{{ systemTitle }}</div>
    </div>

    <!-- Centre: core status metrics (health + link) -->
    <div class="header-center">
      <div class="metric-block">
        <div class="metric-label">系统健康度</div>
        <div class="metric-value" :style="{ color: healthColor }">
          {{ systemHealth !== null ? `${systemHealth}%` : '—' }}
        </div>
      </div>

      <div class="metric-divider"></div>

      <div class="metric-block">
        <div class="metric-label">天地链路</div>
        <div class="metric-value link-status">
          <span class="link-dot" :class="linkDotClass"></span>
          {{ linkLabel }}
        </div>
      </div>

      <div class="metric-divider"></div>

      <div class="metric-block">
        <div class="metric-label">天地延迟</div>
        <div class="metric-value">
          {{ linkLatency !== null && linkLatency !== undefined ? `${linkLatency.toFixed(0)} ms` : '—' }}
        </div>
      </div>
    </div>

    <!-- Right: UTC time -->
    <div class="header-right">
      <div class="utc-block">
        <div class="metric-label">UTC 时间</div>
        <div class="utc-time">{{ utcNow }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.header-bar {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  background: linear-gradient(90deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
  border-bottom: 1px solid #2a3050;
  flex-shrink: 0;
  position: relative;
  z-index: 10;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 280px;
}

.system-logo {
  font-size: 24px;
}

.system-title {
  font-size: 16px;
  font-weight: 500;
  color: #409eff;
  letter-spacing: 1px;
  white-space: nowrap;
}

.header-center {
  display: flex;
  align-items: center;
  gap: 20px;
}

.metric-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  min-width: 100px;
}

.metric-label {
  font-size: 11px;
  color: #7a85a8;
  letter-spacing: 1px;
}

.metric-value {
  font-size: 14px;
  font-weight: 500;
  color: #e0e6ed;
}

.link-status {
  display: flex;
  align-items: center;
  gap: 6px;
}

.link-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.dot-online {
  background: #67c23a;
  box-shadow: 0 0 8px #67c23a;
}

.dot-degraded {
  background: #e6a23c;
  box-shadow: 0 0 8px #e6a23c;
}

.dot-offline {
  background: #f56c6c;
  box-shadow: 0 0 8px #f56c6c;
}

.dot-waiting {
  background: #7a85a8;
}

.metric-divider {
  width: 1px;
  height: 30px;
  background: #2a3050;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 280px;
  justify-content: flex-end;
}

.utc-block {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}

.utc-time {
  font-size: 14px;
  font-family: 'Consolas', 'Monaco', monospace;
  color: #409eff;
  font-weight: 500;
  letter-spacing: 0.5px;
}
</style>
