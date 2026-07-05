<script setup lang="ts">
import { computed } from 'vue'
import { useLinkStore } from '@/stores/link'
import { useHealthStore } from '@/stores/health'

const link = useLinkStore()
const health = useHealthStore()

const ringDeg = computed(() => Math.round((health.systemHealth / 100) * 360))
const ringColor = computed(() => {
  const h = health.systemHealth
  if (h < 60) return 'var(--accent-red)'
  if (h < 80) return 'var(--accent-yellow)'
  return 'var(--accent-green)'
})
const ringStyle = computed(
  () => `conic-gradient(${ringColor.value} 0deg ${ringDeg.value}deg, #2a2f3a ${ringDeg.value}deg 360deg)`,
)
</script>

<template>
  <header class="header">
    <div class="title">🚀 空间站有效载荷预测性维护支持系统</div>
    <div class="status-group">
      <div class="link-status">
        <span class="link-dot" :class="{ loss: link.linkState === 'loss' }"></span>
        <span>{{
          link.linkState === 'normal'
            ? '天地链路 正常'
            : link.linkState === 'loss'
              ? '天地链路 中断'
              : '天地链路 待连接'
        }}</span>
      </div>
      <div class="utc-time">{{ link.utcTime }}</div>
      <div class="health-score">
        <span style="font-size: 0.8rem">系统健康</span>
        <div class="health-ring" :style="{ background: ringStyle }">
          {{ Math.round(health.systemHealth) }}
        </div>
      </div>
    </div>
  </header>
</template>
