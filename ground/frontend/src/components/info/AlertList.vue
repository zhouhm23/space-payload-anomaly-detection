<script setup lang="ts">
import { computed } from 'vue'
import { useHealthStore } from '@/stores/health'

const health = useHealthStore()

const recent = computed(() => [...health.alerts].reverse().slice(0, 20))

function fmtTime(t: number): string {
  const d = new Date(t * 1000)
  return d.toTimeString().slice(0, 8)
}
</script>

<template>
  <div class="info-card">
    <h4>🚨 告警栏 — 实测异常 <span class="badge badge-measured">实报</span></h4>
    <ul v-if="recent.length > 0" class="alert-list">
      <li v-for="(a, i) in recent" :key="i">
        <div class="alert-head">
          <span><strong>{{ a.channel }}</strong> · 分数 {{ a.score.toFixed(3) }}</span>
          <span class="alert-time">{{ fmtTime(a.time) }}</span>
        </div>
        <div class="alert-msg">{{ a.message }}</div>
      </li>
    </ul>
    <div v-else class="empty-hint">暂无告警</div>
  </div>
</template>
