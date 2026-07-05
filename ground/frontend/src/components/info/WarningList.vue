<script setup lang="ts">
import { computed } from 'vue'
import { useHealthStore } from '@/stores/health'
import type { WarningItem } from '@/api/types'

const health = useHealthStore()

const recent = computed(() => [...health.warnings].reverse().slice(0, 20))

function fmtTime(t: number): string {
  const d = new Date(t * 1000)
  return d.toTimeString().slice(0, 8)
}

function badgeClass(w: WarningItem): string {
  if (w.type === 'measured') return 'badge-measured'
  if (w.status === 'confirmed') return 'badge-confirmed'
  if (w.status === 'false') return 'badge-false'
  return 'badge-pending'
}

function badgeText(w: WarningItem): string {
  if (w.type === 'measured') return '实报'
  if (w.status === 'confirmed') return '真实'
  if (w.status === 'false') return '虚报'
  return '待核验'
}
</script>

<template>
  <div class="info-card">
    <h4>
      ⚠️ 预警栏 — 预测推导 <span class="badge badge-pending">预测</span>
    </h4>
    <ul v-if="recent.length > 0" class="warning-list">
      <li v-for="(w, i) in recent" :key="i">
        <div class="warning-head">
          <span>
            <strong>{{ w.channel }}</strong>
            · 预测分数 {{ w.max_predict_score.toFixed(3) }}
            <span class="badge" :class="badgeClass(w)">{{ badgeText(w) }}</span>
          </span>
          <span class="warning-time">{{ fmtTime(w.created_at) }}</span>
        </div>
        <div class="warning-msg">{{ w.message }}</div>
      </li>
    </ul>
    <div v-else class="empty-hint">暂无预警（等待 TTM-R3 预测）</div>
  </div>
</template>
