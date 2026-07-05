<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  name: string
  channel: string
  raw: number | null
  score: number
  healthVal: number
  active: boolean
}>()

defineEmits<{ click: [] }>()

const healthClass = computed(() => {
  if (props.healthVal < 60) return 'danger'
  if (props.healthVal < 80) return 'warn'
  return ''
})

const scoreColor = computed(() => (props.score > 0.7 ? 'var(--accent-red)' : 'var(--accent-yellow)'))
</script>

<template>
  <div class="gauge-card" :class="{ active }" @click="$emit('click')">
    <div class="gauge-name">📡 {{ name }} <small style="color: var(--text-secondary)">[{{ channel }}]</small></div>
    <div class="gauge-value">{{ raw !== null ? raw.toFixed(4) : '—' }}</div>
    <div class="gauge-score">
      异常分数 <span :style="{ color: scoreColor }">{{ score.toFixed(3) }}</span>
    </div>
    <div class="gauge-health" :class="healthClass">健康值 {{ healthVal.toFixed(1) }}%</div>
  </div>
</template>
