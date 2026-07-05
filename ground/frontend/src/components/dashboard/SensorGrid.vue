<script setup lang="ts">
import { computed } from 'vue'
import { useHealthStore } from '@/stores/health'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import SensorCard from './SensorCard.vue'

const health = useHealthStore()
const tree = useDeviceTreeStore()

// Build a card for each sensor node that has a channel mapping.
const cards = computed(() => {
  const out: { channel: string; name: string; raw: number | null; score: number; healthVal: number }[] = []
  for (const d of tree.tree) {
    if (d.sourceId) {
      const ch = d.channelName ?? ''
      const s = health.sensors.find((x) => x.channel === ch)
      out.push({
        channel: ch,
        name: d.name,
        raw: s?.latest_raw ?? null,
        score: s?.latest_score ?? 0,
        healthVal: s?.health ?? 100,
      })
    }
    if (d.children) {
      for (const c of d.children) {
        if (c.sourceId) {
          const ch = c.channelName ?? ''
          const s = health.sensors.find((x) => x.channel === ch)
          out.push({
            channel: ch,
            name: c.name,
            raw: s?.latest_raw ?? null,
            score: s?.latest_score ?? 0,
            healthVal: s?.health ?? 100,
          })
        }
      }
    }
  }
  return out
})

function selectChannel(channel: string) {
  // find node by channelName and select it
  for (const d of tree.tree) {
    if (d.channelName === channel) {
      tree.selectedId = d.id
      return
    }
    if (d.children) {
      const c = d.children.find((x) => x.channelName === channel)
      if (c) {
        tree.selectedId = c.id
        return
      }
    }
  }
}
</script>

<template>
  <div class="gauges-grid">
    <SensorCard
      v-for="card in cards"
      :key="card.channel"
      :name="card.name"
      :channel="card.channel"
      :raw="card.raw"
      :score="card.score"
      :health-val="card.healthVal"
      :active="tree.selectedChannelName() === card.channel"
      @click="selectChannel(card.channel)"
    />
    <div v-if="cards.length === 0" class="placeholder-panel">
      <span>🚧 仪表盘 — 等待传感器数据</span>
    </div>
  </div>
</template>
