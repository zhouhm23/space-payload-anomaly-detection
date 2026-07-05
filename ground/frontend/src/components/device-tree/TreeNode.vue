<script setup lang="ts">
import type { DeviceNode } from '@/api/types'
import { useHealthStore } from '@/stores/health'
import { sourceToChannel } from '@/stores/deviceTree'

const props = defineProps<{
  node: DeviceNode
  depth: number
  active: boolean
}>()

const emit = defineEmits<{
  select: []
  delete: [event: MouseEvent]
  dblclick: []
  dragstart: []
  drop: []
}>()

const health = useHealthStore()

// status dot class derived from latest score for this channel
function dotClass(): string {
  if (!props.node.sourceId) return ''
  const ch = props.node.channelName ?? sourceToChannel(props.node.sourceId)
  const sensor = health.sensors.find((s) => s.channel === ch)
  if (!sensor) return ''
  const sc = sensor.latest_score
  if (sc > 0.7) return 'error'
  if (sc > 0.4) return 'warn'
  return ''
}
</script>

<template>
  <li>
    <div
      class="tree-item"
      :class="{ active }"
      :style="{ paddingLeft: 8 + depth * 20 + 'px' }"
      draggable="true"
      @click="emit('select')"
      @dblclick="emit('dblclick')"
      @dragstart="emit('dragstart')"
      @dragover.prevent
      @drop.prevent="emit('drop')"
    >
      <span class="status-dot" :class="dotClass()"></span>
      <span>{{ (node.children || node.type === 'folder') ? '📁' : '📡' }} {{ node.name }}</span>
      <span class="del-btn" title="删除" @click="emit('delete', $event)">✕</span>
    </div>
  </li>
</template>
