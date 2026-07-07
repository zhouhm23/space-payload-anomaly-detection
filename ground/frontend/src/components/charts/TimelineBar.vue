<script setup lang="ts">
/**
 * TimelineBar — a draggable overview scrollbar (ported from the
 * ``debug/虚拟示波器.html`` timeline-track).
 *
 * Shows the full data range (oldest → newest) with a movable "window"
 * highlight indicating the current viewport.  In realtime mode the
 * window auto-sticks to the right edge; in frozen mode the user can
 * drag it to scroll through history.
 *
 * Emits ``pan`` with the new right-edge timestamp (ms) on drag.
 */

import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  /** Oldest timestamp in buffer (ms) */
  bufferStart: number
  /** Newest timestamp in buffer (ms) */
  bufferEnd: number
  /** Current view right-edge (ms) */
  viewEnd: number
  /** Current view span (ms), = viewEnd - viewStart */
  viewSpan: number
  /** Realtime mode auto-sticks window to right */
  realtime: boolean
}>()

const emit = defineEmits<{
  (e: 'pan', newEndMs: number): void
}>()

const trackRef = ref<HTMLDivElement | null>(null)
let dragging = false
let dragStartClientX = 0
let dragStartViewEnd = 0

const totalSpan = computed(() => Math.max(1, props.bufferEnd - props.bufferStart))

const windowLeftPct = computed(() => {
  if (props.realtime) return 100 - windowWidthPct.value
  const viewStart = props.viewEnd - props.viewSpan
  const leftFrac = Math.max(0, Math.min(1, (viewStart - props.bufferStart) / totalSpan.value))
  return leftFrac * 100
})

const windowWidthPct = computed(() => {
  return Math.max(2, Math.min(100, (props.viewSpan / totalSpan.value) * 100))
})

const windowStyle = computed(() => ({
  left: windowLeftPct.value + '%',
  width: windowWidthPct.value + '%',
}))

const tickLabels = computed(() => {
  const span = totalSpan.value
  const labels: string[] = []
  for (let i = 0; i < 5; i++) {
    const ts = props.bufferStart + (span * i) / 4
    labels.push(formatTime(ts))
  }
  return labels
})

function formatTime(ms: number): string {
  const d = new Date(ms)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

// ---- drag logic ----

function onTrackMouseDown(e: MouseEvent) {
  if (props.realtime) return
  e.preventDefault()
  dragging = true
  setViewFromClientX(e.clientX)
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

function onMouseMove(e: MouseEvent) {
  if (!dragging) return
  setViewFromClientX(e.clientX)
}

function onMouseUp() {
  dragging = false
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
}

function setViewFromClientX(clientX: number) {
  const track = trackRef.value
  if (!track) return
  const rect = track.getBoundingClientRect()
  const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
  // Center the window on the click position
  const winHalfSpan = props.viewSpan / 2
  const desiredCenter = props.bufferStart + frac * totalSpan.value
  let newEnd = desiredCenter + winHalfSpan
  newEnd = Math.max(
    props.bufferStart + props.viewSpan,
    Math.min(newEnd, props.bufferEnd),
  )
  emit('pan', Math.round(newEnd))
}

// Touch support
function onTrackTouchStart(e: TouchEvent) {
  if (props.realtime) return
  e.preventDefault()
  dragging = true
  setViewFromClientX(e.touches[0].clientX)
  document.addEventListener('touchmove', onTouchMove)
  document.addEventListener('touchend', onTouchEnd)
}

function onTouchMove(e: TouchEvent) {
  if (!dragging) return
  setViewFromClientX(e.touches[0].clientX)
}

function onTouchEnd() {
  dragging = false
  document.removeEventListener('touchmove', onTouchMove)
  document.removeEventListener('touchend', onTouchEnd)
}

onMounted(() => {})
onUnmounted(() => {
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
  document.removeEventListener('touchmove', onTouchMove)
  document.removeEventListener('touchend', onTouchEnd)
})
</script>

<template>
  <div class="timeline-bar" :class="{ disabled: realtime }">
    <div
      ref="trackRef"
      class="timeline-track"
      @mousedown="onTrackMouseDown"
      @touchstart="onTrackTouchStart"
    >
      <div class="timeline-window" :style="windowStyle"></div>
    </div>
    <div class="timeline-labels">
      <span v-for="(label, i) in tickLabels" :key="i">{{ label }}</span>
    </div>
  </div>
</template>

<style scoped>
.timeline-bar {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px 8px;
}
.timeline-bar.disabled {
  opacity: 0.5;
  pointer-events: none;
}
.timeline-track {
  position: relative;
  height: 20px;
  background: #21262d;
  border-radius: 10px;
  cursor: pointer;
}
.timeline-window {
  position: absolute;
  top: 2px;
  height: 16px;
  background: rgba(56, 139, 253, 0.25);
  border-left: 2px solid #58a6ff;
  border-right: 2px solid #58a6ff;
  border-radius: 8px;
  cursor: grab;
}
.timeline-window:active {
  cursor: grabbing;
}
.timeline-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.65rem;
  color: #8b949e;
}
</style>
