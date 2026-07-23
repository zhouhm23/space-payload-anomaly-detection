<script setup lang="ts">
/**
 * Draggable splitter — turns the fixed-width side panels into user-resizable ones.
 *
 * Usage (see App.vue):
 *   <ResizableSplitter
 *     storage-key="phm.layout.left"
 *     :default-size="240" :min="160" :max="480"
 *     @resize="onResize" />
 *
 * Design notes:
 * - Double-click resets to defaultSize (a user gesture on body, not automatic).
 * - While dragging, body gets `user-select:none` + a locked `cursor` to avoid
 *   cross-element selection flicker.
 * - The size is owned by the parent (single source of truth); this component
 *   only emits resize events and reads storage to suggest an initial size.
 * - Zero offline dependencies, no third-party lib (same convention as vendor/Sortable).
 */
import { onMounted, onUnmounted, ref } from 'vue'

const props = withDefaults(
  defineProps<{
    /** localStorage key used to persist the current size. */
    storageKey: string
    /** Default size (px); also the double-click reset target. */
    defaultSize: number
    /** Minimum size (px). */
    min?: number
    /** Maximum size (px). */
    max?: number
  }>(),
  { min: 120, max: 600 },
)

const emit = defineEmits<{
  /** Size changed (fires continuously while dragging, once on reset). */
  (e: 'resize', size: number): void
}>()

const dragging = ref(false)

function clamp(v: number): number {
  return Math.min(props.max, Math.max(props.min, v))
}

/** Read the initial size from localStorage; return defaultSize on missing/invalid. */
function readStored(): number {
  try {
    const raw = localStorage.getItem(props.storageKey)
    if (raw == null || String(raw).trim() === '') return props.defaultSize
    const v = Number(raw)
    if (!Number.isFinite(v)) return props.defaultSize
    return clamp(v)
  } catch {
    return props.defaultSize
  }
}

function writeStored(v: number) {
  try {
    localStorage.setItem(props.storageKey, String(v))
  } catch {
    /* localStorage unavailable (private mode) — degrade silently, do not block dragging */
  }
}

let startX = 0
let startSize = 0

function onMouseDown(e: MouseEvent) {
  e.preventDefault()
  dragging.value = true
  startX = e.clientX
  startSize = readStored()
  document.body.style.userSelect = 'none'
  document.body.style.cursor = 'col-resize'
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
}

function onMouseMove(e: MouseEvent) {
  if (!dragging.value) return
  const delta = e.clientX - startX
  const next = clamp(startSize + delta)
  writeStored(next)
  emit('resize', next)
}

function onMouseUp() {
  if (!dragging.value) return
  dragging.value = false
  document.body.style.userSelect = ''
  document.body.style.cursor = ''
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
}

function onDblClick() {
  writeStored(props.defaultSize)
  emit('resize', props.defaultSize)
}

onMounted(() => {
  // On mount, notify the parent to apply the persisted size (so width is restored after refresh).
  emit('resize', readStored())
})

onUnmounted(() => {
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
})
</script>

<template>
  <div
    class="resizer"
    :class="{ active: dragging }"
    @mousedown="onMouseDown"
    @dblclick="onDblClick"
    title="Drag to resize, double-click to reset"
  ></div>
</template>

<style scoped>
.resizer {
  width: 4px;
  flex-shrink: 0;
  cursor: col-resize;
  background: transparent;
  position: relative;
  z-index: 10;
  transition: background 0.15s;
}

.resizer:hover,
.resizer.active {
  background: #409eff;
}
</style>
