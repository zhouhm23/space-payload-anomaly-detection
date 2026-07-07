<script setup lang="ts">
/**
 * ExportPanel — modal dialog for batch-exporting telemetry data.
 *
 * Lets the user pick:
 *  - One or more sensor channels (multi-select from device tree)
 *  - A custom time range (start/end)
 *  - Output format: CSV or XLSX
 *
 * Triggers a file download via GET /api/export.
 */

import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useDeviceTreeStore } from '@/stores/deviceTree'

const emit = defineEmits<{ close: [] }>()

const tree = useDeviceTreeStore()

// ---- collect available channels from device tree ----

interface ChannelOption {
  channel: string
  label: string
}

const channelOptions = ref<ChannelOption[]>([])
const selectedChannels = ref<string[]>([])
const exportFormat = ref<'csv' | 'xlsx'>('csv')

// Time range defaults: last 10 minutes
const now = Date.now()
const timeStart = ref(toLocalInput(now - 10 * 60 * 1000))
const timeEnd = ref(toLocalInput(now))

const exporting = ref(false)
const errorMsg = ref<string | null>(null)
const successMsg = ref<string | null>(null)
let clearTimer: ReturnType<typeof setTimeout> | null = null

function toLocalInput(ms: number): string {
  const d = new Date(ms)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function fromLocalInput(val: string): number {
  return new Date(val).getTime()
}

onMounted(() => {
  // Flatten device tree to find all sensor nodes with channelName
  const channels: ChannelOption[] = []
  function walk(nodes: any[]) {
    for (const n of nodes) {
      if (n.channelName) {
        channels.push({ channel: n.channelName, label: n.name || n.channelName })
      }
      if (n.children) walk(n.children)
    }
  }
  walk(tree.tree)
  channelOptions.value = channels
  // Pre-select the currently active channel
  const active = tree.selectedChannelName()
  if (active) {
    selectedChannels.value = [active]
  } else if (channels.length > 0) {
    selectedChannels.value = [channels[0].channel]
  }
})

onUnmounted(() => {
  if (clearTimer) clearTimeout(clearTimer)
})

function toggleChannel(ch: string) {
  const idx = selectedChannels.value.indexOf(ch)
  if (idx >= 0) {
    selectedChannels.value.splice(idx, 1)
  } else {
    selectedChannels.value.push(ch)
  }
}

const canExport = computed(() => {
  return selectedChannels.value.length > 0 && timeStart.value && timeEnd.value && !exporting.value
})

async function doExport() {
  errorMsg.value = null
  successMsg.value = null
  if (!canExport.value) return

  const startSec = fromLocalInput(timeStart.value) / 1000
  const endSec = fromLocalInput(timeEnd.value) / 1000

  if (startSec >= endSec) {
    errorMsg.value = '起始时间必须早于结束时间'
    return
  }

  exporting.value = true
  try {
    const params = new URLSearchParams({
      channels: selectedChannels.value.join(','),
      start: String(startSec),
      end: String(endSec),
      fmt: exportFormat.value,
    })
    const url = `/api/export?${params}`

    // Trigger download via a hidden link
    const a = document.createElement('a')
    a.href = url
    a.download = ''
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)

    successMsg.value = `已触发下载：${selectedChannels.value.length} 个通道数据`
    // Auto-clear success message after 3 seconds
    if (clearTimer) clearTimeout(clearTimer)
    clearTimer = setTimeout(() => { successMsg.value = null }, 3000)
  } catch (e) {
    errorMsg.value = `导出失败: ${e}`
  } finally {
    exporting.value = false
  }
}
</script>

<template>
  <div class="export-overlay" @click.self="emit('close')">
    <div class="export-modal">
      <div class="export-header">
        <h3>📥 时序数据导出</h3>
        <button class="btn-close" @click="emit('close')">×</button>
      </div>

      <div class="export-body">
        <!-- Channel selection -->
        <div class="export-section">
          <label class="export-label">选择传感器通道（可多选）</label>
          <div class="channel-grid">
            <label
              v-for="opt in channelOptions"
              :key="opt.channel"
              class="channel-chip"
              :class="{ active: selectedChannels.includes(opt.channel) }"
            >
              <input
                type="checkbox"
                :checked="selectedChannels.includes(opt.channel)"
                @change="toggleChannel(opt.channel)"
              />
              <span>{{ opt.label }}</span>
              <small>{{ opt.channel }}</small>
            </label>
          </div>
          <p v-if="channelOptions.length === 0" class="hint">设备树中没有传感器节点</p>
        </div>

        <!-- Time range -->
        <div class="export-section">
          <label class="export-label">时间范围</label>
          <div class="time-row">
            <div>
              <small>起始时间</small>
              <input v-model="timeStart" type="datetime-local" class="time-input" />
            </div>
            <div>
              <small>结束时间</small>
              <input v-model="timeEnd" type="datetime-local" class="time-input" />
            </div>
          </div>
        </div>

        <!-- Format -->
        <div class="export-section">
          <label class="export-label">导出格式</label>
          <div class="format-row">
            <label class="format-option" :class="{ active: exportFormat === 'csv' }">
              <input v-model="exportFormat" type="radio" value="csv" />
              <span>CSV</span>
              <small>通用文本格式</small>
            </label>
            <label class="format-option" :class="{ active: exportFormat === 'xlsx' }">
              <input v-model="exportFormat" type="radio" value="xlsx" />
              <span>XLSX</span>
              <small>Excel 表格</small>
            </label>
          </div>
        </div>

        <!-- Messages -->
        <div v-if="errorMsg" class="export-error">{{ errorMsg }}</div>
        <div v-if="successMsg" class="export-success">{{ successMsg }}</div>
      </div>

      <div class="export-footer">
        <button class="btn" @click="emit('close')">取消</button>
        <button class="btn btn-primary" :disabled="!canExport" @click="doExport">
          {{ exporting ? '导出中…' : '开始导出' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.export-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.export-modal {
  background: var(--bg-card, #1e2433);
  border-radius: 12px;
  width: 540px;
  max-width: 90vw;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
.export-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color, #2a3348);
}
.export-header h3 {
  margin: 0;
  font-size: 1.1rem;
  color: var(--text-primary, #e0e6f0);
}
.btn-close {
  background: none;
  border: none;
  color: var(--text-secondary, #8e9bb5);
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0 8px;
  line-height: 1;
}
.btn-close:hover {
  color: var(--text-primary, #e0e6f0);
}
.export-body {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}
.export-section {
  margin-bottom: 20px;
}
.export-label {
  display: block;
  font-size: 0.9rem;
  color: var(--text-secondary, #8e9bb5);
  margin-bottom: 8px;
  font-weight: 500;
}
.channel-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.channel-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid var(--border-color, #2a3348);
  cursor: pointer;
  font-size: 0.85rem;
  color: var(--text-primary, #e0e6f0);
  transition: all 0.2s;
}
.channel-chip.active {
  border-color: var(--accent-blue, #2d8cf0);
  background: rgba(45, 140, 240, 0.15);
}
.channel-chip small {
  color: var(--text-secondary, #8e9bb5);
  font-size: 0.75rem;
}
.channel-chip input {
  display: none;
}
.hint {
  color: var(--text-secondary, #8e9bb5);
  font-size: 0.85rem;
}
.time-row {
  display: flex;
  gap: 16px;
}
.time-row > div {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.time-row small {
  font-size: 0.75rem;
  color: var(--text-secondary, #8e9bb5);
}
.time-input {
  background: var(--bg-secondary, #1a1f2e);
  border: 1px solid var(--border-color, #2a3348);
  color: var(--text-primary, #e0e6f0);
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 0.85rem;
  color-scheme: dark;
}
.format-row {
  display: flex;
  gap: 12px;
}
.format-option {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid var(--border-color, #2a3348);
  cursor: pointer;
  transition: all 0.2s;
}
.format-option.active {
  border-color: var(--accent-blue, #2d8cf0);
  background: rgba(45, 140, 240, 0.15);
}
.format-option span {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary, #e0e6f0);
}
.format-option small {
  font-size: 0.75rem;
  color: var(--text-secondary, #8e9bb5);
}
.format-option input {
  display: none;
}
.export-error {
  color: var(--accent-red, #ed3f14);
  font-size: 0.85rem;
  padding: 8px 12px;
  background: rgba(237, 63, 20, 0.1);
  border-radius: 6px;
}
.export-success {
  color: var(--accent-green, #19be6b);
  font-size: 0.85rem;
  padding: 8px 12px;
  background: rgba(25, 190, 107, 0.1);
  border-radius: 6px;
}
.export-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--border-color, #2a3348);
}
.btn-primary {
  background: var(--accent-blue, #2d8cf0) !important;
  color: white !important;
}
.btn-primary:disabled {
  opacity: 0.4;
}
</style>
