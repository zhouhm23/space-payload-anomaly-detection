<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import TelemetryChart from '@/components/charts/TelemetryChart.vue'
import AnomalyChart from '@/components/charts/AnomalyChart.vue'
import TimelineBar from '@/components/charts/TimelineBar.vue'
import ExportPanel from '@/components/info/ExportPanel.vue'
import { useTelemetryWindowStore } from '@/stores/telemetryWindow'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { useHealthStore } from '@/stores/health'

const win = useTelemetryWindowStore()
const tree = useDeviceTreeStore()
const health = useHealthStore()

const inputYMin = ref(-1)
const inputYMax = ref(1)
const showExport = ref(false)

const telemetryChartRef = ref<InstanceType<typeof TelemetryChart> | null>(null)
const anomalyChartRef = ref<InstanceType<typeof AnomalyChart> | null>(null)

// ---- chart drawing ----

let lastMode: string = ''

function updateCharts() {
  const telChart = telemetryChartRef.value
  const anomChart = anomalyChartRef.value
  if (!telChart || !anomChart) return

  // Only toggle drag-pan when mode actually changes — calling this on
  // every data update would break an in-progress drag (panTo triggers
  // fetchWindow → raw changes → watch → updateCharts → setPanEnabled(false)).
  if (win.mode !== lastMode) {
    telChart.setPanEnabled(win.mode !== 'realtime')
    lastMode = win.mode
  }

  if (win.raw.length === 0) {
    const now = Date.now()
    const reason = !win.channel
      ? '当前未选中传感器'
      : `通道「${win.channel}」暂无数据`
    telChart.showEmpty(now - 30000, now, reason)
    anomChart.showEmpty(now - 30000, now, reason)
    return
  }

  telChart.hideEmpty()
  anomChart.hideEmpty()

  const teleData = win.teleSeries
  const scoreData = win.scoreSeries

  let xMin = win.raw[0].received_at * 1000
  let xMax = win.raw[win.raw.length - 1].received_at * 1000

  const predTele = win.predTeleSeries
  if (predTele.length > 0) {
    xMax = Math.max(xMax, predTele[predTele.length - 1][0])
  }
  const predScores = win.predScoreSeries
  if (predScores.length > 0) {
    xMax = Math.max(xMax, predScores[predScores.length - 1][0])
  }

  telChart.update({
    xMin,
    xMax,
    yMin: inputYMin.value,
    yMax: inputYMax.value,
    telemetry: teleData,
    prediction: predTele,
  })

  anomChart.update({
    xMin,
    xMax,
    scores: scoreData,
    predScores,
  })
}

// ---- mode controls ----

function onPlay() {
  if (win.mode === 'realtime') {
    win.freeze()
  } else {
    win.startRealtime()
  }
}

function onReset() {
  win.reset()
}

function onWindowSizeChange(e: Event) {
  const val = parseInt((e.target as HTMLInputElement).value)
  if (!isNaN(val)) win.setWindowSize(val)
}

const playLabel = computed(() => {
  if (win.mode === 'realtime') return '⏸ 冻结'
  return '▶ 实时'
})

const statusText = computed(() => {
  if (win.error) return `错误: ${win.error}`
  const pts = win.raw.length
  const preds = win.predictions.length
  let s = `${pts} 点`
  if (preds > 0) s += ` | ${preds} 批预测`
  if (win.mode === 'realtime') s += ' | 实时滚动'
  else if (win.mode === 'frozen') s += ' | 冻结'
  else s += ' | 已重置'
  return s
})

// ---- drag pan (frozen mode) ----

function onChartPan(newEndTsMs: number) {
  win.panTo(newEndTsMs)
}

// ---- timeline bar ----

/** Buffer earliest timestamp (ms) — falls back to view start when empty */
const bufferStartMs = computed(() => {
  if (win.raw.length === 0) return Date.now() - 30_000
  return win.raw[0].received_at * 1000
})

/** Buffer latest timestamp (ms) */
const bufferEndMs = computed(() => {
  if (win.raw.length === 0) return Date.now()
  let end = win.raw[win.raw.length - 1].received_at * 1000
  const predTele = win.predTeleSeries
  if (predTele.length > 0) end = Math.max(end, predTele[predTele.length - 1][0])
  return end
})

/** Current view right-edge (ms) for the timeline highlight */
const viewEndMs = computed(() => {
  if (win.viewEndTs !== null) return win.viewEndTs * 1000
  return bufferEndMs.value
})

/** Current view span (ms) */
const viewSpanMs = computed(() => {
  if (win.raw.length < 2) return 10_000
  return (win.raw[win.raw.length - 1].received_at - win.raw[0].received_at) * 1000
})

function onTimelinePan(newEndMs: number) {
  win.panTo(newEndMs)
}

// ---- watch store changes → redraw ----

watch(
  () => [win.raw, win.predictions, win.mode],
  () => updateCharts(),
  { deep: true },
)

// ---- channel switching ----

watch(
  () => [tree.selectedId, tree.tree],
  () => {
    const ch = tree.selectedChannelName()
    if (ch && ch !== win.channel) {
      win.setChannel(ch)
    }
  },
  { deep: true },
)

// ---- lifecycle ----

let healthTimer: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  health.refreshAll().catch(() => {})
  healthTimer = setInterval(() => {
    health.refreshAll().catch(() => {})
  }, 3000)
  // Wire up drag pan callback (only TelemetryChart, since both charts
  // share the same x-axis range controlled by the store).
  telemetryChartRef.value?.onPan(onChartPan)
  // Initial load
  const ch = tree.selectedChannelName()
  if (ch) {
    win.setChannel(ch)
  }
})

onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
  win.stopPoll()
})
</script>

<template>
  <div class="bottom-panel">
    <div class="control-bar">
      <label style="font-size: 0.8rem; color: var(--text-secondary)">窗口</label>
      <input
        :value="win.windowSize"
        @change="onWindowSizeChange"
        type="number"
        min="100"
        max="1000"
        step="64"
        :disabled="win.mode === 'realtime'"
        title="窗口长度（点数），仅冻结时可修改"
        style="
          width: 60px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          color: var(--text-primary);
          padding: 3px 5px;
          border-radius: 4px;
          font-size: 0.8rem;
          text-align: center;
        "
      />
      <label style="font-size: 0.8rem; color: var(--text-secondary); margin-left: 6px">Y轴</label>
      <input
        v-model.number="inputYMin"
        type="number"
        step="0.1"
        style="
          width: 50px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          color: var(--text-primary);
          padding: 3px 5px;
          border-radius: 4px;
          font-size: 0.8rem;
          text-align: center;
        "
      />
      <span style="color: var(--text-secondary); font-size: 0.8rem">~</span>
      <input
        v-model.number="inputYMax"
        type="number"
        step="0.1"
        style="
          width: 50px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          color: var(--text-primary);
          padding: 3px 5px;
          border-radius: 4px;
          font-size: 0.8rem;
          text-align: center;
        "
      />
      <span style="flex: 1"></span>
      <div class="control-buttons">
        <button
          class="btn"
          :class="{ 'btn-pause': win.mode === 'realtime' }"
          @click="onPlay"
        >
          {{ playLabel }}
        </button>
        <button
          class="btn"
          :disabled="win.mode === 'realtime'"
          @click="onReset"
          title="跳到最新（但不是实时）"
        >
          ↺ 重置
        </button>
        <button class="btn" @click="showExport = true" title="导出时序数据">
          📥 导出数据
        </button>
      </div>
      <span class="chunk-info">{{ statusText }}</span>
    </div>
    <div class="chart-row">
      <div class="chart-main">
        <div class="chart-section top">
          <div class="chart-title">📈 遥测 &amp; 预测 — 实线为遥测，虚线为预测</div>
          <TelemetryChart ref="telemetryChartRef" />
        </div>
        <div class="chart-section bottom">
          <div class="chart-title">📉 异常分数 — 实线为实测，虚线为预测</div>
          <AnomalyChart ref="anomalyChartRef" />
        </div>
        <TimelineBar
          :buffer-start="bufferStartMs"
          :buffer-end="bufferEndMs"
          :view-end="viewEndMs"
          :view-span="viewSpanMs"
          :realtime="win.mode === 'realtime'"
          @pan="onTimelinePan"
        />
      </div>
    </div>
    <ExportPanel v-if="showExport" @close="showExport = false" />
  </div>
</template>
