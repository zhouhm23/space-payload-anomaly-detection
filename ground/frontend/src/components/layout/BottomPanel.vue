<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import TelemetryChart from '@/components/charts/TelemetryChart.vue'
import AnomalyChart from '@/components/charts/AnomalyChart.vue'
import { usePoll } from '@/composables/usePoll'
import { useTelemetryStore } from '@/stores/telemetry'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { useHealthStore } from '@/stores/health'
import { useForecast } from '@/composables/useForecast'
import { api } from '@/api/client'

const telemetry = useTelemetryStore()
const tree = useDeviceTreeStore()
const health = useHealthStore()
const { togglePlayPause, resetStream, fetchBlock } = usePoll()
const forecast = useForecast()

const inputBlockSize = ref(512)
const inputYMin = ref(-1)
const inputYMax = ref(1)

const telemetryChartRef = ref<InstanceType<typeof TelemetryChart> | null>(null)
const anomalyChartRef = ref<InstanceType<typeof AnomalyChart> | null>(null)

// forecast cache (mirrors legacy cachedPredTele / cachedPredScores)
let cachedPredTele: number[][] = []
let cachedPredScores: number[][] = []

function updateNavButtons() {
  // nothing — driven reactively in template
}

async function updateCharts() {
  const now = Date.now()
  const telChart = telemetryChartRef.value
  const anomChart = anomalyChartRef.value
  if (!telChart || !anomChart) return

  if (telemetry.currentBlock < 0 || telemetry.currentBlock >= telemetry.blocks.length) {
    telChart.clear(now - 30000, now)
    anomChart.clear(now - 30000, now)
    return
  }

  const block = telemetry.blocks[telemetry.currentBlock]
  const chName = tree.selectedChannelName()
  const chData = chName ? block.channels[chName] : null

  if (!chData) {
    const reason = !chName
      ? '当前未选中传感器'
      : `通道「${chName}」暂无数据（可能未在空间段启用采集）`
    telChart.showEmpty(now - 30000, now, reason)
    anomChart.showEmpty(now - 30000, now, reason)
    return
  }

  telChart.hideEmpty()
  anomChart.hideEmpty()

  const teleData = chData.telemetry
  const scoreData = chData.scores
  const startIdx = block.startIdx || 0

  // [ts, value, idx]
  const teleTime = teleData.map((p, i) => [p[0], p[1], startIdx + i])
  const scoreTime = scoreData.map((p, i) => [p[0], p[1], startIdx + i])

  // Forecast — only on new block
  if (telemetry.currentBlock !== forecast.lastForecastBlockIdx.value) {
    forecast.lastForecastBlockIdx.value = telemetry.currentBlock
    const teleForPredict = teleTime.map((p) => [p[2], p[1]])
    const result = await forecast.computePredict(teleForPredict, telemetry.currentBlock)
    if (result && teleData.length > 0) {
      const lastTs = teleData[teleData.length - 1][0]
      const lastIdx = teleTime[teleTime.length - 1][2]
      const interval =
        teleData.length > 1 ? teleData[teleData.length - 1][0] - teleData[teleData.length - 2][0] : 20
      cachedPredTele = (result.predValues || []).map((v, i) => [
        lastTs + (i + 1) * interval,
        v,
        lastIdx + 1 + i,
      ])
      cachedPredScores = []
    } else {
      cachedPredTele = []
      cachedPredScores = []
    }
  }

  // X axis range
  let xMin = 0
  let xMax = 100
  if (teleTime.length > 0) {
    xMin = teleTime[0][0]
    xMax = teleTime[teleTime.length - 1][0]
  }
  if (cachedPredTele.length > 0) {
    xMax = cachedPredTele[cachedPredTele.length - 1][0]
  }
  const windowMs = 30 * 1000
  if (xMax - xMin > windowMs) xMin = xMax - windowMs

  const predMarkLine: any =
    teleTime.length > 0
      ? {
          silent: true,
          symbol: 'none',
          data: [{ xAxis: teleTime[teleTime.length - 1][0] }],
          label: {
            show: true,
            formatter: '预测起点',
            color: '#f5a623',
            fontSize: 11,
            position: 'insideEndTop',
          },
        }
      : undefined

  telChart.update({
    xMin,
    xMax,
    yMin: inputYMin.value,
    yMax: inputYMax.value,
    telemetry: teleTime,
    prediction: cachedPredTele,
    predMarkLine,
  })

  // Fetch predict scores for the selected channel
  const psChannel = tree.selectedChannelName()
  let predScoresData: number[][] = []
  if (psChannel) {
    try {
      const ps = await api.predictScores(psChannel)
      if (ps.timestamps && ps.timestamps.length > 0) {
        // Convert seconds to milliseconds for ECharts time axis
        predScoresData = ps.timestamps.map((ts, i) => [ts * 1000, ps.scores[i], 0])
      }
    } catch {
      // ignore — predict scores may not be available yet
    }
  }

  anomChart.update({
    xMin,
    xMax,
    scores: scoreTime,
    predScores: predScoresData,
  })
}

// Watch currentBlock / selectedId changes → refresh charts
let healthTimer: ReturnType<typeof setInterval> | null = null

async function onPollTick() {
  await updateCharts()
}

// Expose for child button clicks
function onPlay() {
  togglePlayPause()
}
function onReset() {
  resetStream()
  forecast.invalidate()
  cachedPredTele = []
  cachedPredScores = []
  updateCharts()
}
function onPrev() {
  telemetry.prevBlock()
  updateCharts()
}
function onNext() {
  telemetry.nextBlock()
  updateCharts()
}

// Block navigation info text
function navInfo(): string {
  const total = telemetry.blocks.length
  const cur = telemetry.currentBlock
  return total > 0 ? `块 ${cur + 1}/${total}` : '块 0/0'
}

// Re-render charts when block index or selection changes
import { watch } from 'vue'
watch(
  () => [telemetry.currentBlock, telemetry.blocks.length, tree.selectedId],
  () => {
    forecast.invalidate()
    cachedPredTele = []
    cachedPredScores = []
    updateCharts()
  },
)

// Periodic health refresh (drives dashboard cards)
onMounted(() => {
  health.refreshAll().catch(() => {})
  healthTimer = setInterval(() => {
    health.refreshAll().catch(() => {})
  }, 3000)
})
onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
})

// Handle a manual fetch (used when selecting a sensor or clicking a card)
async function manualFetch() {
  const bs = parseInt(String(inputBlockSize.value)) || 512
  await fetchBlock(bs)
  await onPollTick()
}

defineExpose({ manualFetch, updateCharts })
</script>

<template>
  <div class="bottom-panel">
    <div class="control-bar">
      <label style="font-size: 0.8rem; color: var(--text-secondary)">区块</label>
      <input
        v-model.number="inputBlockSize"
        type="number"
        min="64"
        max="65536"
        step="64"
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
      <button class="btn" :disabled="telemetry.currentBlock <= 0 || telemetry.playing" title="前一块" @click="onPrev">
        ◀
      </button>
      <span class="block-nav-info">{{ navInfo() }}</span>
      <button
        class="btn"
        :disabled="telemetry.currentBlock >= telemetry.blocks.length - 1 || telemetry.playing"
        title="后一块"
        @click="onNext"
      >
        ▶
      </button>
      <span style="flex: 1"></span>
      <button class="btn" :class="{ 'btn-pause': telemetry.playing }" @click="onPlay">
        {{ telemetry.playing ? '⏸️ 暂停' : '▶ 开始' }}
      </button>
      <button class="btn" :disabled="telemetry.playing" @click="onReset">↺ 重置</button>
      <span class="chunk-info">{{ telemetry.chunkInfo }}</span>
    </div>
    <div class="chart-row">
      <div class="chart-main">
        <div class="chart-section top">
          <div class="chart-title">📈 遥测 &amp; 预测 — 实线为遥测，虚线为预测</div>
          <TelemetryChart ref="telemetryChartRef" />
        </div>
        <div class="chart-section bottom">
          <div class="chart-title">📉 异常分数 — 阈值 0.7</div>
          <AnomalyChart ref="anomalyChartRef" />
        </div>
      </div>
    </div>
  </div>
</template>
