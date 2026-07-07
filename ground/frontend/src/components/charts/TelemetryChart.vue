<script setup lang="ts">
/**
 * TelemetryChart — canvas-based telemetry + prediction line chart.
 *
 * Wraps CanvasChart, preserving the same imperative API that
 * BottomPanel calls (update / clear / showEmpty / hideEmpty /
 * setPanEnabled / onPan) so the parent doesn't need to change.
 */

import { ref, computed } from 'vue'
import CanvasChart, { type Channel, type ChartConfig } from './CanvasChart.vue'

const chartRef = ref<InstanceType<typeof CanvasChart> | null>(null)

const inputYMin = ref(-1)
const inputYMax = ref(1)

// Mutable state pushed into CanvasChart via computed props
const telemetryData = ref<[number, number][]>([])
const predictionData = ref<[number, number][]>([])
const panEnabled = ref(false)
const emptyMsg = ref<string | undefined>(undefined)
const xMin = ref(0)
const xMax = ref(1)

const channels = computed<Channel[]>(() => {
  const list: Channel[] = [
    {
      name: '遥测值',
      color: '#2d8cf0',
      width: 1.5,
      data: telemetryData.value,
      glow: true,
    },
  ]
  if (predictionData.value.length > 0) {
    list.push({
      name: '预测值',
      color: '#19be6b',
      width: 2,
      dash: [6, 4],
      data: predictionData.value,
    })
  }
  return list
})

const config = computed<ChartConfig>(() => ({
  yMin: inputYMin.value,
  yMax: inputYMax.value,
  xMin: xMin.value,
  xMax: xMax.value,
  yLabel: '遥测值',
  xTicks: 10,
}))

// ---- imperative API (called by BottomPanel) ----

interface UpdateParams {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  telemetry: number[][]
  prediction: number[][]
}

function update(p: UpdateParams) {
  xMin.value = p.xMin
  xMax.value = p.xMax
  inputYMin.value = p.yMin
  inputYMax.value = p.yMax
  telemetryData.value = p.telemetry as [number, number][]
  predictionData.value = p.prediction as [number, number][]
  emptyMsg.value = undefined
}

function clear(xMinVal: number, xMaxVal: number) {
  xMin.value = xMinVal
  xMax.value = xMaxVal
  telemetryData.value = []
  predictionData.value = []
}

function showEmpty(xMinVal: number, xMaxVal: number, reason: string) {
  xMin.value = xMinVal
  xMax.value = xMaxVal
  telemetryData.value = []
  predictionData.value = []
  emptyMsg.value = reason
}

function hideEmpty() {
  emptyMsg.value = undefined
}

function setPanEnabled(enabled: boolean) {
  panEnabled.value = enabled
}

let panCallback: ((newEndTsMs: number) => void) | null = null

function onPan(cb: (newEndTsMs: number) => void) {
  panCallback = cb
}

function onCanvasPan(newEndMs: number) {
  if (panEnabled.value) panCallback?.(newEndMs)
}

defineExpose({ update, clear, showEmpty, hideEmpty, setPanEnabled, onPan })
</script>

<template>
  <CanvasChart
    ref="chartRef"
    :channels="channels"
    :config="config"
    :pan-enabled="panEnabled"
    :empty-message="emptyMsg"
    :height="280"
    @pan="onCanvasPan"
  />
</template>
