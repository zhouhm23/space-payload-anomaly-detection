<script setup lang="ts">
/**
 * AnomalyChart — canvas-based anomaly score + threshold + predicted
 * anomaly score chart.  Wraps CanvasChart with the same imperative
 * API (update / clear / showEmpty / hideEmpty).
 */

import { ref, computed } from 'vue'
import CanvasChart, { type Channel, type MarkLine, type ChartConfig } from './CanvasChart.vue'

const chartRef = ref<InstanceType<typeof CanvasChart> | null>(null)

const scoreData = ref<[number, number][]>([])
const predScoreData = ref<[number, number][]>([])
const emptyMsg = ref<string | undefined>(undefined)
const xMin = ref(0)
const xMax = ref(1)

const THRESHOLD = 0.7

const channels = computed<Channel[]>(() => {
  const list: Channel[] = [
    {
      name: '异常分数',
      color: '#f5a623',
      width: 1.5,
      data: scoreData.value,
      glow: true,
    },
  ]
  if (predScoreData.value.length > 0) {
    list.push({
      name: '预测异常分数',
      color: '#19be6b',
      width: 1.5,
      dash: [6, 4],
      data: predScoreData.value,
    })
  }
  return list
})

const markLines = computed<MarkLine[]>(() => [
  {
    axis: 'y',
    value: THRESHOLD,
    color: '#ed3f14',
    dash: [6, 4],
    label: '0.7',
  },
])

const config = computed<ChartConfig>(() => ({
  yMin: 0,
  yMax: 1,
  xMin: xMin.value,
  xMax: xMax.value,
  yLabel: '异常分数',
  xTicks: 10,
}))

// ---- imperative API ----

interface UpdateParams {
  xMin: number
  xMax: number
  scores: number[][]
  predScores?: number[][]
}

function update(p: UpdateParams) {
  xMin.value = p.xMin
  xMax.value = p.xMax
  scoreData.value = p.scores as [number, number][]
  predScoreData.value = (p.predScores || []) as [number, number][]
  emptyMsg.value = undefined
}

function clear(xMinVal: number, xMaxVal: number) {
  xMin.value = xMinVal
  xMax.value = xMaxVal
  scoreData.value = []
  predScoreData.value = []
}

function showEmpty(xMinVal: number, xMaxVal: number, reason: string) {
  xMin.value = xMinVal
  xMax.value = xMaxVal
  scoreData.value = []
  predScoreData.value = []
  emptyMsg.value = reason
}

function hideEmpty() {
  emptyMsg.value = undefined
}

defineExpose({ update, clear, showEmpty, hideEmpty })
</script>

<template>
  <CanvasChart
    ref="chartRef"
    :channels="channels"
    :config="config"
    :mark-lines="markLines"
    :empty-message="emptyMsg"
    :height="140"
  />
</template>
