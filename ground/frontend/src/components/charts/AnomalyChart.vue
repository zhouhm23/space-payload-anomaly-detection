<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { getAnomalyOption, THRESHOLD_LINE } from './options'

const dom = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

interface UpdateParams {
  xMin: number
  xMax: number
  scores: number[][]
  predScores?: number[][]
}

function ensureChart() {
  if (!chart && dom.value) {
    chart = echarts.init(dom.value)
    chart.setOption(getAnomalyOption())
  }
}

function update(p: UpdateParams) {
  ensureChart()
  chart?.setOption({
    xAxis: { min: p.xMin, max: p.xMax },
    series: [
      { data: p.scores },
      {
        data: [],
        markLine: { silent: true, symbol: 'none', data: THRESHOLD_LINE.data, label: THRESHOLD_LINE.label },
      },
      { data: p.predScores || [] },
    ],
  })
}

function clear(xMin: number, xMax: number) {
  ensureChart()
  chart?.setOption({
    xAxis: { min: xMin, max: xMax },
    series: [
      { data: [] },
      {
        data: [],
        markLine: { silent: true, symbol: 'none', data: THRESHOLD_LINE.data, label: THRESHOLD_LINE.label },
      },
    ],
  })
}

function showEmpty(xMin: number, xMax: number, reason: string) {
  ensureChart()
  chart?.setOption({
    xAxis: { min: xMin, max: xMax },
    series: [
      { data: [] },
      {
        data: [],
        markLine: { silent: true, symbol: 'none', data: THRESHOLD_LINE.data, label: THRESHOLD_LINE.label },
      },
    ],
    title: {
      show: true,
      textStyle: { color: '#8e9bb5', fontSize: 14 },
      subtext: reason,
      subtextStyle: { color: '#f5a623' },
      left: 'center',
      top: 'middle',
    },
  })
}

function hideEmpty() {
  chart?.setOption({ title: { show: false } })
}

function onResize() {
  chart?.resize()
}

onMounted(async () => {
  await nextTick()
  ensureChart()
  window.addEventListener('resize', onResize)
})
onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  chart?.dispose()
  chart = null
})

defineExpose({ update, clear, showEmpty, hideEmpty })
</script>

<template>
  <div ref="dom" class="chart-anomaly"></div>
</template>
