<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { getTelemetryOption } from './options'

const dom = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

interface UpdateParams {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  telemetry: number[][]
  prediction: number[][]
  predMarkLine?: any
}

function ensureChart() {
  if (!chart && dom.value) {
    chart = echarts.init(dom.value)
    chart.setOption(getTelemetryOption())
  }
}

function update(p: UpdateParams) {
  ensureChart()
  if (!chart) return
  chart.setOption({
    xAxis: { min: p.xMin, max: p.xMax },
    yAxis: { min: p.yMin, max: p.yMax },
    series: [
      { data: p.telemetry },
      { data: p.prediction, markLine: p.predMarkLine },
    ],
  })
}

function clear(xMin: number, xMax: number) {
  ensureChart()
  chart?.setOption({
    xAxis: { min: xMin, max: xMax },
    yAxis: { min: 0, max: 1 },
    series: [{ data: [] }, { data: [], markLine: { data: [] } }],
  })
}

function showEmpty(xMin: number, xMax: number, reason: string) {
  ensureChart()
  chart?.setOption({
    xAxis: { min: xMin, max: xMax },
    yAxis: { min: 0, max: 1 },
    series: [{ data: [] }, { data: [] }],
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
  <div ref="dom" class="chart-telemetry"></div>
</template>
