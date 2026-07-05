/**
 * ECharts option factories — ported 1:1 from the legacy HTML
 * ``getTelemetryOption`` / ``getAnomalyOption``.
 */

import type { EChartsOption } from 'echarts'

export function getTelemetryOption(): EChartsOption {
  return {
    backgroundColor: 'transparent',
    grid: { left: 55, right: 35, top: 10, bottom: 25 },
    xAxis: {
      type: 'time',
      name: '时间',
      nameTextStyle: { color: '#8e9bb5', fontSize: 12 },
      axisLine: { lineStyle: { color: '#2a3348' } },
      axisLabel: { color: '#e0e6f0', fontSize: 11 },
      splitLine: { show: false },
      axisPointer: { type: 'line' },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: '遥测值',
      nameTextStyle: { color: '#8e9bb5', fontSize: 12 },
      axisLine: { lineStyle: { color: '#2a3348' } },
      splitLine: { lineStyle: { color: '#2d3a5c' } },
      axisLabel: { color: '#e0e6f0', fontSize: 11 },
    },
    series: [
      {
        name: '遥测值',
        type: 'line',
        showSymbol: false,
        lineStyle: { color: '#2d8cf0', width: 1.5 },
        data: [],
      },
      {
        name: '预测值',
        type: 'line',
        showSymbol: false,
        lineStyle: { color: '#19be6b', type: 'dashed', width: 2 },
        data: [],
      },
    ],
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        if (!params || params.length === 0) return ''
        let tip = ''
        params.forEach((p: any) => {
          if (p.data && p.data.length >= 3) {
            const idx = p.data[2]
            const ts = new Date(p.data[0])
            const timeStr = ts.toTimeString().slice(0, 8)
            const val = p.data[1]
            tip += `<b>${p.seriesName}</b><br/>序号: ${idx}<br/>时间: ${timeStr}<br/>值: ${val.toFixed(4)}<br/>`
          }
        })
        return tip
      },
    },
  }
}

export function getAnomalyOption(): EChartsOption {
  return {
    backgroundColor: 'transparent',
    grid: { left: 55, right: 35, top: 10, bottom: 25 },
    xAxis: {
      type: 'time',
      name: '时间',
      nameTextStyle: { color: '#8e9bb5', fontSize: 12 },
      axisLine: { lineStyle: { color: '#2a3348' } },
      axisLabel: { color: '#e0e6f0', fontSize: 11 },
      splitLine: { show: false },
      axisPointer: { type: 'line' },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: '异常分数',
      nameTextStyle: { color: '#8e9bb5', fontSize: 12 },
      axisLine: { lineStyle: { color: '#2a3348' } },
      splitLine: { lineStyle: { color: '#2d3a5c' } },
      axisLabel: { color: '#e0e6f0', fontSize: 11 },
    },
    series: [
      {
        name: '异常分数',
        type: 'line',
        showSymbol: false,
        lineStyle: { color: '#f5a623', width: 1.5 },
        data: [],
      },
      {
        name: '阈值线',
        type: 'line',
        showSymbol: false,
        lineStyle: { color: '#ed3f14', type: 'dashed', width: 1, opacity: 0.7 },
        markLine: {
          silent: true,
          symbol: 'none',
          data: [{ yAxis: 0.7 }],
          label: {
            show: true,
            formatter: '0.7',
            color: '#ed3f14',
            fontSize: 10,
            position: 'end',
          },
        },
        data: [],
      },
      {
        name: '预测异常分数',
        type: 'line',
        showSymbol: false,
        lineStyle: { color: '#19be6b', type: 'dashed', width: 1.5 },
        data: [],
      },
    ],
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        if (!params || params.length === 0) return ''
        let tip = ''
        params.forEach((p: any) => {
          if (p.data && p.data.length >= 3) {
            const idx = p.data[2]
            const ts = new Date(p.data[0])
            const timeStr = ts.toTimeString().slice(0, 8)
            const val = p.data[1]
            tip += `<b>${p.seriesName}</b><br/>序号: ${idx}<br/>时间: ${timeStr}<br/>值: ${val.toFixed(4)}<br/>`
          }
        })
        return tip
      },
    },
  }
}

export const THRESHOLD_LINE = {
  silent: true,
  symbol: 'none',
  data: [{ yAxis: 0.7 }],
  label: { show: true, formatter: '0.7', color: '#ed3f14', fontSize: 10, position: 'end' },
}
