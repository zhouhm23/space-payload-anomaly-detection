<script setup lang="ts">
/**
 * CanvasChart — generic 2D-canvas waveform renderer.
 *
 * Replaces ECharts to eliminate animation/merge artefacts (zig-zag,
 * blank charts).  Borrows the rendering strategy from
 * ``debug/虚拟示波器.html``:
 *
 *   - ``requestAnimationFrame`` full redraw every frame
 *   - DPR-aware canvas sizing (crisp on HiDPI)
 *   - caller pushes data via ``setData()``; the rAF loop draws it
 *   - optional drag-pan (left mouse) and mousewheel zoom
 *
 * This component is display-only — it knows nothing about the store.
 * The parent owns data and passes it in.
 */

import { ref, onMounted, onUnmounted, nextTick } from 'vue'

// ---- types ----

export interface Channel {
  /** Display name (for legend / tooltip) */
  name: string
  /** CSS color string, e.g. '#2d8cf0' */
  color: string
  /** Line width in CSS px (default 1.5) */
  width?: number
  /** Dash pattern (empty = solid) */
  dash?: number[]
  /** Data: array of [x_pixels_in_data_space, y_value] — but we use
   *  [ts_ms, value] and let the renderer map to pixels */
  data: [number, number][]
  /** If true, draw a semi-transparent glow under the line */
  glow?: boolean
}

export interface MarkLine {
  /** 'x' = vertical line at given xAxis value, 'y' = horizontal at yAxis */
  axis: 'x' | 'y'
  /** value in data coordinates */
  value: number
  /** CSS color */
  color: string
  /** dash pattern */
  dash?: number[]
  /** optional label text */
  label?: string
}

export interface ChartConfig {
  /** Y-axis range */
  yMin: number
  yMax: number
  /** X-axis range (epoch ms) */
  xMin: number
  xMax: number
  /** Y-axis label, e.g. '遥测值' */
  yLabel?: string
  /** number of x grid ticks (default 10) */
  xTicks?: number
  /** number of y grid ticks (default 7) */
  yTicks?: number
  /** Fixed nice y tick values; overrides yTicks if provided */
  yTickValues?: number[]
}

// ---- props / emits ----

const props = defineProps<{
  channels: Channel[]
  config: ChartConfig
  markLines?: MarkLine[]
  /** Enable drag-to-pan on the canvas (emits 'pan') */
  panEnabled?: boolean
  /** Optional message to show when there's no data (overlay) */
  emptyMessage?: string
  /** Canvas height in CSS pixels (default 280) */
  height?: number
}>()

const emit = defineEmits<{
  /** User dragged horizontally; newEndMs is the desired right-edge ts */
  (e: 'pan', newEndMs: number): void
}>()

const dom = ref<HTMLDivElement | null>(null)
let canvas: HTMLCanvasElement | null = null
let ctx: CanvasRenderingContext2D | null = null

// Canvas dimensions track the actual container size (not fixed).
// CSS width = 100% of parent; canvas pixel buffer = CSS size × DPR.
const DPR = Math.min(window.devicePixelRatio || 1, 2)
const MARGIN = { top: 12, right: 35, bottom: 28, left: 55 }

// Current logical (CSS-pixel) dimensions — updated by resizeCanvas().
let cssW = 900
let cssH = 280

// Latest data snapshot (reactive via props, but we copy to avoid Vue
// reactivity overhead inside the rAF hot loop).
let snapshot = {
  channels: [] as Channel[],
  config: { yMin: 0, yMax: 1, xMin: 0, xMax: 1 } as ChartConfig,
  markLines: [] as MarkLine[],
  emptyMessage: undefined as string | undefined,
}

let rafId: number | null = null
let needsRedraw = true

// ---- drag-pan state ----
let panning = false
let panStartX = 0
let panStartXMin = 0
let panStartXMax = 0
let panLastEmitTs = 0
const PAN_THROTTLE_MS = 80

// ---- lifecycle ----

function ensureCanvas() {
  if (canvas || !dom.value) return
  canvas = document.createElement('canvas')
  // CSS height: 100% so canvas fills the flex container (which sizes
  // via chart-section flex ratios).  Width is set in resizeCanvas().
  canvas.style.width = '100%'
  canvas.style.height = '100%'
  dom.value.appendChild(canvas)
  ctx = canvas.getContext('2d')
  resizeCanvas()
  // Always bind mousedown — the handler checks props.panEnabled at
  // event time, so toggling realtime/frozen works without re-binding.
  canvas.addEventListener('mousedown', onMouseDown)
  canvas.addEventListener('mousemove', onHover)
  canvas.addEventListener('mouseleave', onHoverEnd)
}

function resizeCanvas() {
  if (!canvas || !dom.value) return
  // Read the ACTUAL container size (flex layout decides width + height).
  // We must measure dom.value, not canvas, because canvas style is 100%
  // and its clientHeight would be circular.
  cssW = dom.value.clientWidth || 900
  cssH = dom.value.clientHeight || 200
  // CSS size stays 100% × 100% (set once in ensureCanvas); only the
  // pixel buffer changes.
  canvas.width = Math.floor(cssW * DPR)
  canvas.height = Math.floor(cssH * DPR)
  ctx?.setTransform(1, 0, 0, 1, 0, 0)
  ctx?.scale(DPR, DPR)
  needsRedraw = true
}

function takeSnapshot() {
  snapshot.channels = props.channels.map((ch) => ({ ...ch, data: ch.data }))
  // During an active drag, takeSnapshot runs every frame and would
  // overwrite the shifted xAxis range we set in onMouseMove, causing
  // the chart to flicker back and forth (visual zig-zag).  Guard it.
  if (!panning) {
    snapshot.config = { ...props.config }
  } else {
    // Only update data + markLines, preserve the panned x-axis range
    snapshot.config = {
      ...props.config,
      xMin: panXMin,
      xMax: panXMax,
    }
  }
  snapshot.markLines = props.markLines ? [...props.markLines] : []
  snapshot.emptyMessage = props.emptyMessage
  needsRedraw = true
}

// ---- coordinate mapping ----

function getPlotWidth() {
  return cssW - MARGIN.left - MARGIN.right
}
function getPlotHeight() {
  return cssH - MARGIN.top - MARGIN.bottom
}

function dataToPixel(xVal: number, yVal: number) {
  const { xMin, xMax, yMin, yMax } = snapshot.config
  const w = getPlotWidth()
  const h = getPlotHeight()
  const xRange = xMax - xMin || 1
  const yRange = yMax - yMin || 1
  const x = MARGIN.left + ((xVal - xMin) / xRange) * w
  const y = MARGIN.top + h - ((yVal - yMin) / yRange) * h
  return { x, y }
}

// ---- drawing ----

function drawGrid() {
  const c = ctx
  if (!c) return
  const left = MARGIN.left
  const top = MARGIN.top
  const w = getPlotWidth()
  const h = getPlotHeight()
  const { xMin, xMax, yMin, yMax } = snapshot.config

  // background
  c.fillStyle = 'transparent'
  c.clearRect(0, 0, cssW, cssH)
  c.fillStyle = 'transparent'
  c.fillRect(left, top, w, h)

  // Y grid + labels
  const yTicks = snapshot.config.yTickValues ?? defaultTicks(yMin, yMax, snapshot.config.yTicks ?? 7)
  c.font = '11px "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif'
  c.textAlign = 'right'
  yTicks.forEach((yVal) => {
    if (yVal < yMin || yVal > yMax) return
    const { y } = dataToPixel(xMin, yVal)
    const isZero = Math.abs(yVal) < 0.001
    c.strokeStyle = isZero ? '#3a4368' : '#21262d'
    c.lineWidth = isZero ? 1 : 0.5
    c.setLineDash(isZero ? [] : [3, 5])
    c.beginPath()
    c.moveTo(left, y)
    c.lineTo(left + w, y)
    c.stroke()
    c.setLineDash([])
    c.fillStyle = '#8b949e'
    c.fillText(formatY(yVal), left - 6, y + 4)
  })

  // Y axis label
  if (snapshot.config.yLabel) {
    c.save()
    c.translate(14, top + h / 2)
    c.rotate(-Math.PI / 2)
    c.fillStyle = '#8b949e'
    c.font = '11px "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif'
    c.textAlign = 'center'
    c.fillText(snapshot.config.yLabel, 0, 0)
    c.restore()
  }

  // X grid + labels (time-based)
  const xTickCount = snapshot.config.xTicks ?? 10
  c.textAlign = 'center'
  for (let i = 0; i <= xTickCount; i++) {
    const xVal = xMin + ((xMax - xMin) * i) / xTickCount
    const { x } = dataToPixel(xVal, 0)
    c.strokeStyle = '#21262d'
    c.lineWidth = 0.5
    c.setLineDash([3, 5])
    c.beginPath()
    c.moveTo(x, top)
    c.lineTo(x, top + h)
    c.stroke()
    c.setLineDash([])
    c.fillStyle = '#8b949e'
    c.font = '10px "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif'
    c.fillText(formatTime(xVal), x, top + h + 16)
  }

  // border
  c.strokeStyle = '#3a4368'
  c.lineWidth = 1
  c.setLineDash([])
  c.strokeRect(left, top, w, h)
}

function drawMarkLines() {
  const c = ctx
  if (!c) return
  const left = MARGIN.left
  const top = MARGIN.top
  const w = getPlotWidth()
  const h = getPlotHeight()
  for (const ml of snapshot.markLines) {
    if (ml.axis === 'y') {
      const { y } = dataToPixel(0, ml.value)
      if (y < top || y > top + h) continue
      c.strokeStyle = ml.color
      c.lineWidth = 1
      c.setLineDash(ml.dash ?? [6, 4])
      c.beginPath()
      c.moveTo(left, y)
      c.lineTo(left + w, y)
      c.stroke()
      c.setLineDash([])
      if (ml.label) {
        c.fillStyle = ml.color
        c.font = '10px "Segoe UI",sans-serif'
        c.textAlign = 'left'
        c.fillText(ml.label, left + w - 40, y - 4)
      }
    } else {
      // vertical line at ml.value (x-axis = ts ms)
      const { x } = dataToPixel(ml.value, 0)
      if (x < left || x > left + w) continue
      c.strokeStyle = ml.color
      c.lineWidth = 1
      c.setLineDash(ml.dash ?? [6, 4])
      c.beginPath()
      c.moveTo(x, top)
      c.lineTo(x, top + h)
      c.stroke()
      c.setLineDash([])
      if (ml.label) {
        c.fillStyle = ml.color
        c.font = '10px "Segoe UI",sans-serif'
        c.textAlign = 'left'
        c.fillText(ml.label, x + 3, top + 12)
      }
    }
  }
}

function drawChannels() {
  const c = ctx
  if (!c) return
  const left = MARGIN.left
  const top = MARGIN.top
  const w = getPlotWidth()
  const h = getPlotHeight()
  const right = left + w

  c.save()
  c.beginPath()
  c.rect(left - 1, top - 1, w + 2, h + 2)
  c.clip()

  for (const ch of snapshot.channels) {
    if (!ch.data || ch.data.length === 0) continue
    const lineW = ch.width ?? 1.5
    // glow pass
    if (ch.glow) {
      c.strokeStyle = hexToRgba(ch.color, 0.25)
      c.lineWidth = lineW + 3
      c.lineCap = 'round'
      c.lineJoin = 'round'
      c.setLineDash([])
      c.beginPath()
      let first = true
      for (const [tx, ty] of ch.data) {
        const { x, y } = dataToPixel(tx, ty)
        if (x < left - 5 || x > right + 5) {
          first = true
          continue
        }
        if (first) {
          c.moveTo(x, y)
          first = false
        } else {
          c.lineTo(x, y)
        }
      }
      c.stroke()
    }
    // main line
    c.strokeStyle = ch.color
    c.lineWidth = lineW
    c.setLineDash(ch.dash ?? [])
    c.beginPath()
    let first = true
    for (const [tx, ty] of ch.data) {
      const { x, y } = dataToPixel(tx, ty)
      if (x < left - 5 || x > right + 5) {
        first = true
        continue
      }
      if (first) {
        c.moveTo(x, y)
        first = false
      } else {
        c.lineTo(x, y)
      }
    }
    c.stroke()
  }
  c.setLineDash([])
  c.restore()
}

function drawEmpty() {
  const c = ctx
  if (!c || !snapshot.emptyMessage) return
  const left = MARGIN.left
  const top = MARGIN.top
  const w = getPlotWidth()
  const h = getPlotHeight()
  c.fillStyle = '#8b9bb5'
  c.font = '13px "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif'
  c.textAlign = 'center'
  c.fillText(snapshot.emptyMessage, left + w / 2, top + h / 2)
}

function drawAll() {
  const c = ctx
  if (!c) return
  c.clearRect(0, 0, cssW, cssH)
  drawGrid()
  drawMarkLines()
  drawChannels()
  drawEmpty()
}

// ---- helpers ----

function defaultTicks(min: number, max: number, count: number): number[] {
  const step = (max - min) / (count - 1)
  return Array.from({ length: count }, (_, i) => min + i * step)
}

function formatY(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0)
  if (Math.abs(v) >= 1) return v.toFixed(1)
  return v.toFixed(2)
}

function formatTime(tsMs: number): string {
  const d = new Date(tsMs)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.substring(0, 2), 16)
  const g = parseInt(h.substring(2, 4), 16)
  const b = parseInt(h.substring(4, 6), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

// ---- drag-pan ----

// During drag we store the shifted x-axis here so takeSnapshot can
// preserve it instead of overwriting with the un-panned props.config.
let panXMin = 0
let panXMax = 0

function onMouseDown(e: MouseEvent) {
  // Check panEnabled at EVENT TIME (not at canvas creation time), so
  // toggling realtime/frozen dynamically enables/disables drag.
  if (!canvas || !props.panEnabled) return
  e.preventDefault()
  panning = true
  panStartX = e.clientX
  panStartXMin = snapshot.config.xMin
  panStartXMax = snapshot.config.xMax
  panXMin = panStartXMin
  panXMax = panStartXMax
  panLastEmitTs = 0
  if (canvas) canvas.style.cursor = 'grabbing'
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

function onMouseMove(e: MouseEvent) {
  if (!panning || !canvas) return
  const dx = e.clientX - panStartX
  const canvasW = canvas.clientWidth || 800
  const span = panStartXMax - panStartXMin
  const timePerPx = span / canvasW
  // drag right → chart shifts left → earlier data → smaller end
  const newEnd = panStartXMax - dx * timePerPx
  // Store shifted range in panXMin/panXMax — takeSnapshot will use these
  panXMin = newEnd - span
  panXMax = newEnd
  // Also update snapshot immediately for instant visual feedback
  snapshot.config.xMin = panXMin
  snapshot.config.xMax = panXMax
  needsRedraw = true
  // throttled data request
  const now = Date.now()
  if (now - panLastEmitTs >= PAN_THROTTLE_MS) {
    panLastEmitTs = now
    emit('pan', Math.round(newEnd))
  }
}

function onMouseUp() {
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
  if (!panning) return
  panning = false
  if (canvas) canvas.style.cursor = props.panEnabled ? 'grab' : 'crosshair'
  // final precise fetch
  emit('pan', Math.round(snapshot.config.xMax))
}

// ---- hover tooltip ----

let hoverX = -1
let hoverY = -1

function onHover(e: MouseEvent) {
  if (!canvas || panning) return
  const rect = canvas.getBoundingClientRect()
  hoverX = e.clientX - rect.left
  hoverY = e.clientY - rect.top
  needsRedraw = true
}

function onHoverEnd() {
  hoverX = -1
  hoverY = -1
  needsRedraw = true
}

/** Find the data point closest to the cursor and draw a tooltip box. */
function drawTooltip() {
  const c = ctx
  if (!c || hoverX < 0 || hoverY < 0) return
  const left = MARGIN.left
  const top = MARGIN.top
  const w = getPlotWidth()
  const h = getPlotHeight()
  // Ignore if cursor is outside the plot area
  if (hoverX < left || hoverX > left + w || hoverY < top || hoverY > top + h) return

  const { xMin, xMax } = snapshot.config
  const xRange = xMax - xMin || 1
  // Convert cursor X to data timestamp
  const tsVal = xMin + ((hoverX - left) / w) * xRange

  // For each channel, find the closest data point to tsVal
  const tips: { name: string; color: string; ts: number; val: number }[] = []
  for (const ch of snapshot.channels) {
    if (!ch.data || ch.data.length === 0) continue
    // Binary search for closest timestamp
    let lo = 0
    let hi = ch.data.length - 1
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1
      if (ch.data[mid][0] < tsVal) lo = mid
      else hi = mid
    }
    const p1 = ch.data[lo]
    const p2 = ch.data[hi]
    const closest = Math.abs(p1[0] - tsVal) <= Math.abs(p2[0] - tsVal) ? p1 : p2
    // Skip channels whose nearest point is too far from the cursor
    const maxDist = xRange * 0.05
    if (Math.abs(closest[0] - tsVal) > maxDist) continue
    tips.push({ name: ch.name, color: ch.color, ts: closest[0], val: closest[1] })
  }

  if (tips.length === 0) return

  // Draw vertical crosshair line at cursor
  c.strokeStyle = 'rgba(255,255,255,0.25)'
  c.lineWidth = 1
  c.setLineDash([4, 4])
  c.beginPath()
  c.moveTo(hoverX, top)
  c.lineTo(hoverX, top + h)
  c.stroke()
  c.setLineDash([])

  // Build tooltip text
  const d = new Date(tips[0].ts)
  const pad = (n: number) => String(n).padStart(2, '0')
  const timeStr = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${String(d.getMilliseconds()).padStart(3, '0')}`
  const lines = [`⏱ ${timeStr}`]
  for (const t of tips) {
    const valStr = typeof t.val === 'number' ? t.val.toFixed(4) : String(t.val)
    lines.push(`● ${t.name}: ${valStr}`)
  }

  // Measure text
  c.font = '12px "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif'
  const lineH = 16
  const padX = 8
  const padY = 6
  let maxW = 0
  for (const ln of lines) {
    maxW = Math.max(maxW, c.measureText(ln).width)
  }
  const boxW = maxW + padX * 2
  const boxH = lines.length * lineH + padY * 2

  // Position: prefer right of cursor, flip to left if overflow
  let boxX = hoverX + 12
  if (boxX + boxW > left + w) boxX = hoverX - boxW - 12
  let boxY = hoverY + 12
  if (boxY + boxH > top + h) boxY = hoverY - boxH - 12

  // Draw tooltip box
  c.fillStyle = 'rgba(13, 17, 23, 0.92)'
  c.strokeStyle = 'rgba(255,255,255,0.15)'
  c.lineWidth = 1
  c.beginPath()
  c.roundRect(boxX, boxY, boxW, boxH, 6)
  c.fill()
  c.stroke()

  // Draw text lines
  c.textAlign = 'left'
  for (let i = 0; i < lines.length; i++) {
    if (i === 0) {
      c.fillStyle = '#8b949e'
    } else {
      // Use the channel color for the dot, white for text
      c.fillStyle = tips[i - 1].color
      c.fillText('●', boxX + padX, boxY + padY + (i + 1) * lineH - 4)
      c.fillStyle = '#e0e6f0'
      const text = lines[i].replace('● ', '')
      c.fillText(text, boxX + padX + 14, boxY + padY + (i + 1) * lineH - 4)
      continue
    }
    c.fillText(lines[i], boxX + padX, boxY + padY + (i + 1) * lineH - 4)
  }
}

// ---- rAF loop ----

function loop() {
  // Detect container size changes (panel collapse/expand, sidebar toggle,
  // window resize that doesn't fire the event, etc.)
  if (dom.value) {
    const w = dom.value.clientWidth
    const h = dom.value.clientHeight
    if (w > 0 && h > 0 && (Math.abs(w - cssW) > 1 || Math.abs(h - cssH) > 1)) {
      resizeCanvas()
    }
  }
  // Update cursor based on panEnabled state
  if (canvas && !panning) {
    canvas.style.cursor = props.panEnabled ? 'grab' : 'crosshair'
  }
  takeSnapshot()
  if (needsRedraw) {
    drawAll()
    drawTooltip()
    needsRedraw = false
  }
  rafId = requestAnimationFrame(loop)
}

function onResize() {
  resizeCanvas()
}

onMounted(async () => {
  await nextTick()
  ensureCanvas()
  window.addEventListener('resize', onResize)
  rafId = requestAnimationFrame(loop)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
  if (rafId) cancelAnimationFrame(rafId)
  rafId = null
  if (canvas) {
    canvas.removeEventListener('mousedown', onMouseDown)
    canvas.removeEventListener('mousemove', onHover)
    canvas.removeEventListener('mouseleave', onHoverEnd)
  }
  canvas = null
  ctx = null
})
</script>

<template>
  <div ref="dom" class="canvas-chart-host"></div>
</template>

<style scoped>
.canvas-chart-host {
  width: 100%;
  height: 100%;
  overflow: hidden;
}
.canvas-chart-host :deep(canvas) {
  display: block;
}
</style>
