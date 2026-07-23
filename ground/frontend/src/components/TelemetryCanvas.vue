<script setup lang="ts">
/**
 * Telemetry chart (native Canvas, ported from the main branch's monitor.js
 * drawChart, with event red dots added).
 *
 * Three vertical regions: telemetry:anomaly-score:all-channel-alert-map = 4:1:2.
 *
 * Core features (validated on the main branch):
 * 1. Time-axis collapse (gaps are compressed to a fixed width so the axis
 *    does not blow up).
 * 2. Solid measured + dashed predicted (null points break the line).
 * 3. Fixed Y axis (telemetry uses the device-tree yMin/yMax; score is fixed [0,1]).
 * 4. Threshold line (red dashed in the anomaly-score region).
 * 5. Gap annotation (grey vertical dashed line + a "中断Xh" label).
 * 6. Data-not-arrived (pred-only contiguous spans, red translucent rectangle).
 *
 * Added (not in the demo):
 * 7. Event red dots: red circles drawn at alert times on the telemetry chart
 *    (provided by alert_points).
 */
import { computed, ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useSystemStore } from '@/stores/system'
import { api } from '@/api'
import { usePoll } from '@/composables/usePoll'
import { computeRedDots } from '@/utils/alertGeometry'

const store = useSystemStore()
const currentChannel = computed(() => store.currentChannel)

// Canvas refs
const canvasRef = ref<HTMLCanvasElement | null>(null)
const tooltipRef = ref<HTMLDivElement | null>(null)
// Plain variables (not refs): set once at init, never reassigned. The closures
// inside drawChart can then narrow them to non-null correctly.
let canvas: HTMLCanvasElement | null = null
let ctx: CanvasRenderingContext2D | null = null

// Palette (aligned with the main branch)
const C = {
  bg: '#0f1530',
  border: '#2a3050',
  textSec: '#7a85a8',
  textPri: '#e0e6ed',
  blue: '#409eff',
  green: '#67c23a',
  yellow: '#e6a23c',
  red: '#f56c6c',
  cyan: '#00c9db',
}

// Configuration
const CHART_CFG = {
  topRatio: 4 / 7,        // telemetry region gets 4 shares
  midRatio: 1 / 7,        // anomaly-score region gets 1 share
  botRatio: 2 / 7,        // all-channel alert-map region gets 2 shares
  padding: { top: 16, right: 16, bottom: 24, left: 50 },
  gapWidthPx: 40,
}
const VIEW_COUNT = 512  // number of points in the visible window

// Data state
const lastFullData = ref<any[]>([])
const gaps = ref<any[]>([])
const threshold = ref<number>(0.5)
const alertPoints = ref<any[]>([])  // all-channel alert points (for the map + telemetry red dots)

// Channel data cache: on channel switch, fill from the cache first to kill the
// "waiting for data" blank. Standard practice (the demo does it too) — the
// channel count is bounded (usually < 20), so no memory risk.
// Write: whenever windowPoll fetches fresh data. Read: the instant a channel is switched to.
interface CachedWindow { data: any[]; gaps: any[]; threshold: number; ts: number }
const channelCache = new Map<string, CachedWindow>()

// ── Preload: after the current channel's data arrives, async-prefetch the next carousel channel
// Visual optimisation for demos: the next channel's cache always hits, zero blank, zero "loading".
// Fire-and-forget; failures are silent (a cache miss falls back to the normal tick fetch).
let preloadingChannels = new Set<string>()  // guard against prefetching the same channel twice
async function preloadNextCarousel() {
  const channels = store.carouselChannels
  if (!channels || channels.length === 0) return
  const cur = currentChannel.value
  const curIdx = channels.indexOf(cur)
  if (curIdx < 0) return
  const nextCh = channels[(curIdx + 1) % channels.length]
  if (!nextCh || nextCh === cur) return
  if (channelCache.has(nextCh) || preloadingChannels.has(nextCh)) return
  preloadingChannels.add(nextCh)
  try {
    const v = await api.window(nextCh, 2048)
    if (v && v.data) {
      channelCache.set(nextCh, {
        data: v.data, gaps: v.gaps || [],
        threshold: v.threshold || 0.5,
        ts: Date.now(),
      })
    }
  } catch (e) {
    // Silent failure: prefetch must not affect the main flow
  } finally {
    preloadingChannels.delete(nextCh)
  }
}

// ── Channel-switch animation (vertical slide, 500ms; fires only on channel change)
// Single canvas + CSS: on switch the chart-wrapper does a translateY + opacity
// transition. The canvas content is already repainted for the new channel, and
// the transition sells the "new data slides in from below" effect.
const chartWrapperRef = ref<HTMLDivElement | null>(null)
let animTimer: ReturnType<typeof setTimeout> | null = null
function triggerChannelSwitchAnim() {
  const el = chartWrapperRef.value
  if (!el) return
  // Reset → force reflow → add the anim class (so the transition replays)
  el.classList.remove('phm-chart-switch-anim')
  // Force a reflow (reading offsetWidth triggers it)
  void el.offsetWidth
  el.classList.add('phm-chart-switch-anim')
  if (animTimer) clearTimeout(animTimer)
  animTimer = setTimeout(() => el.classList.remove('phm-chart-switch-anim'), 600)
}

// Mouse interaction
let mouseX = -1, mouseY = -1
let dpr = 1
let hoveredIdx = -1

// Device-tree sensor config (read yMin/yMax/unit/threshold)
interface SensorCfg { yMin: number; yMax: number; unit: string; threshold: number; name: string }
const sensorCfg = computed<SensorCfg | null>(() => {
  if (!store.deviceTree || !currentChannel.value) return null
  function walk(nodes: any[]): any | null {
    for (const n of nodes || []) {
      if (n.type === 'sensor' && n.channelName === currentChannel.value) return n
      if (n.children) {
        const r = walk(n.children)
        if (r) return r
      }
    }
    return null
  }
  const n = walk(store.deviceTree.device_tree)
  if (!n) return null
  return {
    yMin: n.yMin ?? -1, yMax: n.yMax ?? 1,
    unit: n.unit || '', threshold: n.threshold ?? 0.5,
    name: n.name || currentChannel.value,
  }
})

// Alert points for the current channel (for the telemetry red-dot annotations)
const currentChannelAlertPoints = computed(() => {
  if (!currentChannel.value) return []
  return alertPoints.value.filter(p => p.channel === currentChannel.value)
})

// Poll: telemetry window (2s, event-driven)
const windowPoll = usePoll(
  async () => {
    if (!currentChannel.value) return null
    return await api.window(currentChannel.value, 2048)
  },
  2000,
  { immediate: false, autoStart: false }
)

// Poll: alert points (3s, for the telemetry red dots + all-channel map)
const alertPoll = usePoll(
  async () => await api.alertPoints(),
  3000,
  { immediate: false, autoStart: false }
)

// Event-driven repaint (only redraw when the data signature changes)
let lastWindowSig = ''
let lastAlertSig = ''
let drawScheduled = false

function scheduleDraw() {
  if (drawScheduled) return
  drawScheduled = true
  requestAnimationFrame(() => {
    drawScheduled = false
    drawChart()
  })
}

watch(() => windowPoll.data.value, (v) => {
  if (!v || !v.data) return
  const sig = `${v.channel}|${v.count}|${v.data.length}|${v.data[v.data.length-1]?.timestamp ?? 0}`
  if (sig === lastWindowSig) return
  lastWindowSig = sig
  lastFullData.value = v.data
  gaps.value = v.gaps || []
  threshold.value = v.threshold || sensorCfg.value?.threshold || 0.5
  // Write the cache pool: so a later switch back to this channel fills instantly
  if (v.channel) {
    channelCache.set(v.channel, {
      data: v.data, gaps: v.gaps || [],
      threshold: v.threshold || sensorCfg.value?.threshold || 0.5,
      ts: Date.now(),
    })
    // Preload the next carousel channel (kills the switch blank — demo-facing visual optimisation)
    preloadNextCarousel()
  }
  scheduleDraw()
})

watch(() => alertPoll.data.value, (v) => {
  if (!v) return
  const red = v.red_points || []
  const sig = `${red.length}|${red[0]?.timestamp ?? 0}`
  if (sig === lastAlertSig) return
  lastAlertSig = sig
  alertPoints.value = red
  scheduleDraw()
})

watch(currentChannel, async (ch) => {
  lastWindowSig = ''
  if (ch) {
    // Channel switch: trigger the vertical-slide animation (only on switch, not on new data)
    triggerChannelSwitchAnim()
    // Cache hit → fill instantly (preload usually guarantees a hit, zero blank)
    const cached = channelCache.get(ch)
    if (cached) {
      lastFullData.value = cached.data
      gaps.value = cached.gaps
      threshold.value = cached.threshold || sensorCfg.value?.threshold || 0.5
    } else {
      // Cache miss → clear (do not keep the previous channel's data; avoids the "flash of another channel" bug).
      // During the animation the canvas is masked by the translate+opacity transition, so the blank is not jarring.
      lastFullData.value = []
      gaps.value = []
    }
    await windowPoll.tick()
  }
  scheduleDraw()
})

watch(sensorCfg, () => scheduleDraw())

// ── Canvas init ───────────────────────────────────────────────────────────
function initCanvas() {
  if (!canvasRef.value) return
  canvas = canvasRef.value
  ctx = canvas.getContext('2d')
  resizeCanvas()
}

function resizeCanvas() {
  if (!canvas || !ctx) return
  const parent = canvas.parentElement
  if (!parent) return
  dpr = window.devicePixelRatio || 1
  const w = parent.clientWidth
  const h = parent.clientHeight
  canvas.width = w * dpr
  canvas.height = h * dpr
  canvas.style.width = w + 'px'
  canvas.style.height = h + 'px'
  scheduleDraw()
}

// ── Time formatting ──────────────────────────────────────────────────────
function formatTime(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0')
}

// X-axis tick formatting with adaptive precision.
// At 100Hz over a 5.12s window, second-level ticks have no resolution, so we
// use SS.mmm (seconds.milliseconds). When the span is ≥ 10s it degrades to
// MM:SS (avoids seconds ≥ 60 looking confusing).
// spanSec = data time span (seconds); ts = epoch seconds.
function formatAxisTime(ts: number, spanSec: number): string {
  const d = new Date(ts * 1000)
  if (spanSec < 10) {
    // SS.mmm: seconds + milliseconds only, matching the data precision
    const ss = String(d.getSeconds()).padStart(2, '0')
    const mmm = String(d.getMilliseconds()).padStart(3, '0')
    return `${ss}.${mmm}`
  }
  // MM:SS
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  return `${mm}:${ss}`
}

function fmt(v: number | null | undefined, digits = 3): string {
  if (v == null || isNaN(v)) return '—'
  return Number(v).toFixed(digits)
}

// ── Main draw function (ported from the main branch's drawChart) ──────────
function drawChart() {
  const cv = canvas
  const c = ctx
  if (!cv || !c) return
  const w = cv.width, h = cv.height
  c.clearRect(0, 0, w, h)
  c.fillStyle = C.bg
  c.fillRect(0, 0, w, h)

  // Slice the visible window (last VIEW_COUNT rows)
  const fullData = lastFullData.value
  if (!fullData.length) {
    c.fillStyle = C.textSec
    c.font = `${14 * dpr}px sans-serif`
    c.textAlign = 'center'
    c.fillText('🚧 等待数据', w / 2, h / 2)
    return
  }
  const data = fullData.slice(-VIEW_COUNT)

  // Three region heights
  const topH = h * CHART_CFG.topRatio
  const midH = h * CHART_CFG.midRatio
  const botH = h * CHART_CFG.botRatio
  const pad = {
    top: CHART_CFG.padding.top * dpr,
    right: CHART_CFG.padding.right * dpr,
    bottom: CHART_CFG.padding.bottom * dpr,
    left: CHART_CFG.padding.left * dpr,
  }

  // ── X-axis time collapse ────────────────────────────────────────────────
  const tsMs = data.map((d: any) => (d.timestamp || 0) * 1000)
  const plotW = w - pad.left - pad.right
  const gapList = gaps.value
  const gapIndices: any[] = []
  for (const g of gapList) {
    const idx = data.findIndex((d: any) => Math.abs(d.timestamp * 1000 - g.end * 1000) < 1)
    if (idx > 0) gapIndices.push({ index: idx, duration_s: g.duration })
  }

  const GAP_W = CHART_CFG.gapWidthPx * dpr
  let totalGapW = gapIndices.length * GAP_W
  let dataW = plotW - totalGapW
  if (dataW < plotW * 0.3) dataW = plotW * 0.3

  const segments: any[] = []
  let segStart = 0
  for (const gi of gapIndices) {
    segments.push({ start: segStart, end: gi.index, dur: tsMs[gi.index - 1] - tsMs[segStart] })
    segStart = gi.index
  }
  segments.push({ start: segStart, end: data.length, dur: tsMs[data.length - 1] - tsMs[segStart] })
  const totalDur = segments.reduce((s, seg) => s + Math.max(seg.dur, 1), 0)

  const foldedX = new Array(data.length)
  let cursorX = pad.left
  let segIdx = 0
  for (let i = 0; i < data.length; i++) {
    if (segIdx < segments.length && i >= segments[segIdx].end) {
      cursorX += GAP_W
      segIdx++
    }
    const seg = segments[segIdx]
    const segW = dataW * Math.max(seg.dur, 1) / totalDur
    const frac = seg.dur > 0 ? (tsMs[i] - tsMs[seg.start]) / seg.dur : 0
    foldedX[i] = cursorX + frac * segW
    if (i === seg.end - 1) cursorX += segW
  }

  // Gap-detection threshold (3× the median interval)
  const gapThreshold = (() => {
    if (tsMs.length < 3) return Infinity
    const diffs: number[] = []
    for (let i = 1; i < tsMs.length; i++) {
      if (gapIndices.some(gi => gi.index === i)) continue
      diffs.push(tsMs[i] - tsMs[i - 1])
    }
    if (!diffs.length) return Infinity
    diffs.sort((a, b) => a - b)
    return diffs[Math.floor(diffs.length / 2)] * 3
  })()

  const xOf = (i: number) => foldedX[i]

  // Hover index
  hoveredIdx = -1
  if (mouseX >= pad.left && mouseX <= pad.left + plotW) {
    let best = 0, bestDist = Infinity
    for (let i = 0; i < foldedX.length; i++) {
      const d = Math.abs(foldedX[i] - mouseX)
      if (d < bestDist) { bestDist = d; best = i }
    }
    hoveredIdx = best
  }

  // ── Sub-chart draw function ─────────────────────────────────────────────
  function drawSubChart(y0: number, height: number, rawKey: string, predKey: string,
                        rawColor: string, predColor: string, thresholdVal: number | null,
                        fixedYRange: [number, number] | null) {
    // Force non-null assertion (drawChart above already checked)
    const c = ctx!
    const w = canvas!.width
    let minV: number, maxV: number
    if (fixedYRange) {
      minV = fixedYRange[0]; maxV = fixedYRange[1]
    } else {
      // Use the device-tree-configured yMin/yMax (no auto-ranging)
      minV = sensorCfg.value?.yMin ?? -1
      maxV = sensorCfg.value?.yMax ?? 1
    }
    const plotH = Math.max(10, height - pad.top - pad.bottom)
    const yScale = plotH / Math.max(0.001, maxV - minV)
    const yOf = (v: number) => y0 + pad.top + plotH - (v - minV) * yScale

    // Grid + Y-axis labels
    c.strokeStyle = C.border
    c.lineWidth = 0.5 * dpr
    c.fillStyle = C.textSec
    c.font = `${10 * dpr}px monospace`
    c.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      const y = y0 + pad.top + plotH * i / 4
      c.beginPath(); c.moveTo(pad.left, y); c.lineTo(w - pad.right, y); c.stroke()
      c.fillText((maxV - (maxV - minV) * i / 4).toFixed(3), pad.left - 6 * dpr, y + 3 * dpr)
    }

    // Measured solid line (breaks on null)
    c.beginPath()
    c.strokeStyle = rawColor
    c.lineWidth = 1.5 * dpr
    let rStarted = false, rPrevTs: number | null = null
    for (let i = 0; i < data.length; i++) {
      const v = data[i][rawKey]
      if (v == null) { rStarted = false; continue }
      const x = xOf(i), y = yOf(v)
      if (!rStarted || (rPrevTs != null && tsMs[i] - rPrevTs > gapThreshold)) {
        c.moveTo(x, y); rStarted = true
      } else {
        c.lineTo(x, y)
      }
      rPrevTs = tsMs[i]
    }
    c.stroke()

    // Predicted dashed line (breaks on null)
    c.beginPath()
    c.strokeStyle = predColor
    c.lineWidth = 1.5 * dpr
    c.setLineDash([5 * dpr, 3 * dpr])
    let pStarted = false, pPrevTs: number | null = null
    for (let i = 0; i < data.length; i++) {
      const v = data[i][predKey]
      if (v == null) { pStarted = false; continue }
      const x = xOf(i), y = yOf(v)
      if (!pStarted || (pPrevTs != null && tsMs[i] - pPrevTs > gapThreshold)) {
        c.moveTo(x, y); pStarted = true
      } else {
        c.lineTo(x, y)
      }
      pPrevTs = tsMs[i]
    }
    c.stroke()
    c.setLineDash([])

    // Threshold line
    if (thresholdVal != null) {
      c.beginPath()
      c.strokeStyle = C.red
      c.lineWidth = 1 * dpr
      c.setLineDash([4 * dpr, 4 * dpr])
      const yTh = yOf(thresholdVal)
      c.moveTo(pad.left, yTh); c.lineTo(w - pad.right, yTh)
      c.stroke()
      c.setLineDash([])
      // Threshold label
      c.fillStyle = C.red
      c.font = `${9 * dpr}px sans-serif`
      c.textAlign = 'left'
      c.fillText(`阈值 ${thresholdVal.toFixed(2)}`, pad.left + 4 * dpr, yTh - 4 * dpr)
    }
    return { y0, minV, maxV, yScale, plotH, yOf }
  }

  // ── Unified time→X mapping shared by every region (keeps red-dot X consistent)
  // Takes any timestamp (ms) and returns the collapsed X coordinate.
  // Algorithm: linear interpolation inside each segment; between segments
  // (gaps) it linearly extrapolates from the adjacent segment endpoints.
  function buildTsToX() {
    // Build a [tsMin, tsMax] → [xStart, xEnd] linear map for each segment
    const segMaps = segments.map((seg: any) => {
      const tsStart = tsMs[seg.start]
      const tsEnd = tsMs[Math.max(seg.start, seg.end - 1)]
      const xStart = foldedX[seg.start]
      const xEnd = foldedX[Math.max(seg.start, seg.end - 1)]
      return { tsStart, tsEnd, xStart, xEnd }
    })
    return (ts: number): number | null => {
      // Inside a segment → intra-segment linear interpolation
      for (const m of segMaps) {
        if (ts >= m.tsStart && ts <= m.tsEnd) {
          if (m.tsEnd === m.tsStart) return m.xStart
          const frac = (ts - m.tsStart) / (m.tsEnd - m.tsStart)
          return m.xStart + frac * (m.xEnd - m.xStart)
        }
      }
      // Inside a gap (between segments) → linear extrapolation from adjacent endpoints
      if (segMaps.length >= 2) {
        for (let i = 0; i < segMaps.length - 1; i++) {
          const cur = segMaps[i], next = segMaps[i + 1]
          if (ts > cur.tsEnd && ts < next.tsStart) {
            // Inside a gap: linearly interpolate between cur.xEnd and next.xStart (the gap occupies GAP_W)
            const tsSpan = next.tsStart - cur.tsEnd
            if (tsSpan <= 0) return cur.xEnd
            const frac = (ts - cur.tsEnd) / tsSpan
            return cur.xEnd + frac * (next.xStart - cur.xEnd)
          }
        }
        // Earlier than the first segment or later than the last → extrapolate
        if (ts < segMaps[0].tsStart) {
          const m = segMaps[0]
          if (m.tsEnd === m.tsStart) return m.xStart
          const frac = (ts - m.tsStart) / (m.tsEnd - m.tsStart)
          return m.xStart + frac * (m.xEnd - m.xStart)
        }
        const last = segMaps[segMaps.length - 1]
        if (last.tsEnd === last.tsStart) return last.xStart
        const frac = (ts - last.tsStart) / (last.tsEnd - last.tsStart)
        return last.xStart + frac * (last.xEnd - last.xStart)
      }
      return null
    }
  }
  const tsToX = buildTsToX()
  const dataTsMin = tsMs[0], dataTsMax = tsMs[tsMs.length - 1]

  // Three regions: telemetry / anomaly score / all-channel alert map
  const topLayout = drawSubChart(0, topH, 'raw_value', 'predicted_value', C.blue, C.green, null, null)
  const midLayout = drawSubChart(topH, midH, 'anomaly_score', 'predicted_anomaly_score', C.yellow, C.green, threshold.value, [0, 1])

  // ── Region titles (drawn inside the Canvas to avoid HTML being hidden behind it)
  function drawRegionTitle(y0: number, title: string, color: string, hint: string) {
    const cc = ctx!
    cc.font = `${11 * dpr}px sans-serif`
    cc.textAlign = 'left'
    cc.fillStyle = color
    cc.fillText(title, pad.left + 4 * dpr, y0 + 14 * dpr)
    cc.fillStyle = C.textSec
    cc.font = `${10 * dpr}px sans-serif`
    cc.fillText(hint, pad.left + 4 * dpr + title.length * 9 * dpr + 12 * dpr, y0 + 14 * dpr)
  }
  const unit = sensorCfg.value?.unit || ''
  drawRegionTitle(0, '遥测曲线', C.blue,
    `原值(蓝实) 预测值(绿虚)${unit ? ' 单位:' + unit : ''} 告警时刻(红点)` +
    (sensorCfg.value ? ` Y:[${sensorCfg.value.yMin.toFixed(1)},${sensorCfg.value.yMax.toFixed(1)}]` : ''))
  drawRegionTitle(topH, '异常分数', C.yellow, '实测(黄实) 预测(绿虚) 阈值(红虚)')
  drawRegionTitle(topH + midH, '全通道告警点图', C.red, '实测告警(红) 预测预警(黄)')

  // ── Alert-time red dots: marked in BOTH the telemetry region and the anomaly-score region
  // Both regions draw red circles at the alert moment and share the same X (tsToX).
  // ★ Key fix (3.3a): the original only drew red dots in the telemetry region; the score curve had none.
  // ★ Key fix (3.3e): all three red-dot radii unified to 4*dpr (telemetry / score / all-channel map).
  // ★ Key fix (3.3b/c): the timestamp is now the real sample time (acq_ts), same origin as the telemetry axis.
  // The coordinate maths lives in utils/alertGeometry.ts (pure, unit-testable); here we only draw.
  function drawAlertDot(x: number, y: number) {
    if (!c) return
    c.beginPath()
    c.fillStyle = 'rgba(245, 108, 108, 0.3)'
    c.arc(x, y, 8 * dpr, 0, Math.PI * 2)  // halo (visual emphasis only; not counted toward "size")
    c.fill()
    c.beginPath()
    c.fillStyle = C.red
    c.arc(x, y, 4 * dpr, 0, Math.PI * 2)  // solid red dot (unified radius across all three regions)
    c.fill()
  }
  const dots = computeRedDots(
    data, currentChannelAlertPoints.value, tsToX,
    (v) => topLayout.yOf(v), (v) => midLayout.yOf(v),
  )
  for (const d of dots) drawAlertDot(d.x, d.y)

  // ── Third region: all-channel alert map ─────────────────────────────────
  drawAlertScatter(topH + midH, botH, tsToX, dataTsMin, dataTsMax)

  // ── Gap vertical dashed line + "中断Xh" label ───────────────────────────
  for (const gi of gapIndices) {
    const xGap = foldedX[gi.index] - GAP_W / 2
    c.strokeStyle = C.textSec
    c.lineWidth = 1 * dpr
    c.setLineDash([4 * dpr, 4 * dpr])
    c.beginPath()
    c.moveTo(xGap, pad.top)
    c.lineTo(xGap, topH + midH + botH - pad.bottom)
    c.stroke()
    c.setLineDash([])
    const dur_h = gi.duration_s / 3600
    const label = dur_h >= 1 ? `中断 ${dur_h.toFixed(1)}h` : `中断 ${(gi.duration_s / 60).toFixed(0)}min`
    c.fillStyle = C.textSec
    c.font = `${9 * dpr}px sans-serif`
    c.textAlign = 'center'
    c.fillText(label, xGap, pad.top - 4 * dpr)
  }

  // ── Data-not-arrived spans (pred-only contiguous ≥ 5 points) ──────────────
  const predOnlyRanges: any[] = []
  let poStart = -1
  for (let i = 0; i < data.length; i++) {
    const hasPred = data[i].predicted_value != null
    const hasRaw = data[i].raw_value != null
    if (hasPred && !hasRaw) {
      if (poStart < 0) poStart = i
    } else {
      if (poStart >= 0 && i - poStart >= 5) predOnlyRanges.push({ start: poStart, end: i - 1 })
      poStart = -1
    }
  }
  if (poStart >= 0 && data.length - poStart >= 5) predOnlyRanges.push({ start: poStart, end: data.length - 1 })
  for (const r of predOnlyRanges) {
    const x1 = foldedX[r.start], x2 = foldedX[r.end]
    c.fillStyle = 'rgba(245, 108, 108, 0.08)'
    c.fillRect(x1, pad.top, x2 - x1, topH + midH - pad.top - pad.bottom)
    c.strokeStyle = C.red
    c.lineWidth = 1.5 * dpr
    c.setLineDash([3 * dpr, 3 * dpr])
    c.beginPath()
    c.moveTo(x1, pad.top); c.lineTo(x1, topH + midH - pad.bottom)
    c.moveTo(x2, pad.top); c.lineTo(x2, topH + midH - pad.bottom)
    c.stroke()
    c.setLineDash([])
    c.fillStyle = C.red
    c.font = `${9 * dpr}px sans-serif`
    c.textAlign = 'center'
    c.fillText('数据未到达', (x1 + x2) / 2, pad.top - 4 * dpr)
  }

  // ── X-axis time ticks (in the bottom pad.bottom band) ─────────────────────
  // At 100Hz over a 5.12s window, use SS.mmm (seconds.milliseconds) to match the data precision.
  // Adaptive tick count: roughly one per 80px, at least 3. Uses tsToX (with gap collapse).
  {
    const axisY = topH + midH + botH - pad.bottom  // bottom edge of the chart area
    const labelY = axisY + 14 * dpr                 // tick-label baseline (inside pad.bottom)
    const spanSec = (dataTsMax - dataTsMin) / 1000   // data span (seconds)
    const tickCount = Math.max(3, Math.round(plotW / (80 * dpr)))
    c.strokeStyle = C.border
    c.lineWidth = 0.5 * dpr
    c.fillStyle = C.textSec
    c.font = `${9 * dpr}px sans-serif`
    c.textAlign = 'center'
    for (let i = 0; i <= tickCount; i++) {
      const tsMsTick = dataTsMin + (dataTsMax - dataTsMin) * i / tickCount
      const x = tsToX(tsMsTick)
      if (x == null) continue
      // Short tick line
      c.beginPath()
      c.moveTo(x, axisY)
      c.lineTo(x, axisY + 4 * dpr)
      c.stroke()
      // Time label (SS.mmm)
      c.fillText(formatAxisTime(tsMsTick / 1000, spanSec), x, labelY)
    }
  }

  // ── Hover crosshair + tooltip ───────────────────────────────────────────
  if (hoveredIdx >= 0 && tooltipRef.value) {
    const i = hoveredIdx
    const d = data[i]
    const x = xOf(i)
    c.strokeStyle = C.textSec
    c.lineWidth = 1 * dpr
    c.setLineDash([3 * dpr, 3 * dpr])
    c.beginPath()
    c.moveTo(x, pad.top)
    c.lineTo(x, topH + midH + botH - pad.bottom)
    c.stroke()
    c.setLineDash([])
    // Highlight points
    if (d.raw_value != null) {
      c.fillStyle = C.blue
      c.beginPath(); c.arc(x, topLayout.yOf(d.raw_value), 4 * dpr, 0, Math.PI * 2); c.fill()
    }
    if (d.predicted_value != null) {
      c.fillStyle = C.green
      c.beginPath(); c.arc(x, topLayout.yOf(d.predicted_value), 4 * dpr, 0, Math.PI * 2); c.fill()
    }
    if (d.anomaly_score != null) {
      c.fillStyle = C.yellow
      c.beginPath(); c.arc(x, midLayout.yOf(d.anomaly_score), 4 * dpr, 0, Math.PI * 2); c.fill()
    }
    // tooltip
    const tooltip = tooltipRef.value
    tooltip.style.display = 'block'
    const unit = sensorCfg.value?.unit || ''
    let html = `<div class="tt-label">${formatTime(d.timestamp)}</div>`
    if (d.raw_value != null) html += `<div>遥测: <span class="tt-val">${fmt(d.raw_value, 4)}${unit ? ' ' + unit : ''}</span></div>`
    if (d.predicted_value != null) html += `<div style="color:${C.green}">预测: <span class="tt-val">${fmt(d.predicted_value, 4)}</span></div>`
    if (d.anomaly_score != null) html += `<div>分数: <span class="tt-val">${fmt(d.anomaly_score, 4)}</span></div>`
    if (d.predicted_anomaly_score != null) html += `<div style="color:${C.green}">预测分数: <span class="tt-val">${fmt(d.predicted_anomaly_score, 4)}</span></div>`
    tooltip.innerHTML = html
    const cssX = x / dpr + 12
    const cssY = mouseY / dpr + 12
    const wrapW = cv.parentElement?.clientWidth || 800
    tooltip.style.left = (cssX + 150 > wrapW ? cssX - 160 : cssX) + 'px'
    tooltip.style.top = Math.max(0, cssY) + 'px'
  } else if (tooltipRef.value) {
    tooltipRef.value.style.display = 'none'
  }
}

// ── All-channel alert map (third region, native drawing) ─────────────────────
// ★ Key fix: receives the tsToX function (shared with the telemetry region and
//   red dots) so the three regions' alert-point X coordinates align exactly
//   with the telemetry curve (including gap collapse).
function drawAlertScatter(
  y0: number,
  height: number,
  tsToX: (ts: number) => number | null,
  dataTsMin: number,
  dataTsMax: number,
) {
  const cv = canvas, c = ctx
  if (!cv || !c) return
  const w = cv.width
  const pad = {
    top: CHART_CFG.padding.top * dpr,
    right: CHART_CFG.padding.right * dpr,
    bottom: CHART_CFG.padding.bottom * dpr,
    left: CHART_CFG.padding.left * dpr,
  }
  const plotH = Math.max(10, height - pad.top - pad.bottom)

  // All-channel list (Y-axis categories, deduped, order-preserving, using display names)
  const seenCh = new Set<string>()
  const channels: string[] = []
  for (const p of alertPoints.value) {
    const disp = store.displayName(p.channel)
    if (!seenCh.has(disp)) { seenCh.add(disp); channels.push(disp) }
  }

  // Draw the Y-axis category labels
  c.strokeStyle = C.border
  c.lineWidth = 0.5 * dpr
  c.fillStyle = C.textSec
  c.font = `${10 * dpr}px sans-serif`
  c.textAlign = 'right'
  for (let i = 0; i < channels.length; i++) {
    const y = y0 + pad.top + plotH * (i + 0.5) / Math.max(channels.length, 1)
    c.fillText(channels[i], pad.left - 6 * dpr, y + 3 * dpr)
  }

  // Draw points (using the unified tsToX mapping, aligned exactly with the telemetry red dots)
  for (const p of alertPoints.value) {
    const tsMsPt = p.timestamp * 1000
    if (tsMsPt < dataTsMin || tsMsPt > dataTsMax) continue
    const x = tsToX(tsMsPt)
    if (x == null) continue
    const disp = store.displayName(p.channel)
    const chIdx = channels.indexOf(disp)
    if (chIdx < 0) continue
    const y = y0 + pad.top + plotH * (chIdx + 0.5) / Math.max(channels.length, 1)
    // ★ 3.3e: the three red-dot radii are unified to 4*dpr (same as the
    //   telemetry/score regions); no more the 8+score*8 gradient size (which
    //   used to make the all-channel map dots far larger than the other two regions).
    const size = 4 * dpr
    c.beginPath()
    c.fillStyle = p.type === 'measured' ? 'rgba(245, 108, 108, 0.85)' : 'rgba(230, 162, 60, 0.85)'
    c.arc(x, y, size, 0, Math.PI * 2)
    c.fill()
  }
}

// ── Mouse interaction ─────────────────────────────────────────────────────
function handleMouseMove(e: MouseEvent) {
  const cv = canvas
  if (!cv) return
  const rect = cv.getBoundingClientRect()
  mouseX = (e.clientX - rect.left) * dpr
  mouseY = (e.clientY - rect.top) * dpr
  scheduleDraw()
}
function handleMouseLeave() {
  mouseX = -1; mouseY = -1; hoveredIdx = -1
  if (tooltipRef.value) tooltipRef.value.style.display = 'none'
}

function handleResize() {
  resizeCanvas()
}

onMounted(async () => {
  await nextTick()
  initCanvas()
  window.addEventListener('resize', handleResize)
  windowPoll.start()
  alertPoll.start()
  if (currentChannel.value) await windowPoll.tick()
  await alertPoll.tick()
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<template>
  <div class="chart-wrapper" ref="chartWrapperRef">
    <canvas
      ref="canvasRef"
      class="chart-canvas"
      @mousemove="handleMouseMove"
      @mouseleave="handleMouseLeave"
    ></canvas>
    <div ref="tooltipRef" class="chart-tooltip"></div>
  </div>
</template>

<style scoped>
.chart-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  background: #0f1530;
  overflow: hidden;
}

/* Channel-switch vertical-slide animation (500ms): the old content slides up
   and fades out while the new content slides in from below.
   Single-canvas approach: the canvas is already repainted for the new channel,
   and the whole wrapper does a translate+opacity transition to sell the
   "new channel data slides in from below" effect (direction matches the
   left-side sensor vertical carousel). */
@keyframes phm-chart-slide-in {
  0%   { transform: translateY(40px); opacity: 0; }
  100% { transform: translateY(0);    opacity: 1; }
}
.chart-wrapper.phm-chart-switch-anim {
  animation: phm-chart-slide-in 0.5s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}

.chart-canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.chart-tooltip {
  position: absolute;
  display: none;
  background: rgba(15, 21, 48, 0.95);
  border: 1px solid #2a3050;
  border-radius: 4px;
  padding: 6px 10px;
  font-size: 11px;
  color: #e0e6ed;
  pointer-events: none;
  z-index: 10;
  white-space: nowrap;
  font-family: 'Consolas', monospace;
}

:deep(.tt-label) {
  color: #7a85a8;
  font-size: 10px;
  margin-bottom: 2px;
}

:deep(.tt-val) {
  color: #409eff;
  font-weight: 500;
}
</style>
