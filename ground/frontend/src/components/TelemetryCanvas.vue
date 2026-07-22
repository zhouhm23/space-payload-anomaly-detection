<script setup lang="ts">
/**
 * 遥测图表（原生 Canvas 实现，照搬主分支 monitor.js drawChart + 新增事件红点）。
 *
 * 三区纵向布局：遥测区:异常分数区:全通道告警点图区 = 4:1:2
 *
 * 核心特性（主分支已验证）：
 * 1. 时间轴折叠（gap 压成固定宽度，避免时间轴爆炸）
 * 2. 实测实线 + 预测虚线（null 点断开，不连接）
 * 3. Y 轴固定（遥测用设备树 yMin/yMax，分数固定 [0,1]）
 * 4. 阈值线（异常分数区红色虚线）
 * 5. 缺口标注（灰色竖虚线 + "中断Xh"）
 * 6. 数据未到达（pred-only 连续段，红色半透明矩形）
 *
 * 新增（demo 没有）：
 * 7. 事件开始红点：告警时刻在遥测图上画红色圆点（alert_points 提供）
 */
import { computed, ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useSystemStore } from '@/stores/system'
import { api } from '@/api'
import { usePoll } from '@/composables/usePoll'
import { computeRedDots } from '@/utils/alertGeometry'

const store = useSystemStore()
const currentChannel = computed(() => store.currentChannel)

// Canvas 引用
const canvasRef = ref<HTMLCanvasElement | null>(null)
const tooltipRef = ref<HTMLDivElement | null>(null)
// 用普通变量（非 ref），初始化后不再变。drawChart 内闭包能正确推断非空。
let canvas: HTMLCanvasElement | null = null
let ctx: CanvasRenderingContext2D | null = null

// 配色（对齐主分支）
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

// 配置
const CHART_CFG = {
  topRatio: 4 / 7,        // 遥测区 4 份
  midRatio: 1 / 7,        // 异常分数区 1 份
  botRatio: 2 / 7,        // 全通道告警点图区 2 份
  padding: { top: 16, right: 16, bottom: 24, left: 50 },
  gapWidthPx: 40,
}
const VIEW_COUNT = 512  // 可视窗口点数

// 数据状态
const lastFullData = ref<any[]>([])
const gaps = ref<any[]>([])
const threshold = ref<number>(0.5)
const alertPoints = ref<any[]>([])  // 全通道告警点（用于点图 + 遥测图红点）

// 通道数据缓存池：切换通道时先用缓存瞬时填充，消除"等待数据"空窗。
// 业界标准做法（demo 亦有）——通道数有限（通常 < 20），无容量风险。
// 缓存写入时机：每次 windowPoll 拉到新数据；读取时机：切换通道瞬间。
interface CachedWindow { data: any[]; gaps: any[]; threshold: number; ts: number }
const channelCache = new Map<string, CachedWindow>()

// ── 预加载：当前通道数据到达后，异步预取下一个轮播通道 ──────────────────
// 面向领导展示的视觉优化：切到下一通道时缓存必然命中，零空窗、零"加载中"。
// 预取是 fire-and-forget，失败静默（缓存未命中会 fallback 到正常 tick 拉取）。
let preloadingChannels = new Set<string>()  // 防重复预取同一通道
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
    // 静默失败：预取不影响主流程
  } finally {
    preloadingChannels.delete(nextCh)
  }
}

// ── 通道切换动画（垂直滑动 500ms，仅换通道触发） ──────────────────────────
// 单 canvas + CSS：切换时 chart-wrapper 做 translateY + opacity 过渡，
// canvas 内容已重画为新通道，过渡期间给出"新数据从下方滑入"的视觉。
const chartWrapperRef = ref<HTMLDivElement | null>(null)
let animTimer: ReturnType<typeof setTimeout> | null = null
function triggerChannelSwitchAnim() {
  const el = chartWrapperRef.value
  if (!el) return
  // 重置 → 触发回流 → 加 anim 类（确保 transition 重新播放）
  el.classList.remove('phm-chart-switch-anim')
  // 强制回流（offsetWidth 读取触发 reflow）
  void el.offsetWidth
  el.classList.add('phm-chart-switch-anim')
  if (animTimer) clearTimeout(animTimer)
  animTimer = setTimeout(() => el.classList.remove('phm-chart-switch-anim'), 600)
}

// 鼠标交互
let mouseX = -1, mouseY = -1
let dpr = 1
let hoveredIdx = -1

// 设备树传感器配置（取 yMin/yMax/unit/threshold）
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

// 当前通道的告警点（用于遥测图红点标注）
const currentChannelAlertPoints = computed(() => {
  if (!currentChannel.value) return []
  return alertPoints.value.filter(p => p.channel === currentChannel.value)
})

// 轮询：遥测窗口（2s，事件驱动）
const windowPoll = usePoll(
  async () => {
    if (!currentChannel.value) return null
    return await api.window(currentChannel.value, 2048)
  },
  2000,
  { immediate: false, autoStart: false }
)

// 轮询：告警点（3s，用于遥测红点 + 全通道点图）
const alertPoll = usePoll(
  async () => await api.alertPoints(),
  3000,
  { immediate: false, autoStart: false }
)

// 事件驱动重绘（数据签名变化才重画）
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
  // 写入缓存池：供下次切回该通道时瞬时填充
  if (v.channel) {
    channelCache.set(v.channel, {
      data: v.data, gaps: v.gaps || [],
      threshold: v.threshold || sensorCfg.value?.threshold || 0.5,
      ts: Date.now(),
    })
    // 预加载下一个轮播通道（消除切换空窗，面向领导展示的视觉优化）
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
    // 通道切换：触发垂直滑动动画（仅换通道才动画，新数据到达不动画）
    triggerChannelSwitchAnim()
    // 缓存命中→瞬时填充（预加载保证通常命中，零空窗）
    const cached = channelCache.get(ch)
    if (cached) {
      lastFullData.value = cached.data
      gaps.value = cached.gaps
      threshold.value = cached.threshold || sensorCfg.value?.threshold || 0.5
    } else {
      // 缓存未命中→清空（不再保留前一通道数据，避免"闪现其他通道" bug）
      // 动画期间 canvas 被位移+透明度过渡盖住，留白不突兀
      lastFullData.value = []
      gaps.value = []
    }
    await windowPoll.tick()
  }
  scheduleDraw()
})

watch(sensorCfg, () => scheduleDraw())

// ── Canvas 初始化 ───────────────────────────────────────────────────────────
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

// ── 时间格式化 ──────────────────────────────────────────────────────────────
function formatTime(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0')
}

// 横坐标刻度格式化：自适应精度。
// 采样率 100Hz、数据窗 5.12s，秒级刻度无区分度，故用 SS.mmm（秒.毫秒）。
// 跨度 ≥ 10s 时退化为 MM:SS（避免秒数超过 60 显示混乱）。
// spanSec = 数据时间跨度（秒），ts = epoch 秒。
function formatAxisTime(ts: number, spanSec: number): string {
  const d = new Date(ts * 1000)
  if (spanSec < 10) {
    // SS.mmm：仅取秒+毫秒，与数据精度匹配
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

// ── 主绘制函数（照搬主分支 drawChart） ──────────────────────────────────────
function drawChart() {
  const cv = canvas
  const c = ctx
  if (!cv || !c) return
  const w = cv.width, h = cv.height
  c.clearRect(0, 0, w, h)
  c.fillStyle = C.bg
  c.fillRect(0, 0, w, h)

  // 截取可视窗口（最后 VIEW_COUNT 行）
  const fullData = lastFullData.value
  if (!fullData.length) {
    c.fillStyle = C.textSec
    c.font = `${14 * dpr}px sans-serif`
    c.textAlign = 'center'
    c.fillText('🚧 等待数据', w / 2, h / 2)
    return
  }
  const data = fullData.slice(-VIEW_COUNT)

  // 三区高度
  const topH = h * CHART_CFG.topRatio
  const midH = h * CHART_CFG.midRatio
  const botH = h * CHART_CFG.botRatio
  const pad = {
    top: CHART_CFG.padding.top * dpr,
    right: CHART_CFG.padding.right * dpr,
    bottom: CHART_CFG.padding.bottom * dpr,
    left: CHART_CFG.padding.left * dpr,
  }

  // ── X 轴时间折叠 ──────────────────────────────────────────────────────────
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

  // 缺口检测阈值（3 倍中位数间隔）
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

  // 悬停索引
  hoveredIdx = -1
  if (mouseX >= pad.left && mouseX <= pad.left + plotW) {
    let best = 0, bestDist = Infinity
    for (let i = 0; i < foldedX.length; i++) {
      const d = Math.abs(foldedX[i] - mouseX)
      if (d < bestDist) { bestDist = d; best = i }
    }
    hoveredIdx = best
  }

  // ── 子图绘制函数 ──────────────────────────────────────────────────────────
  function drawSubChart(y0: number, height: number, rawKey: string, predKey: string,
                        rawColor: string, predColor: string, thresholdVal: number | null,
                        fixedYRange: [number, number] | null) {
    // 强制断言非空（外层 drawChart 已检查过）
    const c = ctx!
    const w = canvas!.width
    let minV: number, maxV: number
    if (fixedYRange) {
      minV = fixedYRange[0]; maxV = fixedYRange[1]
    } else {
      // 用设备树配置的 yMin/yMax（不自动量程）
      minV = sensorCfg.value?.yMin ?? -1
      maxV = sensorCfg.value?.yMax ?? 1
    }
    const plotH = Math.max(10, height - pad.top - pad.bottom)
    const yScale = plotH / Math.max(0.001, maxV - minV)
    const yOf = (v: number) => y0 + pad.top + plotH - (v - minV) * yScale

    // 网格 + 纵轴标签
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

    // 实测实线（null 断开）
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

    // 预测虚线（null 断开）
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

    // 阈值线
    if (thresholdVal != null) {
      c.beginPath()
      c.strokeStyle = C.red
      c.lineWidth = 1 * dpr
      c.setLineDash([4 * dpr, 4 * dpr])
      const yTh = yOf(thresholdVal)
      c.moveTo(pad.left, yTh); c.lineTo(w - pad.right, yTh)
      c.stroke()
      c.setLineDash([])
      // 阈值标签
      c.fillStyle = C.red
      c.font = `${9 * dpr}px sans-serif`
      c.textAlign = 'left'
      c.fillText(`阈值 ${thresholdVal.toFixed(2)}`, pad.left + 4 * dpr, yTh - 4 * dpr)
    }
    return { y0, minV, maxV, yScale, plotH, yOf }
  }

  // ── 统一时间→X 映射函数（所有区域共用，保证红点 x 轴一致） ────────────────
  // 输入任意时间戳（毫秒），返回折叠后的 X 坐标。
  // 算法：对 segments 内每个段做线性插值；段间（gap）用前后段端点线性外推。
  function buildTsToX() {
    // 为每个段建立 [tsMin, tsMax] → [xStart, xEnd] 的线性映射
    const segMaps = segments.map((seg: any) => {
      const tsStart = tsMs[seg.start]
      const tsEnd = tsMs[Math.max(seg.start, seg.end - 1)]
      const xStart = foldedX[seg.start]
      const xEnd = foldedX[Math.max(seg.start, seg.end - 1)]
      return { tsStart, tsEnd, xStart, xEnd }
    })
    return (ts: number): number | null => {
      // 在某段范围内 → 段内线性插值
      for (const m of segMaps) {
        if (ts >= m.tsStart && ts <= m.tsEnd) {
          if (m.tsEnd === m.tsStart) return m.xStart
          const frac = (ts - m.tsStart) / (m.tsEnd - m.tsStart)
          return m.xStart + frac * (m.xEnd - m.xStart)
        }
      }
      // 在 gap 范围内（段间）→ 用相邻段端点线性外推
      if (segMaps.length >= 2) {
        for (let i = 0; i < segMaps.length - 1; i++) {
          const cur = segMaps[i], next = segMaps[i + 1]
          if (ts > cur.tsEnd && ts < next.tsStart) {
            // gap 内：在 cur.xEnd 和 next.xStart 之间线性插值（gap 占 GAP_W）
            const tsSpan = next.tsStart - cur.tsEnd
            if (tsSpan <= 0) return cur.xEnd
            const frac = (ts - cur.tsEnd) / tsSpan
            return cur.xEnd + frac * (next.xStart - cur.xEnd)
          }
        }
        // 早于第一段或晚于最后一段 → 外推
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

  // 三区：遥测 / 异常分数 / 全通道告警点图
  const topLayout = drawSubChart(0, topH, 'raw_value', 'predicted_value', C.blue, C.green, null, null)
  const midLayout = drawSubChart(topH, midH, 'anomaly_score', 'predicted_anomaly_score', C.yellow, C.green, threshold.value, [0, 1])

  // ── 三区标题（Canvas 内部绘制，避免 HTML 定位被 Canvas 遮挡） ────────────
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

  // ── 告警时刻红点：遥测区 + 异常分数区双标 ──────────────────────────────────
  // 在遥测区和异常分数区都画红色圆点标注告警发生时刻，两区共享同一 X（tsToX）。
  // ★ 关键修复（3.3a）：原版只在遥测区画红点，异常分数曲线无标注。
  // ★ 关键修复（3.3e）：三处红点半径统一为 4*dpr（遥测区/分数区/全通道点图）。
  // ★ 关键修复（3.3b/c）：timestamp 现在是真实采样时刻（acq_ts），与遥测轴同源。
  // 坐标计算抽到 utils/alertGeometry.ts（纯函数，可单测）；这里只负责绘制。
  function drawAlertDot(x: number, y: number) {
    if (!c) return
    c.beginPath()
    c.fillStyle = 'rgba(245, 108, 108, 0.3)'
    c.arc(x, y, 8 * dpr, 0, Math.PI * 2)  // 光晕（仅视觉强调，不计入"大小"）
    c.fill()
    c.beginPath()
    c.fillStyle = C.red
    c.arc(x, y, 4 * dpr, 0, Math.PI * 2)  // 实心红点（三区统一半径）
    c.fill()
  }
  const dots = computeRedDots(
    data, currentChannelAlertPoints.value, tsToX,
    (v) => topLayout.yOf(v), (v) => midLayout.yOf(v),
  )
  for (const d of dots) drawAlertDot(d.x, d.y)

  // ── 第三区：全通道告警点图 ────────────────────────────────────────────────
  drawAlertScatter(topH + midH, botH, tsToX, dataTsMin, dataTsMax)

  // ── 缺口竖虚线 + "中断Xh" 标注 ────────────────────────────────────────────
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

  // ── 数据未到达区间（pred-only 连续 ≥5 点） ────────────────────────────────
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

  // ── 横坐标时间刻度（底部 pad.bottom 区域） ────────────────────────────────
  // 采样率 100Hz、窗 5.12s，用 SS.mmm（秒.毫秒）格式匹配数据精度。
  // 刻度数自适应：每约 80px 一个，最少 3 个。用 tsToX 映射（含 gap 折叠）。
  {
    const axisY = topH + midH + botH - pad.bottom  // 图表区底边
    const labelY = axisY + 14 * dpr                 // 刻度文字基线（pad.bottom 内）
    const spanSec = (dataTsMax - dataTsMin) / 1000   // 数据跨度（秒）
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
      // 短刻度线
      c.beginPath()
      c.moveTo(x, axisY)
      c.lineTo(x, axisY + 4 * dpr)
      c.stroke()
      // 时间标签（SS.mmm）
      c.fillText(formatAxisTime(tsMsTick / 1000, spanSec), x, labelY)
    }
  }

  // ── 悬停十字准线 + tooltip ────────────────────────────────────────────────
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
    // 高亮点
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

// ── 全通道告警点图（第三区，原生绘制） ────────────────────────────────────────
// ★ 关键修复：接收 tsToX 函数（与遥测区/红点共用同一时间→X 映射），
//   保证三区告警点的 x 坐标与遥测曲线完全对齐（含 gap 折叠）。
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

  // 全通道列表（Y 轴分类，去重保序，用显示名）
  const seenCh = new Set<string>()
  const channels: string[] = []
  for (const p of alertPoints.value) {
    const disp = store.displayName(p.channel)
    if (!seenCh.has(disp)) { seenCh.add(disp); channels.push(disp) }
  }

  // 画 Y 轴分类标签
  c.strokeStyle = C.border
  c.lineWidth = 0.5 * dpr
  c.fillStyle = C.textSec
  c.font = `${10 * dpr}px sans-serif`
  c.textAlign = 'right'
  for (let i = 0; i < channels.length; i++) {
    const y = y0 + pad.top + plotH * (i + 0.5) / Math.max(channels.length, 1)
    c.fillText(channels[i], pad.left - 6 * dpr, y + 3 * dpr)
  }

  // 画点（用统一的 tsToX 映射，与遥测区红点完全对齐）
  for (const p of alertPoints.value) {
    const tsMsPt = p.timestamp * 1000
    if (tsMsPt < dataTsMin || tsMsPt > dataTsMax) continue
    const x = tsToX(tsMsPt)
    if (x == null) continue
    const disp = store.displayName(p.channel)
    const chIdx = channels.indexOf(disp)
    if (chIdx < 0) continue
    const y = y0 + pad.top + plotH * (chIdx + 0.5) / Math.max(channels.length, 1)
    // ★ 3.3e：三处红点半径统一为 4*dpr（与遥测区/分数区一致），
    //   不再用 8+score*8 的渐变大小（之前全通道点图红点比另两区大太多）。
    const size = 4 * dpr
    c.beginPath()
    c.fillStyle = p.type === 'measured' ? 'rgba(245, 108, 108, 0.85)' : 'rgba(230, 162, 60, 0.85)'
    c.arc(x, y, size, 0, Math.PI * 2)
    c.fill()
  }
}

// ── 鼠标交互 ────────────────────────────────────────────────────────────────
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

/* 通道切换垂直滑动动画（500ms）：旧内容上移淡出感、新内容从下方滑入。
   单 canvas 方案：canvas 内容已重画为新通道，wrapper 整体做位移+透明度过渡，
   给出"新通道数据从下方滑入"的视觉（与左侧传感器上下轮播方向一致）。 */
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
