<script setup lang="ts">
/**
 * 设备示意图（Slice 1）— 机箱轮廓 + 模块分区 + 传感器点位（健康度着色）。
 *
 * 与 dashboard.html 的 drawDeviceDiagram 逻辑等价，适配 Vue 响应式 + rAF 生命周期。
 * 选中态通过 deviceTree.selectedId（传感器）/ selectedFolderId（文件夹）双向联动。
 */
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { useHealthStore } from '@/stores/health'
import type { DeviceNode } from '@/api/types'

const tree = useDeviceTreeStore()
const health = useHealthStore()

const dom = ref<HTMLDivElement | null>(null)
let canvas: HTMLCanvasElement | null = null
let ctx: CanvasRenderingContext2D | null = null
let rafId = 0
let cssW = 0
let cssH = 0
const DPR = Math.min(window.devicePixelRatio || 1, 2)

// 颜色常量（Canvas 不支持 var()）
const C = {
  bgSecondary: '#131825',
  border: '#2a3348',
  textPri: '#e0e6f0',
  textSec: '#8e9bb5',
  blue: '#2d8cf0',
  green: '#19be6b',
  yellow: '#f5a623',
  red: '#ed3f14',
}

// hit-test 缓存
let diagramLayout: { dots: { node: DeviceNode; x: number; y: number; r: number }[] } | null = null
// tooltip
const tooltipVisible = ref(false)
const tooltipText = ref('')
const tooltipX = ref(0)
const tooltipY = ref(0)

// ---- 数据访问（响应式快照，避免在 rAF 热循环里读 store）----
function flatSensors(): DeviceNode[] {
  const out: DeviceNode[] = []
  const walk = (nodes: DeviceNode[]) => {
    for (const n of nodes) {
      if (n.sourceId) out.push(n)
      if (n.children) walk(n.children)
    }
  }
  walk(tree.tree)
  return out
}

function groupByModule(): Record<string, DeviceNode[]> {
  const groups: Record<string, DeviceNode[]> = {}
  const ungrouped: DeviceNode[] = []
  for (const s of flatSensors()) {
    const mod = s.position?.module
    if (mod) {
      if (!groups[mod]) groups[mod] = []
      groups[mod].push(s)
    } else ungrouped.push(s)
  }
  if (ungrouped.length) groups['未分组'] = ungrouped
  return groups
}

function gridDims(n: number) {
  if (n <= 0) return { cols: 1, rows: 1 }
  const cols = Math.ceil(Math.sqrt(n))
  return { cols, rows: Math.ceil(n / cols) }
}

function healthColorHex(h: number | null | undefined): string {
  if (h == null) return C.textSec
  if (h < 60) return C.red
  if (h < 80) return C.yellow
  return C.green
}

function sensorHealth(ch: string | undefined): number | null {
  if (!ch) return null
  const s = health.sensors.find((x) => x.channel === ch)
  return s ? s.health : null
}

function findFolderIdByName(name: string): string | null {
  const walk = (nodes: DeviceNode[]): string | null => {
    for (const n of nodes) {
      if (n.name === name && (n.type === 'folder' || n.children)) return n.id
      if (n.children) {
        const r = walk(n.children)
        if (r) return r
      }
    }
    return null
  }
  return walk(tree.tree)
}

function roundRect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  c.beginPath()
  c.moveTo(x + r, y)
  c.lineTo(x + w - r, y)
  c.quadraticCurveTo(x + w, y, x + w, y + r)
  c.lineTo(x + w, y + h - r)
  c.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  c.lineTo(x + r, y + h)
  c.quadraticCurveTo(x, y + h, x, y + h - r)
  c.lineTo(x, y + r)
  c.quadraticCurveTo(x, y, x + r, y)
  c.closePath()
}

// ---- 绘制 ----
function draw() {
  if (!ctx || !canvas) return
  const dc = ctx
  const w = canvas.width
  const h = canvas.height
  if (w === 0 || h === 0) return
  dc.clearRect(0, 0, w, h)

  const pad = 10 * DPR
  const chassisX = pad
  const chassisY = pad
  const chassisW = w - pad * 2
  const chassisH = h - pad * 2 - 14 * DPR
  dc.strokeStyle = C.border
  dc.lineWidth = 1.5 * DPR
  roundRect(dc, chassisX, chassisY, chassisW, chassisH, 8 * DPR)
  dc.stroke()

  const groups = groupByModule()
  const moduleNames = Object.keys(groups)
  const { cols, rows } = gridDims(moduleNames.length)
  const gap = 8 * DPR
  const cellW = (chassisW - gap * (cols - 1)) / cols
  const cellH = (chassisH - gap * (rows - 1)) / rows

  const dots: { node: DeviceNode; x: number; y: number; r: number }[] = []
  const selectedCh = tree.selectedId ? tree.selectedChannelName() : null

  moduleNames.forEach((modName, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    const mx = chassisX + col * (cellW + gap)
    const my = chassisY + row * (cellH + gap)
    const folderId = findFolderIdByName(modName)
    const isSelFolder = tree.selectedFolderId === folderId

    dc.fillStyle = isSelFolder ? 'rgba(45,140,240,0.12)' : 'rgba(255,255,255,0.03)'
    roundRect(dc, mx, my, cellW, cellH, 6 * DPR)
    dc.fill()
    dc.strokeStyle = isSelFolder ? C.blue : C.border
    dc.lineWidth = 1 * DPR
    dc.stroke()

    dc.fillStyle = C.textSec
    dc.font = `${11 * DPR}px 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif`
    dc.textAlign = 'left'
    dc.textBaseline = 'top'
    dc.fillText(modName, mx + 6 * DPR, my + 4 * DPR)

    const sensors = groups[modName]
    const titleH = 18 * DPR
    const dotAreaY = my + titleH
    const dotAreaH = cellH - titleH - 6 * DPR
    sensors.forEach((node, si) => {
      const fx = node.position?.x != null ? node.position.x : (si + 0.5) / sensors.length
      const fy = node.position?.y != null ? node.position.y : 0.5
      const dx = mx + 6 * DPR + fx * (cellW - 12 * DPR)
      const dy = dotAreaY + fy * dotAreaH
      const ch = node.channelName
      const isSel = selectedCh === ch
      const dotR = (isSel ? 7 : 5) * DPR
      dots.push({ node, x: dx, y: dy, r: dotR })

      // 点位颜色按异常分数着色（与设备树圆点一致）：score>0.7红/>0.4黄/否则绿/无数据灰
      const sc = ch ? (health.sensors.find((x) => x.channel === ch)?.latest_score ?? null) : null
      const color = sc == null ? C.textSec : (sc > 0.7 ? C.red : (sc > 0.4 ? C.yellow : C.green))
      if (isSel) {
        const pulse = 1 + 0.3 * Math.sin(Date.now() / 200)
        dc.beginPath()
        dc.arc(dx, dy, dotR * 2 * pulse, 0, Math.PI * 2)
        dc.fillStyle = color + '33'
        dc.fill()
      }
      dc.beginPath()
      dc.arc(dx, dy, dotR, 0, Math.PI * 2)
      dc.fillStyle = color
      dc.fill()
      dc.strokeStyle = isSel ? C.textPri : 'rgba(0,0,0,0.3)'
      dc.lineWidth = isSel ? 1.5 * DPR : 0.5 * DPR
      dc.stroke()
      if (isSel && ch) {
        dc.fillStyle = C.textPri
        dc.font = `${10 * DPR}px sans-serif`
        dc.textAlign = 'center'
        dc.textBaseline = 'bottom'
        dc.fillText(ch, dx, dy - dotR - 2 * DPR)
      }
    })
  })

  if (dots.length === 0) {
    dc.fillStyle = C.textSec
    dc.font = `${12 * DPR}px sans-serif`
    dc.textAlign = 'center'
    dc.textBaseline = 'middle'
    dc.fillText('暂无传感器，请在左侧设备树添加', w / 2, h / 2)
  }
  diagramLayout = { dots }
}

function hitTest(mx: number, my: number): DeviceNode | null {
  if (!diagramLayout) return null
  for (const d of diagramLayout.dots) {
    const dx = mx - d.x
    const dy = my - d.y
    if (dx * dx + dy * dy <= (d.r + 3 * DPR) ** 2) return d.node
  }
  return null
}

// ---- sizing ----
function resize() {
  if (!dom.value || !canvas) return
  cssW = dom.value.clientWidth || 400
  cssH = dom.value.clientHeight || 260
  canvas.width = Math.floor(cssW * DPR)
  canvas.height = Math.floor(cssH * DPR)
  canvas.style.width = cssW + 'px'
  canvas.style.height = cssH + 'px'
}

// ---- rAF ----
function loop() {
  if (dom.value) {
    const w = dom.value.clientWidth
    const h = dom.value.clientHeight
    if (w > 0 && h > 0 && (Math.abs(w - cssW) > 1 || Math.abs(h - cssH) > 1)) resize()
  }
  draw()
  rafId = requestAnimationFrame(loop)
}

// ---- 交互 ----
function onMouseMove(e: MouseEvent) {
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const mx = (e.clientX - rect.left) * DPR
  const my = (e.clientY - rect.top) * DPR
  const hit = hitTest(mx, my)
  if (hit) {
    const ch = hit.channelName
    const s = health.sensors.find((x) => x.channel === ch)
    tooltipText.value = `${hit.name} [${ch}]\n健康: ${s?.health != null ? s.health.toFixed(1) + '%' : '—'}\n分数: ${s?.latest_score != null ? s.latest_score.toFixed(3) : '—'}`
    tooltipVisible.value = true
    // 越界翻转：靠近底部时 tooltip 显示在上方，避免被面板截断
    const el = dom.value
    const ttH = 60
    const ttW = 160
    let tx = e.clientX - rect.left + 12
    let ty = e.clientY - rect.top + 12
    if (el && ty + ttH > el.clientHeight) ty = e.clientY - rect.top - ttH - 8
    if (el && tx + ttW > el.clientWidth) tx = el.clientWidth - ttW - 8
    tooltipX.value = tx
    tooltipY.value = ty
    canvas.style.cursor = 'pointer'
  } else {
    tooltipVisible.value = false
    canvas.style.cursor = 'default'
  }
}

function onMouseLeave() {
  tooltipVisible.value = false
}

function onClick(e: MouseEvent) {
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const mx = (e.clientX - rect.left) * DPR
  const my = (e.clientY - rect.top) * DPR
  const hit = hitTest(mx, my)
  if (hit) tree.selectedId = hit.id
}

onMounted(async () => {
  await nextTick()
  if (!dom.value) return
  canvas = document.createElement('canvas')
  dom.value.appendChild(canvas)
  ctx = canvas.getContext('2d')
  resize()
  canvas.addEventListener('mousemove', onMouseMove)
  canvas.addEventListener('mouseleave', onMouseLeave)
  canvas.addEventListener('click', onClick)
  window.addEventListener('resize', resize)
  rafId = requestAnimationFrame(loop)
})

onUnmounted(() => {
  if (rafId) cancelAnimationFrame(rafId)
  if (canvas) {
    canvas.removeEventListener('mousemove', onMouseMove)
    canvas.removeEventListener('mouseleave', onMouseLeave)
    canvas.removeEventListener('click', onClick)
  }
  window.removeEventListener('resize', resize)
  canvas = null
  ctx = null
})
</script>

<template>
  <div class="device-diagram-wrap">
    <div ref="dom" class="device-diagram-canvas"></div>
    <div
      v-show="tooltipVisible"
      class="diagram-tooltip"
      :style="{ left: tooltipX + 'px', top: tooltipY + 'px' }"
    >{{ tooltipText }}</div>
    <div class="diagram-legend">
      <span><i style="background:#19be6b"></i>健康</span>
      <span><i style="background:#f5a623"></i>警告</span>
      <span><i style="background:#ed3f14"></i>危险</span>
    </div>
  </div>
</template>

<style scoped>
.device-diagram-wrap {
  position: relative;
  height: 260px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  overflow: hidden;
}
.device-diagram-canvas {
  width: 100%;
  height: 100%;
}
.device-diagram-canvas :deep(canvas) {
  display: block;
}
.diagram-tooltip {
  position: absolute;
  display: block;
  pointer-events: none;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 6px 10px;
  font-size: 0.75rem;
  color: var(--text-primary);
  white-space: pre-line;
  z-index: 10;
}
.diagram-legend {
  position: absolute;
  bottom: 6px;
  right: 8px;
  display: flex;
  gap: 10px;
  font-size: 0.65rem;
  color: var(--text-secondary);
  pointer-events: none;
}
.diagram-legend i {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 3px;
  vertical-align: middle;
}
</style>
