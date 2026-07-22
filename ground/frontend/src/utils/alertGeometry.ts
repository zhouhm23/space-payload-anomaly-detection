/**
 * 告警点几何计算（纯函数，无 Canvas 依赖，可单测）。
 *
 * 从 TelemetryCanvas.drawChart 抽出，便于用编造的测试数据断言
 * "给定告警点 + 时间轴 → 应该画在哪个 X、对应哪个数据点"。
 *
 * Canvas 绘制本身（fillRect/arc 调用）不可测也不必测；这里聚焦
 * 数据→坐标的映射逻辑，这是红点对齐正确性的核心。
 */

/** 单个数据点（遥测窗口的一行，字段对齐 /api/v2/window 返回） */
export interface TelemetryPoint {
  timestamp: number  // 秒（epoch）
  raw_value?: number | null
  predicted_value?: number | null
  anomaly_score?: number | null
  predicted_anomaly_score?: number | null
}

/** 告警点（对齐 /api/v2/alert-points 的 red_points/yellow_points） */
export interface AlertPoint {
  channel: string
  timestamp: number  // 秒（epoch），应为真实采样时刻（acq_ts）
  score?: number | null
  type: 'measured' | 'predicted'
}

/** 时间轴上一段连续无 gap 的区间（用于 gap 折叠后的 X 映射）。
 *  ★ 单位约定：tsStart/tsEnd 用【毫秒】（与 computeRedDots 内部的
 *    tsMsAp = timestamp * 1000 一致）。TelemetryCanvas 构建 segments 时
 *    也是从 tsMs（毫秒）取值，保持全链路毫秒统一。 */
export interface TsSegment {
  tsStart: number  // 毫秒
  tsEnd: number    // 毫秒
  xStart: number   // 像素
  xEnd: number     // 像素
}

/**
 * 构建时间戳(毫秒) → X 像素的映射函数（含 gap 折叠）。
 *
 * gap 折叠规则：数据被缺口分成多段，每段在 X 轴上等距铺开，
 * 段间用固定 GAP_WIDTH 像素的"折叠带"表示中断。
 *
 * 返回的函数接收【毫秒】时间戳，返回 X 像素；时间戳不在任何段内则返回 null。
 */
export function buildTsToX(segments: TsSegment[]): (ts: number) => number | null {
  if (segments.length === 0) return () => null
  return function tsToX(ts: number): number | null {
    for (const seg of segments) {
      if (ts < seg.tsStart - 1e-9) return null  // 在所有段之前
      if (ts <= seg.tsEnd + 1e-9 || seg.tsEnd === seg.tsStart) {
        // 落在本段内（或本段是单点）
        if (seg.tsEnd === seg.tsStart) return seg.xStart
        const frac = (ts - seg.tsStart) / (seg.tsEnd - seg.tsStart)
        return seg.xStart + frac * (seg.xEnd - seg.xStart)
      }
    }
    return null
  }
}

/**
 * 在数据点序列中找最接近给定时间戳的索引（线性扫描）。
 *
 * 用于把告警时刻对齐到最近的遥测采样点，取该点的 raw_value/anomaly_score
 * 作为红点的 Y 坐标来源。
 *
 * 返回 -1 表示数据为空。
 */
export function findNearestIndex(tsMs: number[], targetTsMs: number): number {
  if (tsMs.length === 0) return -1
  let nearestIdx = 0
  let nearestDist = Math.abs(tsMs[0] - targetTsMs)
  for (let i = 1; i < tsMs.length; i++) {
    const d = Math.abs(tsMs[i] - targetTsMs)
    if (d < nearestDist) {
      nearestDist = d
      nearestIdx = i
    }
  }
  return nearestIdx
}

/**
 * 给定一组告警点和当前通道，过滤出属于该通道的告警点。
 * 对齐 TelemetryCanvas.currentChannelAlertPoints 的计算。
 */
export function filterByChannel(points: AlertPoint[], channel: string): AlertPoint[] {
  if (!channel) return []
  return points.filter(p => p.channel === channel)
}

/**
 * 计算应在遥测图上绘制的红点坐标列表。
 *
 * 输入：数据点 + 告警点 + 时间轴映射 + Y 映射函数。
 * 输出：每个应画红点的 {x, y, source}（source 说明 Y 来自 raw/pred/score）。
 *
 * 用途：单测时可断言"给定 N 个告警点，应返回 N 个坐标，且 X 与 tsToX 一致"。
 * 这是验证红点对齐正确性的核心断言点。
 */
export interface RedDotPosition {
  x: number
  y: number
  source: 'raw' | 'predicted' | 'score' | 'predicted_score'
}

export function computeRedDots(
  data: TelemetryPoint[],
  alertPoints: AlertPoint[],
  tsToX: (ts: number) => number | null,
  yOfRaw: (v: number) => number,
  yOfScore: (v: number) => number,
): RedDotPosition[] {
  if (data.length === 0) return []
  const tsMs = data.map(d => d.timestamp * 1000)
  const tsMin = tsMs[0], tsMax = tsMs[tsMs.length - 1]
  const out: RedDotPosition[] = []
  for (const ap of alertPoints) {
    const tsMsAp = ap.timestamp * 1000
    if (tsMsAp < tsMin || tsMsAp > tsMax) continue
    const x = tsToX(tsMsAp)
    if (x == null) continue
    const idx = findNearestIndex(tsMs, tsMsAp)
    if (idx < 0) continue
    const d = data[idx]
    // 遥测区红点
    const rv = d.raw_value ?? d.predicted_value
    if (rv != null) {
      out.push({ x, y: yOfRaw(rv), source: d.raw_value != null ? 'raw' : 'predicted' })
    }
    // 分数区红点
    const sv = d.anomaly_score ?? d.predicted_anomaly_score
    if (sv != null) {
      out.push({ x, y: yOfScore(sv), source: d.anomaly_score != null ? 'score' : 'predicted_score' })
    }
  }
  return out
}
