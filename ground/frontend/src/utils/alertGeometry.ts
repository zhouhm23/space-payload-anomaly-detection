/**
 * Alert-point geometry (pure functions, no Canvas dependency, unit-testable).
 *
 * Extracted from TelemetryCanvas.drawChart so that, given fabricated test
 * data, we can assert "for these alert points + this time axis → the red
 * dot should land at this X and align with this data point".
 *
 * The Canvas drawing itself (fillRect/arc calls) is neither testable nor
 * worth testing; this module focuses on the data→coordinate mapping,
 * which is the crux of red-dot alignment correctness.
 */

/** A single data point (one row of the telemetry window, fields aligned with /api/v2/window). */
export interface TelemetryPoint {
  timestamp: number  // seconds (epoch)
  raw_value?: number | null
  predicted_value?: number | null
  anomaly_score?: number | null
  predicted_anomaly_score?: number | null
}

/** An alert point (aligned with /api/v2/alert-points red_points/yellow_points). */
export interface AlertPoint {
  channel: string
  timestamp: number  // seconds (epoch); should be the real sample time (acq_ts)
  score?: number | null
  type: 'measured' | 'predicted'
}

/** A contiguous gap-free interval on the time axis (used for gap-collapsed X mapping).
 *  ★ Unit convention: tsStart/tsEnd are in **milliseconds** (consistent with
 *    computeRedDots' internal tsMsAp = timestamp * 1000). TelemetryCanvas also
 *    builds its segments from tsMs (milliseconds), keeping the whole pipeline
 *    in milliseconds. */
export interface TsSegment {
  tsStart: number  // milliseconds
  tsEnd: number    // milliseconds
  xStart: number   // pixels
  xEnd: number     // pixels
}

/**
 * Build a timestamp(milliseconds) → X-pixel mapping (with gap collapsing).
 *
 * Gap-collapse rule: the data is split into segments by gaps; each segment
 * is laid out evenly on the X axis, and a fixed GAP_WIDTH-pixel "collapsed
 * band" represents the break between segments.
 *
 * The returned function takes a **millisecond** timestamp and returns the X
 * pixel, or null if the timestamp falls outside every segment.
 */
export function buildTsToX(segments: TsSegment[]): (ts: number) => number | null {
  if (segments.length === 0) return () => null
  return function tsToX(ts: number): number | null {
    for (const seg of segments) {
      if (ts < seg.tsStart - 1e-9) return null  // before every segment
      if (ts <= seg.tsEnd + 1e-9 || seg.tsEnd === seg.tsStart) {
        // Inside this segment (or this segment is a single point)
        if (seg.tsEnd === seg.tsStart) return seg.xStart
        const frac = (ts - seg.tsStart) / (seg.tsEnd - seg.tsStart)
        return seg.xStart + frac * (seg.xEnd - seg.xStart)
      }
    }
    return null
  }
}

/**
 * Find the index of the data point closest to a given timestamp (linear scan).
 *
 * Used to align an alert time to the nearest telemetry sample, taking that
 * sample's raw_value/anomaly_score as the source of the red dot's Y
 * coordinate.
 *
 * Returns -1 when the data is empty.
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
 * Given a set of alert points and the current channel, filter to the points
 * belonging to that channel. Mirrors TelemetryCanvas.currentChannelAlertPoints.
 */
export function filterByChannel(points: AlertPoint[], channel: string): AlertPoint[] {
  if (!channel) return []
  return points.filter(p => p.channel === channel)
}

/**
 * Compute the red-dot coordinates to draw on the telemetry chart.
 *
 * Inputs: data points + alert points + time-axis mapping + Y-mapping funcs.
 * Output: one {x, y, source} per red dot (source tells whether Y came from
 * raw/pred/score).
 *
 * Use: in a unit test you can assert "given N alert points, return N
 * coordinates whose X matches tsToX". This is the core assertion point for
 * red-dot alignment correctness.
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
    // Telemetry-region red dot
    const rv = d.raw_value ?? d.predicted_value
    if (rv != null) {
      out.push({ x, y: yOfRaw(rv), source: d.raw_value != null ? 'raw' : 'predicted' })
    }
    // Score-region red dot
    const sv = d.anomaly_score ?? d.predicted_anomaly_score
    if (sv != null) {
      out.push({ x, y: yOfScore(sv), source: d.anomaly_score != null ? 'score' : 'predicted_score' })
    }
  }
  return out
}
