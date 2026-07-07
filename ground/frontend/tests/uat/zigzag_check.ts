/**
 * zigzag_check.ts — 检测 /api/window 返回的数据是否存在锯齿（相邻点交替跳变）。
 *
 * 运行方式（在 frontend 目录下）：
 *   npx tsx tests/uat/zigzag_check.ts
 *
 * 或直接用 node（如果装了 ts-node）：
 *   npx ts-node tests/uat/zigzag_check.ts
 *
 * 原理：每 2 秒拉一次 /api/window，检查 raw 值序列中是否有相邻三点
 * 呈 "低-高-低" 或 "高-低-高" 的交替模式（幅度 > THRESHOLD）。
 * 正常的平滑信号相邻差值符号变化很少；锯齿数据符号频繁交替。
 */

const API_BASE = process.env.API_BASE || 'http://localhost:8501'

/** 相邻点差值的绝对值超过此阈值才算"跳变" */
const JUMP_THRESHOLD = 0.3
/** 在窗口内，如果交替符号变化次数占比超过此比例，判定为锯齿 */
const ZIGZAG_RATIO = 0.3
/** 总共采样多少轮 */
const MAX_ROUNDS = 30
/** 每轮间隔（毫秒） */
const INTERVAL_MS = 2000

interface RawPoint {
  raw: number | null
  score: number | null
  received_at: number
}

interface WindowResponse {
  channel: string
  count: number
  raw: RawPoint[]
}

async function fetchJson(url: string): Promise<any> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url} → ${r.status}`)
  return r.json()
}

/**
 * 检测一个数值序列中是否存在锯齿模式。
 *
 * 返回 { zigzagCount, totalJumps, ratio, maxJump, details }
 *  - zigzagCount: 连续三个点呈"大跳上下交替"的次数
 *  - totalJumps:  相邻点差值超过 THRESHOLD 的总次数
 *  - ratio:       zigzagCount / totalJumps（越高越像锯齿）
 *  - maxJump:     最大单步跳变幅度
 */
function detectZigzag(
  values: number[],
  threshold: number = JUMP_THRESHOLD,
): {
  zigzagCount: number
  totalJumps: number
  ratio: number
  maxJump: number
  worstIdx: number
  worstSample: number[]
} {
  if (values.length < 3) {
    return { zigzagCount: 0, totalJumps: 0, ratio: 0, maxJump: 0, worstIdx: -1, worstSample: [] }
  }

  // 计算相邻差值
  const diffs: number[] = []
  for (let i = 1; i < values.length; i++) {
    diffs.push(values[i] - values[i - 1])
  }

  let zigzagCount = 0
  let totalJumps = 0
  let maxJump = 0
  let worstIdx = -1
  let worstSample: number[] = []

  for (let i = 0; i < diffs.length - 1; i++) {
    const d1 = diffs[i]
    const d2 = diffs[i + 1]
    const abs1 = Math.abs(d1)
    const abs2 = Math.abs(d2)

    if (abs1 > threshold) totalJumps++
    if (abs1 > maxJump) {
      maxJump = abs1
      worstIdx = i
      worstSample = [values[i], values[i + 1], values[i + 2]].map((v) => Number(v.toFixed(4)))
    }

    // 锯齿判定：d1 和 d2 符号相反，且幅度都超过阈值
    if (d1 * d2 < 0 && abs1 > threshold && abs2 > threshold) {
      zigzagCount++
    }
  }

  const ratio = totalJumps > 0 ? zigzagCount / totalJumps : 0
  return { zigzagCount, totalJumps, ratio, maxJump, worstIdx, worstSample }
}

/** 检测时间戳是否单调递增 + 无重复 */
function checkTimestamps(points: RawPoint[]): {
  ascending: boolean
  duplicates: number
  outOfOrder: number
} {
  let outOfOrder = 0
  let prev = -Infinity
  const seen = new Set<number>()
  let duplicates = 0
  for (const p of points) {
    const ts = p.received_at
    if (ts < prev) outOfOrder++
    const rounded = Math.round(ts * 1000) / 1000
    if (seen.has(rounded)) duplicates++
    else seen.add(rounded)
    prev = ts
  }
  return { ascending: outOfOrder === 0, duplicates, outOfOrder }
}

async function getChannel(): Promise<string> {
  const cfg = await fetchJson(`${API_BASE}/api/config?t=${Date.now()}`)
  const tree = cfg.device_tree || []
  const first = tree.find((n: any) => n.sourceId) || tree[0]
  return first?.channelName || first?.name || 'C-1'
}

async function main() {
  console.log('=== 锯齿检测脚本 ===')
  console.log(`API: ${API_BASE}`)
  console.log(`跳变阈值: ${JUMP_THRESHOLD}, 锯齿判定比例: ${ZIGZAG_RATIO}`)
  console.log(`采样轮数: ${MAX_ROUNDS}, 间隔: ${INTERVAL_MS}ms`)
  console.log('')

  let channel = 'C-1'
  try {
    channel = await getChannel()
  } catch {
    console.log('⚠ 无法获取配置，使用默认通道 C-1')
  }
  console.log(`检测通道: ${channel}`)
  console.log('')

  let detectedCount = 0
  let totalRounds = 0

  for (let round = 1; round <= MAX_ROUNDS; round++) {
    try {
      const url = `${API_BASE}/api/window?channel=${encodeURIComponent(channel)}&count=512`
      const data: WindowResponse = await fetchJson(url)
      const raw = data.raw.filter((p): p is RawPoint & { raw: number } => p.raw !== null)
      const scores = data.raw.filter((p): p is RawPoint & { score: number } => p.score !== null)

      if (raw.length < 10) {
        console.log(`[轮 ${String(round).padStart(2)}/${MAX_ROUNDS}] 数据不足 (${raw.length} 点)，跳过`)
        await sleep(INTERVAL_MS)
        continue
      }

      totalRounds++

      // ---- 检查遥测值 ----
      const teleVals = raw.map((p) => p.raw)
      const teleResult = detectZigzag(teleVals)

      // ---- 检查异常分数 ----
      const scoreVals = scores.map((p) => p.score)
      const scoreResult = detectZigzag(scoreVals, 0.15) // 分数跳变阈值更小

      // ---- 检查时间戳 ----
      const tsCheck = checkTimestamps(raw)

      const teleZigzag = teleResult.ratio > ZIGZAG_RATIO && teleResult.zigzagCount >= 3
      const scoreZigzag = scoreResult.ratio > ZIGZAG_RATIO && scoreResult.zigzagCount >= 3
      const tsBad = !tsCheck.ascending || tsCheck.duplicates > 0

      if (teleZigzag || scoreZigzag || tsBad) {
        detectedCount++
        console.log(`[轮 ${String(round).padStart(2)}/${MAX_ROUNDS}] ⚠ 检测到问题！`)
        if (teleZigzag) {
          console.log(`  遥测锯齿: ${teleResult.zigzagCount} 处交替跳变 / ${teleResult.totalJumps} 处跳变 (ratio=${teleResult.ratio.toFixed(2)})`)
          console.log(`  最大跳变: idx=${teleResult.worstIdx} 值=[${teleResult.worstSample.join(', ')}]`)
        }
        if (scoreZigzag) {
          console.log(`  分数锯齿: ${scoreResult.zigzagCount} 处交替跳变 / ${scoreResult.totalJumps} 处跳变 (ratio=${scoreResult.ratio.toFixed(2)})`)
          console.log(`  最大跳变: idx=${scoreResult.worstIdx} 值=[${scoreResult.worstSample.join(', ')}]`)
        }
        if (tsBad) {
          console.log(`  时间戳问题: 升序=${tsCheck.ascending} 重复=${tsCheck.duplicates} 乱序=${tsCheck.outOfOrder}`)
        }
        // 导出问题数据的完整值序列
        console.log(`  遥测前30个值: [${teleVals.slice(0, 30).map((v) => v.toFixed(3)).join(', ')}]`)
        console.log(`  时间戳前10个: [${raw.slice(0, 10).map((p) => p.received_at.toFixed(3)).join(', ')}]`)
      } else {
        process.stdout.write('.')
      }
    } catch (e) {
      console.log(`[轮 ${String(round).padStart(2)}/${MAX_ROUNDS}] 请求失败: ${e}`)
    }
    await sleep(INTERVAL_MS)
  }

  console.log('')
  console.log('=== 总结 ===')
  console.log(`总采样轮数: ${totalRounds}`)
  console.log(`检测到问题轮数: ${detectedCount}`)
  console.log(`问题发生率: ${totalRounds > 0 ? ((detectedCount / totalRounds) * 100).toFixed(1) : 0}%`)

  if (detectedCount === 0) {
    console.log('✅ 未检测到锯齿/时间戳问题')
    process.exit(0)
  } else {
    console.log('❌ 检测到锯齿问题，需要修复')
    process.exit(1)
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}

main().catch((e) => {
  console.error('Fatal:', e)
  process.exit(2)
})
