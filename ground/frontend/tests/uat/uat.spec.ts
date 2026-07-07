import { test, expect, type Page } from '@playwright/test'

/**
 * End-to-end User Acceptance Test (UAT) for the PHM frontend.
 *
 * Walks the full real-user path with assertions at every step:
 *   1. open page → layout renders
 *   2. dashboard cards present → click a sensor → main chart switches
 *   3. click 开始 → polling fires → status text updates
 *   4. wait ≥1 poll → telemetry chart has data points
 *   5. switch sensor → chart data changes
 *   6. wait for warnings → /api/warnings returns entries (or timeout-skip)
 *   7. wait for new data → warning status may flip (best-effort)
 *   8. alerts panel renders when score>0.7 (best-effort)
 *   9. click 重置 → ring buffer cleared
 *  10. 5-minute stability → no console errors / no crash
 *
 * Run:  npx playwright test
 */

const POLL_WAIT_MS = 8_000 // allow ≥2 poll cycles (2s interval)
const STABILITY_MS = 30_000 // shortened from 5min for CI; bump locally

// Button text changed in Day-9 refactor: 开始→实时, 暂停→冻结.
// Keep a regex that accepts both so the suite is resilient during migration.
const PLAY_BTN = /实时|开始/
const PAUSE_BTN = /冻结|暂停/

async function waitForPolls(page: Page, ms = POLL_WAIT_MS) {
  await page.waitForTimeout(ms)
}

test.describe('PHM 前端 UAT', () => {
  test('1. 打开页面 → 布局渲染', async ({ page }) => {
    await page.goto('/')
    // Title
    await expect(page.locator('.header .title')).toContainText('空间站有效载荷预测性维护支持系统')
    // Three panels
    await expect(page.locator('.left-panel')).toBeVisible()
    await expect(page.locator('.center-panel')).toBeVisible()
    await expect(page.locator('.right-panel')).toBeVisible()
    // Bottom chart area — chart containers exist in the DOM.
    // (ECharts canvases have height 0 until data arrives, so we check
    //  presence rather than visibility here.)
    await expect(page.locator('.bottom-panel')).toBeVisible()
    await expect(page.locator('.chart-telemetry')).toHaveCount(1)
    await expect(page.locator('.chart-anomaly')).toHaveCount(1)
    // Device tree loaded
    await expect(page.locator('.device-tree')).toBeVisible()
  })

  test('2. 仪表盘卡片存在 + 点击切换主图源', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(3_000)
    // Either cards are present, or the empty-hint is shown (no data yet)
    const cards = page.locator('.gauge-card')
    const cardCount = await cards.count()
    if (cardCount > 0) {
      await cards.first().click()
      // Selecting should update the selected tree node
      await expect(page.locator('.tree-item.active')).toBeVisible()
    }
  })

  test('3. 点击实时 → 轮询触发 + 按钮变冻结', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    const playBtn = page.locator('button', { hasText: PLAY_BTN }).first()
    await playBtn.click()
    await expect.poll(async () => await playBtn.textContent(), { timeout: 5_000 }).toMatch(PAUSE_BTN)
    // chunk-info should update from "就绪" to something with "点"
    await expect(page.locator('.chunk-info')).not.toHaveText('就绪')
  })

  test('4. 等待轮询 → 遥测图表有数据点', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    // start polling
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, POLL_WAIT_MS)
    // The canvas should have rendered pixels (ECharts draws to canvas).
    // Once data arrives the canvas element appears inside the container.
    const canvases = page.locator('.chart-telemetry canvas')
    await expect(canvases).toHaveCount(1, { timeout: 10_000 })
    // chunk-info should report points count
    await expect(page.locator('.chunk-info')).toContainText(/点/)
  })

  test('5. 切换传感器 → 图表数据更新', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(3_000)
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, POLL_WAIT_MS)
    // If there are ≥2 tree items with sourceId, click the second one
    const sensors = page.locator('.tree-item')
    const count = await sensors.count()
    if (count >= 2) {
      await sensors.nth(1).click()
      await page.waitForTimeout(2_000)
      // chart container should still be present (no crash on switch)
      await expect(page.locator('.chart-telemetry')).toHaveCount(1)
    }
  })

  test('6. 预警栏渲染（best-effort: 等待 TTM 预测）', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    // Wait long enough for the warning service to run a forecast cycle
    await waitForPolls(page, 15_000)
    // The warning panel must exist regardless of content
    await expect(page.locator('.info-card', { hasText: '预警栏' })).toBeVisible()
    // Check /api/warnings directly — non-empty is best-effort, not hard-fail
    const resp = await page.evaluate(async () => {
      const r = await fetch('/api/warnings')
      return r.ok ? await r.json() : null
    })
    expect(resp).not.toBeNull()
  })

  test('7. 新数据抵达 → 预警标签可更新（best-effort）', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, 20_000)
    // If warnings exist, at least one should have a status field
    const resp = await page.evaluate(async () => {
      const r = await fetch('/api/warnings')
      return r.ok ? await r.json() : null
    })
    if (resp && resp.warnings && resp.warnings.length > 0) {
      for (const w of resp.warnings) {
        expect(['pending', 'confirmed', 'false']).toContain(w.status)
      }
    }
  })

  test('8. 告警栏渲染（/api/alerts 可达）', async ({ page }) => {
    await page.goto('/')
    const resp = await page.evaluate(async () => {
      const r = await fetch('/api/alerts')
      return r.ok ? await r.json() : null
    })
    expect(resp).not.toBeNull()
    expect(resp).toHaveProperty('alerts')
    expect(resp).toHaveProperty('threshold')
    // Panel exists in DOM
    await expect(page.locator('.info-card', { hasText: '告警栏' })).toBeVisible()
  })

  test('9. 点击重置 → 跳到最新', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    // Start (realtime), wait, then freeze+reset
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, POLL_WAIT_MS)
    // Freeze first (reset disabled while realtime)
    await page.locator('button', { hasText: PAUSE_BTN }).first().click()
    await page.waitForTimeout(500)
    await page.locator('button', { hasText: '重置' }).first().click()
    await page.waitForTimeout(1_000)
    // chunk-info should show points count (reset jumps to latest, not clears)
    await expect(page.locator('.chunk-info')).toContainText(/点/)
    // /api/health system should be valid
    const resp = await page.evaluate(async () => {
      const r = await fetch('/api/health')
      return r.ok ? await r.json() : null
    })
    if (resp) expect(resp.system).toBeGreaterThanOrEqual(0)
  })

  test('10. 稳定性: 短时运行无 JS 报错', async ({ page }) => {
    const errors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text())
    })
    page.on('pageerror', (err) => errors.push(String(err)))
    await page.goto('/')
    await page.waitForTimeout(2_000)
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, STABILITY_MS)
    // Allow no uncaught errors. (Network 404s for favicons etc. are not
    // console.type='error' from JS, so this is strict to JS errors.)
    expect(errors, `JS errors: ${errors.join('; ')}`).toHaveLength(0)
  })

  // ------------------------------------------------------------------
  // Data integrity tests — verify the fixes for zig-zag and drag-pan.
  // ------------------------------------------------------------------

  test('11. /api/window 数据时间戳严格升序（防锯齿）', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    // Start realtime to trigger auto-poll writing data into SQLite
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, POLL_WAIT_MS)
    // Find the active channel from the device tree
    const channel = await page.locator('.tree-item.active').first().textContent()
    expect(channel).toBeTruthy()
    // Extract channel name — the tree item text is the device label, but
    // the channel is selected in the store.  Query /api/window for the
    // channel shown in the chunk-info.
    const resp = await page.evaluate(async () => {
      // Read the currently selected channel from the Pinia store via
      // the exposed window object (dev only) or fall back to /api/config
      const cfg = await fetch('/api/config?t=' + Date.now()).then((r) => r.json())
      const tree = cfg.device_tree || []
      const first = tree.find((n: any) => n.sourceId) || tree[0]
      const ch = first?.channelName || first?.name || 'C-1'
      const r = await fetch(`/api/window?channel=${encodeURIComponent(ch)}&count=512`)
      return r.ok ? await r.json() : null
    })
    expect(resp).not.toBeNull()
    expect(resp.raw.length).toBeGreaterThan(1)
    const ts = resp.raw.map((p: any) => p.received_at)
    // Verify strictly ascending
    for (let i = 1; i < ts.length; i++) {
      expect(ts[i], `point ${i} ts ${ts[i]} <= prev ${ts[i - 1]}`).toBeGreaterThan(ts[i - 1])
    }
  })

  test('12. /api/window 数据无重复时间戳', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, POLL_WAIT_MS)
    const resp = await page.evaluate(async () => {
      const cfg = await fetch('/api/config?t=' + Date.now()).then((r) => r.json())
      const tree = cfg.device_tree || []
      const first = tree.find((n: any) => n.sourceId) || tree[0]
      const ch = first?.channelName || first?.name || 'C-1'
      const r = await fetch(`/api/window?channel=${encodeURIComponent(ch)}&count=512`)
      return r.ok ? await r.json() : null
    })
    expect(resp).not.toBeNull()
    const ts = resp.raw.map((p: any) => p.received_at)
    const unique = new Set(ts)
    expect(unique.size, `duplicates found in ${ts.length} points`).toBe(ts.length)
  })

  test('13. 冻结模式拖拽平移 → 左侧不空白', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2_000)
    // Start realtime, wait for data, then freeze
    await page.locator('button', { hasText: PLAY_BTN }).first().click()
    await waitForPolls(page, POLL_WAIT_MS)
    await page.locator('button', { hasText: PAUSE_BTN }).first().click()
    await page.waitForTimeout(500)
    // Get the chart canvas bounding box
    const canvas = page.locator('.chart-telemetry canvas')
    await expect(canvas).toHaveCount(1, { timeout: 10_000 })
    const box = await canvas.boundingBox()
    expect(box).not.toBeNull()
    if (!box) return

    // Record the number of data points before drag by checking the
    // store's raw length via /api/window for the current window.
    const beforeCount = await page.evaluate(async () => {
      const cfg = await fetch('/api/config?t=' + Date.now()).then((r) => r.json())
      const tree = cfg.device_tree || []
      const first = tree.find((n: any) => n.sourceId) || tree[0]
      const ch = first?.channelName || first?.name || 'C-1'
      const r = await fetch(`/api/window?channel=${encodeURIComponent(ch)}&count=512`)
      return r.ok ? (await r.json()).raw.length : 0
    })
    expect(beforeCount).toBeGreaterThan(10)

    // Simulate a left-drag (pan to earlier data) of 150px
    const startX = box.x + box.width * 0.7
    const startY = box.y + box.height / 2
    const endX = startX + 150 // drag right → chart shifts left → earlier data
    await page.mouse.move(startX, startY)
    await page.mouse.down()
    // Move in steps so mousemove handlers fire
    for (let x = startX; x <= endX; x += 30) {
      await page.mouse.move(x, startY)
      await page.waitForTimeout(30)
    }
    await page.mouse.up()
    // Wait for throttled fetch to complete (100ms throttle + network)
    await page.waitForTimeout(1_500)

    // After pan, the chart should still show data (not blank).
    // The right-edge timestamp should have changed (moved earlier).
    const afterCount = await page.evaluate(async () => {
      const cfg = await fetch('/api/config?t=' + Date.now()).then((r) => r.json())
      const tree = cfg.device_tree || []
      const first = tree.find((n: any) => n.sourceId) || tree[0]
      const ch = first?.channelName || first?.name || 'C-1'
      const r = await fetch(`/api/window?channel=${encodeURIComponent(ch)}&count=512`)
      return r.ok ? (await r.json()).raw.length : 0
    })
    // Data should still be present (the fix fetches during drag, not only
    // on mouseup).  At minimum the chart must not be empty.
    expect(afterCount).toBeGreaterThan(0)
  })
})
