import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright config for the PHM frontend UAT.
 *
 * The UAT boots the full stack (space segment + ground FastAPI) via the
 * `webServer` fixtures, so it can run on a fresh CI machine with no manual
 * setup.  Tests live in ./uat.spec.ts.
 */
export default defineConfig({
  testDir: './tests/uat',
  fullyParallel: false, // single space segment — must run serially
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? 'line' : 'list',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: 'http://localhost:8501',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    actionTimeout: 15_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: [
    {
      // Space segment — produces telemetry over TCP
      command:
        'cd /d "d:\\Office\\生产实习\\src" && set PYTHONPATH=d:\\Office\\生产实习\\src&& ".\\.conda-env\\python.exe" -m space.main --port 9876',
      port: 9876,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      // Ground FastAPI — serves the built frontend + API
      command:
        'cd /d "d:\\Office\\生产实习\\src\\ground" && set PYTHONPATH=d:\\Office\\生产实习\\src&& "..\\.conda-env\\python.exe" server.py',
      port: 8501,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
})
