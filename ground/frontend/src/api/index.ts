/**
 * PHM 后端 API 客户端封装
 *
 * 所有调用走 /api/v2/（DRF 新接口），通过 vite proxy 转发到 Django :8501。
 * 错误处理：503（Container 未就绪）单独抛出，让调用方决定是否重试。
 */
import axios from 'axios'

const client = axios.create({
  baseURL: '/api/v2/',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

// 统一错误转换：503 → 后端初始化中，可重试
export class BackendNotReadyError extends Error {
  constructor(public state: string, message: string) {
    super(message)
    this.name = 'BackendNotReadyError'
  }
}

client.interceptors.response.use(
  (resp) => resp,
  (error) => {
    if (error.response?.status === 503) {
      const detail = error.response.data?.detail || '后端初始化中'
      const state = error.response.data?.state || 'unknown'
      return Promise.reject(new BackendNotReadyError(state, detail))
    }
    return Promise.reject(error)
  }
)

// ── 系统类 ──────────────────────────────────────────────────────────────────
export const api = {
  /** 健康探针（纯 Django 进程） */
  ping: () => client.get('/ping/').then((r) => r.data),

  /** 启动状态探针 */
  startupStatus: () => client.get('/startup-status/').then((r) => r.data),

  /** 前台主题 */
  theme: () => client.get('/theme/').then((r) => r.data),

  /** 顶栏系统信息 */
  systemInfo: () => client.get('/system-info/').then((r) => r.data),

  /** 设备树 + 健康度聚合 */
  deviceTree: () => client.get('/device-tree/').then((r) => r.data),

  /** 实时遥测窗口（ring buffer + pred 合并，2s 刷新） */
  window: (channel: string, count = 512) =>
    client
      .get('/window/', { params: { channel, count } })
      .then((r) => r.data),

  /** 全通道告警点图（红点实测 + 黄点预测） */
  alertPoints: () => client.get('/alert-points/').then((r) => r.data),

  /** 实测告警列表 */
  alerts: (limit = 20) =>
    client.get('/alerts/', { params: { limit } }).then((r) => r.data),

  /** 预测预警列表 */
  warnings: (limit = 20) =>
    client.get('/warnings/', { params: { limit } }).then((r) => r.data),

  /** RUL 退化预测 */
  rul: (channel?: string) =>
    client.get('/rul/', { params: channel ? { channel } : {} }).then((r) => r.data),
}

export type Api = typeof api
