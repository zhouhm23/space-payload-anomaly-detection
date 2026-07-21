<script setup lang="ts">
/**
 * 右详情区（需求书 §右详情区）。
 *
 * 上下两部分：
 * - 上半：通道详情（量程、单位、阈值，跟随轮播）
 * - 下半：全通道告警和预警列表（遥测值、异常分数、阈值、综合状态、时间）
 *         告警用红色、预警用黄色，不轮播，仅随数据块更新，最多 20 条按时间倒序
 */
import { computed, watch } from 'vue'
import { useSystemStore, type DeviceNode } from '@/stores/system'
import { api } from '@/api'
import { usePoll } from '@/composables/usePoll'

const store = useSystemStore()

// 上半：当前通道详情（从 deviceTree 找当前选中的 sensor 节点）
interface SensorDetail {
  name: string
  unit: string
  yMin: number
  yMax: number
  threshold: number
  description: string
  isSpecial: boolean
  channelName: string
}

function findSensor(nodes: any[], channel: string): SensorDetail | null {
  for (const n of nodes || []) {
    if (n.type === 'sensor' && n.channelName === channel) {
      return {
        name: n.name,
        unit: n.unit && n.unit.trim() ? n.unit : '无',
        yMin: n.yMin ?? 0,
        yMax: n.yMax ?? 0,
        threshold: n.threshold ?? 0.5,
        description: n.description || '',
        isSpecial: !!n.isSpecial,
        channelName: n.channelName,
      }
    }
    if (n.type === 'folder' && n.children) {
      const found = findSensor(n.children, channel)
      if (found) return found
    }
  }
  return null
}

const currentSensor = computed<SensorDetail | null>(() => {
  if (!store.deviceTree || !store.currentChannel) return null
  return findSensor(store.deviceTree.device_tree, store.currentChannel)
})

// 显示名（设备树里可能没 VS-sine 而是 S1，用 store.displayName 转换）
const currentDisplayName = computed(() => store.displayName(store.currentChannel))

// 下半：告警预警合并列表（最多 20 条，3s 轮询）
// 实测告警（红色）+ 预测预警（黄色）
const alertsPoll = usePoll(() => api.alerts(20), 3000, { immediate: true, autoStart: true })
const warningsPoll = usePoll(() => api.warnings(20), 3000, { immediate: true, autoStart: true })

interface AlertRow {
  id: number | string
  type: 'alert' | 'warning'  // alert=实测告警(红), warning=预测预警(黄)
  channel: string
  // 结构化告警原因：遥测值（最新/触发值）+ 异常分数（可能为空，如常数突变）+ 告警原因文本
  raw_value: number | null      // 遥测值（来自 raw_window 最后一个点）
  score: number | null          // 异常分数（null 表示该告警无分数，如常数突变）
  reason: string                // 告警原因（message，结构化）
  threshold: number | null      // 阈值（仅在告警由分数触发时显示）
  final_status: string          // 综合状态
  created_at: number
}

// 从 raw_window 取最新遥测值
function extractRawValue(rawWindow: any): number | null {
  if (!Array.isArray(rawWindow) || rawWindow.length === 0) return null
  const last = rawWindow[rawWindow.length - 1]
  return typeof last === 'number' ? last : null
}

const alertRows = computed<AlertRow[]>(() => {
  const rows: AlertRow[] = []

  // 实测告警（来自内存 AlertStore，含 raw_window/message/score）
  const alertsResp = alertsPoll.data.value
  const alerts = alertsResp?.alerts || []
  const threshold = alertsResp?.threshold ?? 0.5
  for (const a of alerts) {
    rows.push({
      id: `${a.channel}_${a.time}`,
      type: 'alert',
      channel: store.displayName(a.channel),
      raw_value: extractRawValue(a.raw_window ?? a.raw_snapshot),
      score: a.score != null ? a.score : null,
      reason: a.message || '异常告警',
      threshold: a.score != null ? threshold : null,
      final_status: a.final_status || 'active',
      created_at: a.time || a.created_at || 0,
    })
  }

  // 预测预警（来自内存 WarningStore）
  const warnings = warningsPoll.data.value?.warnings || []
  for (const w of warnings) {
    rows.push({
      id: w.id || `${w.channel}_${w.start_ts}`,
      type: 'warning',
      channel: store.displayName(w.channel),
      raw_value: extractRawValue(w.raw_snapshot),
      score: w.max_score != null ? w.max_score : (w.score != null ? w.score : null),
      reason: w.message || '预测异常',
      threshold: w.max_score != null ? 0.5 : null,
      final_status: w.final_status || w.verify_status || 'pending',
      created_at: w.created_at || w.start_ts || 0,
    })
  }

  return rows.sort((a, b) => b.created_at - a.created_at).slice(0, 20)
})

function fmtTime(epoch: number): string {
  if (!epoch) return '—'
  const d = new Date(epoch * 1000)
  return d.toISOString().replace('T', ' ').replace(/\.\d+Z$/, '')
}

function statusLabel(s: string): string {
  const map: Record<string, string> = {
    real: '实警',
    false_alarm: '虚警',
    uncertain: '待定',
    confirmed: '已证实',
    false: '已证伪',
    pending: '待核验',
    unverifiable: '无法核验',
    active: '实报',
  }
  return map[s] || s || '—'
}

function statusClass(s: string): string {
  if (['real', 'confirmed', 'active'].includes(s)) return 'st-real'
  if (['false_alarm', 'false'].includes(s)) return 'st-false'
  if (['uncertain', 'pending'].includes(s)) return 'st-pending'
  if (s === 'unverifiable') return 'st-uncertain'
  return 'st-default'
}
</script>

<template>
  <div class="right-detail">
    <!-- 上半：通道详情 -->
    <div class="detail-top">
      <div class="section-header">
        <span class="section-title">通道详情</span>
        <span class="section-sub">{{ currentDisplayName || '—' }}</span>
      </div>
      <div v-if="currentSensor" class="detail-body">
        <div class="detail-row">
          <span class="row-label">通道名</span>
          <span class="row-value mono">{{ currentSensor.name }} <span class="row-value-sub">({{ currentSensor.channelName }})</span></span>
        </div>
        <div class="detail-row">
          <span class="row-label">量程</span>
          <span class="row-value mono">
            [{{ currentSensor.yMin.toFixed(2) }}, {{ currentSensor.yMax.toFixed(2) }}]
          </span>
        </div>
        <div class="detail-row">
          <span class="row-label">单位</span>
          <span class="row-value">{{ currentSensor.unit }}</span>
        </div>
        <div class="detail-row">
          <span class="row-label">异常阈值</span>
          <span class="row-value mono threshold">{{ currentSensor.threshold.toFixed(2) }}</span>
        </div>
        <div v-if="currentSensor.description" class="detail-desc">
          {{ currentSensor.description }}
        </div>
      </div>
      <div v-else class="detail-empty">
        请在左侧选择通道
      </div>
    </div>

    <!-- 下半：告警预警列表 -->
    <div class="detail-bottom">
      <div class="section-header">
        <span class="section-title">告警 / 预警</span>
        <span class="section-sub">最多 20 条，按时间倒序</span>
      </div>
      <div class="alert-list">
        <div v-if="!alertRows.length" class="alert-empty">
          <span class="empty-icon">✓</span>
          <span>暂无告警 / 预警</span>
        </div>
        <div
          v-for="row in alertRows"
          :key="row.id"
          class="alert-row"
          :class="row.type"
        >
          <div class="row-main">
            <span class="row-tag">{{ row.type === 'alert' ? '告警' : '预警' }}</span>
            <span class="row-channel">{{ row.channel }}</span>
            <span class="meta-status" :class="statusClass(row.final_status)">
              {{ statusLabel(row.final_status) }}
            </span>
          </div>
          <!-- 遥测值（主信息，含单位） -->
          <div class="row-raw-value">
            遥测值: <span class="value-num">{{ row.raw_value != null ? row.raw_value.toFixed(3) : '—' }}</span>
          </div>
          <!-- 结构化告警原因（小字） -->
          <div class="row-reason" :title="row.reason">{{ row.reason }}</div>
          <div class="row-time mono">{{ fmtTime(row.created_at) }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.right-detail {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #0f1530;
}

/* 上半（约 40%） */
.detail-top {
  flex: 0 0 40%;
  display: flex;
  flex-direction: column;
  border-bottom: 1px solid #2a3050;
  overflow: hidden;
}

/* 下半（约 60%） */
.detail-bottom {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.section-header {
  padding: 10px 14px;
  border-bottom: 1px solid #2a3050;
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-shrink: 0;
  background: #1a1f3a;
}

.section-title {
  font-size: 13px;
  font-weight: 500;
  color: #409eff;
  letter-spacing: 1px;
}

.section-sub {
  font-size: 11px;
  color: #7a85a8;
}

.detail-body {
  flex: 1;
  overflow-y: auto;
  padding: 10px 14px;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  padding: 5px 0;
  font-size: 12px;
  border-bottom: 1px dashed rgba(42, 48, 80, 0.5);
}

.row-label {
  color: #7a85a8;
}

.row-value {
  color: #e0e6ed;
}

.row-value.mono {
  font-family: 'Consolas', monospace;
}

.row-value-sub {
  color: #7a85a8;
  font-size: 10px;
}

.row-value.threshold {
  color: #f56c6c;
  font-weight: 500;
}

.detail-desc {
  margin-top: 10px;
  padding: 8px;
  background: rgba(42, 48, 80, 0.3);
  border-radius: 4px;
  font-size: 11px;
  color: #a0aec0;
  line-height: 1.6;
}

.detail-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #4a5278;
  font-size: 12px;
}

.alert-list {
  flex: 1;
  overflow-y: auto;
  padding: 6px;
}

.alert-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 8px;
  color: #4a5278;
  font-size: 12px;
}

.empty-icon {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: rgba(103, 194, 58, 0.15);
  color: #67c23a;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}

.alert-row {
  padding: 8px 10px;
  margin-bottom: 4px;
  border-radius: 4px;
  border-left: 3px solid;
  background: rgba(42, 48, 80, 0.4);
  font-size: 12px;
}

.alert-row.alert {
  border-left-color: #f56c6c;
}

.alert-row.warning {
  border-left-color: #e6a23c;
}

.row-main {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.row-tag {
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 2px;
  font-weight: 500;
}

.alert .row-tag {
  background: rgba(245, 108, 108, 0.2);
  color: #f56c6c;
}

.warning .row-tag {
  background: rgba(230, 162, 60, 0.2);
  color: #e6a23c;
}

.row-channel {
  color: #e0e6ed;
  font-weight: 500;
  flex: 1;
}

.row-raw-value {
  font-size: 12px;
  color: #e0e6ed;
  margin-bottom: 2px;
}

.value-num {
  font-family: 'Consolas', monospace;
  color: #409eff;
  font-weight: 500;
}

.row-reason {
  font-size: 11px;
  color: #7a85a8;
  line-height: 1.4;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.row-meta {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: #7a85a8;
  margin-bottom: 2px;
}

.meta-status {
  padding: 1px 5px;
  border-radius: 2px;
  font-weight: 500;
}

.st-real {
  background: rgba(245, 108, 108, 0.2);
  color: #f56c6c;
}

.st-false {
  background: rgba(103, 194, 58, 0.2);
  color: #67c23a;
}

.st-pending,
.st-uncertain {
  background: rgba(230, 162, 60, 0.2);
  color: #e6a23c;
}

.st-default {
  background: rgba(122, 133, 168, 0.2);
  color: #7a85a8;
}

.row-time {
  font-size: 10px;
  color: #4a5278;
}

.mono {
  font-family: 'Consolas', monospace;
}
</style>
