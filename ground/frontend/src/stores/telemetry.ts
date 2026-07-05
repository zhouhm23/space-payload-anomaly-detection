/**
 * Telemetry store — block history + play/pause/reset state.
 *
 * Mirrors the legacy HTML ``state`` object: ``playing``, ``blocks``,
 * ``currentBlock``, ``channelNames``, ``sampleOffset``.
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '@/api/client'
import type { ChannelData, PollResponse } from '@/api/types'

export interface Block {
  channels: Record<string, ChannelData>
  time: number
  startIdx: number
}

export const useTelemetryStore = defineStore('telemetry', () => {
  const playing = ref(false)
  const blocks = ref<Block[]>([])
  const currentBlock = ref(-1)
  const channelNames = ref<string[]>([])
  const sampleOffset = ref(0)
  const chunkInfo = ref('就绪')
  const exhausted = ref(false)

  /** One poll cycle.  Returns true if new data arrived. */
  async function pollOnce(sourceId: string, sampleRate: number, blockSize: number): Promise<boolean> {
    try {
      const json: PollResponse = await api.poll(sourceId, sampleRate, blockSize)
      const chNames = Object.keys(json.channels)
      if (chNames.length === 0) return false

      // Dedup: skip if first timestamp identical to last block
      const firstCh = chNames[0]
      const newFirstTs = json.channels[firstCh].telemetry[0]?.[0] ?? null
      if (blocks.value.length > 0 && newFirstTs !== null) {
        const lastBlock = blocks.value[blocks.value.length - 1]
        const lastChData = lastBlock.channels[firstCh]
        if (lastChData && lastChData.telemetry.length > 0) {
          if (newFirstTs === lastChData.telemetry[0][0]) return false
        }
      }

      const block: Block = {
        channels: {},
        time: Date.now(),
        startIdx: sampleOffset.value,
      }
      for (const ch of chNames) {
        const chData = json.channels[ch]
        block.channels[ch] = {
          telemetry: chData.telemetry || [],
          scores: chData.scores || [],
        }
      }
      const maxPts = Math.max(...chNames.map((ch) => (json.channels[ch].telemetry || []).length))
      sampleOffset.value += maxPts
      blocks.value.push(block)
      channelNames.value = chNames

      if (playing.value) {
        currentBlock.value = blocks.value.length - 1
      } else if (currentBlock.value < 0) {
        currentBlock.value = 0
      }

      const totalPts = chNames.reduce((s, ch) => s + (json.channels[ch].telemetry || []).length, 0)
      let status = `块 ${blocks.value.length} | ${chNames.length}通道 | ${totalPts} 点`
      if (json.exhausted) status += ' ⚠️ 数据源耗尽'
      chunkInfo.value = status
      exhausted.value = json.exhausted
      return true
    } catch (err) {
      console.warn('API poll failed:', err)
      return false
    }
  }

  function reset(): void {
    blocks.value = []
    currentBlock.value = -1
    sampleOffset.value = 0
    chunkInfo.value = '就绪'
    exhausted.value = false
    api.reset().catch(() => {})
  }

  function prevBlock(): void {
    if (playing.value || currentBlock.value <= 0) return
    currentBlock.value--
  }

  function nextBlock(): void {
    if (playing.value || currentBlock.value >= blocks.value.length - 1) return
    currentBlock.value++
  }

  return {
    playing,
    blocks,
    currentBlock,
    channelNames,
    sampleOffset,
    chunkInfo,
    exhausted,
    pollOnce,
    reset,
    prevBlock,
    nextBlock,
  }
})
