# 数据库层映射（前端）

前端不直接访问数据库，而是通过 `/api/*` 端点获取后端 RingBuffer 的快照。

- **后端实体**：`src/ground/phm/database/ring_buffer.py`
- **前端对应**：`src/stores/telemetry.ts`（blocks 数组 = RingBuffer 的客户端镜像）
- **数据流**：后端 RingBuffer → `/api/poll` JSON → `telemetry.pollOnce()` → `blocks[]`
