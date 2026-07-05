# 数据库层 (Database Layer)

实时 PHM 数据存储。本轮迭代仅实现**实时层**，历史/中间/结果分区预留扩展位。

## 已实现

| 模块 | 职责 | 说明 |
|------|------|------|
| `ring_buffer.py` | 实时遥测环形缓冲 | 多通道线程安全，迁移自 `server.py` 的 `ring_buffers` |
| `alert_store.py` | 实测告警内存队列 | 确认异常（score>0.7），来自天基检测 |
| `warning_store.py` | 预测预警内存队列+状态机 | pending→confirmed/false 生命周期 |

## 预留扩展（本轮不实现）

- **历史分区**：长期遥测归档（目前仅 RingBuffer 滚动 20000 点）
- **中间分区**：模型推理中间结果缓存
- **结果分区**：健康报告、RUL 预测结果持久化

后续接入真实数据库（SQLite/InfluxDB）时，仅需新增 `historical.py` / `intermediate.py` / `results.py` 实现，业务层通过依赖注入切换。
