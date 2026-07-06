# 数据库层 (Database Layer)

实时 PHM 数据存储，包含内存实时层 + SQLite 持久化层。

## 已实现

| 模块 | 职责 | 存储方式 |
|------|------|----------|
| `ring_buffer.py` | 实时遥测环形缓冲 | 多通道线程安全内存，滚动 20000 点 |
| `alert_store.py` | 实测告警内存队列 | 确认异常（score>0.7），来自天基检测 |
| `warning_store.py` | 预测预警内存队列+状态机 | pending→confirmed/false 生命周期 |
| `sqlite_store.py` | **持久化存储** | SQLite WAL 模式，异步批量写入 |

### SQLite 持久化

**双写策略**：RingBuffer/AlertStore/WarningStore 保持同步内存写入（低延迟，供前端实时显示），SQLiteStore 通过后台 daemon 线程异步批量 flush（每 200 条或每 2 秒）。

**三张表**：
- `raw_telemetry` — 原始遥测数据（channel, raw, score, timestamps）
- `detection_results` — 三层级联检测详情（L1/L2/L3 决策、分数、规则、最终分数）
- `alert_records` — 告警+预警记录（类型、分数、消息、生命周期状态）

**DB 路径**：`src/ground/data/phm.db`（已加入 .gitignore）

**查询 API**（同步，供 `/api/history` 和 `/api/detection` 调用）：
- `query_history(channel, start_time, end_time, limit)` — 原始遥测历史
- `query_detection(channel, limit)` — 三层检测详情
- `query_alerts(limit)` — 告警记录
- `stats()` — 各表行数 + 队列深度
