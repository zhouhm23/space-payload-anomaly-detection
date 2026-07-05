# 前端四层架构映射 (Frontend Layer Mapping)

前端工程通过标准 Vue3 分层（views/components/stores/api/composables）组织业务代码，
论文四层 PHM 架构在后端 `src/ground/phm/` 实体实现。此目录提供前端到后端四层的映射文档占位，
方便理解数据流如何从前端发起、经各层处理后返回。

## 映射关系

| 论文层级 | 后端实体 (`phm/`) | 前端对应模块 | 数据流 |
|---------|------------------|-------------|--------|
| **数据库层** | `database/ring_buffer.py`、`alert_store.py`、`warning_store.py` | `stores/`（Pinia 状态） | 后端 RingBuffer → `/api/*` → 前端 store |
| **数据操作层** | `dataops/preprocessor.py`、`feature_extractor.py` | `api/client.ts`（HTTP 封装） | 前端发请求 → dataops 预处理 → 算法层 |
| **算法库层** | `algorithm/tspulse.py`、`ttm.py`、`base.py` | `composables/useForecast.ts` | 前端触发 → `/api/forecast` → TTM-R3 |
| **模型库层** | `model/README.md`（占位） | —（本轮不实现） | 后续版本接入 |

## 各子目录说明

- `database/` — 映射说明：前端 `stores/telemetry.ts` 持有的 blocks 是后端 RingBuffer 的客户端快照
- `dataops/` — 映射说明：前端 `api/client.ts` 是数据操作层的 HTTP 入口
- `algorithm/` — 映射说明：前端 `composables/useForecast.ts` 调用后端 TTM-R3 预测
- `model/` — 占位，与后端 `phm/model/` 同步预留

详见各子目录 README。
