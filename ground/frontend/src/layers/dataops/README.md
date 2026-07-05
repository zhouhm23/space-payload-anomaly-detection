# 数据操作层映射（前端）

前端 `api/client.ts` 封装全部 HTTP 调用，是数据操作层的客户端入口。预处理（插补+标准化）
在后端 `dataops/preprocessor.py` 完成，前端无需感知。

- **后端实体**：`src/ground/phm/dataops/preprocessor.py`
- **前端对应**：`src/api/client.ts`
