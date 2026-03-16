# Day7 验收清单（Transformer 项目）

## 1. 验收范围

- 后端：`xjtu-back` 的 `/transformer/*` 全链路能力。
- 前端：`xjtu-transformer-front` 的检测台、评测报告页、批次历史对比。
- 交付：部署/回滚文档、评测结论固化文档。

## 2. 环境前提

- Python 虚拟环境已安装 `requirements.txt`。
- `xjtu-back` 可通过 `python scripts/ops.py restart --host 127.0.0.1 --port 8000 --reload` 正常启动。
- 本地模型目录可被识别（例如 `D:/xjtu/local_models/Qwen/Qwen2___5-1___5B-Instruct`）。
- 新前端可通过 `npm run dev` 启动在 `5175`。

## 3. 接口验收项

使用同一管理员 token 验证：

- 基础能力
  - `GET /transformer/runtime`
  - `POST /transformer/chat/completions`
  - `POST /transformer/classify`
  - `POST /transformer/cluster`
  - `POST /transformer/rag/analyze`
  - `POST /transformer/eval/run`
- 专题测试与报告
  - `GET /transformer/topics/templates`
  - `POST /transformer/topics/quick-test`
  - `POST /transformer/topics/quick-test/export`
  - `POST /transformer/topics/quick-test/report-markdown`
- Day6 历史能力
  - `POST /transformer/topics/batch-run`
  - `GET /transformer/topics/snapshots`
  - `POST /transformer/topics/compare`

验收标准：

- 所有接口返回 2xx（参数合法时）。
- `chat/rag` 返回 `diagnostics` 字段。
- `quick-test` 返回 02-12 专题明细（11 条）。
- `batch-run` 可落库快照，`snapshots` 可查出新记录。
- `compare` 能输出 topic delta。

## 4. 前端验收项

- 登录后可读取 runtime。
- 一键测试可展示评分表（含检索块、模式、时延）。
- 评测报告页可展示自动汇总图。
- 可导出 CSV 与 Markdown 报告。
- Day6 区域可执行批次、加载快照并进行历史对比。

## 5. 稳定性验收项

- 本地模型并发受限（队列满返回 429，不出现进程卡死）。
- GPU OOM 可自动降级 CPU 继续生成。
- quick-test 快速模式默认不触发重型生成，避免长时间阻塞。
- `ops.py` 支持 `start/stop/restart/check`，可恢复端口占用异常。

## 6. 验收记录（填写区）

- 验收日期：`____-__-__`
- 验收人：`________`
- 版本标识（commit/tag）：`________`
- 结论：`通过 / 有条件通过 / 不通过`
- 备注：`________`
