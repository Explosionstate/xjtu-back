# Transformer 检测系统部署与回滚说明

## 1. 部署对象

- 后端：`D:/xjtu/xjtu-back`
- 前端：`D:/xjtu/xjtu-transformer-front`

## 2. 标准部署步骤

### 2.1 后端部署

```bash
cd D:/xjtu/xjtu-back
pip install -r requirements.txt
python scripts/ops.py restart --host 127.0.0.1 --port 8000 --reload
python scripts/ops.py check --probe --base http://127.0.0.1:8000
```

### 2.2 前端部署

```bash
cd D:/xjtu/xjtu-transformer-front
npm install
npm run build
npm run dev
```

### 2.3 联调验证

- 打开 `http://127.0.0.1:5175`
- 登录后依次执行：runtime、quick-test、batch-run、compare

## 3. 运行维护

- 启动：`python scripts/ops.py start --host 127.0.0.1 --port 8000 --reload`
- 停止：`python scripts/ops.py stop --port 8000`
- 重启：`python scripts/ops.py restart --host 127.0.0.1 --port 8000 --reload`
- 健康检查：`python scripts/ops.py check --probe --base http://127.0.0.1:8000`

## 4. 回滚策略

### 4.1 代码回滚

- 回滚后端目录 `xjtu-back` 到上一个稳定版本。
- 回滚前端目录 `xjtu-transformer-front` 到上一个稳定版本。
- 回滚后重新执行部署步骤 2.1 和 2.2。

### 4.2 数据回滚（快照表）

- Day6 新增表：`transformer_eval_snapshot`
- 若需清空该功能数据，可仅清理该表，不影响主业务表。
- 若需彻底回退 Day6 功能，建议：
  1. 回滚代码。
  2. 备份后删除 `transformer_eval_snapshot` 表。

## 5. 故障处理速查

- 端口占用：
  - `python scripts/ops.py stop --port 8000`
  - `python scripts/ops.py restart --host 127.0.0.1 --port 8000 --reload`
- 本地模型找不到：检查 `D:/xjtu/local_models` 路径与模型目录命名。
- 生成慢/超时：先使用 quick-test 快速模式（`run_generation=false`），并降低 `max_new_tokens`。
- CORS 提示但 OPTIONS=200：优先排查后端 500 traceback。
