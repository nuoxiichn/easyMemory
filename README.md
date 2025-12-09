# EasyMemory


本README由ai生成 如有错误直接来问我。

简易 RAG 记忆基线，支持 HotpotQA / 2Wiki / MQuAKE 的本地小样本适配，提供统一的 ingest/query/update/forget/stats/export 接口，可配置随机/外部 embedding 和 LLM，持久化可选 Postgres。

## 快速开始
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 运行示例
- Hotpot 本地样本 → 生成报告：`python runners/run_hotpot_with_memory.py --limit 50 --top-k 5 --log-interval 5`
- 2Wiki 本地样本：`python runners/run_twiki_with_memory.py --limit 50 --top-k 5 --log-interval 5`
- MQuAKE 本地样本：`python runners/run_mquake_with_memory.py --limit 50 --top-k 5 --log-interval 5`
- LongMemEval: `python runners/run_LongMemEval_with_memory.py --limit 5 --top-k 5`
- Locomo: `runners/run_locomo_with_memory.py --limit 1 --top-k 5`
- 接口验证（可选随机向量）：`python runners/verify_memory_ops.py [--random-embed] [--hard-clean]`

报告输出至 `reports/<dataset>/report_<timestamp>.txt`，数据写入配置中的 Postgres 表（分别为 hotpot/twiki/mquake 专用表）。

## 配置
`config/memory_config.yaml` 控制 embedding/LLM/数据路径和 DB 表名。例如：
```yaml
embedding:
  provider: openai_compat
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: text-embedding-v4
  api_key: ...

llm:
  provider: deepseek
  base_url: https://api.deepseek.com
  model: deepseek-chat
  api_key: ...

db:
  dsn: postgresql://memo:memo@192.168.240.1:5432/ezmemo
  tables:
    hotpot: memory_hotpot
    twiki: memory_twiki
    mquake: memory_mquake
```

## 清理
按表清空：`python scripts/clean.py --table memory_hotpot`；清空配置的所有表：`python scripts/clean.py --table all`。
