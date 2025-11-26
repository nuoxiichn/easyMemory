## Memory 使用速查

以下示例展示如何在已有记忆数据的情况下调用主要接口（适用于 Postgres 持久化或内存版）。

### 初始化

```python
from config_loader import load_config, init_from_config

config = load_config()  # 默认读取 config/memory_config.yaml
memory, llm_fn = init_from_config(config)  # memory 可能是 PostgresMemory 或 SimpleRAGMemory
```

如需显式指定 Postgres DSN：

```python
from memory.postgres_memory import PostgresMemory
from embedding_providers import build_embedding_from_config

cfg = load_config()
embed_fn = build_embedding_from_config(cfg.get("embedding", {}))
memory = PostgresMemory(embed_fn=embed_fn, dsn="postgresql://memo:memo@192.168.240.1:5432/ezmemo")
```

### Ingest（写入）

```python
rec_id = memory.ingest(
    "Paris is the capital of France.",
    {"dataset": "custom", "sample_id": "123", "title": "fact1"},
)
```

### Query（检索）

```python
results = memory.query(
    question="What is the capital of France?",
    constraints={"dataset": "custom", "sample_id": "123"},
    top_k=5,
)
for rec in results:
    print(rec.id, rec.metadata, rec.content[:80])
```

### Update（更新）

```python
memory.update(
    selector={"metadata.title": "fact1"},
    new_content="UPDATED CONTENT",
    new_metadata={"note": "corrected"},
)
```

### Forget（删除/软删）

```python
# 软删（deleted=True，检索时忽略）
memory.forget({"id": rec_id}, hard_delete=False)

# 硬删（从存储中移除）
memory.forget({"metadata.sample_id": "123"}, hard_delete=True)
```

### Stats（统计）

```python
print(memory.stats())
# 形如：{"total_records": 100, "alive_records": 95, "avg_content_len_chars": 420.5}
```

### Export（导出快照）

```python
snap = memory.export_artifacts(include_embedding=False)
# 可 json.dump 写盘；include_embedding=True 时导出向量
```

### RAG 直接问答

```python
result = memory.rag_answer(
    question="What is the capital of France?",
    llm_fn=llm_fn,  # 由 init_from_config 提供
    constraints={"dataset": "custom"},
    top_k=3,
)
print(result["answer"], result["used_records"])
```
