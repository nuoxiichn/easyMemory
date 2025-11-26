from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from embedding_providers import build_embedding_from_config
from llm_providers import build_llm_from_config
from memory.simple_memory import SimpleRAGMemory
from memory.postgres_memory import PostgresMemory
import logging


def load_config(path: str = "config/memory_config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def init_from_config(config: Dict[str, Any], table_name: Optional[str] = None) -> Tuple[SimpleRAGMemory, Any]:
    logger = logging.getLogger(__name__)
    embedding_cfg = config.get("embedding", {})
    llm_cfg = config.get("llm", {})
    db_cfg = config.get("db", {})
    tables_cfg = db_cfg.get("tables", {}) if isinstance(db_cfg, dict) else {}
    default_table = db_cfg.get("default_table", "memory_records") if isinstance(db_cfg, dict) else "memory_records"

    embed_fn = build_embedding_from_config(embedding_cfg)
    llm_fn = build_llm_from_config(llm_cfg)

    dsn = db_cfg.get("dsn") if isinstance(db_cfg, dict) else None
    # table selection can be overridden by callers (runners) via argument or config tables map
    table = table_name or default_table
    if dsn:
        memory = PostgresMemory(embed_fn=embed_fn, dsn=dsn, table_name=table)
        logger.info("Initialized PostgresMemory with DSN=%s table=%s", dsn, table)
    else:
        memory = SimpleRAGMemory(embed_fn=embed_fn)
        logger.info("Initialized SimpleRAGMemory (in-memory)")
    return memory, llm_fn
