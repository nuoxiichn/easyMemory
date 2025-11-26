"""
Utility script to clean memory storage. Currently truncates memory_records table.
"""

import logging
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psycopg2

from config_loader import load_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Clean memory tables.")
    parser.add_argument("--table", type=str, default=None, help="Table to truncate (default_table if omitted). Use 'all' to truncate default + configured tables.")
    args = parser.parse_args()

    cfg = load_config()
    dsn = (cfg.get("db", {}) if isinstance(cfg, dict) else {}).get("dsn")
    default_table = (cfg.get("db", {}) if isinstance(cfg, dict) else {}).get("default_table", "memory_records")
    tables_cfg = (cfg.get("db", {}) if isinstance(cfg, dict) else {}).get("tables", {}) or {}

    if not dsn:
        logging.getLogger(__name__).error("No DSN found in config.db.dsn; aborting.")
        return

    targets = []
    if args.table == "all":
        targets = [default_table] + list(tables_cfg.values())
    elif args.table:
        targets = [args.table]
    else:
        targets = [default_table]

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        for table in targets:
            cur.execute(f"TRUNCATE TABLE {table};")
            logging.getLogger(__name__).info("Truncated table %s", table)
    conn.close()
    logging.getLogger(__name__).info("Completed truncation using DSN=%s", dsn)


if __name__ == "__main__":
    main()
