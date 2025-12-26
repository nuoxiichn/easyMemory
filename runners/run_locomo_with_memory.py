import argparse
import sys
import time
from pathlib import Path
import logging

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入 LoCoMo loader
from benchmarks.locomo_loader import ingest_locomo_sample, load_locomo
from config_loader import init_from_config, load_config


def main() -> None:
    """
    主函数: 运行数据集的导入和 RAG 测试流程
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="运行 LoCoMo 数据集的导入和 RAG 测试"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="限制处理的样本数量（默认: 全部）"
    )
    parser.add_argument(
        "--log-interval", 
        type=int, 
        default=10, 
        help="每处理 N 个样本输出一次进度（0 表示不输出）"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5, 
        help="RAG 检索时返回的 top-k 上下文数量"
    )
    parser.add_argument(
        "--rag-mode",
        type=str,
        default="observation",
        choices=["dialog", "observation", "summary"],
        help="选择 RAG 的检索源模式 (默认: observation)"
    )
    
    parser.add_argument(
        "--qa-categories",
        type=int,
        nargs="+",
        default=None,
        help="仅测试指定类别的 QA (例如: --qa-categories 1 3)，默认全部"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="跳过导入步骤，直接使用已有数据进行查询"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # 1. 加载配置
    # ========================================================================
    print("=" * 60)
    print("步骤 1: 加载配置")
    print("=" * 60)
    
    config = load_config()
    
    datasets_cfg = config.get("datasets", {}).get("locomo", {}) if isinstance(config, dict) else {}
    path = datasets_cfg.get("path")
    split = datasets_cfg.get("split", "validation")
    
    print(f"数据集: LoCoMo")
    print(f"分割: {split}")
    print(f"路径: {path or '使用默认/虚拟数据'}")
    print(f"RAG 模式: {args.rag_mode}")
    qa_filter_msg = str(args.qa_categories) if args.qa_categories else "All (所有类别)"
    print(f"测试 QA 类别: {qa_filter_msg}")
    
    db_tables = config.get("db", {}).get("tables", {}) if isinstance(config, dict) else {}
    table_name = db_tables.get("locomo")
    
    if not table_name:
        print("警告: 未在配置中找到表名，使用默认表")
        table_name = "memory_locomo"
    
    print(f"数据库表: {table_name}")
    
    # ========================================================================
    # 2. 初始化记忆系统和 LLM
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 2: 初始化记忆系统和 LLM")
    print("=" * 60)
    
    memory, llm_fn = init_from_config(config, table_name=table_name)
    print(f"✓ 记忆系统初始化完成")
    
    # ========================================================================
    # 3. 加载数据集
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 3: 加载数据集")
    print("=" * 60)
    
    ds = load_locomo(split=split, path=path, fallback_to_dummy=True)
    
    if not ds:
        print("✗ 加载数据集失败")
        return
    
    if args.limit is not None:
        ds = list(ds)[: args.limit]
    else:
        ds = list(ds)
    
    print(f"✓ 成功加载 {len(ds)} 个样本")
    
    # ========================================================================
    # 4. 导入数据到记忆系统
    # ========================================================================
    if not args.skip_ingest:
        print("\n" + "=" * 60)
        print(f"步骤 4: 导入数据到记忆系统 (模式: {args.rag_mode})")
        print("=" * 60)
        
        for idx, sample in enumerate(ds, start=1):
            ingest_locomo_sample(memory, sample, mode=args.rag_mode)
            
            if args.log_interval and idx % args.log_interval == 0:
                print(f"  进度: {idx}/{len(ds)} 样本已导入", flush=True)
        
        print(f"✓ 完成导入 {len(ds)} 个样本")
        print(f"记忆系统统计: {memory.stats()}")
    else:
        print("\n跳过导入步骤（使用已有数据）")
    
    # ========================================================================
    # 5. 执行 RAG 查询并生成报告
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 5: 执行 RAG 查询并生成报告")
    print("=" * 60)
    
    reports_dir = ROOT / "reports" / "locomo"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"report_{args.rag_mode}_{int(time.time())}.txt"
    
    lines = []
    total_questions = 0
    for sample in ds:
        for qa in sample.get("qa", []):
            cat = qa.get("category", 0)
            if args.qa_categories is None or cat in args.qa_categories:
                total_questions += 1
                
    print(f"计划测试问题总数: {total_questions} (过滤条件: {qa_filter_msg})")
    
    question_idx = 0
    
    for sample_idx, sample in enumerate(ds, start=1):
        sample_id = sample.get("id") or sample.get("sample_id")
        qa_list = sample.get("qa", [])
        
        for qa in qa_list:
            question = qa.get("question", "")
            gold_answer = qa.get("answer", "")
            category = qa.get("category", 0)
            evidence = qa.get("evidence", [])
            
            if args.qa_categories is not None and category not in args.qa_categories:
                continue

            if not question:
                continue
            
            question_idx += 1
            
            query_constraints = {
                "dataset": "locomo", 
                "sample_id": sample_id,
                "rag_mode": args.rag_mode 
            }
            
            result = memory.rag_answer(
                question=question,
                llm_fn=llm_fn,
                constraints=query_constraints,
                top_k=args.top_k,
            )
            
            line = {
                "index": question_idx,
                "sample_id": sample_id,
                "question": question,
                "gold_answer": str(gold_answer),
                "pred_answer": result["answer"],
                "category": category,
                "evidence": evidence,
                "used_records": result["used_records"],
                "metadata": result["metadata"],
            }
            lines.append(line)
            
            if args.log_interval and question_idx % args.log_interval == 0:
                print(f"  进度: {question_idx}/{total_questions} 问题已回答", flush=True)
    
    print(f"✓ 完成 {len(lines)} 个问题的回答")
    
    # ========================================================================
    # 6. 写入报告
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 6: 生成报告")
    print("=" * 60)
    
    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LoCoMo 数据集 RAG 测试报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试模式: {args.rag_mode}\n")
        f.write(f"QA类别筛选: {qa_filter_msg}\n") # 记录筛选条件
        f.write(f"样本数量: {len(ds)}\n")
        f.write(f"问题数量: {len(lines)}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"数据库表: {table_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # ... (后续无需更改) ...
        category_counts = {}
        for line in lines:
            cat = line.get("category", 0)
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        f.write("问题分类统计:\n")
        for cat, count in sorted(category_counts.items()):
            f.write(f"  Category {cat}: {count} 个问题\n")
        f.write("\n")
        
        for line in lines:
            f.write(f"{'=' * 80}\n")
            f.write(f"问题序号: {line['index']}\n")
            f.write(f"样本ID: {line['sample_id']}\n")
            f.write(f"分类: Category {line['category']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"问题:\n{line['question']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"标准答案:\n{line['gold_answer']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"预测答案:\n{line['pred_answer']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"证据: {', '.join(line['evidence'])}\n")
            f.write(f"使用的记录ID: {line['used_records']}\n")
            f.write(f"元数据: {line['metadata']}\n")
            f.write(f"{'=' * 80}\n\n")
    
    print(f"✓ 报告已保存到: {report_path}")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"处理样本数: {len(ds)}")
    print(f"处理问题数: {len(lines)}")
    print(f"报告路径: {report_path}")
    print(f"记忆系统统计: {memory.stats()}")
    print("=" * 60)


if __name__ == "__main__":
    main()