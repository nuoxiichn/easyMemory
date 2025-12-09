import argparse
import sys
import time
from pathlib import Path
import logging

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入 LongMemEval loader
from benchmarks.LongMemEval_loader import ingest_longmemeval_sample, load_longmemeval
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
        description="运行 LongMemEval 数据集的导入和 RAG 测试"
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
        default=5, 
        help="每处理 N 个样本输出一次进度（0 表示不输出）"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5, 
        help="RAG 检索时返回的 top-k 上下文数量"
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
    
    # 获取 LongMemEval 数据集配置
    datasets_cfg = config.get("datasets", {}).get("longmemeval", {}) if isinstance(config, dict) else {}
    path = datasets_cfg.get("path")
    split = datasets_cfg.get("split", "validation")
    
    print(f"数据集: LongMemEval")
    print(f"分割: {split}")
    print(f"路径: {path or '使用默认/虚拟数据'}")
    
    # 获取数据库表配置
    db_tables = config.get("db", {}).get("tables", {}) if isinstance(config, dict) else {}
    table_name = db_tables.get("longmemeval")
    
    if not table_name:
        print("警告: 未在配置中找到表名，使用默认表")
        table_name = "memory_longmemeval"
    
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
    
    # 加载 LongMemEval 数据集
    ds = load_longmemeval(split=split, path=path, fallback_to_dummy=True)
    
    if not ds:
        print("✗ 加载数据集失败")
        return
    
    # 限制样本数量
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
        print("步骤 4: 导入数据到记忆系统")
        print("=" * 60)
        
        for idx, sample in enumerate(ds, start=1):
            # 导入 LongMemEval 样本
            ingest_longmemeval_sample(memory, sample)
            
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
    
    # 创建 LongMemEval 报告目录
    reports_dir = ROOT / "reports" / "longmemeval"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"report_{int(time.time())}.txt"
    
    lines = []
    for idx, sample in enumerate(ds, start=1):
        q = sample.get("question", "")
        gold = sample.get("answer", "")
        sample_id = sample.get("id") or sample.get("question_id")
        
        # 执行 RAG 查询
        result = memory.rag_answer(
            question=q,
            llm_fn=llm_fn,
            constraints={"dataset": "longmemeval", "sample_id": sample_id},
            top_k=args.top_k,
        )
        
        line = {
            "index": idx,
            "sample_id": sample_id,
            "question": q,
            "gold_answer": gold,
            "pred_answer": result["answer"],
            "used_records": result["used_records"],
            "metadata": result["metadata"],
        }
        lines.append(line)
        
        if args.log_interval and idx % args.log_interval == 0:
            print(f"  进度: {idx}/{len(ds)} 问题已回答", flush=True)
    
    print(f"✓ 完成 {len(ds)} 个问题的回答")
    
    # ========================================================================
    # 6. 写入报告
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 6: 生成报告")
    print("=" * 60)
    
    with report_path.open("w", encoding="utf-8") as f:
        # 写入报告头
        f.write("=" * 80 + "\n")
        f.write("LongMemEval 数据集 RAG 测试报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: {len(lines)}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"数据库表: {table_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入每个样本的结果
        for line in lines:
            f.write(f"{'=' * 80}\n")
            f.write(f"样本序号: {line['index']}\n")
            f.write(f"样本ID: {line['sample_id']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"问题:\n{line['question']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"标准答案:\n{line['gold_answer']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"预测答案:\n{line['pred_answer']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"使用的记录ID: {line['used_records']}\n")
            f.write(f"元数据: {line['metadata']}\n")
            f.write(f"{'=' * 80}\n\n")
    
    print(f"✓ 报告已保存到: {report_path}")
    
    # ========================================================================
    # 7. 显示最终统计
    # ========================================================================
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"处理样本数: {len(lines)}")
    print(f"报告路径: {report_path}")
    print(f"记忆系统统计: {memory.stats()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
