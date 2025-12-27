import argparse
import sys
import time
import string
import re
import logging
from pathlib import Path
from collections import Counter

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入 LongMemEval loader
from benchmarks.LongMemEval_loader import ingest_longmemeval_sample, load_longmemeval
from config_loader import init_from_config, load_config


# ==============================================================================
# 评估辅助函数
# ==============================================================================

def normalize_answer(s):
    """标准化答案文本：去标点、转小写、去多余空格"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def compute_f1(prediction, ground_truth):
    """计算预测答案和标准答案的 F1 Token Overlap 分数"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def check_session_recall(retrieved_records, gold_session_ids):
    """
    检查检索到的记录是否来自标准答案所在的 Session
    
    LongMemEval 的 ground truth 提供了 'answer_session_ids'。
    Loader 在存入 Memory 时，将原始 session id 存为了 'origin_session_id'。
    """
    if not gold_session_ids:
        # 有些题目可能没有明确的 answer_session_ids，或者全历史都相关
        # 这种情况下无法严格评估检索，暂时视为 True 或忽略
        return False, []

    retrieved_sessions = set()
    for rec in retrieved_records:
        meta = rec.get("metadata", {})
        # 获取 loader 存入的原始 session id
        sid = meta.get("origin_session_id")
        if sid:
            retrieved_sessions.add(sid)
    
    # 检查命中情况：只要检索到了任意一个包含答案的 session 就算命中
    hits = [gid for gid in gold_session_ids if gid in retrieved_sessions]
    success = len(hits) > 0
    return success, hits


# ==============================================================================
# 主程序
# ==============================================================================

def main() -> None:
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="运行 LongMemEval 数据集的导入和 RAG 测试 (含详细评估)")
    
    parser.add_argument("--limit", type=int, default=None, help="限制样本数量 (随机抽样)")
    parser.add_argument("--log-interval", type=int, default=5, help="日志输出间隔")
    parser.add_argument("--top-k", type=int, default=5, help="RAG 上下文 Top-K")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过导入步骤")
    
    # [新增] 筛选参数
    parser.add_argument(
        "--target-types", 
        type=str, 
        nargs="+", 
        default=None, 
        help="筛选特定的问题类型 (e.g. --target-types temporal-reasoning multi-session)"
    )
    
    # [新增] 评估参数
    parser.add_argument("--f1-threshold", type=float, default=0.3, help="判定答案正确的 F1 阈值")

    args = parser.parse_args()
    
    # ========================================================================
    # 1. 加载配置
    # ========================================================================
    print("=" * 60)
    print("步骤 1: 加载配置")
    print("=" * 60)
    
    config = load_config()
    datasets_cfg = config.get("datasets", {}).get("longmemeval", {}) if isinstance(config, dict) else {}
    path = datasets_cfg.get("path")
    split = datasets_cfg.get("split", "validation")
    
    # 获取表名
    db_tables = config.get("db", {}).get("tables", {}) if isinstance(config, dict) else {}
    table_name = db_tables.get("longmemeval", "memory_longmemeval")
    
    print(f"数据集: LongMemEval")
    print(f"目标类型: {args.target_types if args.target_types else 'All'}")
    print(f"数据库表: {table_name}")
    
    # ========================================================================
    # 2. 初始化
    # ========================================================================
    memory, llm_fn = init_from_config(config, table_name=table_name)
    print(f"✓ 记忆系统初始化完成")
    
    # ========================================================================
    # 3. 加载与筛选数据
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 3: 加载数据集")
    print("=" * 60)
    
    # 使用新版 loader，传入筛选参数
    ds = load_longmemeval(
        path=path,
        target_types=args.target_types,
        limit=args.limit,
        fallback_to_dummy=True
    )
    
    if not ds:
        print("✗ 加载数据集失败")
        return
    
    print(f"✓ 成功加载并筛选出 {len(ds)} 个样本")
    
    # ========================================================================
    # 4. 导入数据
    # ========================================================================
    if not args.skip_ingest:
        print("\n" + "=" * 60)
        print("步骤 4: 导入数据到记忆系统")
        print("=" * 60)
        
        for idx, sample in enumerate(ds, start=1):
            ingest_longmemeval_sample(memory, sample)
            if args.log_interval and idx % args.log_interval == 0:
                print(f"  进度: {idx}/{len(ds)} 样本已导入", flush=True)
        
        print(f"✓ 完成导入 {len(ds)} 个样本")
    else:
        print("\n跳过导入步骤（使用已有数据）")
    
    # ========================================================================
    # 5. RAG 查询与详细评估
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 5: 执行 RAG 查询并生成报告")
    print("=" * 60)
    
    reports_dir = ROOT / "reports" / "longmemeval"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"report_longmemeval_{int(time.time())}.txt"
    
    lines = []
    
    # 统计计数器
    stats = {
        "total": 0,
        "metrics": {
            "success": 0,          # R+ G+
            "reasoning_error": 0,  # R+ G-
            "retrieval_error": 0,  # R- G-
            "lucky_guess": 0       # R- G+
        },
        "scores": {
            "total_f1": 0.0,
            "session_recall_count": 0
        },
        "by_type": {} # 按 question_type 统计 F1
    }
    
    for idx, sample in enumerate(ds, start=1):
        sample_id = sample.get("id") or sample.get("question_id")
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")
        q_type = sample.get("question_type", "unknown")
        
        # LongMemEval 提供的证据 session IDs
        gold_session_ids = sample.get("answer_session_ids", []) 
        # 如果 dataset 里没提供 answer_session_ids，尝试 fallback 到 haystack_session_ids
        if not gold_session_ids:
             gold_session_ids = sample.get("haystack_session_ids", [])

        # 构造约束
        constraints = {"dataset": "longmemeval", "sample_id": sample_id}
        
        # ---------------------------------------------------------
        # A. 显式检索 (用于评估检索质量)
        # ---------------------------------------------------------
        retrieved_context = []
        try:
            # 调用 query 获取 MemoryRecord 对象列表
            retrieved_objs = memory.query(
                question=question,
                constraints=constraints,
                top_k=args.top_k
            )
            # 转为字典以便分析
            for rec in retrieved_objs:
                retrieved_context.append({
                    "content": rec.content,
                    "metadata": rec.metadata,
                    "id": rec.id
                })
        except Exception as e:
            logging.error(f"检索出错: {e}")
            retrieved_context = []
            
        # ---------------------------------------------------------
        # B. RAG 生成 (生成答案)
        # ---------------------------------------------------------
        result = memory.rag_answer(
            question=question,
            llm_fn=llm_fn,
            constraints=constraints,
            top_k=args.top_k
        )
        pred_answer = result["answer"]
        
        # ---------------------------------------------------------
        # C. 自动评估
        # ---------------------------------------------------------
        # 1. 计算 F1
        f1 = compute_f1(pred_answer, gold_answer)
        is_correct = f1 >= args.f1_threshold
        
        # 2. 检查 Session 召回
        has_evidence, hits = check_session_recall(retrieved_context, gold_session_ids)
        
        # 3. 归类状态
        status = "UNKNOWN"
        if has_evidence:
            stats["scores"]["session_recall_count"] += 1
            if is_correct:
                status = "✅ Success"
                stats["metrics"]["success"] += 1
            else:
                status = "⚠️ Reasoning Error"
                stats["metrics"]["reasoning_error"] += 1
        else:
            if is_correct:
                status = "❓ Lucky Guess"
                stats["metrics"]["lucky_guess"] += 1
            else:
                status = "❌ Retrieval Error"
                stats["metrics"]["retrieval_error"] += 1
        
        # 4. 更新统计
        stats["total"] += 1
        stats["scores"]["total_f1"] += f1
        
        # 按类型统计 F1
        if q_type not in stats["by_type"]:
            stats["by_type"][q_type] = {"sum_f1": 0.0, "count": 0}
        stats["by_type"][q_type]["sum_f1"] += f1
        stats["by_type"][q_type]["count"] += 1
        
        # 记录
        line = {
            "index": idx,
            "sample_id": sample_id,
            "type": q_type,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "f1_score": round(f1, 2),
            "status": status,
            "gold_sessions": gold_session_ids,
            "hit_sessions": hits,
            "metadata": result.get("metadata", [])
        }
        lines.append(line)
        
        if args.log_interval and idx % args.log_interval == 0:
            print(f"  进度: {idx}/{len(ds)} | Status: {status} | F1: {f1:.2f}", flush=True)

    # ========================================================================
    # 6. 生成报告
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤 6: 生成统计报告")
    print("=" * 60)
    
    total = stats["total"] if stats["total"] > 0 else 1
    avg_f1 = stats["scores"]["total_f1"] / total
    recall_rate = stats["scores"]["session_recall_count"] / total
    m = stats["metrics"]
    
    summary = [
        "================================================================================",
        "                           LongMemEval 评估报告                                 ",
        "================================================================================",
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"样本数: {total}",
        f"筛选类型: {args.target_types}",
        f"F1 阈值: {args.f1_threshold}",
        "",
        "--------------------- 核心指标 ---------------------",
        f"平均 F1 分数   : {avg_f1:.4f}",
        f"Session 召回率 : {recall_rate:.2%}",
        "",
        "--------------------- 详细分布 ---------------------",
        f"✅ Success (R+G+)       : {m['success']} ({m['success']/total:.1%})",
        f"⚠️ Reasoning Error (R+G-): {m['reasoning_error']} ({m['reasoning_error']/total:.1%})",
        f"❌ Retrieval Error (R-)   : {m['retrieval_error']} ({m['retrieval_error']/total:.1%})",
        f"❓ Lucky Guess (R-G+)     : {m['lucky_guess']} ({m['lucky_guess']/total:.1%})",
        "",
        "--------------------- 类型分析 (Avg F1) ------------"
    ]
    
    for q_t, val in stats["by_type"].items():
        type_avg = val["sum_f1"] / val["count"]
        summary.append(f"{q_t:<25}: {type_avg:.4f} (n={val['count']})")
        
    summary.append("================================================================================")
    
    # 打印和保存
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with report_path.open("w", encoding="utf-8") as f:
        f.write(summary_text + "\n\n")
        f.write("详细记录:\n")
        for line in lines:
            f.write(f"[{line['status']}] [Type: {line['type']}] F1:{line['f1_score']}\n")
            f.write(f"Q: {line['question']}\n")
            f.write(f"A_Gold: {line['gold_answer']}\n")
            f.write(f"A_Pred: {line['pred_answer']}\n")
            f.write(f"Sessions Hit: {line['hit_sessions']} / {line['gold_sessions']}\n")
            f.write("-" * 40 + "\n")
            
    print(f"\n✓ 报告已保存: {report_path}")
    print(f"✓ 记忆系统统计: {memory.stats()}")

if __name__ == "__main__":
    main()