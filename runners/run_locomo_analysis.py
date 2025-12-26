import argparse
import sys
import time
import string
import re
from pathlib import Path
import logging
from collections import Counter

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入 LoCoMo loader
from benchmarks.locomo_loader import ingest_locomo_sample, load_locomo
from config_loader import init_from_config, load_config

# ==============================================================================
# 评估辅助函数 (F1 Score & Normalization)
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
    
    # 如果标准答案为空，无法计算
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

def check_evidence_recall(retrieved_records, gold_evidence_ids):
    """
    检查检索到的记录中是否包含标准答案所需的证据ID
    
    retrieved_records: 列表，每个元素包含 metadata
    gold_evidence_ids: 列表，例如 ['D1:3', 'D2:5']
    """
    if not gold_evidence_ids:
        return False, [] # 无标准证据，无法评估检索

    # 提取检索到的所有 evidence_id (前提是 metadata 里存了)
    retrieved_ids = set()
    for rec in retrieved_records:
        meta = rec.get("metadata", {})
        # 兼容 observation 模式存的 evidence_id
        if "evidence_id" in meta and meta["evidence_id"]:
            retrieved_ids.add(meta["evidence_id"])
        
        # 兼容 dialog 模式 (如果 metadata 有 turn_id 范围等，这里简化处理)
        # 如果是 Dialog 模式，可能难以精确匹配 ID，这里主要针对 Observation 模式优化

    # 检查是否命中
    # 命中逻辑：只要找回了 Gold Evidence 中的任意一条，就算检索成功 (宽松模式)
    # 也可以改为必须找回所有 (严格模式)
    hits = [gid for gid in gold_evidence_ids if gid in retrieved_ids]
    success = len(hits) > 0
    return success, hits

# ==============================================================================
# 主程序
# ==============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="运行 LoCoMo 数据集的导入和 RAG 测试 (含详细评估)")
    parser.add_argument("--limit", type=int, default=None, help="限制处理样本数")
    parser.add_argument("--log-interval", type=int, default=5, help="日志间隔")
    parser.add_argument("--top-k", type=int, default=5, help="RAG 上下文 Top-K")
    parser.add_argument("--rag-mode", type=str, default="observation", choices=["dialog", "observation", "summary"])
    parser.add_argument("--qa-categories", type=int, nargs="+", default=None, help="指定 QA 类别 (如 1 3)")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过导入")
    # [新增参数] 判定答案正确的 F1 阈值
    parser.add_argument("--f1-threshold", type=float, default=0.3, help="判定答案正确的 F1 分数阈值 (0.0-1.0)")
    
    args = parser.parse_args()

    # 处理 qa_categories 类型安全问题
    if args.qa_categories is not None:
        if isinstance(args.qa_categories, int):
            args.qa_categories = [args.qa_categories]
        args.qa_categories = set(args.qa_categories)

    print("=" * 60)
    print("步骤 1: 加载配置")
    print("=" * 60)
    
    config = load_config()
    datasets_cfg = config.get("datasets", {}).get("locomo", {}) if isinstance(config, dict) else {}
    path = datasets_cfg.get("path")
    split = datasets_cfg.get("split", "validation")
    table_name = config.get("db", {}).get("tables", {}).get("locomo", "memory_locomo")
    
    print(f"数据集: LoCoMo ({split})")
    print(f"RAG 模式: {args.rag_mode}")
    print(f"QA 类别: {args.qa_categories if args.qa_categories else 'All'}")
    
    # 初始化
    memory, llm_fn = init_from_config(config, table_name=table_name)
    
    # 加载数据
    ds = load_locomo(split=split, path=path, fallback_to_dummy=True)
    if not ds: return
    ds = list(ds)[:args.limit] if args.limit else list(ds)
    print(f"✓ 加载 {len(ds)} 个样本")

    # 导入数据
    if not args.skip_ingest:
        print("\n" + "=" * 60)
        print(f"步骤 4: 导入数据 (模式: {args.rag_mode})")
        print("=" * 60)
        for idx, sample in enumerate(ds, start=1):
            ingest_locomo_sample(memory, sample, mode=args.rag_mode)
            if args.log_interval and idx % args.log_interval == 0:
                print(f"  进度: {idx}/{len(ds)}...", flush=True)
    
    # --------------------------------------------------------------------------
    # 执行 RAG 查询与评估
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 5: RAG 查询与详细评估")
    print("=" * 60)
    
    reports_dir = ROOT / "reports" / "locomo"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"report_{args.rag_mode}_eval_{int(time.time())}.txt"
    report_path = reports_dir / report_filename
    
    lines = []
    
    # 统计计数器
    stats = {
        "total_questions": 0,
        "metrics": {
            "success": 0,          # 找到证据且回答正确
            "reasoning_error": 0,  # 找到证据但回答错误
            "retrieval_error": 0,  # 没找到证据
            "lucky_guess": 0       # 没找到证据但蒙对了 (少见)
        },
        "scores": {
            "total_f1": 0.0,
            "evidence_recall_count": 0 # 找回了至少一条证据的问题数
        }
    }

    # 预计算总数
    total_qs = 0
    for s in ds:
        for q in s.get("qa", []):
            if args.qa_categories is None or q.get("category", 0) in args.qa_categories:
                total_qs += 1

    question_idx = 0
    print(f"开始测试 {total_qs} 个问题...")

    for sample in ds:
        sample_id = sample.get("id") or sample.get("sample_id")
        
        for qa in sample.get("qa", []):
            category = qa.get("category", 0)
            if args.qa_categories is not None and category not in args.qa_categories:
                continue
            
            question = qa.get("question", "")
            gold_answer = str(qa.get("answer", ""))
            gold_evidence = qa.get("evidence", []) # List of strings e.g. ["D1:3"]
            
            if not question: continue
            question_idx += 1
            query_constraints = {
                "dataset": "locomo", 
                "sample_id": sample_id,
                "rag_mode": args.rag_mode 
            }
            try:
                retrieved_records_objs = memory.query(
                    question=question,
                    constraints=query_constraints,
                    top_k=args.top_k
                )
                
                # [关键步骤] 将 MemoryRecord 对象转换为字典列表
                # 因为 check_evidence_recall 可能期望 {'metadata': ...} 格式
                retrieved_context = []
                for rec in retrieved_records_objs:
                    retrieved_context.append({
                        "content": rec.content,
                        "metadata": rec.metadata, # 这里有我们需要的 evidence_id
                        "id": rec.id
                    })
                    
            except Exception as e:
                logging.error(f"检索失败: {e}")
                retrieved_context = []
            
            result = memory.rag_answer(
                question=question,
                llm_fn=llm_fn,
                constraints=query_constraints,
                top_k=args.top_k
            )
            
            pred_answer = result["answer"]
            
            # 2.1 计算 F1 分数
            f1_score = compute_f1(pred_answer, gold_answer)
            is_correct = f1_score >= args.f1_threshold
            
            # 2.2 检查证据是否找到 
            has_evidence, hits = check_evidence_recall(retrieved_context, gold_evidence)

            # 3. 结果分类
            eval_status = "UNKNOWN"
            if has_evidence:
                stats["scores"]["evidence_recall_count"] += 1
                if is_correct:
                    eval_status = "✅ Success (Found & Correct)"
                    stats["metrics"]["success"] += 1
                else:
                    eval_status = "⚠️ Reasoning Error (Found but Wrong)"
                    stats["metrics"]["reasoning_error"] += 1
            else:
                if is_correct:
                    eval_status = "❓ Lucky Guess (Missed Evidence but Correct)"
                    stats["metrics"]["lucky_guess"] += 1
                else:
                    eval_status = "❌ Retrieval Error (Evidence Missed)"
                    stats["metrics"]["retrieval_error"] += 1

            stats["total_questions"] += 1
            stats["scores"]["total_f1"] += f1_score

            # 4. 记录日志行
            line = {
                "index": question_idx,
                "sample_id": sample_id,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "f1_score": round(f1_score, 2),
                "gold_evidence": gold_evidence,
                "retrieved_evidence_hit": hits,
                "status": eval_status,
                "category": category
            }
            lines.append(line)

            if args.log_interval and question_idx % args.log_interval == 0:
                print(f"  进度: {question_idx}/{total_qs} | Status: {eval_status}", flush=True)

    # ==============================================================================
    # 生成详细报告
    # ==============================================================================
    
    # 计算统计百分比
    total = stats["total_questions"] if stats["total_questions"] > 0 else 1
    avg_f1 = stats["scores"]["total_f1"] / total
    recall_rate = stats["scores"]["evidence_recall_count"] / total
    
    m = stats["metrics"]
    
    summary_text = [
        "================================================================================",
        "                               综合评估报告                                     ",
        "================================================================================",
        f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"RAG 模式: {args.rag_mode}",
        f"总问题数: {stats['total_questions']}",
        f"F1 阈值 : {args.f1_threshold}",
        "",
        "--------------------- 核心指标 ---------------------",
        f"平均 F1 分数   : {avg_f1:.4f} (越高越好)",
        f"证据召回率 (R) : {recall_rate:.2%} (检索器能否找到答案所在)",
        "",
        "--------------------- 错误分析 ---------------------",
        f"✅ (R+G+)       : {m['success']} ({m['success']/total:.1%})",
        "",
        f"⚠️ (R+G-)   : {m['reasoning_error']} ({m['reasoning_error']/total:.1%})",
        "",
        f"❌ (R-)     : {m['retrieval_error']} ({m['retrieval_error']/total:.1%})",
        "",
        f"❓ (R-G+)   : {m['lucky_guess']} ({m['lucky_guess']/total:.1%})",
        ""
    ]
    
    summary_text.append("================================================================================")
    
    # 打印到控制台
    print("\n".join(summary_text))
    
    # 写入文件
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_text) + "\n\n")
        f.write("详细测试记录:\n")
        for line in lines:
            f.write(f"[{line['status']}] (F1: {line['f1_score']})\n")
            f.write(f"Q: {line['question']}\n")
            f.write(f"A(Gold): {line['gold_answer']}\n")
            f.write(f"A(Pred): {line['pred_answer']}\n")
            f.write(f"Evidence(Gold): {line['gold_evidence']} | Hit: {line['retrieved_evidence_hit']}\n")
            f.write("-" * 40 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")

if __name__ == "__main__":
    main()