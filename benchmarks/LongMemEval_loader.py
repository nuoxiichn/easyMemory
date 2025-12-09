import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

# SimpleRAGMemory 在需要时导入（避免测试时的循环依赖）
try:
    from memory.simple_memory import SimpleRAGMemory
except ImportError:
    SimpleRAGMemory = None


def _dummy_longmemeval_sample() -> Dict[str, Any]:
    """
    返回一个虚拟的 LongMemEval 样本，用于测试和演示。
    
    LongMemEval 数据集格式:
    - question_id: 问题唯一标识符
    - question: 问题文本
    - answer: 答案文本
    - haystack_sessions: 对话历史会话列表（每个会话是一个对话轮次列表）
    - question_type: 问题类型
    - question_date: 问题提出的日期
    - haystack_dates: 每个会话的日期列表
    """
    return {
        "question_id": "dummy-longmemeval-0",
        "question": "What is the main topic discussed in the conversation?",
        "answer": "Transportation management technology",
        "question_type": "single-session-user",
        "question_date": "2023/05/30 (Tue) 22:36",
        "haystack_dates": ["2023/05/20 (Sat) 04:48", "2023/05/20 (Sat) 10:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Tell me about transportation management."},
                {"role": "assistant", "content": "Transportation management involves optimizing logistics."}
            ],
            [
                {"role": "user", "content": "What technologies are used?"},
                {"role": "assistant", "content": "Common technologies include TMS and GPS tracking."}
            ]
        ],
    }


def _normalize_sample_ids(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    标准化样本ID字段。
    LongMemEval 使用 'question_id'，将其转换为标准的 'id' 字段。
    """
    normalized = []
    for s in samples:
        s = dict(s)
        if "id" not in s:
            if "question_id" in s:
                s["id"] = s["question_id"]
            elif "_id" in s:
                s["id"] = s["_id"]
        normalized.append(s)
    return normalized


def load_longmemeval(
    split: str = "train",
    fallback_to_dummy: bool = True,
    path: Optional[Union[str, Path]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    加载 LongMemEval 数据集。
    
    参数:
        split: 数据集分割（train/validation/test）
        fallback_to_dummy: 如果加载失败，是否返回虚拟样本
        path: 本地数据文件路径
    
    返回:
        样本列表，如果失败且 fallback_to_dummy=False 则返回 None
    
    实现方式:
    1. 从本地文件加载（如果提供了 path）
    2. 如果加载失败且 fallback_to_dummy=True，返回虚拟样本
    """
    
    # 从本地 JSON 文件加载
    if path:
        try:
            p = Path(path)
            if not p.exists():
                print(f"警告: 文件不存在: {path}")
            else:
                data = json.loads(p.read_text(encoding='utf-8'))
                print(f"✓ 从本地文件加载了 {len(data)} 个样本: {path}")
                return _normalize_sample_ids(data)
        except Exception as e:
            print(f"警告: 从本地文件加载失败: {e}")
    
    # 返回虚拟样本
    if fallback_to_dummy:
        print("使用虚拟样本进行测试")
        return [_dummy_longmemeval_sample()]
    
    return None


def ingest_longmemeval_sample(
    memory, 
    sample: Dict[str, Any]
) -> List[str]:
    """
    将 LongMemEval 样本的对话历史导入到记忆系统中。
    
    参数:
        memory: 记忆系统实例
        sample: 要导入的样本
    
    返回:
        导入的记录ID列表
    
    实现说明:
    - LongMemEval 包含多个对话会话（haystack_sessions）
    - 每个会话包含多轮对话（user 和 assistant 的交互）
    - 我们将每个会话作为一个完整的上下文导入
    - 不使用 haystack_session_ids 和 answer_session_ids 字段
    """
    ingested_ids: List[str] = []
    
    sample_id = sample.get("id") or sample.get("question_id")
    question = sample.get("question", "")
    question_type = sample.get("question_type", "unknown")
    question_date = sample.get("question_date", "")
    
    # 获取对话会话列表
    haystack_sessions = sample.get("haystack_sessions", [])
    haystack_dates = sample.get("haystack_dates", [])
    
    # 为每个对话会话创建一个记录
    for idx, session in enumerate(haystack_sessions):
        # 将对话轮次组合成一个连贯的文本
        conversation_parts = []
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conversation_parts.append(f"{role.capitalize()}: {content}")
        
        # 组合成完整的对话文本
        conversation_text = "\n\n".join(conversation_parts)
        
        if conversation_text:
            # 获取会话日期
            session_date = haystack_dates[idx] if idx < len(haystack_dates) else "unknown"
            
            metadata = {
                "dataset": "longmemeval",
                "sample_id": sample_id,
                "question": question,
                "question_type": question_type,
                "question_date": question_date,
                "session_date": session_date,
                "session_index": idx,
                "type": "conversation",
            }
            
            rec_id = memory.ingest(conversation_text, metadata)
            ingested_ids.append(rec_id)
    
    return ingested_ids


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    测试 loader 功能
    """
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    
    # 1. 测试加载数据
    print("测试1: 加载虚拟数据")
    samples = load_longmemeval(fallback_to_dummy=True)
    if samples:
        print(f"✓ 成功加载 {len(samples)} 个样本")
        print(f"第一个样本的键: {list(samples[0].keys())}")
        print(f"问题: {samples[0].get('question')}")
        print(f"对话会话数: {len(samples[0].get('haystack_sessions', []))}")
    
    # 2. 测试导入到内存（不使用数据库）
    print("\n测试2: 导入到内存系统")
    from memory.simple_memory import SimpleRAGMemory
    
    def dummy_embed(text):
        import numpy as np
        return np.random.rand(128)  # 随机向量用于测试
    
    memory = SimpleRAGMemory(embed_fn=dummy_embed)
    
    for sample in samples:
        ids = ingest_longmemeval_sample(memory, sample)
        sample_id = sample.get('id') or sample.get('question_id')
        print(f"✓ 样本 {sample_id} 导入了 {len(ids)} 条对话记录")
    
    print(f"\n记忆系统统计: {memory.stats()}")
