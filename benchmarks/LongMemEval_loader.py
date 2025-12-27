import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

# SimpleRAGMemory 在需要时导入
try:
    from memory.simple_memory import SimpleRAGMemory
except ImportError:
    SimpleRAGMemory = None


def _dummy_longmemeval_sample() -> Dict[str, Any]:
    """
    返回一个符合 LongMemEval 真实结构的虚拟样本。
    """
    return {
        "question_id": "dummy-001",
        "question_type": "multi-session",
        "question": "How much more did I have to pay?",
        "question_date": "2023/05/30 (Tue) 20:07",
        "answer": "$300",
        "haystack_dates": [
            "2023/05/20 (Sat) 01:24",
            "2023/05/21 (Sun) 05:47"
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Tell me about Eran Stark."},
                {"role": "assistant", "content": "I could not find specific information."}
            ],
            [
                {"role": "user", "content": "I've booked a trip... corrected price was $2,800.", "has_answer": True},
                {"role": "assistant", "content": "I'm happy to help... be cautious about fraud.", "has_answer": False}
            ]
        ]
    }


def _normalize_sample_ids(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 question_id 统一为 standard id 字段"""
    normalized = []
    for s in samples:
        s = dict(s)
        if "id" not in s:
            s["id"] = s.get("question_id") or s.get("_id") or str(uuid.uuid4())
        normalized.append(s)
    return normalized


def load_longmemeval(
    path: Optional[Union[str, Path]] = None,
    split: str = "train",
    target_types: Optional[Union[List[str], str]] = None,
    limit: Optional[int] = None,
    random_seed: int = 42,
    fallback_to_dummy: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """
    加载并筛选 LongMemEval 数据集。

    参数:
        path: 本地数据文件路径 (json格式)
        split: (预留字段，LongMemEval通常只有一个文件)
        target_types: 想要筛选的问题类型，可以是单个字符串或字符串列表。
                      例如: ["multi-session", "temporal-reasoning"]
                      如果为 None，则选择所有类型。
        limit: 限制返回的样本数量（随机抽样）。如果为 None，返回所有符合条件的样本。
        random_seed: 随机抽样种子，保证实验可复现。
        fallback_to_dummy: 如果加载失败，是否返回虚拟数据。

    返回:
        筛选和抽样后的样本列表。
    """
    data = []
    
    # 1. 加载原始数据
    if path:
        try:
            p = Path(path)
            if not p.exists():
                print(f"警告: 文件不存在: {path}")
            else:
                raw_data = json.loads(p.read_text(encoding='utf-8'))
                print(f"✓ 从文件加载了 {len(raw_data)} 条原始数据")
                data = raw_data
        except Exception as e:
            print(f"警告: 加载失败: {e}")

    # 如果没加载到数据且允许fallback，使用 dummy
    if not data and fallback_to_dummy:
        print("警告: 未提供路径或加载失败，使用虚拟样本进行测试")
        data = [_dummy_longmemeval_sample()]

    if not data:
        return None

    # 2. 筛选 (Filter) - 根据 question_type
    if target_types:
        if isinstance(target_types, str):
            target_types = [target_types]
        
        # 过滤数据
        original_count = len(data)
        data = [
            item for item in data 
            if item.get("question_type") in target_types
        ]
        print(f"✓ 类型筛选: 保留 {target_types}, {original_count} -> {len(data)} 条")

    # 3. 抽样 (Sample) - 根据 limit
    if limit is not None:
        if limit < len(data):
            random.seed(random_seed)
            data = random.sample(data, limit)
            print(f"✓ 随机抽样: 限制为 {limit} 条 (Seed: {random_seed})")
        else:
            print(f"提示: limit ({limit}) >= 数据量 ({len(data)})，返回全部数据")

    return _normalize_sample_ids(data)


def ingest_longmemeval_sample(
    memory, 
    sample: Dict[str, Any]
) -> List[str]:
    """
    将 LongMemEval 样本导入记忆系统。
    
    数据结构映射:
    - haystack_sessions: 包含多个 Session (List[List[Turn]])
    - haystack_dates: 对应每个 Session 的时间戳 (List[str])
    """
    ingested_ids: List[str] = []
    
    sample_id = sample.get("id")
    question_type = sample.get("question_type", "unknown")
    
    # 获取核心数据
    sessions = sample.get("haystack_sessions", [])
    dates = sample.get("haystack_dates", [])
    session_ids = sample.get("haystack_session_ids", []) # 可能不存在，做兼容处理
    
    # 遍历每个 Session
    for idx, session_turns in enumerate(sessions):
        conversation_lines = []
        for turn in session_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conversation_lines.append(f"{role}: {content}")
        
        full_text = "\n".join(conversation_lines)
        
        if not full_text.strip():
            continue

        session_date = dates[idx] if idx < len(dates) else "unknown"
        origin_sess_id = session_ids[idx] if idx < len(session_ids) else f"sess_{idx}"

        # 3. 构建 Metadata
        # 这些 Metadata 对于后续 RAG 检索时的过滤非常重要
        metadata = {
            "dataset": "longmemeval",
            "sample_id": sample_id,       
            "question_type": question_type,
            "session_date": session_date,
            "session_index": idx,        
            "origin_session_id": origin_sess_id,
            "type": "conversation_session"
        }
        
        # 4. 存入 Memory
        rec_id = memory.ingest(full_text, metadata)
        ingested_ids.append(rec_id)
    
    return ingested_ids


# ============================================================================
# 测试主程序
# ============================================================================

if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    
    # 假设我们有一个本地文件（这里为了测试，如果没有文件会走 dummy）
    # 你可以修改这里的 path 指向你下载的真实 json 文件
    TEST_PATH = "dataset/LongMemEval/longmemeval_s_cleaned.json"
    
    print("-" * 50)
    print("测试场景 1: 筛选 'multi-session' 类型，抽取 2 条")
    samples = load_longmemeval(
        path=TEST_PATH,
        target_types=["multi-session"], 
        limit=2,
        fallback_to_dummy=True
    )
    
    if samples:
        s = samples[0]
        print(f"样本ID: {s['id']}")
        print(f"类型: {s.get('question_type')}")
        print(f"Session数量: {len(s.get('haystack_sessions', []))}")
        print(f"Date数量: {len(s.get('haystack_dates', []))}")
    
    print("\n" + "-" * 50)
    print("测试场景 2: 导入到 Memory")
    
    # 模拟 Memory
    class MockMemory:
        def ingest(self, text, meta):
            # 简单打印一下正在存什么
            print(f"  -> Ingesting Session [Date: {meta['session_date']}]")
            return "mock_uuid"

    memory = MockMemory()
    
    if samples:
        print(f"正在导入样本 {samples[0]['id']} ...")
        ingest_longmemeval_sample(memory, samples[0])