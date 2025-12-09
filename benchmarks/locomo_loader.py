import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

# SimpleRAGMemory 在需要时导入（避免测试时的循环依赖）
try:
    from memory.simple_memory import SimpleRAGMemory
except ImportError:
    SimpleRAGMemory = None


def _dummy_locomo_sample() -> Dict[str, Any]:
    """
    返回一个虚拟的 LoCoMo 样本，用于测试和演示。
    
    LoCoMo 数据集格式:
    - qa: 问答对列表
    - conversation: 对话会话字典，包含 speaker_a, speaker_b 和多个 session_N
    - event_summary: 事件摘要
    - session_summary: 会话摘要
    - sample_id: 样本ID
    """
    return {
        "sample_id": "dummy-locomo-0",
        "qa": [
            {
                "question": "What did Person A talk about?",
                "answer": "Their recent trip",
                "evidence": ["D1:3"],
                "category": 1
            }
        ],
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "2:00 pm on 1 May, 2023",
            "session_1": [
                {
                    "speaker": "Alice",
                    "dia_id": "D1:1",
                    "text": "Hi Bob! How are you?"
                },
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "Hi Alice! I'm doing great. How about you?"
                },
                {
                    "speaker": "Alice",
                    "dia_id": "D1:3",
                    "text": "I just came back from a wonderful trip to the mountains!"
                }
            ]
        },
        "session_summary": {
            "session_1_summary": "Alice and Bob greeted each other. Alice mentioned her recent trip to the mountains."
        }
    }


def _normalize_sample_ids(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    标准化样本ID字段。
    LoCoMo 使用 'sample_id'，将其转换为标准的 'id' 字段。
    """
    normalized = []
    for s in samples:
        s = dict(s)
        if "id" not in s:
            if "sample_id" in s:
                s["id"] = s["sample_id"]
            elif "_id" in s:
                s["id"] = s["_id"]
        normalized.append(s)
    return normalized


def load_locomo(
    split: str = "train",
    fallback_to_dummy: bool = True,
    path: Optional[Union[str, Path]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    加载 LoCoMo 数据集。
    
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
        return [_dummy_locomo_sample()]
    
    return None


def ingest_locomo_sample(
    memory, 
    sample: Dict[str, Any]
) -> List[str]:
    """
    将 LoCoMo 样本的对话历史导入到记忆系统中。
    
    参数:
        memory: 记忆系统实例
        sample: 要导入的样本
    
    返回:
        导入的记录ID列表
    
    实现说明:
    - LoCoMo 包含多个对话会话（session_1, session_2, ...）
    - 每个会话包含多轮对话，每轮有 speaker, dia_id, text 字段
    - 我们将每个会话作为一个完整的对话导入
    """
    ingested_ids: List[str] = []
    
    sample_id = sample.get("id") or sample.get("sample_id")
    conversation = sample.get("conversation", {})
    
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    
    # 查找所有会话（session_1, session_2, ...）
    session_keys = sorted([k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")])
    
    for session_key in session_keys:
        session_dialogue = conversation.get(session_key, [])
        session_date_time = conversation.get(f"{session_key}_date_time", "unknown")
        
        if not session_dialogue:
            continue
        
        # 将对话轮次组合成一个连贯的文本
        conversation_parts = []
        for turn in session_dialogue:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            
            # 格式: Speaker: text
            conversation_parts.append(f"{speaker}: {text}")
        
        # 组合成完整的对话文本
        conversation_text = "\n\n".join(conversation_parts)
        
        if conversation_text:
            # 获取会话摘要（如果有）
            session_summary_key = f"{session_key}_summary"
            session_summaries = sample.get("session_summary", {})
            session_summary = session_summaries.get(session_summary_key, "")
            
            metadata = {
                "dataset": "locomo",
                "sample_id": sample_id,
                "session_key": session_key,
                "session_date_time": session_date_time,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "type": "conversation",
                "session_summary": session_summary,
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
    samples = load_locomo(fallback_to_dummy=True)
    if samples:
        print(f"✓ 成功加载 {len(samples)} 个样本")
        print(f"第一个样本的键: {list(samples[0].keys())}")
        if 'conversation' in samples[0]:
            conv = samples[0]['conversation']
            print(f"对话者: {conv.get('speaker_a')} 和 {conv.get('speaker_b')}")
            session_keys = [k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')]
            print(f"会话数量: {len(session_keys)}")
    
    # 2. 测试导入到内存（不使用数据库）
    print("\n测试2: 导入到内存系统")
    from memory.simple_memory import SimpleRAGMemory
    
    def dummy_embed(text):
        import numpy as np
        return np.random.rand(128)  # 随机向量用于测试
    
    memory = SimpleRAGMemory(embed_fn=dummy_embed)
    
    for sample in samples:
        ids = ingest_locomo_sample(memory, sample)
        sample_id = sample.get('id') or sample.get('sample_id')
        print(f"✓ 样本 {sample_id} 导入了 {len(ids)} 条对话记录")
    
    print(f"\n记忆系统统计: {memory.stats()}")
