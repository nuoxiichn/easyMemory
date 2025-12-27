import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

# SimpleRAGMemory
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
    sample: Dict[str, Any],
    mode: str = "observation"
) -> List[str]:

    ingested_ids: List[str] = []
    
    sample_id = sample.get("id") or sample.get("sample_id")
    conversation = sample.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    
    base_metadata = {
        "dataset": "locomo",
        "sample_id": sample_id,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "rag_mode": mode
    }

    # === 获取时间 ===
    def get_session_time(sess_key):
        time_key = f"{sess_key}_date_time"
        return conversation.get(time_key, "unknown")

    # === 模式 1: Observations ===
    if mode == "observation":
        observations_data = sample.get("observation", {})
        for session_key_raw, speaker_data in observations_data.items():
            clean_session_key = session_key_raw.replace("_observation", "")
            session_time = get_session_time(clean_session_key)

            if not isinstance(speaker_data, dict): continue

            for speaker_name, obs_list in speaker_data.items():
                for obs_item in obs_list:
                    if isinstance(obs_item, list) and len(obs_item) >= 1:
                        obs_text = obs_item[0]
                        evidence_id = obs_item[1] if len(obs_item) > 1 else None

                        if obs_text:
                            meta = base_metadata.copy()
                            meta["type"] = "observation"
                            meta["session_key"] = clean_session_key
                            meta["relevant_speaker"] = speaker_name
                            meta["timestamp"] = session_time
                            
                            if evidence_id:
                                meta["evidence_id"] = evidence_id
                                meta["evidence_ids"] = [evidence_id]
                            
                            rec_id = memory.ingest(obs_text, meta)
                            ingested_ids.append(rec_id)

    # === 模式 2: Session Summaries ===
    elif mode == "summary":
        summaries = sample.get("session_summary", {})
        for key, summary_text in summaries.items():
            session_key = key.replace("_summary", "")
            session_time = get_session_time(session_key)
            
            if summary_text:
                meta = base_metadata.copy()
                meta["type"] = "summary"
                meta["session_key"] = session_key
                meta["timestamp"] = session_time
                
                rec_id = memory.ingest(summary_text, meta)
                ingested_ids.append(rec_id)

    # === 模式 3: Raw Dialogs ===
    else: 
        session_keys = sorted([k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")])
        
        for session_key in session_keys:
            session_time = get_session_time(session_key)
            
            session_dialogue = conversation.get(session_key, [])
            conversation_parts = []
            contained_ids = []
            
            for turn in session_dialogue:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                conversation_parts.append(f"{speaker}: {text}")
                
                dia_id = turn.get("dia_id")
                if dia_id:
                    contained_ids.append(dia_id)

            conversation_text = "\n\n".join(conversation_parts)

            if conversation_text:
                meta = base_metadata.copy()
                meta["type"] = "conversation"
                meta["session_key"] = session_key
                meta["timestamp"] = session_time 
                
                if contained_ids:
                    meta["evidence_ids"] = contained_ids
                
                rec_id = memory.ingest(conversation_text, meta)
                ingested_ids.append(rec_id)
                
    return ingested_ids


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    测试 loader 功能及不同 RAG 模式的数据导入
    """
    import sys
    import numpy as np
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from memory.simple_memory import SimpleRAGMemory

    # 1. 构造一个符合 LoCoMo 真实结构的富测试样本
    # (包含 conversation, session_summary 以及你刚提供的嵌套 observation)
    rich_test_sample = {
        "sample_id": "test_sample_001",
        "qa": [],
        # 1. 原始对话数据
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "May 1, 2023",
            "session_1": [
                {"speaker": "Caroline", "text": "Hi Melanie!", "dia_id": "D1:1"},
                {"speaker": "Melanie", "text": "Hey Caroline, long time no see.", "dia_id": "D1:2"}
            ]
        },
        # 2. 摘要数据
        "session_summary": {
            "session_1_summary": "Caroline and Melanie greeted each other after a long time."
        },
        # 3. 观察数据 (复杂的嵌套结构)
        "observation": {
            "session_1_observation": {
                "Caroline": [
                    [
                        "Caroline attended an LGBTQ support group recently.",
                        "D1:3"
                    ],
                    [
                        "The support group has made Caroline feel accepted.",
                        "D1:7"
                    ]
                ],
                "Melanie": [
                    [
                        "Melanie is currently managing kids and work.",
                        "D1:2"
                    ]
                ]
            }
        }
    }

    # 定义 dummy embedding 函数
    def dummy_embed(text):
        return np.random.rand(128)

    # 2. 循环测试三种模式
    test_modes = ["dialog", "observation", "summary"]

    print(f"开始测试样本 ID: {rich_test_sample['sample_id']}")
    print("-" * 60)

    for mode in test_modes:
        print(f"\n>>> 正在测试模式: [ {mode.upper()} ]")
        
        # 为每个模式初始化一个新的干净内存
        memory = SimpleRAGMemory(embed_fn=dummy_embed)
        
        # 执行导入
        ids = ingest_locomo_sample(memory, rich_test_sample, mode=mode)
        
        # 输出结果统计
        print(f"✓ 成功导入记录数: {len(ids)}")
        
        # 验证导入的内容类型（通过 hack memory 的内部存储来查看，假设是 list）
        # 注意：这里假设 SimpleRAGMemory 有 .storage 或类似结构，如果没有请根据实际情况调整
        if hasattr(memory, 'storage') and len(memory.storage) > 0:
            first_item = memory.storage[0]
            print(f"  第一条记录预览 (Metadata): {first_item.get('metadata', {})}")
            print(f"  第一条记录预览 (Text片段): {first_item.get('text', '')[:50]}...")
        
        # 预期结果验证
        if mode == "observation":
            if len(ids) == 3:
                print("  [验证通过] Observation 模式正确拆分了 3 条原子事实。")
            else:
                print(f"  [验证警告] 预期导入 3 条 Observation，实际导入 {len(ids)} 条。")
        elif mode == "dialog":
            if len(ids) == 1: # 只有一个 session
                print("  [验证通过] Dialog 模式正确合并为了 1 个会话片段。")
        elif mode == "summary":
            if len(ids) == 1:
                print("  [验证通过] Summary 模式正确导入了 1 条会话摘要。")

    print("\n" + "="*60)
    print("测试结束")
