# compression_engine.py — 上下文压缩引擎
"""
非侵入式上下文压缩引擎。

核心设计原则：
  - messages 数组始终保持完整，压缩是叠加在消息上的"视图层"
  - 压缩状态独立存储在 compression_state 字典中
  - 上下文组装和 UI 渲染时动态替换压缩区间
  - 取消压缩只需清除对应范围记录，无需修改 messages

使用流程：
  1. 用户在 UI 中选择压缩区间和提示词
  2. 调用 compress_messages_range() 获取 LLM 摘要
  3. 调用 add_compression_range() 将摘要持久化到 compression_state
  4. API 调用路径中调用 get_compressed_context_messages() 获取压缩后的 context
  5. UI 渲染时调用 get_message_compression_info() 标注压缩区间的消息
"""

import time
from datetime import datetime
from google import genai
from google.genai import types


# --- 工具函数：规范化 Gemini 模型名称 ---

def _normalize_gemini_model(model_id):
    if not model_id:
        return model_id
    if model_id.startswith("models/"):
        return model_id
    return f"models/{model_id}"


# ============================================================
# 1. 上下文组装 — 供 API 调用路径使用
# ============================================================

def get_compressed_context_messages(messages, compression_state):
    """
    返回压缩后的消息列表，压缩区间被替换为单条摘要消息。

    用于 Gemini 和 OpenAI 两种 API 路径中构建 context 时调用，
    替代直接遍历 st.session_state.messages。

    Args:
        messages: st.session_state.messages 完整消息列表
        compression_state: st.session_state.get("compression_state", {})

    Returns:
        list of {"role": str, "content": str}
        — role 保持原始值 ("user" / "assistant")，由各 API 路径自行转换
        — content 使用 full_payload（首选）或 content
        — 压缩区间被替换为一条 summary 消息
    """
    if not compression_state or not compression_state.get("ranges"):
        return [
            {
                "role": m["role"],
                "content": m.get("full_payload", m.get("content", "")),
            }
            for m in messages
        ]

    # msg_id -> 所属压缩区间 快速查找
    compressed_ids = {}
    for r in compression_state["ranges"]:
        for mid in r["msg_ids"]:
            compressed_ids[mid] = r

    result = []
    emitted_ranges = set()

    for m in messages:
        msg_id = m.get("msg_id", "")
        rng = compressed_ids.get(msg_id)

        if rng is None:
            text = m.get("full_payload", m.get("content", ""))
            result.append({"role": m["role"], "content": text})
        elif rng["id"] not in emitted_ranges:
            result.append({
                "role": rng.get("summary_role", "user"),
                "content": rng["summary"],
            })
            emitted_ranges.add(rng["id"])

    return result


# ============================================================
# 2. UI 渲染辅助 — 供消息渲染循环使用
# ============================================================

def get_message_compression_info(messages, compression_state, idx):
    """
    查询 messages[idx] 是否在某个压缩区间中。

    供消息渲染循环调用，决定是否为该消息添加压缩标注。

    Returns:
        {
            "compressed": bool,
            "is_start": bool,     # 是此压缩区间的第一条消息
            "is_end": bool,       # 是此压缩区间的最后一条消息
            "range_info": {       # 仅 compressed=True 时有值
                "id": str,
                "summary": str,
                "model": str,
                "prompt": str,
                "count": int,     # 本区间包含的消息数
            } | None
        }
    """
    info = {
        "compressed": False,
        "is_start": False,
        "is_end": False,
        "range_info": None,
    }

    if (
        not compression_state
        or not compression_state.get("ranges")
        or idx >= len(messages)
    ):
        return info

    msg = messages[idx]
    msg_id = msg.get("msg_id", "")

    for r in compression_state["ranges"]:
        mids = r["msg_ids"]
        if msg_id in mids:
            info["compressed"] = True
            info["is_start"] = (mids[0] == msg_id)
            info["is_end"] = (mids[-1] == msg_id)
            info["range_info"] = {
                "id": r["id"],
                "summary": r.get("summary", ""),
                "model": r.get("model", ""),
                "prompt": r.get("prompt", ""),
                "count": len(mids),
            }
            break

    return info


# ============================================================
# 3. 压缩执行 — 将消息区间发送给 LLM 生成摘要
# ============================================================

def compress_messages_range(
    messages,
    range_start,
    range_end,
    compression_prompt,
    model_channel,
    model_id,
    api_configs,
):
    """
    将 messages[range_start:range_end+1] 发送给压缩 LLM，返回摘要文本。

    消息使用 raw_payload（无时间戳/搜索结果污染）拼接发送以求最佳质量。

    Args:
        messages: 完整消息列表 (st.session_state.messages)
        range_start: 压缩区间起始索引
        range_end: 压缩区间结束索引 (包含)
        compression_prompt: 用户自定义的压缩提示词
        model_channel: 执行压缩的模型通道 (e.g. "Gemini", "SiliconFlow")
        model_id: 执行压缩的具体模型 ID
        api_configs: st.session_state.api_configs (用于获取 API key)

    Returns:
        str: 摘要文本

    Raises:
        ValueError: 区间无效或模型通道不支持
        Exception: API 调用失败
    """
    if range_start < 0 or range_end >= len(messages) or range_start > range_end:
        raise ValueError(f"无效的压缩区间: [{range_start}, {range_end}] (共 {len(messages)} 条消息)")

    msg_texts = []
    for i in range(range_start, range_end + 1):
        m = messages[i]
        role_label = "用户" if m["role"] == "user" else "助手"
        text = m.get("raw_payload", m.get("content", ""))
        msg_texts.append(f"[{role_label}]: {text}")

    conversation_text = "\n\n".join(msg_texts)

    compress_content = (
        f"{compression_prompt}\n\n"
        f"以下是需要压缩的对话内容：\n\n"
        f"{conversation_text}\n\n"
        f"请输出压缩后的摘要："
    )

    api_key = lookup_api_key(api_configs, model_channel)

    if _is_gemini_type(model_channel):
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=_normalize_gemini_model(model_id),
            contents=compress_content,
            generation_config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
        summary = resp.text.strip()

    elif _is_openai_type(model_channel):
        from provider_registry import create_openai_client
        client = create_openai_client(model_channel, api_key)
        from chat_utils import retry_api_call
        resp = retry_api_call(
            lambda: client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": compress_content}],
                temperature=0.3,
                max_tokens=2048,
            )
        )
        summary = resp.choices[0].message.content.strip()

    else:
        raise ValueError(f"不支持的模型通道: {model_channel}")

    return summary


# ============================================================
# 4. 压缩状态管理
# ============================================================

def add_compression_range(
    compression_state, msg_ids, summary, model_channel, model_id, compression_prompt,
    summary_role="user",
):
    """
    向压缩状态中添加一个新的压缩区间。

    Args:
        compression_state: st.session_state.compression_state (会被原地修改)
        msg_ids: 被压缩消息的 msg_id 列表 (保持原始顺序)
        summary: 摘要文本
        model_channel: 执行压缩的模型通道
        model_id: 执行压缩的模型 ID
        compression_prompt: 使用的压缩提示词
        summary_role: 摘要消息在 context 中的角色 (默认 "user")

    Returns:
        str: 新创建的 range_id
    """
    if "ranges" not in compression_state:
        compression_state["ranges"] = []

    range_id = f"comp_{int(time.time_ns())}"

    compression_state["ranges"].append({
        "id": range_id,
        "msg_ids": list(msg_ids),
        "summary": summary,
        "model_channel": model_channel,
        "model": model_id,
        "prompt": compression_prompt,
        "summary_role": summary_role,
        "timestamp": datetime.now().isoformat(),
    })

    return range_id


def remove_compression_range(compression_state, range_id):
    """
    从压缩状态中移除指定的压缩区间。

    Returns:
        bool: 是否成功移除 (False 表示 range_id 不存在)
    """
    if not compression_state or not compression_state.get("ranges"):
        return False

    before_len = len(compression_state["ranges"])
    compression_state["ranges"] = [
        r for r in compression_state["ranges"] if r["id"] != range_id
    ]
    return len(compression_state["ranges"]) < before_len


def get_compressed_msg_ids(compression_state):
    """
    返回所有已被压缩的 msg_id 集合。

    用于：
      - 判断某条消息是否已被压缩
      - 重叠检测
      - 删除消息时清理相关压缩区间
    """
    ids = set()
    if compression_state and compression_state.get("ranges"):
        for r in compression_state["ranges"]:
            ids.update(r["msg_ids"])
    return ids


def check_range_overlap(compression_state, proposed_msg_ids):
    """
    检查拟压缩的消息集合是否与已有压缩区间重叠。

    Returns:
        (bool, str | None):
          - (True, range_id)  — 有重叠，返回冲突的 range_id
          - (False, None)     — 无重叠
    """
    existing_ids = get_compressed_msg_ids(compression_state)
    overlap = existing_ids.intersection(set(proposed_msg_ids))
    if not overlap:
        return False, None

    for r in compression_state.get("ranges", []):
        if any(mid in r["msg_ids"] for mid in overlap):
            return True, r["id"]
    return False, None


def cleanup_deleted_messages(messages, compression_state):
    """
    当 messages 数组发生变化（删除消息）后，清理包含已不存在的 msg_id 的压缩区间。

    如果一个压缩区间的所有 msg_id 都不在 messages 中了，整个区间被移除。
    如果部分还在，保留区间但更新 msg_ids 列表。

    Args:
        messages: 当前的 st.session_state.messages
        compression_state: 当前的 compression_state

    Returns:
        int: 被移除的压缩区间数量
    """
    if not compression_state or not compression_state.get("ranges"):
        return 0

    existing_ids = {m.get("msg_id", "") for m in messages}
    removed = 0
    new_ranges = []

    for r in compression_state["ranges"]:
        remaining = [mid for mid in r["msg_ids"] if mid in existing_ids]
        if not remaining:
            removed += 1
            continue
        r["msg_ids"] = remaining
        new_ranges.append(r)

    compression_state["ranges"] = new_ranges
    return removed


# ============================================================
# 5. 内部工具
# ============================================================

def _is_gemini_type(channel_id):
    from provider_registry import is_gemini_type as _igt
    return _igt(channel_id)


def _is_openai_type(channel_id):
    from provider_registry import is_openai_type as _iot
    return _iot(channel_id)


def lookup_api_key(api_configs, channel_id):
    from provider_registry import get_api_key as _gak
    return _gak(api_configs, channel_id)
