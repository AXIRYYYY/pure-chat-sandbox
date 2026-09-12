import os
import json
import hashlib
import shutil
import random
import time
import requests  # 🌟 新增：用于调用博查(Bocha)搜索 API
import streamlit as st
from datetime import datetime
from tavily import TavilyClient
from config_manager import LOG_DIR, save_config


# ============================================================
# 分支功能与确定性序列化工具
# ============================================================

def deterministic_json_dumps(data, indent=2):
    """
    确定性 JSON 序列化：强制 sort_keys 确保相同数据产生相同的字节流，
    这是 Gemini Context Cache 100% 命中的关键保障。
    """
    return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True)


def compute_payload_hash(payload_str):
    """
    计算 Payload 的 SHA-256 哈希值，用于缓存前缀一致性审计。
    返回 64 位十六进制字符串。
    """
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def save_chat_history(save_path, messages, search_cache=None, backup=True):
    """
    统一保存对话历史：
    1. 自动备份原文件（首次写入时生成 .bak）
    2. 使用确定性序列化（sort_keys=True）
    3. 集中处理写入逻辑，避免散落各处的 json.dump 不一致
    """
    if backup and os.path.exists(save_path):
        bak_path = save_path + ".bak"
        if not os.path.exists(bak_path):
            try:
                shutil.copy2(save_path, bak_path)
            except Exception:
                pass  # 备份失败不应阻塞主流程

    data = {
        "messages": messages,
        "search_cache": search_cache or {},
    }
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(deterministic_json_dumps(data))


def normalize_messages(data):
    """
    延迟加载归一化：将旧版线性格式在内存中转换为新版结构，不操作磁盘。
    
    旧版格式（线性数组）：
        [msg1, msg2, ...]
    
    新版格式（包装对象，与 save_chat_history 兼容）：
        {"messages": [msg1, msg2, ...], "search_cache": {...}}
    
    返回：归一化后的 messages 列表
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("messages", [])
    return []


def msg_has_branches(msg):
    """检查 assistant 消息是否存在多个生成分支"""
    return (
        msg.get("role") == "assistant"
        and "branches" in msg
        and isinstance(msg["branches"], dict)
        and len(msg["branches"]) > 1
    )


def get_save_file_path(workspace_name):
    """
    根据工作区名称生成合法的本地持久化 JSON 文件路径。
    """
    # 仅保留字母和数字
    safe_name = "".join([c for c in workspace_name if c.isalnum()]).strip()
    return os.path.join(LOG_DIR, f"auto_chat_{safe_name}.json")


def clear_transient_states():
    """
    统一清理所有属于中间态或需要重置的会话状态键，
    防止 UI 重跑时残留干扰逻辑。
    """
    for key in [
        "failed_msg",
        "trigger_rerun",
        "trigger_gem_init",
        "edit_state",
        "pending_msg",
        "last_turn_logs",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def retry_api_call(
    func, max_attempts=3, base_delay=1.0, max_delay=16.0, context="API 请求"
):
    """通用自动重试工具，采用指数退避策略。"""
    attempt = 0
    while True:
        try:
            return func()
        except Exception as e:
            attempt += 1
            if max_attempts != -1 and attempt > max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.2)
            sleep_time = delay + jitter
            st.warning(
                f"{context} 第 {attempt} 次失败：{e}，{sleep_time:.1f}s 后重试..."
            )
            time.sleep(sleep_time)


def retry_api_stream(
    stream_func, max_attempts=3, base_delay=1.0, max_delay=16.0, context="API 请求"
):
    """
    流式 API 专用自动重试生成器。

    与 retry_api_call 不同，此函数将流创建 AND 迭代都包裹在重试逻辑内。
    如果流在迭代过程中抛异常，会从头重新建立连接并重试。

    参数:
        stream_func: 返回可迭代流对象的函数（如 lambda: chat.send_message_stream(...)）
        max_attempts: 最大重试次数，-1 表示无限重试
        base_delay: 初始等待秒数
        max_delay: 最大等待秒数
        context: 用于显示的错误上下文名称

    Yields:
        stream 中的每个 chunk 对象
    """
    attempt = 0
    while True:
        try:
            stream = stream_func()
            for chunk in stream:
                yield chunk
            break  # 流正常消费完毕，退出重试循环
        except Exception as e:
            attempt += 1
            if max_attempts != -1 and attempt > max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.2)
            sleep_time = delay + jitter
            st.warning(
                f"{context} 流读取第 {attempt} 次失败：{e}，{sleep_time:.1f}s 后重试..."
            )
            time.sleep(sleep_time)


def handle_feedback(idx):
    """
    处理点赞/点踩反馈的回调函数，
    负责状态更新、本地持久化以及审计日志的记录。
    """
    if "messages" not in st.session_state or idx >= len(st.session_state.messages):
        return

    val = st.session_state.get(f"fb_{idx}")
    fb_str = "👍" if val == 1 else ("👎" if val == 0 else None)
    if fb_str:
        st.session_state.messages[idx]["feedback"] = fb_str
        st.toast(f"已记录反馈: {fb_str}")

        # 持久化到对话文件（使用确定性序列化）
        save_path = get_save_file_path(
            st.session_state.api_configs["current_workspace"]
        )
        save_chat_history(
            save_path,
            st.session_state.messages,
            st.session_state.get("search_cache", {}),
        )

        # 写入审计日志 audit_trail.jsonl
        with open(
            os.path.join(LOG_DIR, "audit_trail.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(
                deterministic_json_dumps(
                    {
                        "ts": str(datetime.now()),
                        "action": "feedback",
                        "workspace": st.session_state.api_configs["current_workspace"],
                        "case": st.session_state.chat_title,
                        "msg_index": idx,
                        "feedback": fb_str,
                    },
                    indent=None,
                )
                + "\n"
            )


def delete_turn(index):
    """
    删除某一轮完整的对话（即对应的用户提问和助理回答）。
    """
    if "messages" not in st.session_state or index >= len(st.session_state.messages):
        return

    # 确定要删除的轮次的起始索引（即用户消息的索引）
    turn_start_index = (
        index if st.session_state.messages[index]["role"] == "user" else index - 1
    )

    if turn_start_index < 0:
        return

    # 确认要删除的两条索引（如有的话）
    indices_to_remove = [turn_start_index]
    if (
        turn_start_index + 1 < len(st.session_state.messages)
        and st.session_state.messages[turn_start_index + 1]["role"] == "assistant"
    ):
        indices_to_remove.append(turn_start_index + 1)

    # 过滤掉这些消息，生成新的消息列表
    new_messages = [
        msg
        for i, msg in enumerate(st.session_state.messages)
        if i not in indices_to_remove
    ]

    # 清除 fb_ 前缀的旧反馈状态，防止错乱
    for key in [k for k in st.session_state if k.startswith("fb_")]:
        del st.session_state[key]

    # 🌟 清理被删除消息的 search_cache（通过 msg_id 精准删除）
    if "search_cache" in st.session_state:
        for i in indices_to_remove:
            if i < len(st.session_state.messages):
                msg = st.session_state.messages[i]
                if "msg_id" in msg:
                    st.session_state.search_cache.pop(msg["msg_id"], None)

    st.session_state.messages = new_messages

    # 根据新列表重新同步反馈状态（由 1 和 0 映射会 👍 和 👎）
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "assistant" and "feedback" in msg:
            fb_key = f"fb_{i}"
            st.session_state[fb_key] = 1 if msg["feedback"] == "👍" else 0

    # 保存更新后的会话到磁盘（使用确定性序列化）
    save_path = get_save_file_path(st.session_state.api_configs["current_workspace"])
    save_chat_history(
        save_path,
        st.session_state.messages,
        st.session_state.get("search_cache", {}),
    )

    clear_transient_states()
    st.toast(f"已删除第 {turn_start_index // 2 + 1} 轮对话。")


def change_workspace():
    """
    切换工作区时的核心处理函数。
    负责保存旧工作区的偏好、加载新工作区的偏好，以及重新读取会话记录。
    """
    new_ws = st.session_state.ws_selector
    old_ws = st.session_state.api_configs.get("current_workspace", "默认沙盒 (Normal)")

    if "workspace_prefs" not in st.session_state.api_configs:
        st.session_state.api_configs["workspace_prefs"] = {}

    # 1. 保存当前状态作为旧工作区的偏好
    # 获取当前工作区的现有偏好设置（如果存在）
    old_workspace_prefs = st.session_state.api_configs["workspace_prefs"].get(
        old_ws, {}
    )

    # 获取或初始化 platform_models
    platform_models = old_workspace_prefs.get("platform_models", {})

    # 更新当前平台的模型选择
    current_choice = st.session_state.api_configs.get("last_choice", "Gemini")
    current_model = st.session_state.api_configs.get("last_model", "")
    if current_choice and current_model:
        platform_models[current_choice] = current_model

    st.session_state.api_configs["workspace_prefs"][old_ws] = {
        "last_choice": current_choice,
        "last_model": current_model,
        "platform_models": platform_models,  # 新增：保存每个平台最后选择的模型
        "system_prompt": st.session_state.api_configs.get("system_prompt", ""),
        "temperature": st.session_state.api_configs.get("temperature", 0.7),
        "top_p": st.session_state.api_configs.get("top_p", 0.9),
        "max_tokens": st.session_state.api_configs.get("max_tokens", 8192),
    }

    # 2. 设置新的工作区名
    st.session_state.api_configs["current_workspace"] = new_ws
    st.session_state.api_configs["auto_name_triggered"] = False

    # 3. 加载新工作区的偏好（如果之前有记录的话）
    prefs = st.session_state.api_configs["workspace_prefs"].get(new_ws, {})
    if prefs:
        st.session_state.api_configs["last_choice"] = prefs.get("last_choice", "Gemini")
        st.session_state.api_configs["last_model"] = prefs.get("last_model", "")
        st.session_state.api_configs["system_prompt"] = prefs.get("system_prompt", "")
        st.session_state.api_configs["temperature"] = prefs.get("temperature", 0.7)
        st.session_state.api_configs["top_p"] = prefs.get("top_p", 0.9)
        st.session_state.api_configs["max_tokens"] = prefs.get("max_tokens", 8192)

    # 4. 持久化全局配置并更新当前标题
    save_config(st.session_state.api_configs)
    st.session_state.chat_title = st.session_state.api_configs.get(
        "workspace_titles", {}
    ).get(new_ws, "未命名案例")

    # 5. 清理缓存状态
    for key in list(st.session_state.keys()):
        if key.startswith("fb_"):
            del st.session_state[key]
    clear_transient_states()

    # 6. 读取新工作区的会话文件
    save_path = get_save_file_path(new_ws)
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 🌟 使用 normalize_messages 兼容旧版数组与新版包装对象
                st.session_state.messages = normalize_messages(data)
                if isinstance(data, dict) and "search_cache" in data:
                    st.session_state.search_cache = data["search_cache"]
        except:
            st.session_state.messages = []
    else:
        st.session_state.messages = []


# ============================================================
# 提示词管理（JSON持久化）
# ============================================================
PROMPT_CONFIG_FILE = "prompts_config.json"

# 默认关键词提取提示词（当 JSON 文件不存在或损坏时使用）
DEFAULT_KEYWORD_EXTRACTION_PROMPT = (
    "你是一个关键词提取引擎。请从用户的输入中提取1-3个最相关的搜索关键词，"
    "用逗号分隔。只回复关键词，不要有其他内容。"
    "例如：'最新AI技术,GPT发展,大模型动态'"
)


def load_prompt(key="keyword_extraction"):
    """
    从 prompts_config.json 加载指定 key 的提示词。

    参数:
        key: 提示词的键名，默认为 "keyword_extraction"

    返回:
        str: 提示词内容。如果 JSON 文件不存在或 key 不存在，自动创建/修复并返回默认值。
    """
    # 如果 JSON 文件不存在，用默认值自动创建
    if not os.path.exists(PROMPT_CONFIG_FILE):
        default_data = {
            "keyword_extraction": {
                "system_prompt": DEFAULT_KEYWORD_EXTRACTION_PROMPT,
                "description": "用于从用户问题中提取联网搜索关键词的系统提示词",
            }
        }
        with open(PROMPT_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)
        return DEFAULT_KEYWORD_EXTRACTION_PROMPT

    # 读取 JSON 文件
    try:
        with open(PROMPT_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(key, {}).get("system_prompt", DEFAULT_KEYWORD_EXTRACTION_PROMPT)
    except Exception:
        # 解析失败时返回默认值，不中断程序
        return DEFAULT_KEYWORD_EXTRACTION_PROMPT


def save_prompt(content, key="keyword_extraction"):
    """
    将提示词保存到 prompts_config.json。

    参数:
        content: 新的提示词内容
        key: 提示词的键名，默认为 "keyword_extraction"
    """
    # 确保 JSON 文件存在
    if not os.path.exists(PROMPT_CONFIG_FILE):
        load_prompt(key)

    try:
        with open(PROMPT_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    # 更新指定 key 的提示词内容
    if key not in data:
        data[key] = {}
    data[key]["system_prompt"] = content

    # 写回 JSON 文件
    with open(PROMPT_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def reset_prompt(key="keyword_extraction"):
    """
    将指定 key 的提示词恢复为默认值。

    参数:
        key: 提示词的键名，默认为 "keyword_extraction"

    返回:
        str: 恢复后的默认提示词内容
    """
    save_prompt(DEFAULT_KEYWORD_EXTRACTION_PROMPT, key)
    return DEFAULT_KEYWORD_EXTRACTION_PROMPT


def perform_web_search(client, prompt_text, tavily_api_key, model_id, retry_attempts=3):
    """
    执行联网搜索的核心工具函数。

    ⚡ 专为 SiliconFlow 和 DeepSeek 两个平台共用而设计，
    封装了"关键词提取 → Tavily搜索 → 结果格式化"的完整流程，
    避免两处重复代码。

    参数:
        client: OpenAI 兼容的客户端实例 (SiliconFlow 或 DeepSeek 均可)
        prompt_text: 用户输入的原始问题文本，用于提取搜索关键词
        tavily_api_key: Tavily 搜索引擎的 API 密钥
        model_id: 用于关键词提取的 AI 模型 ID
        retry_attempts: 发生错误时最大重试次数，-1 表示无限重试

    返回:
        tuple: (keywords, search_context, results_count)
            - keywords (str): AI 提取出的搜索关键词（逗号分隔）
            - search_context (str): 格式化后的搜索结果文本，每行一条
            - results_count (int): 搜索结果的总条数

    异常:
        向上传播任何联网或解析异常，由调用方 (app.py) 负责捕获和展示友好的错误提示
    """
    # ----- 第1步：从 prompts_config.json 加载提示词，提取搜索关键词 -----
    # 从 JSON 配置文件中读取关键词提取提示词（用户可在侧边栏中自定义编辑）
    system_prompt = load_prompt("keyword_extraction")
    extract_keywords_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    def extract_keywords_request():
        # 兼容 OpenAI 格式的客户端 (SiliconFlow, DeepSeek)
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return client.chat.completions.create(
                model=model_id,
                messages=extract_keywords_messages,
                max_tokens=50,
                temperature=0.1,
            )
        # 兼容 Gemini SDK (google-genai)
        elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
            # 转换消息格式为 Gemini 格式
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"System: {system_prompt}\n\nUser: {prompt_text}"}
                    ],
                }
            ]
            response = client.models.generate_content(
                model=model_id,
                contents=prompt_text,
                config={
                    "system_instruction": system_prompt,
                    "max_output_tokens": 50,
                    "temperature": 0.1,
                },
            )

            # 模拟 OpenAI 的返回结构，以便后续统一处理
            class MockResponse:
                class Choice:
                    class Message:
                        def __init__(self, content):
                            self.content = content

                    def __init__(self, content):
                        self.message = self.Message(content)

                def __init__(self, content):
                    self.choices = [self.Choice(content)]

            return MockResponse(response.text)
        else:
            raise ValueError(f"不支持的客户端类型: {type(client)}")

    keywords_res = retry_api_call(
        extract_keywords_request,
        max_attempts=retry_attempts,
        context="关键词提取请求",
    )
    keywords = keywords_res.choices[0].message.content.strip()

    # 安全检查：AI必须返回有效的搜索关键词，拒绝用空字符串搜索
    if not keywords:
        raw_keywords = keywords_res.choices[
            0
        ].message.content  # 保留原始返回内容，用于调试
        raise ValueError(
            f"AI未能提取出有效搜索关键词（原始返回: {repr(raw_keywords)}）"
        )

    # ----- 第2步：用提取到的关键词调用 Tavily 搜索引擎 -----
    def tavily_search_request():
        return TavilyClient(api_key=tavily_api_key).search(query=keywords)

    search_res = retry_api_call(
        tavily_search_request,
        max_attempts=retry_attempts,
        context="Tavily 搜索请求",
    )

    # ----- 第3步：将搜索结果格式化为易读的文本块 -----
    search_context = "\n".join(
        [f"- {r['content']} ({r['url']})" for r in search_res["results"]]
    )

    # 返回关键词、格式化文本、结果条数（条数用于界面状态提示）
    return keywords, search_context, len(search_res["results"])


# ============================================================
# 🌟 新增：博查(Bocha)搜索函数
# ============================================================
def perform_bocha_search(
    client, prompt_text, bocha_api_key, model_id, retry_attempts=3,
    freshness="noLimit", summary=True, count=10,
    predefined_keywords=None,
):
    """
    使用博查(Bocha)搜索引擎执行联网搜索。

    与 perform_web_search() 结构相同，但将 Tavily 替换为博查 API。
    博查 API 文档：https://api.bocha.cn/v1/web-search

    参数:
        client: OpenAI 兼容的客户端实例 (用于关键词提取)
        prompt_text: 用户输入的原始问题文本
        bocha_api_key: 博查搜索的 API Key
        model_id: 用于关键词提取的 AI 模型 ID
        retry_attempts: 发生错误时最大重试次数，-1 表示无限重试
        freshness: 搜索时间范围，默认 "noLimit"（不限）
        summary: 是否显示文本摘要，默认 True
        count: 返回结果条数（1-50），默认 10
        predefined_keywords: 可选，预定义搜索关键词。
            如果提供则跳过 AI 关键词提取步骤，直接使用此关键词搜索。
            用于"多次联网"功能中 AI 已建议关键词的场景。

    返回:
        tuple: (keywords, search_context, results_count)
            格式与 perform_web_search() 完全一致，方便调用方无缝切换
    """
    # ----- 第1步：提取搜索关键词（或使用预定义关键词）-----
    if predefined_keywords:
        # 🌟 如果提供了预定义关键词，跳过 AI 提取步骤
        keywords = predefined_keywords
    else:
        # 从 prompts_config.json 加载提示词，让 AI 提取搜索关键词
        system_prompt = load_prompt("keyword_extraction")
        extract_keywords_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        def extract_keywords_request():
            # 兼容 OpenAI 格式的客户端 (SiliconFlow, DeepSeek)
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                return client.chat.completions.create(
                    model=model_id,
                    messages=extract_keywords_messages,
                    max_tokens=50,
                    temperature=0.1,
                )
            # 兼容 Gemini 客户端 (google-genai)
            elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt_text,
                    config={
                        "system_instruction": system_prompt,
                        "max_output_tokens": 50,
                        "temperature": 0.1,
                    },
                )
                # 模拟 OpenAI 的返回结构，以便后续统一处理
                class MockResponse:
                    class Choice:
                        class Message:
                            def __init__(self, content):
                                self.content = content
                        def __init__(self, content):
                            self.message = self.Message(content)
                    def __init__(self, content):
                        self.choices = [self.Choice(content)]
                return MockResponse(response.text)
            else:
                raise ValueError(f"不支持的客户端类型: {type(client)}")

        keywords_res = retry_api_call(
            extract_keywords_request,
            max_attempts=retry_attempts,
            context="关键词提取请求",
        )
        keywords = keywords_res.choices[0].message.content.strip()

        # 安全检查：AI必须返回有效的搜索关键词
        if not keywords:
            raw_keywords = keywords_res.choices[0].message.content
            raise ValueError(
                f"AI未能提取出有效搜索关键词（原始返回: {repr(raw_keywords)}）"
            )

    # ----- 第2步：用提取到的关键词调用博查 API -----
    def bocha_search_request():
        url = "https://api.bocha.cn/v1/web-search"
        headers = {
            "Authorization": f"Bearer {bocha_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": keywords,
            "freshness": freshness,
            "summary": summary,
            "count": count,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()  # 非 2xx 状态码会抛出异常
        return resp.json()

    search_res = retry_api_call(
        bocha_search_request,
        max_attempts=retry_attempts,
        context="博查搜索请求",
    )

    # ----- 第3步：解析博查响应，格式化为易读的文本块 -----
    # 博查返回结构：data.webPages.value 是网页结果列表
    web_pages = search_res.get("data", {}).get("webPages", {})
    results_list = web_pages.get("value", [])

    # 格式化为与 Tavily 兼容的文本格式
    search_context = "\n".join(
        [
            f"- {item.get('snippet', item.get('summary', '无摘要'))} ({item.get('url', '无链接')})"
            for item in results_list
        ]
    )

    # 返回关键词、格式化文本、结果条数
    return keywords, search_context, len(results_list)
