import streamlit as st
import json
import os
import shutil
import time
from datetime import datetime, timezone
from google import genai
from google.genai import types
from openai import OpenAI
from tavily import TavilyClient
import tiktoken
import fitz  # PyMuPDF
from export_utils import export_chat_to_md
import ai_models_fetcher

# --- 导入自定义模块 ---
from config_manager import (
    CONFIG_FILE,
    MODEL_CONFIG_FILE,
    LOG_DIR,
    ATTACH_DIR,
    load_model_config,
    load_config,
    save_config,
    get_model_options,
)
from provider_registry import (
    initialize as init_provider_registry,
    get_visible_channels,
    get_channel_display_name,
    get_provider_info,
    get_api_key as lookup_api_key,
    create_openai_client,
    is_gemini_type,
    is_openai_type,
    load_provider_registry as load_provider_reg,
    save_provider_registry as save_provider_reg,
)
from token_utils import calculate_prompt_tokens, update_token_estimate
from file_utils import (
    process_uploaded_file,
    get_file_content_summary,
    read_pdf_content,
    read_text_content,
)
from chat_utils import (
    get_save_file_path,
    handle_feedback,
    delete_turn,
    change_workspace,
    clear_transient_states,
    perform_web_search,  # 🌟 共用联网搜索函数（SiliconFlow & DeepSeek）
    perform_bocha_search,  # 🔍 新增：博查(Bocha)搜索函数
    load_prompt,  # 🔍 从 JSON 配置加载关键词提取提示词
    save_prompt,  # 🔍 保存用户自定义的关键词提取提示词
    reset_prompt,  # 🔍 恢复关键词提取提示词为默认值
    retry_api_call,
    retry_api_stream,  # 🌟 新增导入
    deterministic_json_dumps,  # 🌟 确定性序列化（缓存一致性保障）
    compute_payload_hash,  # 🌟 Payload 哈希审计
    save_chat_history,  # 🌟 统一保存函数
    normalize_messages,  # 🌟 旧版数据兼容归一化
    msg_has_branches,  # 🌟 分支检测
)
from compression_engine import (
    get_compressed_context_messages,
    get_message_compression_info,
    compress_messages_range,
    add_compression_range,
    remove_compression_range,
    get_compressed_msg_ids,
    check_range_overlap,
    cleanup_deleted_messages,
)


# --- 工具函数：规范化Gemini模型名称 ---
def normalize_gemini_model(model_id):
    """
    规范化Gemini模型ID，确保以models/开头但不会重复添加前缀。

    参数:
        model_id: 模型ID字符串，可能是短格式（如'gemini-2.5-pro'）
                  或完整格式（如'models/gemini-2.5-pro'）

    返回:
        规范化后的模型ID，确保以'models/'开头且不重复
    """
    if not model_id:
        return model_id
    if model_id.startswith("models/"):
        return model_id
    return f"models/{model_id}"


def _get_round_mapping(messages):
    """
    构建轮次映射表。
    返回 [(round_number, msg_idx, preview_text), ...]
    每个 user 消息为一个轮次的起点。
    """
    rounds = []
    rn = 0
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            rn += 1
            raw = m.get("raw_payload", m.get("content", ""))
            preview = raw[:60].replace("\n", " ").strip()
            rounds.append((rn, i, preview))
    return rounds


# --- 1. 版本号定义 ---
__version__ = "5.4.5"

# --- 2. 页面与状态管理 ---
st.set_page_config(
    page_title=f"纯净chat沙盒 V{__version__} - Guardian Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🌟 功能优化 2 & 3: 还原为干净的默认布局，并将顶栏背景设置为透明
st.markdown(
    """
    <style>
        /* 1. 调整主内容区域边距，确保不遮盖顶层 UI */
        .main .block-container {
            padding-top: 4rem !important;
            padding-right: 3rem;
            padding-left: 3rem;
            padding-bottom: 3rem;
        }

        /* 2. 将顶栏背景设置为透明 */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            background: none !important;
        }

        /* 3. 仅微调工具栏位置，保护滚动条顶部区域 */
        div[data-testid="stToolbar"] {
            right: 2rem !important;
        }

        /* 4. 扩大附件上传区域的文件列表显示高度（默认~3行 → 扩展到~10行） */
        /*    st.file_uploader 中已上传文件列表是 dropzone 后的兄弟 div */
        div[data-testid="stFileUploaderDropzone"] + div section {
            max-height: 420px !important;
            overflow-y: auto !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# 🌟 漏洞修复 4: 强化守护态，处理页面刷新导致的中断
if "pending_msg" in st.session_state:
    # 如果存在一个 "正在处理" 的消息，说明上次生成被强制中断（如刷新）
    # 将其转为草稿状态，而不是丢弃或移到未处理的 failed_msg
    if "edit_state" not in st.session_state:
        st.session_state.edit_state = {
            "type": "draft",
            "msg": st.session_state.pending_msg,
        }
    # pending_msg 的使命已经完成，可以删除了
    del st.session_state.pending_msg


if "api_configs" not in st.session_state:
    st.session_state.api_configs = load_config()
# 🌟 加载模型配置

if "model_config" not in st.session_state:
    st.session_state.model_config = load_model_config()

# 🌟 初始化供应商注册表 + 自动迁移已有通道
init_provider_registry()
builtin_reg, custom_reg = load_provider_reg()
_auto_enabled = False
for _pid in ["Gemini", "SiliconFlow", "DeepSeek"]:
    if lookup_api_key(st.session_state.api_configs, _pid) and not builtin_reg.get(
        _pid, {}
    ).get("enabled"):
        builtin_reg[_pid] = {"enabled": True}
        _auto_enabled = True
if _auto_enabled:
    save_provider_reg(builtin_reg, custom_reg)


current_ws = st.session_state.api_configs.get("current_workspace", "默认沙盒 (Normal)")
current_save_file = get_save_file_path(current_ws)


def _opencode_session_id():
    """按工作区生成稳定的 OpenCode Go session id（用于 x-opencode-session 请求头）。"""
    import hashlib
    _ws_key = f"opencode_session_{current_ws}"
    if _ws_key not in st.session_state:
        _seed = f"pure-chat-sandbox|{current_ws}"
        st.session_state[_ws_key] = f"ocg-{hashlib.md5(_seed.encode()).hexdigest()[:24]}"
    return st.session_state[_ws_key]

if "messages" not in st.session_state:
    if os.path.exists(current_save_file):
        try:
            with open(current_save_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 🌟 使用 normalize_messages 统一兼容旧版数组和包装对象
                st.session_state.messages = normalize_messages(data)
                if isinstance(data, dict) and "search_cache" in data:
                    st.session_state.search_cache = data["search_cache"]
        except:
            st.session_state.messages = []
    else:
        st.session_state.messages = []

# 🌟 漏洞修复 6: 从磁盘恢复 pending_msg（应对手机浏览器刷新丢失 session 的情况）
#     当 session 丢失时，st.session_state.pending_msg 不存在，
#     但磁盘上的 pending 文件可能还存在（因为 API 请求超时/卡住还没返回）。
_ws_safe_name = "".join([c for c in current_ws if c.isalnum()]).strip()
_pending_file = os.path.join(LOG_DIR, f"pending_msg_{_ws_safe_name}.json")

# 仅当 session 中没有 pending_msg 且没有编辑状态时，尝试从文件恢复
if "pending_msg" not in st.session_state and "edit_state" not in st.session_state:
    if os.path.exists(_pending_file):
        try:
            with open(_pending_file, "r", encoding="utf-8") as _f:
                _pending_data = json.load(_f)
            # 转为草稿状态，让用户修改后重试
            st.session_state.edit_state = {
                "type": "draft",
                "msg": _pending_data,
            }
        except Exception as _e:
            print(f"从磁盘恢复 pending_msg 失败: {_e}")
        finally:
            # 无论成功还是失败，都删除磁盘文件，防止下次重复恢复
            if os.path.exists(_pending_file):
                os.remove(_pending_file)

if "chat_title" not in st.session_state:
    st.session_state.chat_title = st.session_state.api_configs.get(
        "workspace_titles", {}
    ).get(current_ws, "未命名案例")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(time.time())

TIMESTAMP_PREFIX = "\n\n[提交于："
TIMESTAMP_SUFFIX = "]"

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = str(time.time() + 1)

# 🌟 搜索结果独立缓存（key=msg_id, value={question, searches[]}）
#    搜索结果不再塞进 full_payload，从缓存独立管理
if "search_cache" not in st.session_state:
    st.session_state.search_cache = {}

# 🌟 上下文压缩状态（非侵入式：messages 数组不被修改）
if "compression_state" not in st.session_state:
    st.session_state.compression_state = {"ranges": []}

# --- 3. 侧边栏：资产控制台 ---
with st.sidebar:
    st.title("🛡️ 审计级实验室")

    workspaces = st.session_state.api_configs.get(
        "workspaces", ["默认沙盒 (Normal)", "Gem 专属区 (Gem)", "草稿区 (Draft)"]
    )
    if "默认沙盒 (Normal)" not in workspaces:
        workspaces.insert(0, "默认沙盒 (Normal)")
        st.session_state.api_configs["workspaces"] = workspaces
        save_config(st.session_state.api_configs)

    def_ws_idx = workspaces.index(current_ws) if current_ws in workspaces else 0
    st.selectbox(
        "🗂️ 当前工作区 (互相隔离)",
        workspaces,
        index=def_ws_idx,
        key="ws_selector",
        on_change=change_workspace,
        format_func=lambda ws: f"{st.session_state.api_configs.get('workspace_titles', {}).get(ws, '未命名案例')} [{ws}]",
    )
    st.caption("⚠️ 切换工作区后如页面空白，请刷新网页。")

    with st.expander("⚙️ 工作区管理"):
        new_ws_name = st.text_input("新增工作区名称", key="new_ws_input")
        if st.button("➕ 添加工作区 (复制当前配置)", use_container_width=True):
            if new_ws_name and new_ws_name not in workspaces:
                workspaces.append(new_ws_name)
                st.session_state.api_configs["workspaces"] = workspaces

                if "workspace_prefs" not in st.session_state.api_configs:
                    st.session_state.api_configs["workspace_prefs"] = {}
                st.session_state.api_configs["workspace_prefs"][new_ws_name] = {
                    "last_choice": st.session_state.api_configs.get(
                        "last_choice", "Gemini"
                    ),
                    "last_model": st.session_state.api_configs.get("last_model", ""),
                    "system_prompt": st.session_state.api_configs.get(
                        "system_prompt", ""
                    ),
                    "temperature": st.session_state.api_configs.get("temperature", 0.7),
                    "top_p": st.session_state.api_configs.get("top_p", 0.9),
                    "max_tokens": st.session_state.api_configs.get("max_tokens", 8192),
                }

                save_config(st.session_state.api_configs)
                st.toast(f"已添加工作区: {new_ws_name}")
                st.rerun()
            elif new_ws_name in workspaces:
                st.warning("工作区已存在！")
        del_ws_opts = [ws for ws in workspaces if ws != "默认沙盒 (Normal)"]
        if del_ws_opts:
            del_ws_name = st.selectbox(
                "选择要删除的工作区", del_ws_opts, key="del_ws_select"
            )
            if st.button("🗑️ 删除选定工作区", use_container_width=True):
                if del_ws_name:
                    workspaces.remove(del_ws_name)
                    st.session_state.api_configs["workspaces"] = workspaces
                    if (
                        "workspace_prefs" in st.session_state.api_configs
                        and del_ws_name
                        in st.session_state.api_configs["workspace_prefs"]
                    ):
                        del st.session_state.api_configs["workspace_prefs"][del_ws_name]
                    if current_ws == del_ws_name:
                        st.session_state.api_configs["current_workspace"] = (
                            "默认沙盒 (Normal)"
                        )
                    save_config(st.session_state.api_configs)
                    st.toast(f"已删除工作区: {del_ws_name}")
                    st.rerun()

    st.divider()

    st.subheader("📝 案例属性")

    # 🌟 新增：自动命名配置
    auto_name_prefs = st.session_state.api_configs.get("auto_name_prefs", {})
    auto_name_enabled = auto_name_prefs.get("enabled", False)
    auto_name_model = auto_name_prefs.get("model", "Qwen/Qwen2.5-72B-Instruct")

    with st.expander("⚙️ 自动命名设置", expanded=auto_name_enabled):
        auto_name_channel = auto_name_prefs.get("channel", "SiliconFlow")
        auto_name_enabled = st.toggle("✅ 开启自动命名", value=auto_name_enabled)
        if auto_name_enabled:
            auto_name_channel_ids = get_visible_channels(st.session_state.api_configs)
            if not auto_name_channel_ids:
                auto_name_channel_ids = ["SiliconFlow", "Gemini", "DeepSeek"]
            auto_name_labels = [get_channel_display_name(pid) for pid in auto_name_channel_ids]
            _auto_def_idx = auto_name_channel_ids.index(auto_name_channel) if auto_name_channel in auto_name_channel_ids else 0
            auto_name_label = st.selectbox(
                "命名模型通道",
                auto_name_labels,
                index=_auto_def_idx,
                help="推荐使用便宜模型来执行命名任务",
            )
            auto_name_channel = auto_name_channel_ids[auto_name_labels.index(auto_name_label)]
            if auto_name_channel == "SiliconFlow":
                sf_models = [
                    "Qwen/Qwen2.5-72B-Instruct",
                    "deepseek-ai/DeepSeek-V3",
                    "deepseek-ai/DeepSeek-R1",
                ]
                auto_name_model = st.selectbox(
                    "选择命名模型",
                    sf_models,
                    index=(
                        sf_models.index(auto_name_model)
                        if auto_name_model in sf_models
                        else 0
                    ),
                )
            else:
                gem_models = [
                    "gemini-2.0-flash-thinking-exp",
                    "gemini-2.5-pro",
                    "gemini-1.5-pro",
                ]
                auto_name_model = st.selectbox(
                    "选择命名模型",
                    gem_models,
                    index=(
                        gem_models.index(auto_name_model)
                        if auto_name_model in gem_models
                        else 0
                    ),
                )

        if st.button("💾 保存自动命名设置", use_container_width=True):
            st.session_state.api_configs["auto_name_prefs"] = {
                "enabled": auto_name_enabled,
                "channel": auto_name_channel,
                "model": auto_name_model,
            }
            save_config(st.session_state.api_configs)
            st.toast("✅ 自动命名设置已保存")
            st.rerun()

    current_title = st.session_state.api_configs.get("workspace_titles", {}).get(
        current_ws, "未命名案例"
    )
    new_title = st.text_input("案例名称", value=current_title)

    if new_title != current_title:
        st.session_state.chat_title = new_title
        if "workspace_titles" not in st.session_state.api_configs:
            st.session_state.api_configs["workspace_titles"] = {}
        st.session_state.api_configs["workspace_titles"][current_ws] = new_title
        save_config(st.session_state.api_configs)
    else:
        st.session_state.chat_title = current_title

    # 🌟 新增：手动重新命名按钮
    if st.button(
        "🤖 重新生成案例名称",
        use_container_width=True,
        help="基于当前对话内容重新生成案例名称",
    ):
        if len(st.session_state.messages) > 0:
            try:
                with st.spinner("正在重新生成案例名称..."):
                    # 提取最近的对话内容（最多最近3轮）
                    recent_messages = []
                    for msg in st.session_state.messages[-6:]:
                        # 取最近6条消息（最多3轮对话）
                        role = "用户" if msg["role"] == "user" else "助手"
                        content = msg.get("content", "")
                        # 移除时间戳
                        if TIMESTAMP_PREFIX in content:
                            content = content.rsplit(TIMESTAMP_PREFIX, 1)[0]
                        recent_messages.append(f"{role}：{content}")

                    conversation_text = "\n\n".join(recent_messages)

                    prompt_for_rename = f"""
请根据以下对话内容，生成一个 5-10 个字的简洁案例名称。
要求：
1. 准确概括对话主题
2. 简洁明了
3. 不要超过 10 个字
4. 不要包含标点符号
5. 仅回复案例名称，不要其他内容

对话内容：
{conversation_text}
"""

                    auto_name_prefs = st.session_state.api_configs.get(
                        "auto_name_prefs", {}
                    )
                    channel = auto_name_prefs.get("channel", "SiliconFlow")
                    model = auto_name_prefs.get("model", "Qwen/Qwen2.5-72B-Instruct")

                    # 🌟 通用化：通过供应商注册表路由
                    _rn_key = lookup_api_key(st.session_state.api_configs, channel)
                    if is_gemini_type(channel):
                        _rc = genai.Client(api_key=_rn_key)
                        _resp = _rc.models.generate_content(
                            model=normalize_gemini_model(model),
                            contents=prompt_for_rename,
                            generation_config=types.GenerateContentConfig(
                                temperature=0.3, max_output_tokens=30
                            ),
                        )
                        generated_name = _resp.text.strip()
                    elif is_openai_type(channel):
                        _rc = create_openai_client(channel, _rn_key, _opencode_session_id())
                        _resp = _rc.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "你是一个专业的案例命名助手，请根据对话内容生成简洁的案例名称。"},
                                {"role": "user", "content": prompt_for_rename},
                            ],
                            temperature=0.3,
                            max_tokens=30,
                        )
                        generated_name = _resp.choices[0].message.content.strip()
                    else:
                        generated_name = ""

                    # 清理生成的名称
                    generated_name = (
                        generated_name.replace('"', "")
                        .replace("'", "")
                        .replace("。", "")
                        .replace("，", "")
                        .replace("、", "")
                        .replace("《", "")
                        .replace("》", "")
                        .strip()
                    )

                    if generated_name and len(generated_name) > 0:
                        st.session_state.chat_title = generated_name
                        if "workspace_titles" not in st.session_state.api_configs:
                            st.session_state.api_configs["workspace_titles"] = {}
                        st.session_state.api_configs["workspace_titles"][
                            current_ws
                        ] = generated_name
                        save_config(st.session_state.api_configs)
                        st.toast(f"✅ 已重新命名：{generated_name}")
                        st.rerun()
            except Exception as e:
                st.error(f"重新命名失败: {e}")
        else:
            st.warning("暂无对话内容，无法生成案例名称")

    st.divider()

    # 🌟 防呆机制：仅在 Gem 工作区显示 Gem 专属定制模块
    if "(Gem)" in current_ws:
        with st.expander("💎 Gem 专属定制"):
            gem_instruction = st.text_area(
                "Gem 固定要求 (Fixed Prompt)",
                value=st.session_state.api_configs.get("gem_instruction", ""),
                help="设定该 Gem 的人设、输出格式或固定要求。",
            )
            gem_files = st.file_uploader(
                "Gem 专属知识库 (固定附件)",
                type=["txt", "md", "py", "json", "csv", "pdf", "html"],
                key="gem_uploader",
                accept_multiple_files=True,
            )

            if st.button("🚀 实例化 Gem (建立底层缓存)", type="primary"):
                if not gem_instruction and not gem_files:
                    st.warning("请至少填写固定要求或上传知识库！")
                else:
                    st.session_state.messages = []
                    if os.path.exists(current_save_file):
                        os.remove(current_save_file)
                    for key in list(st.session_state.keys()):
                        if key.startswith("fb_"):
                            del st.session_state[key]
                    clear_transient_states()

                    gem_payload = ""
                    # 🌟 修改：Gem 固定要求不再拼入用户消息开头，而是并入 System Prompt（系统指令）
                    # if gem_instruction:
                    #     gem_payload += f"【Gem 专属固定要求】\n{gem_instruction}\n\n"

                    gem_attach_names = []
                    if gem_files:
                        for idx, g_file in enumerate(gem_files):
                            # 🌟 功能优化 1: 给附件打戳防止相互覆盖，加入 idx 防止同秒并发重名
                            ts = datetime.now().strftime("%Y%m%d%H%M%S")
                            attach_name = (
                                f"Gem_{ts}_{idx}_{os.path.basename(g_file.name)}"
                            )
                            gem_attach_names.append(attach_name)

                            save_path = os.path.join(ATTACH_DIR, attach_name)
                            with open(save_path, "wb") as f:
                                f.write(g_file.getbuffer())

                            # 使用统一的文件解析工具（已移除 >5MB 跳过限制）
                            file_text = process_uploaded_file(g_file)
                            gem_payload += f"【Gem 专属知识库：{attach_name}】\n{file_text}\n\n"

                    gem_payload += "系统初始化指令：请阅读并严格遵循上述专属知识库。如果理解，请仅回复：“💎 Gem 实例化完成！底层 Context Cache 已建立，随时可以开始对话。”"

                    st.session_state.trigger_gem_init = {
                        "payload": gem_payload,
                        "attach_names": gem_attach_names,
                    }

                    st.session_state.api_configs["gem_instruction"] = gem_instruction
                    save_config(st.session_state.api_configs)
                    st.rerun()

    # --- 上下文压缩（折叠，紧随案例属性区） ---
    _comp_has_ranges = bool(
        st.session_state.get("compression_state", {}).get("ranges", [])
    )
    with st.expander("🧹 上下文压缩", expanded=_comp_has_ranges):
        comp_defaults = st.session_state.api_configs.get("compression_defaults", {})

        with st.expander("⚙️ 压缩配置", expanded=False):
            comp_prompt = st.text_area(
                "压缩提示词",
                value=comp_defaults.get(
                    "prompt",
                    "请用中文简要总结以下对话的核心内容和技术要点，保留关键决策和代码片段要点。",
                ),
                height=150,
                help="此提示词将被发送给压缩模型，指导其如何总结对话。",
            )

            comp_ch_ids = get_visible_channels(st.session_state.api_configs)
            if not comp_ch_ids:
                comp_ch_ids = ["Gemini", "SiliconFlow"]
            comp_ch_labels = [get_channel_display_name(pid) for pid in comp_ch_ids]
            comp_ch = comp_defaults.get("channel", "Gemini")
            comp_def_idx = comp_ch_ids.index(comp_ch) if comp_ch in comp_ch_ids else 0
            comp_channel = comp_ch_ids[
                comp_ch_labels.index(
                    st.selectbox(
                        "压缩模型通道",
                        comp_ch_labels,
                        index=comp_def_idx,
                        key="comp_chan",
                    )
                )
            ]

            comp_opts = st.session_state.model_config.get(comp_channel, [])
            if os.path.exists("enabled_models.json"):
                with open("enabled_models.json", "r", encoding="utf-8") as f:
                    comp_enabled = json.load(f)
                comp_extra = comp_enabled.get(comp_channel, [])
                comp_opts = list(set(comp_opts + comp_extra))
                comp_opts.sort()
            if "自定义..." not in comp_opts:
                comp_opts.insert(0, "自定义...")
            if not comp_opts:
                comp_opts = ["自定义..."]

            comp_m = comp_defaults.get("model", comp_opts[0] if comp_opts else "gemini-2.0-flash")
            comp_m_idx = comp_opts.index(comp_m) if comp_m in comp_opts else 0
            comp_model = st.selectbox(
                "压缩模型", comp_opts, index=comp_m_idx, key="comp_model_sel"
            )
            if comp_model == "自定义...":
                comp_model = st.text_input(
                    "手动输入压缩模型ID",
                    value=comp_defaults.get("model", ""),
                    key="comp_model_custom",
                )

            if st.button("💾 保存压缩配置", use_container_width=True):
                st.session_state.api_configs["compression_defaults"] = {
                    "channel": comp_channel,
                    "model": comp_model,
                    "prompt": comp_prompt,
                }
                save_config(st.session_state.api_configs)
                st.toast("✅ 压缩配置已保存")

        rounds = _get_round_mapping(st.session_state.messages)
        if rounds:
            round_options = [f"第{rn}轮: {preview}" for rn, idx, preview in rounds]
            start_sel = st.selectbox(
                "压缩起始轮次", round_options, key="comp_start"
            )
            end_default_idx = round_options.index(start_sel)
            end_sel = st.selectbox(
                "压缩结束轮次",
                round_options[end_default_idx:],
                key="comp_end",
            )

            start_rn_str = start_sel.split(":")[0]
            end_rn_str = end_sel.split(":")[0]
            start_rn = int(start_rn_str.replace("第", "").replace("轮", ""))
            end_rn = int(end_rn_str.replace("第", "").replace("轮", ""))

            start_i = None
            end_i = None
            for rn, idx, _ in rounds:
                if rn == start_rn:
                    start_i = idx
                if rn == end_rn:
                    end_i = idx
                    if (
                        end_i + 1 < len(st.session_state.messages)
                        and st.session_state.messages[end_i + 1]["role"] == "assistant"
                    ):
                        end_i = end_i + 1

            if start_i is not None and end_i is not None:
                msg_ids_to_compress = [
                    st.session_state.messages[j]["msg_id"]
                    for j in range(start_i, end_i + 1)
                    if "msg_id" in st.session_state.messages[j]
                ]
                has_overlap, conflict_id = check_range_overlap(
                    st.session_state.compression_state, msg_ids_to_compress
                )

                count = end_i - start_i + 1
                st.caption(f"将压缩 {count} 条消息 (第{start_rn}-{end_rn}轮)")

                if has_overlap:
                    st.warning(f"⚠️ 所选区间与已有压缩区间 ({conflict_id}) 重叠，请调整范围。")
                elif st.button("🚀 执行压缩", type="primary", use_container_width=True):
                    with st.spinner(f"正在使用 {comp_channel}/{comp_model} 压缩对话..."):
                        try:
                            summary = compress_messages_range(
                                st.session_state.messages,
                                start_i,
                                end_i,
                                comp_prompt,
                                comp_channel,
                                comp_model,
                                st.session_state.api_configs,
                            )
                            add_compression_range(
                                st.session_state.compression_state,
                                msg_ids_to_compress,
                                summary,
                                comp_channel,
                                comp_model,
                                comp_prompt,
                            )
                            save_chat_history(
                                current_save_file,
                                st.session_state.messages,
                                st.session_state.get("search_cache", {}),
                            )
                            st.toast(f"✅ 已压缩第{start_rn}-{end_rn}轮对话")
                            st.rerun()
                        except Exception as e:
                            st.error(f"压缩失败: {e}")

        cs = st.session_state.compression_state
        if cs.get("ranges"):
            st.divider()
            st.caption(f"已压缩 {len(cs['ranges'])} 个区间")
            for r in cs["ranges"]:
                with st.expander(f"📦 {len(r['msg_ids'])} 条消息"):
                    st.caption(f"模型: {r.get('model', 'N/A')}")
                    st.caption(f"提示词: {r.get('prompt', '')[:80]}...")
                    st.info(r["summary"])
                    if st.button(
                        "🗑️ 取消此压缩",
                        key=f"uncmp_{r['id']}",
                        use_container_width=True,
                    ):
                        remove_compression_range(
                            st.session_state.compression_state, r["id"]
                        )
                        save_chat_history(
                            current_save_file,
                            st.session_state.messages,
                            st.session_state.get("search_cache", {}),
                        )
                        st.rerun()
        else:
            st.caption("暂无压缩区间")

    # --- 对话索引 ---
    st.divider()
    st.subheader("📋 对话索引")

    idx_rounds = _get_round_mapping(st.session_state.messages)
    if idx_rounds:
        cs2 = st.session_state.compression_state
        compressed_ids = get_compressed_msg_ids(cs2)

        with st.expander("对话轮次导航", expanded=len(idx_rounds) <= 10):
            i = 0
            while i < len(idx_rounds):
                rn, idx, preview = idx_rounds[i]
                mid = st.session_state.messages[idx].get("msg_id", "")

                if mid in compressed_ids:
                    cur_range = None
                    for r in cs2.get("ranges", []):
                        if mid in r["msg_ids"]:
                            cur_range = r
                            break

                    if cur_range:
                        comp_rnds = []
                        j = i
                        while j < len(idx_rounds):
                            crn, cidx, cprev = idx_rounds[j]
                            cmid = st.session_state.messages[cidx].get("msg_id", "")
                            if cmid in cur_range["msg_ids"]:
                                comp_rnds.append((crn, cidx, cprev))
                                j += 1
                            else:
                                break

                        first_rn = comp_rnds[0][0]
                        last_rn = comp_rnds[-1][0]
                        with st.expander(f"📦 已压缩 (第{first_rn}-{last_rn}轮)"):
                            for crn, cidx, cprev in comp_rnds:
                                short = (
                                    cprev[:30] + "..."
                                    if len(cprev) > 30
                                    else cprev
                                )
                                st.markdown(
                                    f'<a href="#turn_{cidx}" title="{cprev}">第{crn}轮: {short}</a>',
                                    unsafe_allow_html=True,
                                )
                            summ = cur_range.get("summary", "")
                            st.info(
                                summ[:150]
                                + ("..." if len(summ) > 150 else "")
                            )
                        i = j
                        continue

                short = preview[:30] + "..." if len(preview) > 30 else preview
                st.markdown(
                    f'<a href="#turn_{idx}" title="{preview}">第{rn}轮: {short}</a>',
                    unsafe_allow_html=True,
                )
                i += 1
    else:
        st.caption("暂无对话")

    st.divider()

    web_search = st.toggle(
        "🌍 开启全局联网", value=st.session_state.api_configs.get("web_search", True)
    )
    st.session_state.api_configs["web_search"] = web_search

    # 🔍 搜索引擎选择（Tavily / 博查）
    search_engine = st.radio(
        "🔎 搜索引擎",
        ["Tavily", "博查(Bocha)"],
        index=0 if st.session_state.api_configs.get("search_engine", "Tavily") == "Tavily" else 1,
        help="Tavily 是默认搜索引擎；博查(Bocha) 是国产搜索引擎，支持更丰富的搜索参数",
        disabled=not web_search,
    )
    st.session_state.api_configs["search_engine"] = search_engine

    # 🔍 联网搜索关键词提取提示词 - 显性化展示与编辑
    with st.expander("🔍 联网搜索关键词提取提示词"):
        # 从 JSON 配置加载当前提示词
        current_prompt = load_prompt("keyword_extraction")

        # 可编辑文本框，让用户自定义关键词提取提示词
        edited_prompt = st.text_area(
            "编辑提示词（用于从你的问题中提取搜索关键词）",
            value=current_prompt,
            height=150,
            help="修改后点击「保存修改」即生效，下次搜索将使用新提示词",
        )

        # 显示字数统计，帮助用户了解当前提示词长度
        st.caption(f"📝 当前共 {len(edited_prompt)} 个字")

        # 两列布局：保存 + 恢复默认
        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button("💾 保存修改", use_container_width=True):
                save_prompt(edited_prompt, "keyword_extraction")
                st.success("✅ 提示词已保存！下次联网搜索时将使用新提示词。")

        with col_reset:
            if st.button("🔄 恢复默认", use_container_width=True):
                default_prompt = reset_prompt("keyword_extraction")
                st.success("✅ 已恢复为默认提示词，请刷新页面查看。")
                st.rerun()

    channel_ids = get_visible_channels(st.session_state.api_configs)
    if not channel_ids:
        channel_ids = ["Gemini"]
    channel_labels = [get_channel_display_name(pid) for pid in channel_ids]
    last_choice = st.session_state.api_configs.get("last_choice", "Gemini")
    def_ch_idx = channel_ids.index(last_choice) if last_choice in channel_ids else 0
    model_choice = channel_ids[def_ch_idx]
    selected_label = st.selectbox("模型通道", channel_labels, index=def_ch_idx)
    model_choice = channel_ids[channel_labels.index(selected_label)]

    # 到：
    if "model_config" not in st.session_state:
        st.session_state.model_config = load_model_config()

    opts = st.session_state.model_config.get(model_choice, [])

    # 🌟 核心：融合静态配置与用户动态启用的模型
    if os.path.exists("enabled_models.json"):
        with open("enabled_models.json", "r", encoding="utf-8") as f:
            enabled = json.load(f)
        enabled_for_platform = enabled.get(model_choice, [])
        # 融合：enabled_for_platform + 静态配置，去重并排序
        opts = list(set(opts + enabled_for_platform))
        opts.sort()

    if "自定义..." not in opts:
        opts.insert(0, "自定义...")

    if not opts:
        opts = ["自定义..."]  # 配置为空时的兜底

    # 获取当前工作区的偏好设置
    workspace_prefs = st.session_state.api_configs.get("workspace_prefs", {}).get(
        current_ws, {}
    )

    # 优先从 platform_models 中获取当前平台对应的模型
    platform_models = workspace_prefs.get("platform_models", {})
    platform_model = platform_models.get(model_choice)

    # 如果 platform_models 中没有当前平台的模型，则使用 last_model 作为后备
    if platform_model:
        last_m = platform_model
    else:
        last_m = workspace_prefs.get(
            "last_model",
            st.session_state.api_configs.get(
                "last_model", opts[0] if opts else "自定义..."
            ),
        )

    def_m_idx = opts.index(last_m) if last_m in opts else 0
    target_model = st.selectbox("选择具体模型", opts, index=def_m_idx)
    if target_model == "自定义...":
        target_model = st.text_input("手动输入模型ID", value=last_m)

    with st.expander("🎛️ 高级调优参数 (System & Generation)"):
        # 🌟 联网搜索建议指令
        SEARCH_SUGGESTION_INSTRUCTION = (
            "如果搜索结果不完整或需要更多信息，请在回答末尾添加"
            "【联网搜索建议：关键词1,关键词2】的格式明确告诉我需要搜索什么。"
        )

        sys_prompt = st.text_area(
            "System Prompt (全局行为约束)",
            value=st.session_state.api_configs.get("system_prompt", ""),
            help="设置 AI 的人设或全局指令。",
        )

        # 🌟 联网搜索建议开关
        suggestion_enabled = st.toggle(
            "🔍 开启联网搜索建议",
            value=st.session_state.api_configs.get(
                "web_search_suggestion_enabled", False
            ),
        )
        st.session_state.api_configs["web_search_suggestion_enabled"] = (
            suggestion_enabled
        )

        if suggestion_enabled:
            st.info(
                f'📌 已开启，发送时将自动向 System Prompt 追加：\n\n"{SEARCH_SUGGESTION_INSTRUCTION}"'
            )

        col_t, col_p = st.columns(2)
        with col_t:
            temp = st.slider(
                "Temperature",
                0.0,
                2.0,
                st.session_state.api_configs.get("temperature", 0.7),
                0.1,
            )
        with col_p:
            top_p = st.slider(
                "Top P", 0.0, 1.0, st.session_state.api_configs.get("top_p", 0.9), 0.05
            )

        max_tok = st.number_input(
            "Max Output Tokens",
            1024,
            128000,
            st.session_state.api_configs.get("max_tokens", 8192),
            1024,
        )

        # 🌟 思考参数（模型 ID 含 deepseek 即可用，不限通道）
        reasoning_effort = "medium"
        if "deepseek" in target_model.lower():
            reasoning_effort = st.selectbox(
                "思考强度 (Reasoning Effort)",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(
                    st.session_state.api_configs.get("reasoning_effort", "medium")
                ),
                help="设置 DeepSeek 模型的推理深度。",
            )

    if st.button("💾 持久化偏好配置"):
        st.session_state.api_configs["last_choice"] = model_choice
        st.session_state.api_configs["last_model"] = target_model
        st.session_state.api_configs["system_prompt"] = sys_prompt
        st.session_state.api_configs["temperature"] = temp
        st.session_state.api_configs["top_p"] = top_p
        st.session_state.api_configs["max_tokens"] = max_tok
        st.session_state.api_configs["reasoning_effort"] = reasoning_effort

        if "workspace_prefs" not in st.session_state.api_configs:
            st.session_state.api_configs["workspace_prefs"] = {}

        # 获取当前工作区的现有偏好设置（如果存在）
        workspace_prefs = st.session_state.api_configs["workspace_prefs"].get(
            current_ws, {}
        )

        # 获取或初始化 platform_models
        platform_models = workspace_prefs.get("platform_models", {})

        # 更新当前平台的模型选择
        platform_models[model_choice] = target_model

        # 保存工作区偏好设置
        st.session_state.api_configs["workspace_prefs"][current_ws] = {
            "last_choice": model_choice,
            "last_model": target_model,
            "platform_models": platform_models,  # 新增：保存每个平台最后选择的模型
            "system_prompt": sys_prompt,
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_tok,
            "reasoning_effort": reasoning_effort,
        }

        save_config(st.session_state.api_configs)
        st.toast("已更新本地配置")

    st.divider()

    st.subheader("🛠️ 视图与数据操作")
    view_mode = st.toggle("🔍 强制源码显示 (Raw Mode)")

    st.divider()

    # --- 新增：Gemini 缓存监控 ---
    if model_choice == "Gemini":
        st.subheader("📦 Gemini Cache 状态")
        active_cache_name = st.session_state.get("active_gemini_cache")

        if active_cache_name:
            try:
                client = genai.Client(api_key=st.session_state.api_configs["gemini"])
                # --- 核心强化：获取完整的缓存元数据 ---
                if (
                    "cache_metadata" not in st.session_state
                    or st.session_state.cache_metadata.name != active_cache_name
                ):
                    st.session_state.cache_metadata = client.caches.get(
                        name=active_cache_name
                    )

                cache = st.session_state.cache_metadata
                cache_id = cache.name.split("/")[-1]
                st.code(f"ID: {cache_id}", language="bash")

                # --- 核心强化：显示精确到期时间和剩余寿命 ---
                if cache.expire_time:
                    expire_utc = cache.expire_time
                    now_utc = datetime.now(timezone.utc)
                    remaining = expire_utc - now_utc

                    if remaining.total_seconds() > 0:
                        hours, rem = divmod(remaining.total_seconds(), 3600)
                        mins, secs = divmod(rem, 60)
                        st.info(f"⏳ 剩余寿命: {int(hours)}h {int(mins)}m {int(secs)}s")
                        st.caption(
                            f"将于 {expire_utc.astimezone().strftime('%Y-%m-%d %H:%M:%S')} 到期"
                        )
                    else:
                        st.warning("🚨 缓存已过期！")

                cache_status = st.session_state.get("cache_last_hit_status", "未知")
                if cache_status == "命中":
                    st.success("✅ 上次请求: 缓存命中")
                elif cache_status == "未命中":
                    st.warning("❌ 上次请求: 缓存未命中")
                else:
                    st.caption("从未发起过关联请求")

                # --- 核心强化：提供缓存延期功能 ---
                with st.expander("🔧 缓存续期与操作"):
                    st.write("为当前缓存追加新的存留时间 (TTL):")
                    col_ttl_val, col_ttl_unit = st.columns(2)
                    with col_ttl_val:
                        ttl_value = st.number_input(
                            "时长", min_value=1, value=1, step=1, key="extend_ttl_val"
                        )
                    with col_ttl_unit:
                        ttl_unit = st.selectbox(
                            "单位", ["小时", "分钟", "秒"], key="extend_ttl_unit"
                        )

                    if st.button("⚡ 确认延长缓存", type="primary"):
                        unit_map = {"小时": 3600, "分钟": 60, "秒": 1}
                        ttl_seconds = ttl_value * unit_map[ttl_unit]
                        with st.spinner("正在向 Gemini 后台请求延长缓存..."):
                            try:
                                client.caches.update(
                                    name=cache.name,
                                    config=types.UpdateCachedContentConfig(
                                        ttl=f"{ttl_seconds}s"
                                    ),
                                )
                                # 强制刷新元数据
                                st.session_state.cache_metadata = client.caches.get(
                                    name=cache.name
                                )
                                st.toast("✅ 缓存已成功续期！")
                                st.rerun()
                            except Exception as e:
                                st.error(f"续期失败: {e}")

                    if st.button("🗑️ 废弃当前 Cache"):
                        try:
                            client.caches.delete(name=active_cache_name)
                            st.toast("✅ Gemini 后台缓存已删除")
                        except Exception as e:
                            st.error(f"后台删除失败: {e}")

                        for key in [
                            "active_gemini_cache",
                            "cache_metadata",
                            "cache_last_hit_status",
                        ]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
            except Exception as e:
                st.error(f"无法获取缓存状态: {e}")
                if st.button("强制清除本地缓存记录"):
                    for key in [
                        "active_gemini_cache",
                        "cache_metadata",
                        "cache_last_hit_status",
                    ]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        else:
            st.caption("当前无活动的 Cache。")

    st.subheader("📁 审计归档")
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(
        [c for c in st.session_state.chat_title if c.isalnum() or c in (" ", "_", "-")]
    ).strip()

    chat_bundle = {
        "metadata": {
            "title": st.session_state.chat_title,
            "export_time": file_ts,
            "model": target_model,
            "workspace": current_ws,
            "system_prompt": sys_prompt,
            "temperature": temp,
            "top_p": top_p,
        },
        "messages": st.session_state.messages,
    }

    st.download_button(
        "📥 导出 JSON 资产",
        data=json.dumps(chat_bundle, ensure_ascii=False, indent=2),
        file_name=f"纯净chat沙盒_{safe_title}_{file_ts}.json",
    )
    md_data = export_chat_to_md(st.session_state.chat_title, st.session_state.messages)
    st.download_button(
        "📄 导出 MD 资产 (带附件与时间)",
        data=md_data,
        file_name=f"纯净chat沙盒_{safe_title}_{file_ts}.md",
        mime="text/markdown",
    )
    uploaded_json = st.file_uploader(
        "📤 导入历史 JSON", type="json", key=st.session_state.uploader_key
    )
    if uploaded_json and st.button("🔄 载入并覆盖会话"):
        try:
            data = json.load(uploaded_json)
            if isinstance(data, dict):
                st.session_state.messages = data.get("messages", [])
                meta = data.get("metadata", {})

                imported_title = meta.get("title", "已载入案例")
                st.session_state.chat_title = imported_title
                if "workspace_titles" not in st.session_state.api_configs:
                    st.session_state.api_configs["workspace_titles"] = {}
                st.session_state.api_configs["workspace_titles"][
                    current_ws
                ] = imported_title

                if "system_prompt" in meta:
                    st.session_state.api_configs["system_prompt"] = meta[
                        "system_prompt"
                    ]
                if "temperature" in meta:
                    st.session_state.api_configs["temperature"] = meta["temperature"]
                if "top_p" in meta:
                    st.session_state.api_configs["top_p"] = meta["top_p"]
            elif isinstance(data, list):
                st.session_state.messages = data
            else:
                st.session_state.messages = []

            save_chat_history(
                current_save_file,
                st.session_state.messages,
                st.session_state.get("search_cache", {}),
            )
            save_config(st.session_state.api_configs)

            for key in list(st.session_state.keys()):
                if key.startswith("fb_"):
                    del st.session_state[key]
            clear_transient_states()

            st.session_state.uploader_key = str(time.time())
            st.rerun()
        except Exception as e:
            st.error(f"解析 JSON 失败: {str(e)}")

    st.divider()

    has_gem_base = len(st.session_state.messages) >= 2 and st.session_state.messages[
        0
    ].get("is_gem_init")
    if has_gem_base:
        if st.button(
            "🧹 重置到 Gem 初始态 (保留缓存)",
            help="清空后续所有对话，仅保留 Gem 的简历和要求，防止 JD 污染",
        ):
            st.session_state.messages = st.session_state.messages[:2]

            # 🌟 清理被删除消息的 search_cache 条目
            if "search_cache" in st.session_state:
                # 只保留仍在 messages 中的 msg_id
                existing_ids = {
                    m["msg_id"] for m in st.session_state.messages if "msg_id" in m
                }
                st.session_state.search_cache = {
                    k: v
                    for k, v in st.session_state.search_cache.items()
                    if k in existing_ids
                }

            cleanup_deleted_messages(
                st.session_state.messages,
                st.session_state.compression_state,
            )

            save_chat_history(
                current_save_file,
                st.session_state.messages,
                st.session_state.get("search_cache", {}),
            )

            for key in list(st.session_state.keys()):
                if key.startswith("fb_"):
                    idx = int(key.split("_")[1])
                    if idx >= 2:
                        del st.session_state[key]
            clear_transient_states()
            st.rerun()

    if st.button("🗑️ 彻底销毁当前会话", help="清空当前工作区的所有内容"):
        st.session_state.messages = []
        st.session_state.compression_state = {"ranges": []}
        if os.path.exists(current_save_file):
            os.remove(current_save_file)
        for key in list(st.session_state.keys()):
            if key.startswith("fb_"):
                del st.session_state[key]
        clear_transient_states()
        st.rerun()

    st.divider()

    st.divider()
    st.caption(f"🛡️ 纯净chat沙盒 V{__version__}")
    st.caption("⚠️ 附件缓存位于 ./attachments_cache，请定期清理。")

# --- 4. 主界面：流式对话 ---
# 确保侧边栏上下文已在上方 with st.sidebar: 中闭合
# 核心渲染入口
st.header(f"💼 {st.session_state.chat_title} [{current_ws}]")

# 🌟 功能增强 6: 处理删除请求
if "delete_request" in st.session_state:
    delete_turn(st.session_state.delete_request)
    cleanup_deleted_messages(
        st.session_state.messages,
        st.session_state.compression_state,
    )
    del st.session_state.delete_request  # 清理触发器
    st.rerun()  # 删除后立即刷新界面

# 渲染对话序列
for i, msg in enumerate(st.session_state.messages):
    cinfo = get_message_compression_info(
        st.session_state.messages,
        st.session_state.compression_state,
        i,
    )

    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<a id="turn_{i}" style="scroll-margin-top:80px;"></a>', unsafe_allow_html=True)
        if cinfo.get("is_start"):
            ni = cinfo.get("range_info", {})
            st.caption(f"📦 此消息起进入压缩区间 (共 {ni.get('count', '?')} 条消息，API 发送时将替换为摘要)")

        # ──────────────────────────────────────────────────
        # 分支渲染：assistant 且存在多个分支 → 使用 Tabs
        # ──────────────────────────────────────────────────
        if msg["role"] == "assistant" and msg_has_branches(msg):
            branch_keys = list(msg["branches"].keys())
            branch_labels = [
                msg["branches"][k].get("model_display", k) for k in branch_keys
            ]
            tabs = st.tabs(branch_labels)
            for tab_idx, (bk, tab) in enumerate(zip(branch_keys, tabs)):
                with tab:
                    branch = msg["branches"][bk]

                    if branch.get("thinking"):
                        with st.expander("🧠 剥离出的思考过程", expanded=False):
                            st.markdown(branch["thinking"])

                    content = branch.get("content", "")
                    if view_mode:
                        st.code(content, language="markdown")
                    else:
                        st.markdown(content)

                    # 分支级元数据（时间/模型/Latency/Token）
                    raw_time = branch.get("timestamp", "N/A")
                    display_time = (
                        raw_time[:19].replace("T", " ") if raw_time != "N/A" else "N/A"
                    )
                    lat = branch.get("latency", 0)
                    lat_ms = int(lat * 1000) if isinstance(lat, float) else lat
                    model_disp = branch.get(
                        "model_display",
                        f"Model: {branch.get('model', 'Unknown')}",
                    )
                    token_disp = branch.get("token_usage", "Tokens: N/A")

                    st.caption(
                        f"📅 {display_time} | 📦 {model_disp} | ⏱️ Latency: {lat_ms}ms | 📊 {token_disp}"
                    )

        else:
            # ── 非分支消息：原有渲染逻辑 ──
            if msg.get("thinking"):
                with st.expander("🧠 剥离出的思考过程", expanded=False):
                    st.markdown(msg["thinking"])

            if msg.get("is_gem_init"):
                with st.expander("💎 查看 Gem 注入的底层知识库与指令", expanded=False):
                    st.markdown(msg["raw_payload"])
            elif view_mode:
                st.code(msg["content"], language="markdown")
            else:
                st.markdown(msg["content"])

        # 🌟 功能增强 6: 增加删除功能 & 统一操作区
        with st.expander("📋 操作与原始数据", expanded=False):
            # 分支消息显示活跃分支的原始内容
            if msg["role"] == "assistant" and msg_has_branches(msg):
                active_bk = msg.get("active_branch")
                if active_bk and active_bk in msg["branches"]:
                    st.code(
                        msg["branches"][active_bk].get("content", ""),
                        language="markdown",
                    )
                else:
                    st.code(msg["content"], language="markdown")
            else:
                st.code(msg["content"], language="markdown")
            if st.button("🗑️ 删除此轮对话", key=f"del_{i}", use_container_width=True):
                st.session_state.delete_request = i
                st.rerun()

        # 🌟 功能优化 5: 兼容多附件显示
        if msg.get("attachments"):
            st.markdown(f"🖇️ 关联物理附件: `{', '.join(msg['attachments'])}`")
        elif msg.get("attachment"):
            st.markdown(f"🖇️ 关联物理附件: `{msg['attachment']}`")

        # --- 助理消息的专属元数据和反馈 ---
        if msg["role"] == "assistant":
            # 非分支消息显示顶层的元数据；分支消息的元数据已在 Tabs 内展示
            if not msg_has_branches(msg):
                raw_time = msg.get("timestamp", "N/A")
                display_time = (
                    raw_time[:19].replace("T", " ") if raw_time != "N/A" else "N/A"
                )
                lat = msg.get("latency", 0)
                lat_ms = int(lat * 1000) if isinstance(lat, float) else lat
                model_disp = msg.get(
                    "model_display", f"Model: {msg.get('model', 'Unknown')}"
                )
                token_disp = msg.get("token_usage", "Tokens: N/A")
                st.caption(
                    f"📅 {display_time} | 📦 {model_disp} | ⏱️ Latency: {lat_ms}ms | 📊 {token_disp}"
                )

            fb_key = f"fb_{i}"
            if fb_key not in st.session_state and "feedback" in msg:
                st.session_state[fb_key] = 1 if msg["feedback"] == "👍" else 0
            st.feedback("thumbs", key=fb_key, on_change=handle_feedback, args=(i,))

            # ⚖️ 对比模型 UI（仅对 assistant 消息显示）
            with st.expander("⚖️ 对比其他模型生成", expanded=False):
                cmp_channel_ids = get_visible_channels(st.session_state.api_configs)
                if not cmp_channel_ids:
                    cmp_channel_ids = ["Gemini", "SiliconFlow", "DeepSeek"]
                cmp_channel_labels = [get_channel_display_name(pid) for pid in cmp_channel_ids]
                cmp_selected_label = st.selectbox(
                    "选择对比平台",
                    cmp_channel_labels,
                    key=f"cmp_ch_{i}",
                )
                compare_channel = cmp_channel_ids[cmp_channel_labels.index(cmp_selected_label)]

                # 获取选定平台的可用模型列表
                cmp_opts = st.session_state.model_config.get(compare_channel, [])
                if os.path.exists("enabled_models.json"):
                    with open("enabled_models.json", "r", encoding="utf-8") as f:
                        enabled = json.load(f)
                    enabled_platform = enabled.get(compare_channel, [])
                    cmp_opts = list(set(cmp_opts + enabled_platform))
                    cmp_opts.sort()
                if "自定义..." not in cmp_opts:
                    cmp_opts.insert(0, "自定义...")
                if not cmp_opts:
                    cmp_opts = ["自定义..."]

                compare_model = st.selectbox(
                    "选择对比模型",
                    cmp_opts,
                    key=f"cmp_m_{i}",
                )
                if compare_model == "自定义...":
                    compare_model = st.text_input(
                        "手动输入模型ID", key=f"cmp_custom_{i}"
                    )

                if st.button(
                    "🚀 对比生成",
                    key=f"cmp_go_{i}",
                    use_container_width=True,
                ):
                    st.session_state.compare_request = {
                        "assistant_idx": i,
                        "model_choice": compare_channel,
                        "target_model": compare_model,
                    }
                    st.rerun()

        # --- 用户消息的专属缓存建立按钮 ---
        if msg["role"] == "user" and model_choice == "Gemini":
            if st.button(
                "📌 以此为结尾建立 Cache",
                key=f"cache_{i}",
                help="将本轮及之前的所有对话打包，在 Gemini 后台建立一个长效缓存。（注意：Gemini 官方要求缓存内容需大于 32,768 Tokens）",
            ):
                st.session_state.cache_creation_pending = i
                st.rerun()

        if cinfo.get("is_end"):
            ni = cinfo.get("range_info", {})
            st.info(f"📝 **压缩摘要**:\n\n{ni.get('summary', '')}")
            if st.button(
                "🗑️ 取消此压缩",
                key=f"msg_uncmp_{ni.get('id', '')}",
                use_container_width=True,
            ):
                remove_compression_range(
                    st.session_state.compression_state, ni.get("id", "")
                )
                save_chat_history(
                    current_save_file,
                    st.session_state.messages,
                    st.session_state.get("search_cache", {}),
                )
                st.rerun()


# --- 新增：底层 I/O 日志查看器 ---
if "last_turn_logs" in st.session_state:
    with st.expander("🔍 查看本轮对话底层 I/O 日志", expanded=False):
        st.subheader("📥 Input Payload (发往 API)")
        st.json(st.session_state.last_turn_logs["input_payload"])
        st.subheader("📤 Output Payload (来自 API)")
        st.json(st.session_state.last_turn_logs["output_payload"])

if (
    len(st.session_state.messages) >= 2
    and st.session_state.messages[-1]["role"] == "assistant"
):
    col1, col2, col3 = st.columns([1.5, 1.5, 3])
    with col1:
        if st.button(
            "🔁 直接重新生成", key="rerun_direct", help="使用原提示词直接重新生成"
        ):
            idx_to_remove = len(st.session_state.messages) - 1
            if f"fb_{idx_to_remove}" in st.session_state:
                del st.session_state[f"fb_{idx_to_remove}"]
            st.session_state.trigger_rerun = st.session_state.messages[-2]
            st.session_state.messages = st.session_state.messages[:-2]
            st.rerun()
    with col2:
        if not st.session_state.messages[-2].get("is_gem_init"):
            if st.button(
                "✏️ 编辑并重新生成", key="rerun_edit", help="修改提示词后再发送"
            ):
                st.session_state.edit_state = {
                    "type": "rerun",
                    "msg": st.session_state.messages[-2],
                }
                st.rerun()

# --- 新增：缓存创建配置界面 ---
if "cache_creation_pending" in st.session_state:
    idx = st.session_state.cache_creation_pending
    with st.container(border=True):
        st.subheader(f"📦 正在为截至第 {idx + 1} 条消息建立缓存")
        st.markdown("请为该缓存设置一个**存留时间 (TTL)**，过期后它将被自动删除。")

        col_ttl_val, col_ttl_unit = st.columns(2)
        with col_ttl_val:
            ttl_value = st.number_input(
                "时长", min_value=1, value=1, step=1, key="create_ttl_val"
            )
        with col_ttl_unit:
            ttl_unit = st.selectbox(
                "单位", ["小时", "分钟", "秒"], key="create_ttl_unit", index=0
            )

        col_confirm, col_cancel = st.columns(2)
        if col_confirm.button(
            "✅ 确认创建缓存", type="primary", use_container_width=True
        ):
            unit_map = {"小时": 3600, "分钟": 60, "秒": 1}
            ttl_seconds = ttl_value * unit_map[ttl_unit]
            st.session_state.execute_cache_creation = {
                "index": idx,
                "ttl": f"{ttl_seconds}s",
            }
            del st.session_state.cache_creation_pending
            st.rerun()

        if col_cancel.button("❌ 取消", use_container_width=True):
            del st.session_state.cache_creation_pending
            st.rerun()

# 显示上一次联网搜索的异常详情（用 session_state 跨 rerun 持久保留）
if "web_search_error" in st.session_state:
    st.error(f"🔍 上次联网搜索异常: {st.session_state.web_search_error}")


# ============================================================
# 获取有效的 System Prompt（合并全局指令、Gem 固定要求及联网建议）
# ============================================================
def get_effective_system_prompt(base_prompt):
    """
    根据开关状态和 Gem 配置，处理并返回最终的 System Prompt。
    1. 合并全局基础指令 (base_prompt)
    2. 如果存在 Gem 固定要求，则合并
    3. 如果开启了联网搜索建议，则追加相关指令
    """
    effective_prompt = base_prompt

    # 🌟 新增：合并 Gem 固定要求 (Fixed Prompt)
    # 逻辑修正：仅在 Gem 工作区 (名称包含 "(Gem)") 时才拼接
    gem_inst = st.session_state.api_configs.get("gem_instruction", "")
    if gem_inst.strip() and "(Gem)" in current_ws:
        # 如果已有基础指令，用双换行分隔
        sep = "\n\n" if effective_prompt.strip() else ""
        effective_prompt = f"{effective_prompt}{sep}【Gem 固定要求】\n{gem_inst}"

    # 联网搜索建议指令
    SEARCH_SUGGESTION_INSTRUCTION = (
        "如果搜索结果不完整或需要更多信息，请在回答末尾添加"
        "【联网搜索建议：关键词1,关键词2】的格式明确告诉我需要搜索什么。"
    )

    if st.session_state.api_configs.get("web_search_suggestion_enabled", False):
        if SEARCH_SUGGESTION_INSTRUCTION not in effective_prompt:
            # 在末尾优雅追加，保持分隔
            separator = "\n\n" if effective_prompt.strip() else ""
            effective_prompt = (
                f"{effective_prompt}{separator}{SEARCH_SUGGESTION_INSTRUCTION}"
            )

    return effective_prompt


# ============================================================
# 检测 AI 回答中的联网搜索建议
# ============================================================
def extract_ai_search_suggestion(content):
    """
    检测 AI 回答中是否包含联网搜索建议标记。

    当 AI 在 System Prompt 的引导下觉得信息不够时，
    会在回答末尾输出 【联网搜索建议：关键词1,关键词2】 格式的标记。
    此函数负责从回答文本中提取出这个标记里的关键词。

    参数:
        content: AI 回答的文本内容

    返回:
        str or None: 提取到的搜索关键词（逗号分隔），没有则返回 None
    """
    import re

    # 匹配 【联网搜索建议：xxx】 或 【联网搜索建议:xxx】 两种格式
    match = re.search(r"【联网搜索建议[：:](.+?)】", content)
    if match:
        return match.group(1).strip()
    return None


# --- 新增：多次联网功能（支持 AI 主动建议搜索） ---
# 🔍 根据所选搜索引擎检查对应的 API Key
_search_engine = st.session_state.api_configs.get("search_engine", "Tavily")
_search_api_key_available = (
    st.session_state.api_configs.get("tavily")
    if _search_engine == "Tavily"
    else st.session_state.api_configs.get("bocha")
)
if (
    web_search
    and _search_api_key_available
    and len(st.session_state.messages) >= 2
    and st.session_state.messages[-1]["role"] == "assistant"
    and is_openai_type(model_choice)
):
    st.divider()
    st.subheader("🔗 多次联网功能")

    # ----- 检测 AI 回答中是否包含联网搜索建议 -----
    # 扫描最后一条 AI 回答，看是否有 【联网搜索建议：关键词】 格式的标记
    ai_suggested_keywords = extract_ai_search_suggestion(
        st.session_state.messages[-1].get("content", "")
    )

    if ai_suggested_keywords and not st.session_state.get("ai_suggestion_ignored"):
        # ── 情况 A：AI 主动建议了搜索关键词 ──
        # 显示 AI 建议的关键词 + 确认按钮
        st.info(f"🤖 **AI 建议联网搜索：** `{ai_suggested_keywords}`")

        col_agree, col_ignore = st.columns([1, 1])
        with col_agree:
            if st.button("✅ 同意搜索", use_container_width=True, type="primary"):
                # 标记为"AI 建议"来源，触发处理逻辑中的路径A（重生成）
                st.session_state.search_source = "ai_suggested"
                st.session_state.trigger_additional_search = ai_suggested_keywords
                st.rerun()
        with col_ignore:
            if st.button("❌ 忽略", use_container_width=True):
                # 用户忽略 AI 建议，清除建议状态，下次 rerun 不再显示
                st.session_state.ai_suggestion_ignored = True
                st.rerun()
    else:
        # ── 情况 B：AI 没有建议，显示手动输入框 ──
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            additional_keywords = st.text_input(
                "输入新的搜索关键词进行再次联网 (可选)",
                placeholder="例如：最新发展、补充信息、其他方面...",
            )
        with col_btn:
            if st.button("🔎 再次联网", use_container_width=True):
                if additional_keywords.strip():
                    # 标记为"手动"来源，触发处理逻辑中的路径B（新增独立回答）
                    st.session_state.search_source = "manual"
                    st.session_state.trigger_additional_search = additional_keywords
                    st.rerun()
                else:
                    st.warning("请输入搜索关键词")

# --- 5. 核心推理：新版 SDK 适配 ---
# --- 修改：将 Token 测算改为手动触发 ---
# 🌟 功能优化 2 & 5: 调整输入区布局并支持多附件上传
with st.container():
    col1, col2 = st.columns([4, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "📎 添加单次或多个对话附件",
            type=["txt", "md", "py", "json", "csv", "pdf", "html"],
            key=st.session_state.file_uploader_key,
            accept_multiple_files=True,
        )
    with col2:
        st.write(" ")  # 用于垂直对齐
        st.write(" ")  # 用于垂直对齐
        st.button(
            "📊 预估 Token",
            on_click=update_token_estimate,
            use_container_width=True,
            help="计算当前完整上下文的 Token 数量",
        )

# 附件解析详情展示：显示每个已上传文件的字符数、体积和类型
if uploaded_files:
    with st.expander("📋 附件解析详情", expanded=True):
        for uf in uploaded_files:
            file_size_mb = uf.size / (1024 * 1024)
            # 提取文件扩展名作为类型标识
            ext = os.path.splitext(uf.name)[1].upper().lstrip(".") or "未知"
            # 解析文件内容并统计字符数
            file_text = process_uploaded_file(uf)
            char_count = len(file_text)
            # 格式化显示：📁 文件名 — 字符数 | 体积 | 类型
            st.caption(
                f"📁 `{uf.name}` — {char_count:,} 字符 | {file_size_mb:.1f}MB | {ext}"
            )

# 显示预估结果（如果存在）
if "estimated_tokens_display" in st.session_state:
    result = st.session_state.estimated_tokens_display
    if "失败" in str(result):
        st.error(f"预估失败，请检查API Key或模型ID。")
    else:
        st.info(f"📊 **当前上下文 Token: {result}**")

prompt = st.chat_input("输入指令...")

# 4. 进入草稿/编辑编辑模式渲染
if "edit_state" in st.session_state:
    state = st.session_state.edit_state

    # --- 新增：在这里渲染报错信息（如果存在，则报错在草稿上方） ---
    if "persistent_error" in st.session_state:
        st.error(st.session_state.persistent_error)
        del st.session_state.persistent_error

    is_draft = state["type"] == "draft"

    st.warning(
        "⚠️ 请求失败或被中断，已为您保存草稿，请修改后重试："
        if is_draft
        else "✏️ 正在编辑上一条消息..."
    )

    if state["msg"].get("is_gem_init"):
        st.info("💎 这是一个 Gem 实例化请求，为保护底层缓存，不支持修改文本。")
        col_e1, col_e2 = st.columns(2)
        if col_e1.button("🔄 重新发送", use_container_width=True):
            st.session_state.trigger_rerun = state["msg"]
            del st.session_state.edit_state
            st.rerun()
        if col_e2.button("❌ 取消", use_container_width=True):
            # 🌟 漏洞修复 6: 取消编辑时同步清理磁盘上的 pending 文件
            _ws_safe_name = "".join([c for c in current_ws if c.isalnum()]).strip()
            _pending_file = os.path.join(LOG_DIR, f"pending_msg_{_ws_safe_name}.json")
            if os.path.exists(_pending_file):
                os.remove(_pending_file)
            del st.session_state.edit_state
            st.rerun()
    else:
        # --- 修改：剥离时间戳，使其不可编辑 ---
        original_content = state["msg"]["content"]
        user_text = original_content
        original_timestamp = ""
        if TIMESTAMP_PREFIX in original_content:
            parts = original_content.rsplit(TIMESTAMP_PREFIX, 1)
            user_text = parts[0]
            original_timestamp = TIMESTAMP_PREFIX + parts[1]

        new_content = st.text_area("编辑提示词", value=user_text, height=150)
        if original_timestamp:
            st.caption(f"原始时间: {original_timestamp.strip()}")

        # 🌟 功能优化 5: 升级编辑模式以兼容多附件
        attach_names = state["msg"].get("attachments", [])
        # 兼容旧的单附件格式
        if not attach_names and state["msg"].get("attachment"):
            attach_names = [state["msg"]["attachment"]]

        if attach_names:
            st.markdown(
                f"📎 附带文件: `{', '.join(attach_names)}` *(发送时将自动重新挂载)*"
            )

        col_e1, col_e2 = st.columns(2)
        if col_e1.button("🚀 确认发送", use_container_width=True):
            final_prompt_text = new_content
            full_payload = new_content

            if attach_names:
                file_contents = []
                for attach_name in attach_names:
                    save_path = os.path.join(ATTACH_DIR, attach_name)
                    # 从带时间戳的文件名中还原原始名称用于显示
                    original_filename = "_".join(attach_name.split("_")[1:])
                    if os.path.exists(save_path):
                        # 已移除 >5MB 跳过限制，所有附件一律从本地缓存解析
                        if attach_name.lower().endswith(".pdf"):
                            file_text = read_pdf_content(save_path)
                        else:
                            with open(save_path, "rb") as f:
                                file_text = read_text_content(f.read())
                        file_contents.append(
                            f"【附件原文：{original_filename}】\n{file_text}"
                        )

                if file_contents:
                    full_payload = (
                        f"【用户指令】：{new_content}\n\n"
                        + "\n\n---\n\n".join(file_contents)
                    )

            if state["type"] == "rerun":
                idx_to_remove = len(st.session_state.messages) - 1
                if f"fb_{idx_to_remove}" in st.session_state:
                    del st.session_state[f"fb_{idx_to_remove}"]
                st.session_state.messages = st.session_state.messages[:-2]

            new_msg = state["msg"].copy()
            # 注意：这里 content 和 raw_payload 都不带时间戳，时间戳将在主流程中统一注入
            new_msg["content"] = final_prompt_text
            new_msg["raw_payload"] = full_payload
            new_msg["full_payload"] = full_payload

            st.session_state.trigger_rerun = new_msg
            del st.session_state.edit_state
            st.rerun()

        if col_e2.button("❌ 取消", use_container_width=True):
            # 🌟 漏洞修复 6: 取消编辑时同步清理磁盘上的 pending 文件
            _ws_safe_name = "".join([c for c in current_ws if c.isalnum()]).strip()
            _pending_file = os.path.join(LOG_DIR, f"pending_msg_{_ws_safe_name}.json")
            if os.path.exists(_pending_file):
                os.remove(_pending_file)
            del st.session_state.edit_state
            st.rerun()


if prompt and prompt.strip() and "edit_state" in st.session_state:
    del st.session_state.edit_state

trigger_msg = st.session_state.get("trigger_rerun")
trigger_gem = st.session_state.get("trigger_gem_init")
trigger_cache_exec = st.session_state.get("execute_cache_creation")
trigger_additional_search = st.session_state.get(
    "trigger_additional_search"
)  # 多次联网触发器
trigger_compare = st.session_state.get(
    "compare_request"
)  # 🌟 对比模型生成触发器
should_run = False

if trigger_gem:
    should_run = True
elif trigger_msg:
    should_run = True
elif trigger_cache_exec:
    should_run = True
elif trigger_additional_search:
    should_run = True
elif trigger_compare:
    should_run = True
elif (
    prompt is not None
    and "edit_state" not in st.session_state
    and "cache_creation_pending" not in st.session_state
):
    should_run = True

if should_run:
    # 清理临时UI状态标记（新对话开始时重置忽略标记）
    if "ai_suggestion_ignored" in st.session_state:
        del st.session_state.ai_suggestion_ignored

    _provider_key = lookup_api_key(st.session_state.api_configs, model_choice)
    if not _provider_key:
        _display_name = get_channel_display_name(model_choice)
        st.warning(f"⚠️ 拦截：请先在「模型库管理」页面配置 {_display_name} API Key！")
        st.stop()

    # 🔍 根据选择的搜索引擎检查对应的 API Key
    if web_search and is_openai_type(model_choice):
        search_engine = st.session_state.api_configs.get("search_engine", "Tavily")
        if search_engine == "Tavily" and not st.session_state.api_configs.get("tavily"):
            st.warning("⚠️ 拦截：开启联网模式（Tavily）时，请先在「模型库管理」页面配置 Tavily API Key！")
            st.stop()
        elif search_engine == "博查(Bocha)" and not st.session_state.api_configs.get("bocha"):
            st.warning("⚠️ 拦截：开启联网模式（博查）时，请先在「模型库管理」页面配置博查 API Key！")
            st.stop()

    is_rerun = False
    is_gem_init = False
    is_compare_branch = False  # 🌟 对比模型分支标志
    compare_assistant_idx = None  # 🌟 被追加分支的 assistant 消息索引
    attach_names = []  # 🌟 核心修复：在所有逻辑分支前初始化 attach_names
    preserved_msg_id = None  # 🌟 rerun 时保留原 msg_id（指向已有搜索缓存）
    original_timestamp_str = (
        ""  # 🕒 rerun 时保存原始时间戳，避免重新生成时更新为当前时间
    )

    if trigger_cache_exec:
        if not is_gemini_type(model_choice):
            st.error("❌ 仅 Gemini 通道支持手动建立缓存！")
            del st.session_state.execute_cache_creation
            st.stop()

        idx = trigger_cache_exec["index"]
        ttl = trigger_cache_exec["ttl"]

        with st.spinner(
            f"正在将截至第 {idx + 1} 条消息的历史打包并建立缓存 (TTL: {ttl})..."
        ):
            try:
                history_for_cache = []
                for m in st.session_state.messages[: idx + 1]:
                    role = "user" if m["role"] == "user" else "model"
                    # --- 核心修正：显式指定 text 参数 ---
                    history_for_cache.append(
                        types.Content(
                            role=role,
                            parts=[
                                types.Part.from_text(
                                    text=m.get("full_payload", m["content"])
                                )
                            ],
                        )
                    )

                client = genai.Client(api_key=st.session_state.api_configs["gemini"])
                cache_display_name = f"{st.session_state.chat_title}_{datetime.now().strftime('%Y%m%d%H%M')}"
                google_search_tool = types.Tool(google_search=types.GoogleSearch())

                cache_config_params = {
                    "display_name": cache_display_name,
                    "contents": history_for_cache,
                    "ttl": ttl,
                }
                if sys_prompt.strip():
                    # 🌟 修改：创建缓存时也使用有效的系统指令（含 Gem 要求）
                    cache_config_params["system_instruction"] = (
                        get_effective_system_prompt(sys_prompt)
                    )
                if web_search:
                    cache_config_params["tools"] = [google_search_tool]

                cache_config = types.CreateCachedContentConfig(**cache_config_params)
                cache = client.caches.create(
                    model=normalize_gemini_model(target_model), config=cache_config
                )

                st.session_state.active_gemini_cache = cache.name
                st.session_state.cache_metadata = cache
                st.session_state.cache_last_hit_status = "未知"

                st.success(f"✅ 缓存建立成功！ID: ...{cache.name[-10:]}")
                time.sleep(2)

            except Exception as e:
                st.error(f"建立缓存失败: {e}")
                time.sleep(3)
            finally:
                del st.session_state.execute_cache_creation
                st.rerun()
        st.stop()

    if trigger_gem:
        prompt = "💎 [Gem 实例化指令与知识库已注入]"
        full_payload = trigger_gem["payload"]
        gem_attach_names = trigger_gem.get("attach_names", [])
        if gem_attach_names:
            attach_names.extend(gem_attach_names)
        # 兼容旧状态缓存
        old_attach_name = trigger_gem.get("attach_name")
        if old_attach_name and old_attach_name not in attach_names:
            attach_names.append(old_attach_name)

        is_gem_init = True
        del st.session_state.trigger_gem_init
    elif trigger_additional_search:
        # 多次联网：根据来源（AI建议/手动）走不同路径
        search_source = st.session_state.get("search_source", "manual")

        # 清理临时状态标记
        for key in ["search_source", "ai_suggestion_ignored"]:
            if key in st.session_state:
                del st.session_state[key]

        if search_source == "ai_suggested":
            # ═══════════════════════════════════════════════════════════════
            # 路径A：AI 建议搜索 → 删除旧回答 → 合并新旧结果 → 重新生成
            # ═══════════════════════════════════════════════════════════════
            with st.status(
                "🔗 正在按 AI 建议进行补充搜索...", expanded=False
            ) as status:
                try:
                    # 1. 获取旧用户消息（含有第一次搜索的结果）
                    last_user_msg = st.session_state.messages[-2]

                    # 2. 用 AI 建议的关键词进行新搜索（根据搜索引擎选择）
                    _search_engine = st.session_state.api_configs.get("search_engine", "Tavily")
                    if _search_engine == "Tavily":
                        # Tavily 搜索
                        search_res = TavilyClient(
                            api_key=st.session_state.api_configs["tavily"]
                        ).search(query=trigger_additional_search)
                        new_search_context = "\n".join(
                            [
                                f"- {r['content']} ({r['url']})"
                                for r in search_res["results"]
                            ]
                        )
                        _results_count = len(search_res["results"])
                    else:
                        # 博查(Bocha) 搜索 — 使用 perform_bocha_search 并传入预定义关键词
                        _keywords, new_search_context, _results_count = perform_bocha_search(
                            client=None,  # 不需要 AI 提取关键词
                            prompt_text="",
                            bocha_api_key=st.session_state.api_configs["bocha"],
                            model_id="",
                            predefined_keywords=trigger_additional_search,
                        )

                    # 3. 提取原始用户问题（去掉旧搜索结果）
                    original_question = last_user_msg.get("raw_payload", "")
                    if not original_question:
                        original_question = last_user_msg.get("content", "")

                    # 4. 获取旧消息的缓存（如果有），准备合并新旧搜索结果
                    old_msg_id = last_user_msg.get("msg_id")
                    old_searches = []
                    if old_msg_id and "search_cache" in st.session_state:
                        old_searches = st.session_state.search_cache.get(
                            old_msg_id, {}
                        ).get("searches", [])

                    # 5. 生成新 msg_id，将合并后的搜索结果存入独立缓存
                    new_msg_id = f"msg_{time.time_ns()}"
                    combined_searches = list(old_searches) + [
                        {
                            "keywords": trigger_additional_search,
                            "results": new_search_context,
                        }
                    ]
                    if "search_cache" not in st.session_state:
                        st.session_state.search_cache = {}
                    st.session_state.search_cache[new_msg_id] = {
                        "question": original_question,
                        "searches": combined_searches,
                    }

                    # 6. 删除旧 AI 回答（准备重新生成）
                    st.session_state.messages = st.session_state.messages[:-1]

                    # 7. 构建修改后的用户消息
                    #    🌟 不修改 full_payload，保持干净！搜索结果从缓存取
                    modified_msg = last_user_msg.copy()
                    modified_msg["content"] = original_question
                    modified_msg["raw_payload"] = original_question
                    modified_msg["msg_id"] = new_msg_id  # 指向合并后的缓存

                    status.update(
                        label=f"✅ 已获取 {_results_count} 条搜索结果",
                        state="complete",
                    )
                    time.sleep(0.5)

                    # 8. 通过 trigger_rerun 让 AI 重新生成回答
                    st.session_state.trigger_rerun = modified_msg

                except Exception as search_e:
                    st.error(f"⚠️ 补充搜索失败: {str(search_e)}")
                    st.session_state.web_search_error = str(search_e)
                finally:
                    del st.session_state.trigger_additional_search
                    st.rerun()
            st.stop()

        else:
            # ═══════════════════════════════════════════════════════════════
            # 路径B：手动输入关键词 → 保留旧回答 → 新增一轮独立回答
            # ═══════════════════════════════════════════════════════════════
            # 构建一条"假装用户手动搜索"的消息，走正常流程让 AI 生成新回答
            # 注意：让主流程的 web_search 机制自动提取关键词并搜索
            manual_search_msg = {
                "role": "user",
                "content": f"🔍 补充搜索：{trigger_additional_search}",
            }

            del st.session_state.trigger_additional_search
            st.session_state.trigger_rerun = manual_search_msg
            st.rerun()
            st.stop()
    elif trigger_compare:
        compare_assistant_idx = trigger_compare["assistant_idx"]
        compare_model_choice = trigger_compare["model_choice"]
        compare_target_model = trigger_compare["target_model"]

        # 提取对应的用户消息（assistant 的前一条）
        compare_last_user_msg = st.session_state.messages[compare_assistant_idx - 1]
        prompt = compare_last_user_msg.get("content", "")
        full_payload = compare_last_user_msg.get(
            "full_payload", compare_last_user_msg.get("raw_payload", prompt)
        )
        if TIMESTAMP_PREFIX in prompt:
            prompt = prompt.rsplit(TIMESTAMP_PREFIX, 1)[0]
        if TIMESTAMP_PREFIX in full_payload:
            full_payload = full_payload.rsplit(TIMESTAMP_PREFIX, 1)[0]

        preserved_msg_id = compare_last_user_msg.get("msg_id")
        is_compare_branch = True

        # 覆盖当前模型选择为对比目标模型
        model_choice = compare_model_choice
        target_model = compare_target_model

        del st.session_state.compare_request
        st.toast(f"⚖️ 正在用 {model_choice}/{target_model} 对比生成...")
    elif trigger_msg:
        last_user_msg = trigger_msg
        prompt = last_user_msg["content"]

        # 🕒 在清除时间戳之前，先提取原始时间戳，用于重新生成时保持原时间
        if TIMESTAMP_PREFIX in last_user_msg.get("content", ""):
            _parts = last_user_msg["content"].rsplit(TIMESTAMP_PREFIX, 1)
            original_timestamp_str = _parts[1].rstrip(TIMESTAMP_SUFFIX)

        # full_payload 优先从 full_payload 字段取（含搜索结果），
        # 没有则回退到 raw_payload（仅原始问题，编辑用）
        full_payload = last_user_msg.get(
            "full_payload", last_user_msg.get("raw_payload", prompt)
        )
        if TIMESTAMP_PREFIX in full_payload:
            full_payload = full_payload.rsplit(TIMESTAMP_PREFIX, 1)[0]

        if TIMESTAMP_PREFIX in prompt:
            prompt = prompt.rsplit(TIMESTAMP_PREFIX, 1)[0]

        # 🌟 保留原 msg_id（路径A设置了它指向合并后的搜索缓存）
        preserved_msg_id = last_user_msg.get("msg_id")

        # 🌟 核心修复：正确处理重跑消息的附件（兼容新旧格式）
        attach_names = last_user_msg.get("attachments", [])
        if not attach_names and last_user_msg.get("attachment"):
            attach_names = [last_user_msg.get("attachment")]

        is_rerun = True
        is_gem_init = last_user_msg.get("is_gem_init", False)
        del st.session_state.trigger_rerun
    else:
        # 这是一个新消息
        if not prompt or not prompt.strip():
            prompt = "请分析附件内容" if uploaded_files else "继续"

        full_payload = prompt

        if uploaded_files:
            file_contents = []
            for uploaded_file in uploaded_files:
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                unique_name = f"{ts}_{uploaded_file.name}"
                attach_names.append(unique_name)

                save_path = os.path.join(ATTACH_DIR, unique_name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 使用统一的文件解析工具（已移除 >5MB 跳过限制，所有附件一律解析）
                file_text = process_uploaded_file(uploaded_file)
                file_contents.append(
                    f"【附件内容：{uploaded_file.name}】\n{file_text}"
                )

            if file_contents:
                full_payload = f"【用户指令】：{prompt}\n\n" + "\n\n---\n\n".join(
                    file_contents
                )

            st.session_state.file_uploader_key = str(time.time())

    start_time = time.time()

    # --- 修改：为所有用户新请求注入时间戳 ---
    final_prompt_for_display = prompt
    final_full_payload = full_payload
    if not is_gem_init:
        # 🕒 如果是重新生成且有原始时间戳，则复用原始时间戳，不更新为当前时间
        if is_rerun and original_timestamp_str:
            timestamp_str = original_timestamp_str
        else:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_prompt_for_display += (
            f"{TIMESTAMP_PREFIX}{timestamp_str}{TIMESTAMP_SUFFIX}"
        )
        final_full_payload += f"{TIMESTAMP_PREFIX}{timestamp_str}{TIMESTAMP_SUFFIX}"

    # 🌟 生成唯一消息ID，用作 search_cache 的 key（纳秒级精度，不会重复）
    #    如果是 rerun 且保留了原 msg_id（路径A设置的），优先使用它
    msg_id = preserved_msg_id if preserved_msg_id else f"msg_{time.time_ns()}"

    user_entry = {
        "role": "user",
        "content": final_prompt_for_display,
        # 🌟 raw_payload 改用 prompt（始终是干净的原始问题），避免被搜索结果污染
        "raw_payload": prompt,
        "full_payload": final_full_payload,  # full_payload 带时间戳，用于发送
        "msg_id": msg_id,  # 🌟 新增 msg_id，作为搜索缓存的唯一标识
    }
    if attach_names:
        user_entry["attachments"] = attach_names

    if is_gem_init:
        user_entry["is_gem_init"] = True

    with st.chat_message("user"):
        if is_gem_init:
            with st.expander("💎 查看 Gem 注入的底层知识库与指令", expanded=False):
                st.markdown(full_payload)
        elif view_mode:
            st.code(final_prompt_for_display, language="markdown")
        else:
            st.markdown(final_prompt_for_display)
        if "attachments" in user_entry and user_entry["attachments"]:
            st.markdown(f"🖇️ `{len(user_entry['attachments'])} 个附件物理备份成功`")
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        response_placeholder = st.empty()
        full_res = ""
        thinking_process = ""

        raw_usage_data = None

        # 🌟 史诗级漏洞 1 修复：将 user_entry 挂载到守护态，防止 UI 交互导致数据蒸发
        st.session_state.pending_msg = user_entry

        # 🌟 漏洞修复 6: 将 pending_msg 同步写入磁盘文件，防止手机浏览器刷新丢失 session
        #     手机浏览器刷新时可能丢失 session cookie，导致 session_state 被清空。
        #     写入磁盘文件后，即使 session 丢失，也能从文件恢复 pending_msg。
        _ws_safe_name = "".join([c for c in current_ws if c.isalnum()]).strip()
        _pending_file = os.path.join(LOG_DIR, f"pending_msg_{_ws_safe_name}.json")
        with open(_pending_file, "w", encoding="utf-8") as _f:
            json.dump(user_entry, _f, ensure_ascii=False, indent=2)

        try:
            request_log_payload = {}
            response_log_payload = {}
            retry_attempts = st.session_state.api_configs.get("api_retry_attempts", 3)

            if model_choice == "Gemini":
                client = genai.Client(api_key=st.session_state.api_configs["gemini"])

                config_params = {
                    "temperature": temp,
                    "top_p": top_p,
                    "max_output_tokens": max_tok,
                }

                active_cache = st.session_state.get("active_gemini_cache")
                if active_cache:
                    config_params["cached_content"] = active_cache
                else:
                    if sys_prompt.strip():
                        config_params["system_instruction"] = (
                            get_effective_system_prompt(sys_prompt)
                        )
                    if web_search:
                        google_search_tool = types.Tool(
                            google_search=types.GoogleSearch()
                        )
                        config_params["tools"] = [google_search_tool]

                config = types.GenerateContentConfig(**config_params)

                history = []
                last_role = None
                compressed_msgs = get_compressed_context_messages(
                    st.session_state.messages,
                    st.session_state.compression_state,
                )
                for cm in compressed_msgs:
                    role = "user" if cm["role"] == "user" else "model"
                    text_content = cm["content"]
                    if not text_content or not text_content.strip():
                        text_content = " "
                    if not history and role == "model":
                        history.append(
                            {"role": "user", "parts": [{"text": "[System Init]"}]}
                        )
                        last_role = "user"
                    if role == last_role and history:
                        history[-1]["parts"][0]["text"] += f"\n\n{text_content}"
                    else:
                        history.append(
                            {"role": role, "parts": [{"text": text_content}]}
                        )
                        last_role = role

                # --- 捕获请求日志 ---
                request_log_payload = {
                    "model": normalize_gemini_model(target_model),
                    "generation_config": config_params,
                    "history_before_send": history if not active_cache else [],
                    "new_content": final_full_payload,
                }

                chat = client.chats.create(
                    model=normalize_gemini_model(target_model),
                    config=config,
                    history=(
                        [types.Content(**h) for h in history]
                        if not active_cache
                        else []
                    ),
                )

                # 🌟 修复：使用 retry_api_stream 包裹整个流生命周期
                for chunk in retry_api_stream(
                    lambda: chat.send_message_stream(final_full_payload),
                    max_attempts=retry_attempts,
                    context=f"Gemini 请求 ({target_model})",
                ):
                    if chunk.text:
                        full_res += chunk.text
                        if view_mode:
                            response_placeholder.code(
                                full_res + "▌", language="markdown"
                            )
                        else:
                            response_placeholder.markdown(full_res + "▌")
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        raw_usage_data = chunk.usage_metadata

                # --- 捕获响应日志 ---
                response_log_payload = {
                    "final_text": full_res,
                    "usage_metadata": str(raw_usage_data) if raw_usage_data else "N/A",
                }

            else:
                # 🌟 统一 OpenAI 兼容路径 (通过供应商注册表路由)
                _provider_key = lookup_api_key(st.session_state.api_configs, model_choice)
                provider_info = get_provider_info(model_choice)
                if not provider_info or not provider_info.get("api_host"):
                    st.error(f"❌ 供应商 {model_choice} 缺少有效的 API 地址！")
                    st.stop()
                client = create_openai_client(model_choice, _provider_key, _opencode_session_id())

                final_prompt = final_full_payload
                search_context = None
                has_cached = (
                    preserved_msg_id
                    and preserved_msg_id in st.session_state.get("search_cache", {})
                )

                # 🔍 根据所选搜索引擎检查对应的 API Key
                _search_engine = st.session_state.api_configs.get("search_engine", "Tavily")
                _search_api_key_available = (
                    st.session_state.api_configs.get("tavily")
                    if _search_engine == "Tavily"
                    else st.session_state.api_configs.get("bocha")
                )
                if (
                    web_search
                    and _search_api_key_available
                    and not is_gem_init
                    and (not is_rerun or not has_cached)
                ):
                    with st.status(
                        "🔗 正在联网获取最新信息...", expanded=False
                    ) as status:
                        try:
                            if "web_search_error" in st.session_state:
                                del st.session_state.web_search_error

                            extract_cfg = st.session_state.api_configs.get(
                                "web_search_extract_config", {}
                            )
                            extractor_platform = extract_cfg.get("platform")
                            extractor_model = extract_cfg.get("model")

                            if not extractor_model:
                                old_cfg = st.session_state.api_configs.get(
                                    "web_search_model", {}
                                )
                                extractor_model = old_cfg.get(model_choice, target_model)
                                extractor_platform = model_choice

                            # 🌟 通用提取模型 Client 创建
                            if is_gemini_type(extractor_platform):
                                extractor_client = genai.Client(
                                    api_key=lookup_api_key(st.session_state.api_configs, extractor_platform)
                                )
                            elif is_openai_type(extractor_platform):
                                _ext_key = lookup_api_key(st.session_state.api_configs, extractor_platform)
                                extractor_client = create_openai_client(extractor_platform, _ext_key, _opencode_session_id())
                            else:
                                extractor_client = client
                                extractor_model = target_model

                            if _search_engine == "Tavily":
                                keywords, search_context, results_count = (
                                    perform_web_search(
                                        extractor_client,
                                        prompt,
                                        st.session_state.api_configs["tavily"],
                                        extractor_model,
                                        retry_attempts=retry_attempts,
                                    )
                                )
                            else:
                                keywords, search_context, results_count = (
                                    perform_bocha_search(
                                        extractor_client,
                                        prompt,
                                        st.session_state.api_configs["bocha"],
                                        extractor_model,
                                        retry_attempts=retry_attempts,
                                    )
                                )
                            status.update(label=f"🔎 提取关键词: {keywords}", state="running")

                            final_prompt = f"【最新信息参考】(搜索关键词: {keywords}):\n{search_context}\n\n{final_full_payload}"
                            msg_id = user_entry.get("msg_id")
                            if msg_id:
                                if "search_cache" not in st.session_state:
                                    st.session_state.search_cache = {}
                                if msg_id not in st.session_state.search_cache:
                                    st.session_state.search_cache[msg_id] = {
                                        "question": prompt,
                                        "searches": [],
                                    }
                                st.session_state.search_cache[msg_id]["searches"].append(
                                    {"keywords": keywords, "results": search_context}
                                )
                            status.update(
                                label=f"✅ 已获取 {results_count} 条最新信息",
                                state="complete",
                            )
                        except Exception as search_e:
                            status.update(
                                label=f"⚠️ 联网失败，使用本地知识作答", state="error"
                            )
                            st.session_state.web_search_error = str(search_e)

                # 🌟 Rerun 时将缓存搜索结果注入 final_prompt
                if is_rerun and preserved_msg_id:
                    cache_entry = st.session_state.get("search_cache", {}).get(
                        preserved_msg_id
                    )
                    if cache_entry:
                        for s in cache_entry.get("searches", []):
                            final_prompt = (
                                f"【搜索结果】(关键词: {s['keywords']}):\n"
                                f"{s['results']}\n\n{final_prompt}"
                            )

                messages_flow = []
                if sys_prompt.strip():
                    messages_flow.append(
                        {"role": "system", "content": get_effective_system_prompt(sys_prompt)}
                    )
                compressed_ids = get_compressed_msg_ids(
                    st.session_state.compression_state
                )
                emitted_ranges = set()
                for m in st.session_state.messages:
                    msg_id = m.get("msg_id", "")
                    if msg_id in compressed_ids:
                        for r in st.session_state.compression_state.get("ranges", []):
                            if msg_id in r["msg_ids"]:
                                if r["id"] not in emitted_ranges:
                                    messages_flow.append({
                                        "role": r.get("summary_role", "user"),
                                        "content": r["summary"],
                                    })
                                    emitted_ranges.add(r["id"])
                                break
                        continue
                    text_content = m.get("full_payload", m["content"])
                    if not text_content or not text_content.strip():
                        text_content = " "
                    if m["role"] == "user" and "msg_id" in m:
                        cache_entry = st.session_state.get("search_cache", {}).get(m["msg_id"])
                        if cache_entry:
                            for s in cache_entry.get("searches", []):
                                text_content = (
                                    f"【搜索结果】(关键词: {s['keywords']}):\n"
                                    f"{s['results']}\n\n{text_content}"
                                )
                    messages_flow.append({"role": m["role"], "content": text_content})
                messages_flow.append({"role": "user", "content": final_prompt})

                request_log_payload = {
                    "model": target_model,
                    "messages": messages_flow,
                    "temperature": temp,
                    "top_p": top_p,
                    "max_tokens": max_tok,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                # DeepSeek 系模型: 思考强度配置（按模型 ID 判断，不限通道）
                if "deepseek" in target_model.lower():
                    request_log_payload["extra_body"] = {"thinking": {"type": "enabled"}}
                    if target_model in ("deepseek-reasoner", "deepseek-chat"):
                        request_log_payload["reasoning_effort"] = reasoning_effort

                _display_name = get_channel_display_name(model_choice)
                stream = retry_api_stream(
                    lambda: client.chat.completions.create(**request_log_payload),
                    max_attempts=retry_attempts,
                    context=f"{_display_name} 请求 ({target_model})",
                )

                finish_reason = None
                for chunk in stream:
                    if hasattr(chunk, "usage") and chunk.usage:
                        raw_usage_data = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        thinking_process += delta.reasoning_content
                        with thinking_placeholder.container():
                            with st.expander("🧠 剥离出的思考过程", expanded=False):
                                st.markdown(thinking_process + "▌")
                    if delta.content:
                        full_res += delta.content
                        if view_mode:
                            response_placeholder.code(full_res + "▌", language="markdown")
                        else:
                            response_placeholder.markdown(full_res + "▌")

                response_log_payload = {
                    "final_content": {"role": "assistant", "content": full_res},
                    "thinking_process": thinking_process,
                    "finish_reason": finish_reason,
                    "usage": str(raw_usage_data) if raw_usage_data else "N/A",
                }

            if thinking_process:
                with thinking_placeholder.container():
                    with st.expander("🧠 剥离出的思考过程", expanded=False):
                        st.markdown(thinking_process)

            if view_mode:
                response_placeholder.code(full_res, language="markdown")
            else:
                response_placeholder.markdown(full_res)

            latency = time.time() - start_time
            latency_ms = int(latency * 1000)
            current_ts = datetime.now().isoformat()

            display_model = f"Model: {model_choice} API ({target_model})"

            token_info = "Tokens: 计算失败"
            if raw_usage_data:
                if model_choice == "Gemini":
                    p_tokens = getattr(raw_usage_data, "prompt_token_count", 0) or 0
                    c_tokens = getattr(raw_usage_data, "candidates_token_count", 0) or 0
                    t_tokens = getattr(raw_usage_data, "total_token_count", 0) or 0
                    cache_tokens = (
                        getattr(raw_usage_data, "cached_content_token_count", 0) or 0
                    )
                    if st.session_state.get("active_gemini_cache"):
                        st.session_state.cache_last_hit_status = (
                            "命中" if cache_tokens > 0 else "未命中"
                        )

                    # 为Gemini模型设置token_info
                    if cache_tokens > 0:
                        token_info = f"Tokens: {t_tokens} (Prompt: {p_tokens} [⚡Cache: {cache_tokens}] | Response: {c_tokens})"
                    else:
                        token_info = f"Tokens: {t_tokens} (Prompt: {p_tokens} | Response: {c_tokens})"
                else:
                    p_tokens = getattr(raw_usage_data, "prompt_tokens", 0) or 0
                    c_tokens = getattr(raw_usage_data, "completion_tokens", 0) or 0
                    t_tokens = getattr(raw_usage_data, "total_tokens", 0) or 0

                    cache_hit_tokens = 0
                    cache_miss_tokens = 0

                    if model_choice == "DeepSeek":
                        cache_hit_tokens = (
                            getattr(raw_usage_data, "prompt_cache_hit_tokens", 0) or 0
                        )
                        cache_miss_tokens = (
                            getattr(raw_usage_data, "prompt_cache_miss_tokens", 0) or 0
                        )

                    if cache_hit_tokens == 0 and cache_miss_tokens == 0:
                        if (
                            hasattr(raw_usage_data, "prompt_tokens_details")
                            and raw_usage_data.prompt_tokens_details
                        ):
                            cache_hit_tokens = (
                                getattr(
                                    raw_usage_data.prompt_tokens_details,
                                    "cached_tokens",
                                    0,
                                )
                                or 0
                            )

                    if cache_hit_tokens > 0 or cache_miss_tokens > 0:
                        token_info = f"Tokens: {t_tokens} (Prompt: {p_tokens} [⚡Hit: {cache_hit_tokens} | ❌Miss: {cache_miss_tokens}] | Response: {c_tokens})"
                    else:
                        token_info = f"Tokens: {t_tokens} (Prompt: {p_tokens} | Response: {c_tokens})"

            display_time = current_ts[:19].replace("T", " ")
            st.caption(
                f"📅 {display_time} | 📦 {display_model} | ⏱️ Latency: {latency_ms}ms | 📊 {token_info}"
            )

            assistant_entry = {
                "role": "assistant",
                "content": full_res,
                "thinking": thinking_process if thinking_process else None,
                "timestamp": current_ts,
                "model": target_model,
                "model_display": display_model,
                "latency": latency,
                "token_usage": token_info,
            }

            if is_compare_branch:
                # 🌟 对比模型分支：将结果添加到已有的 assistant 消息中
                target_msg = st.session_state.messages[compare_assistant_idx]
                branch_key = (
                    f"{model_choice}_{target_model.replace('/', '_').replace(':', '_')}"
                )

                # 首次添加分支时，将原始响应转为 __original__ 分支
                if "branches" not in target_msg:
                    target_msg["branches"] = {}
                    target_msg["active_branch"] = "__original__"
                    target_msg["branches"]["__original__"] = {
                        "content": target_msg.get("content", ""),
                        "thinking": target_msg.get("thinking"),
                        "timestamp": target_msg.get("timestamp"),
                        "model": target_msg.get("model"),
                        "model_display": target_msg.get("model_display"),
                        "latency": target_msg.get("latency"),
                        "token_usage": target_msg.get("token_usage"),
                    }
                    # 清理顶层 thinking 字段（已迁入分支）
                    target_msg["thinking"] = None

                # 检查是否已存在相同模型的分支
                if branch_key in target_msg["branches"]:
                    st.warning(f"⚠️ 该模型 ({model_choice}/{target_model}) 已有对比结果，已覆盖更新。")

                target_msg["branches"][branch_key] = {
                    "content": full_res,
                    "thinking": thinking_process if thinking_process else None,
                    "timestamp": current_ts,
                    "model": target_model,
                    "model_display": display_model,
                    "latency": latency,
                    "token_usage": token_info,
                }
                target_msg["active_branch"] = branch_key
                target_msg["content"] = full_res  # 同步更新顶层 content
                target_msg["model_display"] = display_model
            else:
                # 正常流程：追加用户 + 助理消息
                st.session_state.messages.append(user_entry)
                st.session_state.messages.append(assistant_entry)

            st.session_state.last_turn_logs = {
                "input_payload": request_log_payload,
                "output_payload": response_log_payload,
            }

            with open(
                os.path.join(LOG_DIR, "audit_trail.jsonl"), "a", encoding="utf-8"
            ) as f:
                f.write(
                    deterministic_json_dumps(
                        {
                            "ts": str(datetime.now()),
                            "action": "generation",
                            "workspace": current_ws,
                            "case": st.session_state.chat_title,
                            "model": target_model,
                            "params": {"temperature": temp, "top_p": top_p},
                            "pair": [user_entry, assistant_entry],
                        },
                        indent=None,
                    )
                    + "\n"
                )

            save_chat_history(
                current_save_file,
                st.session_state.messages,
                st.session_state.get("search_cache", {}),
            )

            st.session_state.file_uploader_key = str(time.time())

            # 🌟 新增：自动命名逻辑
            auto_name_prefs = st.session_state.api_configs.get("auto_name_prefs", {})
            if auto_name_prefs.get("enabled", False):
                # 检查是否已经触发过
                auto_name_triggered = st.session_state.api_configs.get(
                    "auto_name_triggered", False
                )
                # 如果当前案例名称还是默认的"未命名案例"，也触发命名
                current_title = st.session_state.api_configs.get(
                    "workspace_titles", {}
                ).get(current_ws, "未命名案例")
                should_trigger = (
                    not auto_name_triggered or current_title == "未命名案例"
                )

                if should_trigger:
                    # 只有第一条用户消息后才触发
                    user_msg_count = sum(
                        1 for m in st.session_state.messages if m["role"] == "user"
                    )
                    if user_msg_count == 1:
                        try:
                            with st.spinner("🤖 正在生成案例名称..."):
                                # 提取用户第一条消息
                                first_user_msg = next(
                                    m
                                    for m in st.session_state.messages
                                    if m["role"] == "user"
                                )
                                prompt_for_name = f"""
    请根据以下对话内容，生成一个 5-10 个字的简洁案例名称。
    要求：
    1. 准确概括对话主题
    2. 简洁明了
    3. 不要超过 10 个字
    4. 不要包含标点符号
    5. 仅回复案例名称，不要其他内容

    对话内容：
    {first_user_msg.get("content", "继续对话")}
    """

                                # 根据配置选择模型通道
                                channel = auto_name_prefs.get("channel", "SiliconFlow")
                                model = auto_name_prefs.get(
                                    "model", "Qwen/Qwen2.5-72B-Instruct"
                                )

                                # 🌟 通用化：通过供应商注册表路由
                                _name_key = lookup_api_key(st.session_state.api_configs, channel)
                                if is_gemini_type(channel):
                                    _gc = genai.Client(api_key=_name_key)
                                    _resp = _gc.models.generate_content(
                                        model=normalize_gemini_model(model),
                                        contents=prompt_for_name,
                                        generation_config=types.GenerateContentConfig(
                                            temperature=0.3, max_output_tokens=30
                                        ),
                                    )
                                    generated_name = _resp.text.strip()
                                elif is_openai_type(channel):
                                    _name_client = create_openai_client(channel, _name_key, _opencode_session_id())
                                    _resp = _name_client.chat.completions.create(
                                        model=model,
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": "你是一个专业的案例命名助手，请根据对话内容生成简洁的案例名称。",
                                            },
                                            {"role": "user", "content": prompt_for_name},
                                        ],
                                        temperature=0.3,
                                        max_tokens=30,
                                    )
                                    generated_name = _resp.choices[0].message.content.strip()
                                else:
                                    generated_name = ""

                                # 清理生成的名称（移除引号、标点等）
                                generated_name = (
                                    generated_name.replace('"', "")
                                    .replace("'", "")
                                    .replace("。", "")
                                    .replace("，", "")
                                    .replace("、", "")
                                    .replace("《", "")
                                    .replace("》", "")
                                    .strip()
                                )

                                # 更新案例名称
                                if generated_name and len(generated_name) > 0:
                                    st.session_state.chat_title = generated_name
                                    if (
                                        "workspace_titles"
                                        not in st.session_state.api_configs
                                    ):
                                        st.session_state.api_configs[
                                            "workspace_titles"
                                        ] = {}
                                    st.session_state.api_configs["workspace_titles"][
                                        current_ws
                                    ] = generated_name
                                    st.session_state.api_configs[
                                        "auto_name_triggered"
                                    ] = True
                                    save_config(st.session_state.api_configs)
                                    st.toast(f"✅ 已自动命名：{generated_name}")
                        except Exception as e:
                            print(f"自动命名失败: {e}")
                            # 失败时不阻止正常流程

            if "pending_msg" in st.session_state:
                del st.session_state.pending_msg

            # 🌟 漏洞修复 6: API 成功完成后，同步删除磁盘上的 pending 文件
            _ws_safe_name = "".join([c for c in current_ws if c.isalnum()]).strip()
            _pending_file = os.path.join(LOG_DIR, f"pending_msg_{_ws_safe_name}.json")
            if os.path.exists(_pending_file):
                os.remove(_pending_file)

            if "estimated_tokens_display" in st.session_state:
                del st.session_state.estimated_tokens_display

            st.rerun()

        except Exception as e:
            error_msg = f"⚠️ 通道执行异常: {str(e)}"
            st.session_state.persistent_error = error_msg
            # 🌟 必须保留 pending_msg，不能让页面以为所有请求已结束
            st.session_state.edit_state = {"type": "draft", "msg": user_entry}
            st.rerun()
