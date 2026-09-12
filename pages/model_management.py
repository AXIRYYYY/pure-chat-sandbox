import streamlit as st
import json
import os
import ai_models_fetcher
from provider_registry import (
    BUILTIN_PROVIDERS,
    load_provider_registry,
    save_provider_registry,
    ensure_builtin_defaults,
    get_provider_info,
    get_api_key,
    is_builtin,
    is_gemini_type,
    is_openai_type,
    add_custom_provider,
    update_custom_provider,
    delete_custom_provider,
    get_all_custom_providers,
    initialize as init_provider_registry,
)

st.set_page_config(page_title="模型库管理", layout="wide")

# 初始化 session state（存储模型变更信息：新增和减少的模型）
if "model_changes_notification" not in st.session_state:
    st.session_state.model_changes_notification = None

# 自定义 CSS 样式，处理长模型名称显示
st.markdown(
    """
<style>
    /* 优化 multiselect 显示 */
    .stMultiSelect {
        min-height: 100px;
    }
    
    /* 处理长文本溢出 */
    .stMultiSelect [data-testid="multiSelectOptionListContainer"] {
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* 优化选中项显示 */
    .stMultiSelect .stMultiSelectOptions div {
        word-break: break-word;
        white-space: normal;
        padding: 8px 4px;
    }
    
    /* 扩展输入框宽度 */
    [data-testid="stMultiSelect"] {
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.header("🤖 模型库管理")

# 显示模型变更通知（新增和减少的模型，重新加载后依然保存）
if st.session_state.model_changes_notification:
    notification = st.session_state.model_changes_notification
    col1, col2 = st.columns([20, 1])
    with col1:
        # 显示新增模型（用绿色成功框）
        if notification["added"]:
            st.success("🎉 发现新增模型！")
            for platform, models in notification["added"].items():
                with st.expander(f"✨ {platform} 新增 {len(models)} 个模型"):
                    for model in models:
                        st.write(f"• {model}")

        # 显示减少模型（用黄色警告框）
        if notification["removed"]:
            st.warning("⚠️ 发现模型减少！")
            for platform, models in notification["removed"].items():
                with st.expander(f"📉 {platform} 减少 {len(models)} 个模型"):
                    for model in models:
                        st.write(f"• {model}")
    with col2:
        if st.button("✕", key="close_notification"):
            st.session_state.model_changes_notification = None
            st.rerun()
    st.divider()

st.subheader("🔑 联网搜索 Key 设置")
try:
    with open("api_config.json", "r", encoding="utf-8") as f:
        api_config_keys = json.load(f)
except:
    api_config_keys = {}

col_key1, col_key2 = st.columns(2)
with col_key1:
    tavily_key = st.text_input(
        "Tavily Key (联网搜索用)",
        value=api_config_keys.get("tavily", ""),
        type="password",
    )
with col_key2:
    bocha_key = st.text_input(
        "博查 Key (联网搜索用)",
        value=api_config_keys.get("bocha", ""),
        type="password",
        help="前往 https://open.bocha.cn 获取博查搜索 API Key",
    )

if st.button("💾 保存所有 API Key 与供应商配置", use_container_width=True, type="primary"):
    api_config_keys["tavily"] = tavily_key
    api_config_keys["bocha"] = bocha_key

    # 🌟 收集热门供应商区的 Key (读取 widget state)
    for pid, info in BUILTIN_PROVIDERS.items():
        wk = f"prov_keyin_{pid}"
        if wk in st.session_state:
            api_config_keys[info.get("api_key_field", pid)] = st.session_state[wk]

    # 🌟 收集自定义接口的 Key
    for cid in get_all_custom_providers():
        wk = f"cst_k_{cid}"
        if wk in st.session_state:
            api_config_keys[cid] = st.session_state[wk]

    # 🌟 持久化供应商启用状态 (从 widget state 读取 toggle)
    builtin_reg2, custom_reg2 = load_provider_registry()
    for pid in BUILTIN_PROVIDERS:
        tk = f"prov_tg_{pid}"
        if tk in st.session_state:
            builtin_reg2[pid] = {"enabled": st.session_state[tk]}
    save_provider_registry(builtin_reg2, custom_reg2)

    with open("api_config.json", "w", encoding="utf-8") as f:
        json.dump(api_config_keys, f, ensure_ascii=False, indent=4)
    if "api_configs" in st.session_state:
        st.session_state.api_configs.update(api_config_keys)
    st.toast("✅ 所有 API Key 与供应商配置已保存")
    st.rerun()

st.divider()

# ———————————————————————— 🌟 热门供应商 ————————————————————————

st.subheader("🏪 热门供应商")

init_provider_registry()
builtin_reg, custom_reg = load_provider_registry()

try:
    with open("api_config.json", "r", encoding="utf-8") as f:
        api_config_keys = json.load(f)
except:
    api_config_keys = {}

providers_list = list(BUILTIN_PROVIDERS.items())
for col_i in range(0, len(providers_list), 2):
    c1, c2 = st.columns(2)
    for j, (pid, info) in enumerate(providers_list[col_i : col_i + 2]):
        with (c1 if j == 0 else c2):
            enabled = builtin_reg.get(pid, {}).get("enabled", False)
            api_key_field = info.get("api_key_field", pid)
            current_key = api_config_keys.get(api_key_field, "")
            has_key = bool(current_key)

            type_badge = "🟢 Gemini" if info["type"] == "gemini" else "🔵 OpenAI"
            # 开关关闭的供应商折叠收起，启用的展开（expanded 按注册表状态重算）
            with st.expander(f"**{info['name']}**  `{type_badge}`", expanded=enabled):
                st.caption(info.get("description", ""))
                if info.get("api_key_url"):
                    st.caption(f"[🔗 获取 API Key]({info['api_key_url']})")

                new_enabled = st.toggle(
                    "✅ 启用此供应商",
                    value=enabled,
                    key=f"prov_tg_{pid}",
                )

                if new_enabled:
                    st.text_input(
                        "API Key",
                        value=current_key,
                        type="password",
                        key=f"prov_keyin_{pid}",
                        placeholder="填完后点击顶部「💾 保存所有 API Key 与供应商配置」",
                    )
                    if has_key:
                        st.caption("🔑 已配置")
                    else:
                        st.caption("⚠️ 未配置")

st.divider()

# ———————————————————————— 🌟 自定义 OpenAI 接口 ————————————————————————

st.subheader("🔧 自定义 OpenAI 兼容接口")

st.markdown("""
添加任意遵循 OpenAI Chat Completions 格式的 API，例如自建代理、内部服务、
第三方兼容服务等。可无限添加。
""")

cst = get_all_custom_providers()
if cst:
    for cid, cinfo in list(cst.items()):
        with st.expander(f"⚙️ {cinfo.get('name', cid)}", expanded=True):
            ccol1, ccol2 = st.columns([3, 1])
            with ccol1:
                new_name = st.text_input("名称", value=cinfo.get("name", ""), key=f"cst_nm_{cid}")
                new_host = st.text_input("API Host", value=cinfo.get("api_host", ""), key=f"cst_h_{cid}", placeholder="https://your-api.example.com/v1")
                cst_key_field = cid
                cst_key = api_config_keys.get(cst_key_field, "")
                st.text_input("API Key", value=cst_key, type="password", key=f"cst_k_{cid}", placeholder="填完后点击「保存修改」或顶部统一保存按钮")
            with ccol2:
                st.write("")
                st.write("")
                if st.button("🗑️ 删除", key=f"cst_del_{cid}", use_container_width=True):
                    delete_custom_provider(cid)
                    st.toast(f"已删除 {cinfo.get('name', cid)}")
                    st.rerun()
                if st.button("💾 保存修改", key=f"cst_sv_{cid}", use_container_width=True):
                    update_custom_provider(cid, new_name, new_host)
                    # 🌟 从 widget state 读取当前 Key 值
                    wk = f"cst_k_{cid}"
                    _cst_key_val = st.session_state.get(wk, "")
                    if _cst_key_val:
                        try:
                            with open("api_config.json", "r", encoding="utf-8") as f:
                                _save_cfg = json.load(f)
                        except:
                            _save_cfg = {}
                        _save_cfg[cst_key_field] = _cst_key_val
                        with open("api_config.json", "w", encoding="utf-8") as f:
                            json.dump(_save_cfg, f, ensure_ascii=False, indent=4)
                        if "api_configs" in st.session_state:
                            st.session_state.api_configs[cst_key_field] = _cst_key_val
                    st.toast(f"✅ {new_name} 已更新")
                    st.rerun()

st.markdown("")
with st.expander("➕ 添加新的自定义接口", expanded=not cst):
    new_cst_name = st.text_input("供应商名称", placeholder="我的自定义 LLM", key="new_cst_name")
    new_cst_host = st.text_input("API Host", placeholder="https://api.example.com/v1", key="new_cst_host")
    new_cst_key = st.text_input("API Key", type="password", key="new_cst_key")
    if st.button("✅ 确认添加", use_container_width=True):
        if new_cst_name and new_cst_host:
            pid = add_custom_provider(new_cst_name, new_cst_host)
            if new_cst_key:
                api_config_keys[pid] = new_cst_key
                with open("api_config.json", "w", encoding="utf-8") as f:
                    json.dump(api_config_keys, f, ensure_ascii=False, indent=4)
            if "api_configs" in st.session_state:
                st.session_state.api_configs[pid] = new_cst_key
            st.toast(f"✅ 已添加供应商：{new_cst_name}")
            st.rerun()
        else:
            st.warning("请至少填写名称和 API Host")

st.divider()

if st.button("🚀 获取最新模型列表", use_container_width=True):
    # 保存旧的模型列表
    old_models = {}
    if os.path.exists("available_models.json"):
        with open("available_models.json", "r", encoding="utf-8") as f:
            old_all_models = json.load(f)
            for m in old_all_models:
                if m["平台"] not in old_models:
                    old_models[m["平台"]] = []
                old_models[m["平台"]].append(m["模型ID"])

    with st.spinner("正在获取模型数据..."):
        _, report = ai_models_fetcher.main_with_report(interactive=False)

        # 🌟 在 UI 中显示各供应商获取结果
        st.write("")
        st.markdown("**📊 供应商获取结果：**")
        for r in report:
            icon = r["状态"].split(" ")[0] if r["状态"] else "ℹ️"
            detail = f" — {r['详情']}" if r["详情"] else ""
            st.caption(f"{icon} **{r['供应商']}**：{r['状态']}（{r['模型数']} 个模型）{detail}")

        # 加载新的模型列表
        with open("available_models.json", "r", encoding="utf-8") as f:
            new_all_models = json.load(f)
            new_models = {}
            for m in new_all_models:
                if m["平台"] not in new_models:
                    new_models[m["平台"]] = []
                new_models[m["平台"]].append(m["模型ID"])

        # 检测新增和减少的模型
        changes = {"added": {}, "removed": {}}
        # 合并新旧平台，防止漏掉"整个平台消失"的情况
        all_platforms = set(old_models.keys()) | set(new_models.keys())
        for platform in all_platforms:
            old_set = set(old_models.get(platform, []))
            new_set = set(new_models.get(platform, []))
            added = sorted(list(new_set - old_set))  # 新有旧无 = 新增
            removed = sorted(list(old_set - new_set))  # 旧有新无 = 减少
            if added:
                changes["added"][platform] = added
            if removed:
                changes["removed"][platform] = removed

        st.toast("✅ 模型数据更新完毕")

        # 保存模型变更信息到 session state
        if changes["added"] or changes["removed"]:
            st.session_state.model_changes_notification = changes

        st.rerun()

if os.path.exists("available_models.json"):
    with open("available_models.json", "r", encoding="utf-8") as f:
        all_models = json.load(f)

    # 按平台分组，并排序（动态获取所有平台）
    grouped = {}
    for m in all_models:
        platform = m["平台"]
        if platform not in grouped:
            grouped[platform] = []
        grouped[platform].append(m["模型ID"])

    # 对每个平台的模型进行排序
    for platform in grouped:
        grouped[platform].sort()

    # 加载已启用
    if os.path.exists("enabled_models.json"):
        with open("enabled_models.json", "r", encoding="utf-8") as f:
            enabled = json.load(f)
    else:
        enabled = grouped

    new_enabled = {}
    for platform, models in grouped.items():
        if not models:
            continue

        st.write(f"**{platform} 模型** ({len(models)} 个)")

        # 使用列布局改进显示
        col1, col2 = st.columns([1, 4])

        with col1:
            st.write("**操作**")
            select_all = st.checkbox("全选", value=False, key=f"select_all_{platform}")
            deselect_all = st.checkbox(
                "清空", value=False, key=f"deselect_all_{platform}"
            )

        with col2:
            if select_all:
                selected = models
            elif deselect_all:
                selected = []
            else:
                selected = st.multiselect(
                    f"选择要启用的 {platform} 模型",
                    models,
                    default=[m for m in enabled.get(platform, []) if m in models],
                    key=f"ms_{platform}",
                    max_selections=None,
                )

        new_enabled[platform] = selected

        # 显示已选模型详情
        if selected:
            st.write(f"✅ 已选 {len(selected)} 个模型：")
            # 分列显示已选模型，优化长名称显示
            sorted_selected = sorted(selected)
            cols = st.columns(min(2, len(sorted_selected)))
            for idx, model in enumerate(sorted_selected):
                with cols[idx % len(cols)]:
                    st.caption(f"• {model}")

    if st.button("💾 保存已启用模型", use_container_width=True):
        # 排序后保存
        sorted_enabled = {k: sorted(v) for k, v in new_enabled.items()}
        with open("enabled_models.json", "w", encoding="utf-8") as f:
            json.dump(sorted_enabled, f, ensure_ascii=False, indent=4)
        st.toast("✅ 模型列表已更新")
        st.rerun()

    # --- 🌟 重构：联网搜索关键词提取模型配置 (支持跨平台选择) ---
    st.divider()
    st.subheader("🌐 联网搜索 - 关键词提取模型")

    st.markdown("""
    联网搜索时，会先用一个 AI 模型从你的输入中**提取 1-3 个搜索关键词**，
    再传给当前选择的搜索引擎。你可以自由选择任一平台的廉价模型来执行此任务。
    """)

    # 加载当前的 api_config.json
    try:
        with open("api_config.json", "r", encoding="utf-8") as f:
            api_config = json.load(f)
    except:
        api_config = {}

    # 获取当前配置
    # 旧格式兼容：如果以前存的是字典，尝试转换，否则设为空
    current_extract_cfg = api_config.get("web_search_extract_config", {})
    if not isinstance(current_extract_cfg, dict):
        current_extract_cfg = {}

    current_platform = current_extract_cfg.get("platform", "SiliconFlow")
    current_model_id = current_extract_cfg.get("model", "")

    # 1. 选择平台 (动态：显示所有有模型可用的平台)
    available_platforms = sorted(grouped.keys()) if grouped else ["SiliconFlow", "DeepSeek", "Gemini"]
    if not available_platforms:
        available_platforms = ["SiliconFlow", "DeepSeek", "Gemini"]
    plat_idx = (
        available_platforms.index(current_platform)
        if current_platform in available_platforms
        else 0
    )
    selected_platform = st.selectbox(
        "第一步：选择 API 平台",
        available_platforms,
        index=plat_idx,
        key="web_search_platform_select",
        help="你可以使用不同于主聊天的平台来提取关键词，以节省成本。",
    )

    # 2. 选择该平台下的模型
    platform_models = grouped.get(selected_platform, [])

    # 确定模型下拉框的选项
    model_options = platform_models.copy()
    if (
        current_model_id
        and current_model_id not in model_options
        and selected_platform == current_platform
    ):
        model_options.insert(0, current_model_id)
    model_options.append("自定义...")

    # 确定默认索引
    if selected_platform == current_platform and current_model_id in model_options:
        model_def_idx = model_options.index(current_model_id)
    else:
        model_def_idx = 0

    selected_model = st.selectbox(
        f"第二步：选择 {selected_platform} 的具体模型",
        model_options,
        index=model_def_idx,
        key="web_search_model_select",
    )

    final_model_id = selected_model
    if selected_model == "自定义...":
        final_model_id = st.text_input(
            f"手动输入 {selected_platform} 的模型 ID",
            value=(
                current_model_id
                if selected_platform == current_platform
                and current_model_id not in platform_models
                else ""
            ),
            key="web_search_custom_model_input",
        )

    # 保存配置按钮
    if st.button("💾 保存关键词提取模型配置", use_container_width=True):
        if not final_model_id:
            st.error("❌ 模型 ID 不能为空")
        else:
            try:
                with open("api_config.json", "r", encoding="utf-8") as f:
                    api_config_to_save = json.load(f)
            except:
                api_config_to_save = {}

            api_config_to_save["web_search_extract_config"] = {
                "platform": selected_platform,
                "model": final_model_id,
            }
            # 同时保留旧字段以便向下兼容（可选，但为了稳妥可以删掉旧的或保持同步）
            # 这里我们采用新字段，后面在 app.py 中优先读取新字段

            with open("api_config.json", "w", encoding="utf-8") as f:
                json.dump(api_config_to_save, f, ensure_ascii=False, indent=4)

            # 同步更新 session_state
            if "api_configs" in st.session_state:
                st.session_state.api_configs["web_search_extract_config"] = (
                    api_config_to_save["web_search_extract_config"]
                )

            st.toast("✅ 关键词提取模型配置已保存")
            st.rerun()

    st.divider()
    st.subheader("🔁 API 自动重连设置")
    st.markdown(
        """
        当 API 因高负载、临时网络抖动或短期错误失败时，系统会自动重试请求。
        重试采用指数退避策略，等待时间将按 1s、2s、4s、8s 等方式递增。
        """,
    )

    retry_options = [
        ("不重试", 0),
        ("1 次", 1),
        ("2 次", 2),
        ("3 次", 3),
        ("5 次", 5),
        ("10 次", 10),
        ("无限制", -1),
    ]
    retry_labels = [label for label, _ in retry_options]
    current_retry = api_config.get("api_retry_attempts", 3)
    current_label = next(
        (label for label, value in retry_options if value == current_retry), "3 次"
    )
    selected_label = st.selectbox(
        "发生 API 错误时自动重试次数",
        retry_labels,
        index=retry_labels.index(current_label),
        key="api_retry_attempts_select",
    )
    selected_retry = dict(retry_options)[selected_label]

    if st.button("💾 保存自动重连设置", use_container_width=True):
        try:
            with open("api_config.json", "r", encoding="utf-8") as f:
                api_config_to_save = json.load(f)
        except:
            api_config_to_save = {}

        api_config_to_save["api_retry_attempts"] = selected_retry
        with open("api_config.json", "w", encoding="utf-8") as f:
            json.dump(api_config_to_save, f, ensure_ascii=False, indent=4)

        # 🌟 修复：同步更新 session_state
        if "api_configs" in st.session_state:
            st.session_state.api_configs["api_retry_attempts"] = selected_retry

        st.toast("✅ 自动重连设置已保存")
        st.rerun()


else:
    st.warning("⚠️ 请先运行 `ai_models_fetcher.py` 获取最新模型列表")
