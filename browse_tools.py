# browse_tools.py — AI 自主上网工具集
"""
为「AI 自主上网」模式提供两个技能的执行代码：

  技能 A「去搜一下」：Tavily → 博查 → 必应网页版 三级降级链（不依赖任何 Key 也能兜底）
  技能 B「打开网页」：requests 抓取网页正文文字（不引入新依赖）

以及 AI 上网请求标记的解析器。

工作方式（文本协议）：
  - 系统提示词教 AI 输出固定格式的上网请求： 【上网】搜索:关键词  /  【上网】打开:URL  /  【上网】结束
  - app.py 的预循环解析该标记 → 执行技能 → 把结果作为新消息喂回 AI → 直到 AI 输出【上网】结束或正常回答
"""

import re
import time
import html as html_lib
from urllib.parse import quote

import requests

from tavily import TavilyClient


# ============================================================
# 1. 注入给 AI 的上网指令（追加到 system 提示词末尾）
# ============================================================

BROWSE_PROMPT = (
    "【AI 自主上网能力】\n"
    "当你的知识不足以回答当前问题、或问题涉及实时信息时，你可以自主上网获取资料。\n"
    "请按以下格式提出上网请求（每次只提出一个请求，写在回答的最前面）：\n"
    "【上网】搜索:关键词1 关键词2\n"
    "【上网】打开:https://example.com\n"
    "【上网】结束\n\n"
    "规则：\n"
    "1. 需要实时信息时，先提出「搜索」请求，等待系统返回搜索结果。\n"
    "2. 若搜索结果需要深入了解，可再提出「打开」请求查看网页正文。\n"
    "3. 信息足够后，输出【上网】结束，然后给出最终回答。\n"
    "4. 不需要上网时，直接回答，不要输出任何标记。\n"
    "5. 搜索或打开失败时，系统会返回失败说明，你可以换关键词或换网址重试。"
)


# ============================================================
# 2. AI 上网请求标记解析
# ============================================================

_BROWSE_PATTERN = re.compile(r"【上网】\s*(搜索|打开|结束)\s*[:：]?\s*(.*)")


def parse_browse_request(text):
    """
    解析 AI 输出中的上网请求标记。

    Args:
        text: AI 本轮输出文本

    Returns:
        tuple (action, arg) 或 None:
          - ("搜索", "关键词")  — 请求搜索
          - ("打开", "https://...") — 请求打开网页
          - ("结束", "")        — 请求结束上网
          - None                — 未发现上网标记（AI 直接开始回答）
    """
    if not text:
        return None
    matches = _BROWSE_PATTERN.findall(text)
    if not matches:
        return None
    action, arg = matches[-1]
    action = action.strip()
    arg = arg.strip().strip('"').strip("'").strip()
    if action == "结束":
        return ("结束", "")
    if not arg:
        return None
    return (action, arg)


# ============================================================
# 3. 技能 B：打开网页，抓取正文文字
# ============================================================

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_REMOVE_BLOCKS = ("script", "style", "nav", "header", "footer", "noscript", "svg", "iframe", "form")


def fetch_webpage_text(url, max_chars=8000, timeout=20):
    """
    抓取网页正文文字（技能 B）。

    用 requests 直接抓取（TLS 指纹不会被 Cloudflare 拦截，见项目经验教训 #3），
    去掉脚本/样式/导航等噪音后提取可见文字。

    Args:
        url: 网页地址
        max_chars: 返回文本最大长度（防爆上下文）
        timeout: 请求超时秒数

    Returns:
        str: 提取到的正文文字

    Raises:
        ValueError / requests 异常: 抓取失败（由调用方转成错误文本喂回 AI）
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"无效的网址: {url}")

    resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type.lower() and "text" not in content_type.lower():
        raise ValueError(f"该地址返回的不是网页内容: {content_type}")

    raw = resp.text

    # 去掉大块噪音标签及其内容
    for tag in _REMOVE_BLOCKS:
        raw = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", raw, flags=re.S | re.I)

    # 去掉剩余标签，反转义 HTML 实体
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError("网页中没有提取到可见文字（可能是纯脚本渲染页面）")

    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(内容过长已截断)"

    return text


# ============================================================
# 4. 技能 A：搜索（三级降级链）
# ============================================================

def search_with_fallback(
    keywords,
    tavily_api_key,
    bocha_api_key,
    engine_pref="Tavily",
    retry_attempts=3,
    max_results=8,
    allow_api_search=True,
):
    """
    执行搜索，带降级链：首选引擎 → 备选引擎 → 必应网页版（不依赖 Key）。

    Args:
        keywords: 搜索关键词
        tavily_api_key: Tavily API Key（可为空）
        bocha_api_key: 博查 API Key（可为空）
        engine_pref: 首选引擎 ("Tavily" 或 "博查(Bocha)")
        retry_attempts: 每个引擎失败后的重试次数
        max_results: 返回结果条数上限
        allow_api_search: 是否允许使用 Tavily/博查 API。
            False 时跳过付费 API，直接使用必应网页版（对应侧边栏「搜索引擎」开关关闭）。

    Returns:
        tuple (source, formatted_text, results_count) | None:
          - source: 结果来源描述（"Tavily" / "博查" / "必应网页版"）
          - formatted_text: 格式化后的搜索结果文本
          - results_count: 结果条数
          - None: 所有引擎全部失败
    """
    if not allow_api_search:
        try:
            return _search_bing(keywords, max_results)
        except Exception:
            return None

    engines = []
    if engine_pref == "Tavily":
        engines = ["Tavily", "博查(Bocha)"]
    else:
        engines = ["博查(Bocha)", "Tavily"]

    for engine in engines:
        try:
            if engine == "Tavily" and tavily_api_key:
                return _search_tavily(keywords, tavily_api_key, retry_attempts, max_results)
            if engine == "博查(Bocha)" and bocha_api_key:
                return _search_bocha(keywords, bocha_api_key, retry_attempts, max_results)
        except Exception:
            continue  # 当前引擎失败 → 降级下一个

    # 兜底：必应网页版（不依赖任何 Key）
    try:
        return _search_bing(keywords, max_results)
    except Exception:
        return None


def _search_tavily(keywords, api_key, retry_attempts, max_results):
    """Tavily 搜索（无关键词提取步骤，关键词由 AI 直接提供）"""
    from chat_utils import retry_api_call

    def req():
        return TavilyClient(api_key=api_key).search(query=keywords)

    res = retry_api_call(req, max_attempts=retry_attempts, context="Tavily 搜索请求")
    results = res.get("results", [])[:max_results]
    formatted = "\n".join(
        [f"- {r.get('content', '无摘要')} ({r.get('url', '无链接')})" for r in results]
    )
    if not results:
        raise ValueError("Tavily 未返回任何结果")
    return "Tavily", formatted, len(results)


def _search_bocha(keywords, api_key, retry_attempts, max_results):
    """博查搜索（无关键词提取步骤）"""
    from chat_utils import retry_api_call

    def req():
        resp = requests.post(
            "https://api.bocha.cn/v1/web-search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"query": keywords, "freshness": "noLimit", "summary": True, "count": max_results},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    res = retry_api_call(req, max_attempts=retry_attempts, context="博查搜索请求")
    web_pages = res.get("data", {}).get("webPages", {})
    results = web_pages.get("value", [])[:max_results]
    formatted = "\n".join(
        [
            f"- {item.get('snippet', item.get('summary', '无摘要'))} ({item.get('url', '无链接')})"
            for item in results
        ]
    )
    if not results:
        raise ValueError("博查未返回任何结果")
    return "博查", formatted, len(results)


def _search_bing(keywords, max_results):
    """必应网页版搜索（无 Key 兜底方案，直接抓取搜索结果页）"""
    url = f"https://www.bing.com/search?q={quote(keywords)}&setlang=zh-hans"
    resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=15)
    resp.raise_for_status()
    html_text = resp.text

    # 标准解析：<li class="b_algo"> 条目
    items = re.findall(r'<li class="b_algo".*?</li>', html_text, re.S)
    results = []
    for item in items[:max_results]:
        m_link = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', item, re.S)
        if not m_link:
            continue
        link = m_link.group(1)
        title = re.sub(r"<[^>]+>", "", m_link.group(2))
        title = html_lib.unescape(title).strip()
        m_snip = re.search(r"<p[^>]*>(.*?)</p>", item, re.S)
        snippet = ""
        if m_snip:
            snippet = re.sub(r"<[^>]+>", "", m_snip.group(1))
            snippet = html_lib.unescape(snippet).strip()
        results.append((title, link, snippet))

    # 兜底解析：标题+链接全局抓取
    if not results:
        for m in re.finditer(r'<h2[^>]*>\s*<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', html_text, re.S):
            title = re.sub(r"<[^>]+>", "", m.group(2))
            title = html_lib.unescape(title).strip()
            results.append((title, m.group(1), ""))
            if len(results) >= max_results:
                break

    if not results:
        raise ValueError("必应网页版未能解析出结果（页面结构可能已变化）")

    formatted = "\n".join(
        [f"- {title} ({link})" + (f" {snippet}" if snippet else "") for title, link, snippet in results]
    )
    return "必应网页版", formatted, len(results)


# ============================================================
# 5. 执行一次上网请求（供 app.py 预循环调用）
# ============================================================

def execute_browse_action(action, arg, api_configs, retry_attempts=3, allow_api_search=True):
    """
    执行一次上网动作，返回 (source, result_text)。

    失败不抛异常——一律转成错误文本，让 AI 决定换关键词/换网址重试。

    Args:
        action: "搜索" 或 "打开"
        arg: 关键词或网址
        api_configs: st.session_state.api_configs（取 Tavily/博查 Key 与搜索引擎偏好）
        retry_attempts: 搜索重试次数
        allow_api_search: 是否允许使用 Tavily/博查 API（False 时搜索直接走必应网页版）

    Returns:
        tuple (source, result_text):
          - source: 来源描述（如 "Tavily" / "必应网页版" / "网页正文" / "失败"）
          - result_text: 结果文本（失败时为失败说明）
    """
    if action == "搜索":
        engine_pref = api_configs.get("search_engine", "Tavily")
        out = search_with_fallback(
            arg,
            api_configs.get("tavily", ""),
            api_configs.get("bocha", ""),
            engine_pref=engine_pref,
            retry_attempts=retry_attempts,
            allow_api_search=allow_api_search,
        )
        if out is None:
            return "失败", "所有搜索引擎均不可用（Tavily/博查/必应网页版全部失败），请稍后重试或直接回答。"
        source, formatted, count = out
        return source, f"共 {count} 条搜索结果：\n{formatted}"

    if action == "打开":
        try:
            text = fetch_webpage_text(arg)
            return "网页正文", f"网页 {arg} 的正文内容：\n{text}"
        except Exception as e:
            return "失败", f"打开网页失败: {e}。请换一个网址重试，或直接回答。"

    return "失败", f"无法识别的上网请求: {action}"


# ============================================================
# 6. 预循环主体（两条 API 通道共用）
# ============================================================

def run_browse_loop(
    channel_call,
    base_messages,
    current_prompt,
    api_configs,
    max_rounds=3,
    retry_attempts=3,
    status=None,
    allow_api_search=True,
):
    """
    边查边答循环：AI 自主决定上网，直到输出【上网】结束或直接回答。

    Args:
        channel_call: callable(messages) -> str
            接收 OpenAI 格式消息列表（role/content），返回 AI 输出文本。
            由调用方（app.py）按通道实现（Gemini / OpenAI 兼容）。
        base_messages: 基础上下文消息列表（不含上网记录与当前问题）
        current_prompt: 当前用户问题文本
        api_configs: st.session_state.api_configs
        max_rounds: 最多上网次数（防烧钱上限）
        retry_attempts: 搜索重试次数（同时用于 AI 请求的网络抗抖动重试）
        status: st.status 对象（用于展示上网过程）
        allow_api_search: 是否允许使用 Tavily/博查 API
            （False 时搜索直接走必应网页版，对应侧边栏「搜索引擎」开关关闭）

    Returns:
        dict: {
            "rounds": [{"kind": "搜索"|"打开", "arg": str, "result": str, "source": str}, ...],
            "ai_final": str | None,  # AI 在预循环中直接给出的回答（无上网标记时）
            "error": str | None,     # 预循环请求重试耗尽后的错误信息（无 rounds 时由调用方注入提示）
        }
    """
    rounds_log = []
    ai_final = None
    error = None

    for round_idx in range(1, max_rounds + 1):
        messages = list(base_messages) + [
            {"role": "user", "content": current_prompt}
        ]
        for r in rounds_log:
            messages.append(
                {"role": "assistant", "content": f"【上网】{r['kind']}:{r['arg']}"}
            )
            messages.append(
                {"role": "user", "content": f"【上网结果】(来源: {r['source']})\n{r['result']}"}
            )

        # 🌟 网络抗抖动：AI 请求自动重试（复用 retry_attempts 次数，间隔 1 秒），
        #    避免瞬时网关故障导致整个上网循环中断
        ai_out = None
        last_err = None
        for attempt in range(1, max(retry_attempts, 1) + 1):
            try:
                ai_out = channel_call(messages)
                break
            except Exception as e:
                last_err = e
                if attempt < max(retry_attempts, 1):
                    if status is not None:
                        status.update(
                            label=f"⚠️ 上网请求失败（第{attempt}次），正在重试...", state="running"
                        )
                    time.sleep(1)
        if ai_out is None:
            error = f"AI 上网请求失败: {last_err}"
            if status is not None:
                status.update(label=f"⚠️ {error}", state="error")
            break

        parsed = parse_browse_request(ai_out)

        if parsed is None:
            # AI 没有提出上网请求 → 它已经开始回答了（预循环只负责上网，回答交给主流程）
            ai_final = ai_out
            break
        action, arg = parsed
        if action == "结束":
            break

        source, result_text = execute_browse_action(
            action, arg, api_configs, retry_attempts,
            allow_api_search=allow_api_search,
        )
        rounds_log.append(
            {"kind": action, "arg": arg, "result": result_text, "source": source}
        )

        if status is not None:
            label = f"🤖 第{round_idx}轮 上网: {'搜索' if action == '搜索' else '打开'} {arg} (来源: {source})"
            status.update(label=label, state="running")

    return {"rounds": rounds_log, "ai_final": ai_final, "error": error}


def format_browse_context(rounds_log, max_result_chars=3000, reused=False):
    """
    将上网过程格式化为注入主请求的上下文文本。

    Args:
        rounds_log: run_browse_loop 返回的 rounds 列表
        max_result_chars: 每轮结果截断长度（防爆上下文）
        reused: 是否为缓存复用（重新生成/对比时未重新联网），True 时标题加标注

    Returns:
        str: 注入文本（空字符串表示无上网记录）
    """
    if not rounds_log:
        return ""

    lines = []
    for i, r in enumerate(rounds_log, 1):
        result = r["result"]
        if len(result) > max_result_chars:
            result = result[:max_result_chars] + "\n...(已截断)"
        kind_label = "搜索" if r["kind"] == "搜索" else "打开网页"
        lines.append(
            f"第{i}轮 上网（来源: {r['source']}）\n"
            f"动作: {kind_label} {r['arg']}\n"
            f"结果:\n{result}"
        )
    if reused:
        title = "【AI 自主上网过程】(🔁 来自缓存复用，未重新联网)"
    else:
        title = "【AI 自主上网过程】"
    return title + "\n" + "\n\n".join(lines)