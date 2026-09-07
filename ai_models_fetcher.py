#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI模型参数获取脚本
功能：自动获取所有已启用供应商的最新模型列表，并保存为 JSON 文件供程序使用
"""

import json
import os
from datetime import datetime

import requests

# API端点
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/models"
DEEPSEEK_API_URL = "https://api.deepseek.com/models"

# 伪装浏览器 User-Agent（绕过 Cloudflare Bot Fight Mode）
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def get_datetime_string():
    """获取当前日期时间字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def loads_config():
    """尝试从配置文件加载"""
    config_file = "api_config.json"
    config = {}

    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except:
            pass
    return config


def load_all_providers():
    """加载供应商注册表，返回所有已启用的供应商列表"""
    providers = []
    provider_file = "provider_registry.json"
    if os.path.exists(provider_file):
        try:
            with open(provider_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            builtin = data.get("builtin", {})
            custom = data.get("custom", {})
            for pid, info in builtin.items():
                if info.get("enabled", False):
                    providers.append({"id": pid, "is_custom": False})
            for pid, info in custom.items():
                if info.get("enabled", True):
                    providers.append({"id": pid, "is_custom": True, "info": info})
        except:
            pass
    if not providers:
        # 兜底: 回退到硬编码的 3 个通道
        providers = [
            {"id": "Gemini", "is_custom": False},
            {"id": "SiliconFlow", "is_custom": False},
            {"id": "DeepSeek", "is_custom": False},
        ]
    return providers


def get_provider_info(provider_id):
    """获取供应商元数据"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from provider_registry import BUILTIN_PROVIDERS, load_provider_registry
        info = BUILTIN_PROVIDERS.get(provider_id)
        if info:
            return info
        _, custom = load_provider_registry()
        if provider_id in custom:
            return {
                "name": custom[provider_id].get("name", provider_id),
                "type": custom[provider_id].get("type", "openai"),
                "api_host": custom[provider_id].get("api_host", ""),
            }
    except:
        pass
    return None


def http_get(url, headers=None, params=None):
    """使用 requests 发送 GET 请求"""
    merged = _base_headers()
    if headers:
        merged.update(headers)
    try:
        resp = requests.get(url, headers=merged, params=params, timeout=30, allow_redirects=False)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        print(f"   请求错误: HTTP {e.response.status_code}: {e.response.reason}")
        return None
    except Exception as e:
        print(f"   请求错误: {e}")
        return None


def http_post(url, headers=None, json_body=None):
    """使用 requests 发送 POST 请求"""
    merged = _base_headers()
    if headers:
        merged.update(headers)
    try:
        resp = requests.post(url, headers=merged, json=json_body, timeout=15, allow_redirects=False)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        print(f"   请求错误: HTTP {e.response.status_code}: {e.response.reason}")
        return None
    except Exception as e:
        print(f"   请求错误: {e}")
        return None


def http_post_raw(url, headers=None, json_body=None):
    """发送 POST 请求，返回完整 response 对象（用于错误诊断）"""
    merged = _base_headers()
    if headers:
        merged.update(headers)
    return requests.post(url, headers=merged, json=json_body, timeout=15, allow_redirects=False)


def _base_headers():
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }


def get_gemini_models(api_key):
    """获取Gemini模型列表"""
    print("🔍 正在获取Gemini模型列表...")
    if not api_key:
        return []

    data = http_get(GEMINI_API_URL, params={"key": api_key})

    models = []
    if data and "models" in data:
        for m in data["models"]:
            models.append({"模型ID": m.get("name"), "平台": "Gemini"})
        print(f"✅ 成功获取 {len(models)} 个Gemini模型")
    return models


def get_siliconflow_models(api_key):
    """获取硅基流动模型列表"""
    print("🔍 正在获取硅基流动模型列表...")
    if not api_key:
        return []

    headers = {"Authorization": f"Bearer {api_key}"}
    all_models = []
    for mtype in ["text", "image", "audio", "video"]:
        data = http_get(SILICONFLOW_API_URL, headers=headers, params={"type": mtype})
        if data and "data" in data:
            for m in data["data"]:
                all_models.append({"模型ID": m.get("id"), "平台": "SiliconFlow"})

    print(f"✅ 成功获取 {len(all_models)} 个硅基流动模型")
    return all_models


def get_deepseek_models(api_key):
    """获取DeepSeek模型列表"""
    print("🔍 正在获取 DeepSeek 模型列表...")
    if not api_key:
        return []

    headers = {"Authorization": f"Bearer {api_key}"}
    data = http_get(DEEPSEEK_API_URL, headers=headers)

    models = []
    if data and "data" in data:
        for m in data["data"]:
            models.append({"模型ID": m.get("id"), "平台": "DeepSeek"})
        print(f"✅ 成功获取 {len(models)} 个 DeepSeek 模型")
    return models


def save_to_json(all_models, output_dir):
    """保存模型数据到JSON文件"""
    filepath = os.path.join(output_dir, "available_models.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_models, f, ensure_ascii=False, indent=4)
        print(f"🎉 模型数据已保存到: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 保存JSON失败: {e}")
        return False


def fetch_openai_models(api_host, api_key, platform_label):
    """通用 OpenAI 兼容接口模型获取，返回 (models, error_msg)"""
    print(f"🔍 正在获取 {platform_label} 模型列表...")
    if not api_key or not api_host:
        return [], "缺少 api_host 或 api_key"

    base = api_host.rstrip("/")
    urls_to_try = []
    if base.endswith("/v1"):
        urls_to_try.append(f"{base}/models")
    else:
        urls_to_try.append(f"{base}/v1/models")
        urls_to_try.append(f"{base}/models")

    last_error = ""
    for url in urls_to_try:
        headers = {"Authorization": f"Bearer {api_key}"}
        data = http_get(url, headers=headers)

        if data is None:
            continue

        models = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("models")
            if items and isinstance(items, list):
                for m in items:
                    model_id = m.get("id") or m.get("model") or m.get("name") or str(m)
                    models.append({"模型ID": model_id, "平台": platform_label})
        elif isinstance(data, list):
            for m in data:
                model_id = m.get("id") or m.get("model") or m.get("name") or str(m)
                models.append({"模型ID": model_id, "平台": platform_label})

        if models:
            print(f"✅ 成功获取 {len(models)} 个 {platform_label} 模型 (URL: {url})")
            return models, ""
        else:
            last_error = f"URL {url} 返回了数据但格式不匹配"

    last_error = f"所有 URL 均失败"
    print(f"⚠️ {platform_label}: {last_error}")
    return [], last_error


def verify_chat_key(api_host, api_key, platform_label):
    """用最小 chat 请求验证 API Key 是否有效，返回 (ok: bool, detail: str)"""
    print(f"🔑 验证 {platform_label} chat 连通性...")
    if not api_key or not api_host:
        return False, "缺少 api_host 或 api_key"

    base = api_host.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    test_body = {
        "model": "deepseek-v4-flash-free",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }

    try:
        resp = http_post_raw(url, headers=headers, json_body=test_body)
        if resp.status_code == 200:
            print(f"✅ {platform_label} chat 连通性验证成功")
            return True, "Chat 请求成功，Key 有效"
        else:
            err_snippet = resp.text[:300] if resp.text else ""
            detail = f"HTTP {resp.status_code}: {resp.reason}"
            if err_snippet:
                detail += f" — {err_snippet}"
            print(f"❌ {platform_label} chat 失败: {detail}")
            return False, detail
    except Exception as e:
        detail = f"网络错误: {str(e)[:200]}"
        print(f"❌ {platform_label} chat 失败: {detail}")
        return False, detail


def main(interactive=True):
    return main_with_report(interactive=interactive)[0]


def main_with_report(interactive=True):
    "返回 (all_models, report_list) — report_list 每项: {供应商, 状态, 模型数, 详情}"
    report = []

    def _r(name, status, count=0, detail=""):
        report.append({"供应商": name, "状态": status, "模型数": count, "详情": detail})

    print("=" * 60)
    print("🤖 AI模型参数获取工具 (多供应商版)")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    config = loads_config()
    providers = load_all_providers()

    all_models = []

    for prov in providers:
        pid = prov["id"]
        prov_info = get_provider_info(pid)

        if not prov_info:
            msg = f"未知供应商: {pid}"
            print(f"⚠️ {msg}")
            _r(pid, "❌ 未知", 0, msg)
            continue

        prov_type = prov_info.get("type", "openai")
        api_key = config.get(pid, "") or config.get(prov_info.get("api_key_field", pid), "")
        display_name = prov_info.get("name", pid)

        if not api_key:
            msg = "未配置 API Key"
            print(f"⚠️ {display_name} {msg}")
            _r(display_name, "🔒 跳过", 0, msg)
            continue

        if prov_type == "gemini":
            models = get_gemini_models(api_key)
            all_models.extend(models)
            _r(display_name, "✅ 成功", len(models))
        elif prov_type == "openai":
            api_host = prov_info.get("api_host", "")
            if api_host:
                models, err_msg = fetch_openai_models(api_host, api_key, pid)
                all_models.extend(models)
                if err_msg:
                    ok, detail = verify_chat_key(api_host, api_key, pid)
                    if ok:
                        _r(display_name, "⚠️ Key有效但/models不可用", 0,
                           f"Key 通过 chat 验证有效，但 /models 端点获取失败")
                    else:
                        _r(display_name, "❌ 认证失败", 0, f"Chat 验证也失败: {detail}")
                elif models:
                    _r(display_name, "✅ 成功", len(models))
                else:
                    ok, detail = verify_chat_key(api_host, api_key, pid)
                    if ok:
                        _r(display_name, "⚠️ Key有效但/models不可用", 0,
                           f"Key 通过 chat 验证有效，但 /models 返回空")
                    else:
                        _r(display_name, "❌ 认证失败", 0, f"Chat 验证也失败: {detail}")
            else:
                if pid == "SiliconFlow":
                    models = get_siliconflow_models(api_key)
                    all_models.extend(models)
                    _r(display_name, "✅ 成功", len(models))
                elif pid == "DeepSeek":
                    models = get_deepseek_models(api_key)
                    all_models.extend(models)
                    _r(display_name, "✅ 成功", len(models))
                else:
                    msg = "缺少 API Host"
                    print(f"⚠️ {pid} {msg}")
                    _r(display_name, "❌ 失败", 0, msg)
        else:
            msg = f"不支持的供应商类型: {prov_type}"
            print(f"⚠️ {pid} {msg}")
            _r(display_name, "❌ 失败", 0, msg)

    print("-" * 60)
    print(f"📊 总计: {len(all_models)} 个模型")

    save_to_json(all_models, script_dir)
    print("=" * 60)
    print("✅ 程序执行完成!")
    if interactive:
        input("按回车键退出...")

    return all_models, report


if __name__ == "__main__":
    main()