# provider_registry.py — 供应商注册表
"""
管理所有 LLM 供应商（内置热门 + 用户自定义）。
提供通道可见性判断、OpenAI client 工厂、供应商元数据查询等功能。
"""

import json
import os
from openai import OpenAI

PROVIDER_REGISTRY_FILE = "provider_registry.json"

# —————————————————————————— 内置热门供应商预设 ——————————————————————————

BUILTIN_PROVIDERS = {
    "Gemini": {
        "name": "Google Gemini",
        "type": "gemini",
        "api_host": None,
        "api_key_field": "gemini",
        "description": "Google 官方大模型 (Gemini 2.5/3.1 Pro/Flash)",
        "api_key_url": "https://aistudio.google.com/apikey",
    },
    "DeepSeek": {
        "name": "DeepSeek",
        "type": "openai",
        "api_host": "https://api.deepseek.com",
        "api_key_field": "deepseek",
        "description": "深度求索官方 API (V4, R1, Chat, Reasoner)",
        "api_key_url": "https://platform.deepseek.com/api_keys",
    },
    "SiliconFlow": {
        "name": "SiliconFlow 硅基流动",
        "type": "openai",
        "api_host": "https://api.siliconflow.cn/v1",
        "api_key_field": "siliconflow",
        "description": "国产模型聚合平台 (DeepSeek, Qwen, GLM, Kimi 等)",
        "api_key_url": "https://cloud.siliconflow.cn/account/ak",
    },
    "opencode-zen": {
        "name": "OpenCode Zen",
        "type": "openai",
        "api_host": "https://opencode.ai/zen/v1",
        "api_key_field": "opencode-zen",
        "description": "OpenCode 官方模型网关 (含免费模型: DeepSeek V4 Flash Free, MiMo-V2.5 Free 等)",
        "api_key_url": "https://opencode.ai/auth",
    },
    "openai": {
        "name": "OpenAI",
        "type": "openai",
        "api_host": "https://api.openai.com/v1",
        "description": "GPT-4o, GPT-4.1, o4-mini 等旗舰模型",
        "api_key_url": "https://platform.openai.com/api-keys",
    },
    "dashscope": {
        "name": "阿里百炼 (通义千问)",
        "type": "openai",
        "api_host": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云大模型平台 (Qwen3.5, Qwen-Flash 等)",
        "api_key_url": "https://bailian.console.aliyun.com/",
    },
    "zhipu": {
        "name": "智谱AI (GLM)",
        "type": "openai",
        "api_host": "https://open.bigmodel.cn/api/paas/v4",
        "description": "智谱清言大模型 (GLM-5, GLM-4 系列)",
        "api_key_url": "https://open.bigmodel.cn/usercenter/apikeys",
    },
    "moonshot": {
        "name": "月之暗面 Moonshot",
        "type": "openai",
        "api_host": "https://api.moonshot.cn/v1",
        "description": "Kimi K2.5/K2.6 系列大模型",
        "api_key_url": "https://platform.moonshot.cn/console/api-keys",
    },
    "minimax": {
        "name": "MiniMax",
        "type": "openai",
        "api_host": "https://api.minimax.chat/v1",
        "description": "MiniMax-M2.5 等海螺AI模型",
        "api_key_url": "https://platform.minimax.com/user-center/basic-information",
    },
    "groq": {
        "name": "Groq",
        "type": "openai",
        "api_host": "https://api.groq.com/openai/v1",
        "description": "超低延迟推理 (Llama 4, Qwen 等，有免费 tier)",
        "api_key_url": "https://console.groq.com/keys",
    },
    "openrouter": {
        "name": "OpenRouter",
        "type": "openai",
        "api_host": "https://openrouter.ai/api/v1",
        "description": "国际模型聚合平台 (300+ 模型，统一计费)",
        "api_key_url": "https://openrouter.ai/keys",
    },
    "xai": {
        "name": "xAI Grok",
        "type": "openai",
        "api_host": "https://api.x.ai/v1",
        "description": "Elon Musk xAI (Grok-4 等)",
        "api_key_url": "https://console.x.ai/",
    },
    "ollama": {
        "name": "Ollama (本地)",
        "type": "openai",
        "api_host": "http://localhost:11434/v1",
        "description": "本地运行的开源模型 (Llama, Qwen, Mistral 等)",
        "api_key_url": None,
    },
    "volcengine": {
        "name": "火山引擎 (豆包)",
        "type": "openai",
        "api_host": "https://ark.cn-beijing.volces.com/api/v3",
        "description": "字节跳动火山方舟 (Doubao-1.6, DeepSeek 等)",
        "api_key_url": "https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey",
    },
    "stepfun": {
        "name": "阶跃星辰 StepFun",
        "type": "openai",
        "api_host": "https://api.stepfun.com/v1",
        "description": "Step-3 系列多模态大模型",
        "api_key_url": "https://platform.stepfun.com/interface-key",
    },
    "together": {
        "name": "Together AI",
        "type": "openai",
        "api_host": "https://api.together.xyz/v1",
        "description": "开源模型推理平台 (Llama, DeepSeek, Qwen 等)",
        "api_key_url": "https://api.together.ai/settings/api-keys",
    },
    "mistral": {
        "name": "Mistral AI",
        "type": "openai",
        "api_host": "https://api.mistral.ai/v1",
        "description": "Mistral Large 2, Codestral 等欧洲模型",
        "api_key_url": "https://console.mistral.ai/api-keys/",
    },
}


# —————————————————————————— 注册表加载/保存 ——————————————————————————

def load_provider_registry():
    if os.path.exists(PROVIDER_REGISTRY_FILE):
        try:
            with open(PROVIDER_REGISTRY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                builtin = data.get("builtin", {})
                custom = data.get("custom", {})
                return builtin, custom
        except Exception:
            pass
    return {}, {}


def save_provider_registry(builtin, custom):
    with open(PROVIDER_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump({"builtin": builtin, "custom": custom}, f, ensure_ascii=False, indent=4)


def ensure_builtin_defaults():
    builtin, custom = load_provider_registry()
    changed = False
    for pid, info in BUILTIN_PROVIDERS.items():
        if pid not in builtin:
            builtin[pid] = {"enabled": False}
            changed = True
        existing = builtin[pid]
        if "enabled" not in existing:
            existing["enabled"] = False
            changed = True
    if changed:
        save_provider_registry(builtin, custom)
    return builtin, custom


# —————————————————————————— API Key 桥接 ——————————————————————————

def get_api_key(api_config, provider_id):
    info = BUILTIN_PROVIDERS.get(provider_id, {})
    api_key_field = info.get("api_key_field", provider_id)
    return api_config.get(api_key_field, "") or ""


# —————————————————————————— 供应商元数据查询 ——————————————————————————

def get_provider_info(provider_id):
    metadata = BUILTIN_PROVIDERS.get(provider_id)
    if metadata:
        return metadata
    _, custom = load_provider_registry()
    custom_def = custom.get(provider_id, {})
    if custom_def:
        return {
            "name": custom_def.get("name", provider_id),
            "type": custom_def.get("type", "openai"),
            "api_host": custom_def.get("api_host", ""),
            "description": custom_def.get("description", "自定义接口"),
            "api_key_url": None,
            "is_custom": True,
            "api_key_field": provider_id,
        }
    return None


def is_provider_enabled(provider_id):
    builtin, custom = load_provider_registry()
    if provider_id in builtin:
        return builtin[provider_id].get("enabled", False)
    if provider_id in custom:
        return custom[provider_id].get("enabled", True)
    return False


def set_provider_enabled(provider_id, enabled):
    builtin, custom = load_provider_registry()
    if provider_id in BUILTIN_PROVIDERS:
        if provider_id not in builtin:
            builtin[provider_id] = {}
        builtin[provider_id]["enabled"] = enabled
        save_provider_registry(builtin, custom)
        return True
    if provider_id in custom:
        custom[provider_id]["enabled"] = enabled
        save_provider_registry(builtin, custom)
        return True
    return False


def get_all_provider_ids():
    builtin, custom = load_provider_registry()
    ids = []
    for pid in BUILTIN_PROVIDERS:
        if builtin.get(pid, {}).get("enabled", False):
            ids.append(pid)
    for pid, info in custom.items():
        if info.get("enabled", True):
            ids.append(pid)
    return ids


# —————————————————————————— 通道可见性判断 ——————————————————————————

def get_visible_channels(api_config):
    """
    返回在下拉菜单中应显示的通道列表 (按顺序).
    条件：1) 供应商已启用  2) 有 API key  3) 有模型可选
    """
    builtin, custom = load_provider_registry()
    visible = []

    for pid in BUILTIN_PROVIDERS:
        if not builtin.get(pid, {}).get("enabled", False):
            continue
        if not get_api_key(api_config, pid):
            continue
        if not _has_models(pid):
            continue
        visible.append(pid)

    for pid in sorted(custom.keys()):
        info = custom[pid]
        if not info.get("enabled", True):
            continue
        if not get_api_key(api_config, pid):
            continue
        if not _has_models(pid):
            continue
        visible.append(pid)

    return visible


def _has_models(provider_id):
    if os.path.exists("enabled_models.json"):
        try:
            with open("enabled_models.json", "r", encoding="utf-8") as f:
                enabled = json.load(f)
            if enabled.get(provider_id):
                return True
        except Exception:
            pass
    if os.path.exists("model_config.json"):
        try:
            with open("model_config.json", "r", encoding="utf-8") as f:
                mc = json.load(f)
            if mc.get(provider_id):
                return True
        except Exception:
            pass
    return False


def get_channel_display_name(provider_id):
    info = get_provider_info(provider_id)
    if info:
        return info["name"]
    return provider_id


# —————————————————————————— OpenAI Client 工厂 ——————————————————————————

def create_openai_client(provider_id, api_key):
    info = get_provider_info(provider_id)
    if not info:
        return None
    base_url = info.get("api_host", "")
    if not base_url:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def get_api_host(provider_id):
    info = get_provider_info(provider_id)
    if info:
        return info.get("api_host", "")
    return ""


# —————————————————————————— 自定义供应商管理 ——————————————————————————

def add_custom_provider(name, api_host, api_key=""):
    builtin, custom = load_provider_registry()
    import uuid
    pid = f"custom_{uuid.uuid4().hex[:8]}"
    custom[pid] = {
        "name": name,
        "type": "openai",
        "api_host": api_host,
        "enabled": True,
    }
    save_provider_registry(builtin, custom)
    return pid


def update_custom_provider(provider_id, name, api_host):
    builtin, custom = load_provider_registry()
    if provider_id not in custom:
        return False
    custom[provider_id]["name"] = name
    custom[provider_id]["api_host"] = api_host
    save_provider_registry(builtin, custom)
    return True


def delete_custom_provider(provider_id):
    builtin, custom = load_provider_registry()
    if provider_id in custom:
        del custom[provider_id]
        save_provider_registry(builtin, custom)
        return True
    return False


def get_all_custom_providers():
    _, custom = load_provider_registry()
    return dict(custom)


def is_builtin(provider_id):
    return provider_id in BUILTIN_PROVIDERS


def is_gemini_type(provider_id):
    info = get_provider_info(provider_id)
    return info and info.get("type") == "gemini"


def is_openai_type(provider_id):
    info = get_provider_info(provider_id)
    return info and info.get("type") == "openai"


# —————————————————————————— 初始化 ——————————————————————————

def initialize():
    ensure_builtin_defaults()