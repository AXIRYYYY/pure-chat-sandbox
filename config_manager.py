import json
import os

# --- 1. 环境初始化与目录内控 ---

CONFIG_FILE = "api_config.json"
MODEL_CONFIG_FILE = "model_config.json"
PROVIDER_REGISTRY_FILE = "provider_registry.json"
LOG_DIR = "chat_logs"
ATTACH_DIR = "attachments_cache"

# 确保日志和附件缓存目录存在
for d in [LOG_DIR, ATTACH_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


def load_model_config():
    """从本地文件加载模型列表配置，如果文件不存在则返回默认配置"""
    if os.path.exists(MODEL_CONFIG_FILE):
        try:
            with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            # 如果解析失败，回退到默认设置
            pass
    return {
        "Gemini": [
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro",
            "gemini-2.0-flash-thinking-exp",
            "gemini-1.5-pro",
            "自定义...",
        ],
        "SiliconFlow": [
            "deepseek-ai/DeepSeek-V3.2",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
    }


def get_model_options(model_choice):
    """根据选择的通道（Gemini或SiliconFlow）返回可选模型列表"""
    model_config = load_model_config()
    return model_config.get(model_choice, [])


def load_config():
    """加载应用的主配置文件（API Key、偏好设置等）"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            # 解析失败返回默认值
            pass
    return {
        "gemini": "",
        "siliconflow": "",
        "deepseek": "",
        "tavily": "",
        "bocha": "",  # 🔍 博查搜索 API Key（替代/补充 Tavily）
        "last_choice": "Gemini",
        "last_model": "gemini-2.0-flash-thinking-exp",
        "web_search": True,
        "web_search_suggestion_enabled": False,  # 🔔 联网搜索建议开关（默认关闭）
        "system_prompt": "",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 8192,
        "gem_instruction": "",
        "current_workspace": "默认沙盒 (Normal)",
        "workspace_titles": {},
        "workspaces": ["默认沙盒 (Normal)", "Gem 专属区 (Gem)", "草稿区 (Draft)"],
        "workspace_prefs": {},
        "auto_name_prefs": {
            "enabled": False,
            "channel": "SiliconFlow",
            "model": "Qwen/Qwen2.5-72B-Instruct",
        },
        "auto_name_triggered": False,
        # 🌟 API 自动重连配置
        # 发生高负载、网络抖动或临时故障时，系统会自动重试请求。
        # -1 表示无限重试。
        "api_retry_attempts": 3,
        "web_search_model": {
            "SiliconFlow": "Qwen/Qwen2.5-72B-Instruct",
            "DeepSeek": "deepseek-chat",
        },
        "web_search_extract_config": {},
        "search_engine": "Tavily",
        "reasoning_effort": "medium",
        "compression_defaults": {},
    }


def save_config(config):
    """将最新的配置持久化保存到 api_config.json"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
