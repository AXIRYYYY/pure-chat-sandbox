# 纯净 Chat 沙盒 (Pure Chat Sandbox)

[![Version](https://img.shields.io/badge/version-5.4.1-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一个为审计、合规及深度研究设计的 LLM 交互实验室，支持 Gemini、SiliconFlow、DeepSeek 等多供应商通道，具备长效缓存、工作区隔离、多附件解析及智能联网搜索功能。

## 核心特性

- **多模型多通道**: 深度集成 Google Gemini (支持 Context Cache)、SiliconFlow (聚合 DeepSeek, Qwen 等) 和 DeepSeek 原生平台，并可通过供应商注册表扩展更多 OpenAI 兼容通道。
- **动态模型库**: 支持从各平台 API 自动获取最新模型列表，通过图形化界面启用/禁用模型，无需手动修改配置文件。
- **联网搜索统一模块**: SiliconFlow 和 DeepSeek 平台共用同一套联网搜索逻辑，代码更精简，维护更便捷。
- **模块化架构**: 高度解耦的后端服务，易于维护与扩展。
- **审计级工作区**: 完整的案例隔离机制，每个工作区拥有独立的配置、提示词及对话历史。
- **智能附件处理**: 自动解析 PDF (基于 PyMuPDF)、TXT、MD、HTML 等格式，支持超大文件物理备份。
- **联网搜索集成**: 支持 Tavily / 博查(Bocha) 双搜索引擎，可切换，辅助实时联网搜索。
- **长效缓存管理**: 可视化管理 Gemini 后台缓存，节省 Token 并提升长上下文响应速度。
- **自动化命名**: 根据对话内容自动生成简洁的案例标题。
- **多页面架构**: 采用 Streamlit Pages 实现模块化界面，扩展更灵活。

## 项目结构

```text
.
├── app.py                    # Streamlit 主应用入口
├── config_manager.py         # 配置与常量管理
├── provider_registry.py      # 供应商注册表、OpenAI client 工厂
├── compression_engine.py     # 上下文压缩引擎
├── file_utils.py             # 文件解析服务 (PDF/TXT/HTML)
├── token_utils.py            # Token 计算与估算 (Gemini/SiliconFlow)
├── chat_utils.py             # 对话历史、工作区管理 & 共用联网搜索
├── export_utils.py           # Markdown 导出功能
├── ai_models_fetcher.py      # 自动模型列表获取工具
├── api_config.example.json   # API 密钥配置模板
├── requirements.txt          # 依赖清单
├── LICENSE                   # MIT 许可证
├── README.md                 # 本文档
├── CHANGELOG.md              # 更新日志
├── pages/                    # Streamlit 多页面
│   └── model_management.py   # 模型库管理页面
├── chat_logs/                # 存储对话历史 (JSON/JSONL，本地运行生成，不提交)
└── attachments_cache/        # 附件物理备份目录（本地运行生成，不提交）
```

## 快速开始

### 1. 安装依赖

确保已安装 Python 3.9+，然后执行：

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

**方式一：通过 UI 界面填写（推荐）**

启动应用后，访问 **「模型库管理」** 页面（`/model_management`），在对应输入框中填入各平台的 Key，点击保存即可。

**方式二：编辑配置文件**

将 `api_config.example.json` 复制为 `api_config.json`，手动填入密钥：

```bash
copy api_config.example.json api_config.json
# Linux / macOS:
# cp api_config.example.json api_config.json
```

然后编辑 `api_config.json`，替换以下字段：

| 字段 | 获取地址 |
|------|---------|
| `gemini` | [Google AI Studio](https://aistudio.google.com/) |
| `tavily` | [Tavily](https://tavily.com/) |
| `deepseek` | [DeepSeek](https://platform.deepseek.com/) |
| `bocha` | [博查](https://open.bocha.cn) |
| `siliconflow` | [SiliconFlow](https://siliconflow.cn/) |

### 3. 启动应用

```bash
streamlit run app.py
```

### 4. 获取模型列表（首次使用必需）

在已启动的应用中访问 **「模型库管理」** 页面（`/model_management`），点击 **获取最新模型列表** 从各平台拉取最新可用模型，勾选需要的模型后点击保存，此后即可在主页切换使用。

## 导出与归档

系统支持将对话导出为：
- **JSON**: 包含所有元数据（Token 使用、Latency 等）的完整资产。
- **Markdown**: 格式精美的阅读版文档，包含时间戳与附件引用。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 更新记录

详细历史见 [CHANGELOG.md](CHANGELOG.md)。