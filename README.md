# 纯净 Chat 沙盒 (Pure Chat Sandbox)

[![Version](https://img.shields.io/badge/version-5.5.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一个为审计、合规及深度研究设计的 LLM 交互实验室，支持 18 个内置供应商与任意自定义 OpenAI 兼容通道，具备长效缓存、工作区隔离、多附件解析及智能联网搜索功能。

## 核心特性

### 与常见对话页的区别
- **中间段可压的非侵入式压缩**: 按轮次选任意连续区间（含中间段），多区间可并存，摘要替换进上下文而原文完整保留，取消即恢复——区别于常见的从头截断。
- **审计级时间感知**: 每条用户消息自动注入 `[提交于：YYYY-MM-DD HH:MM:SS]` 时间戳发往 API，AI 每轮都能感知对话发生的时间与时序——配合联网搜索返回北京时间，可做时效性判断与跨天对话的时间线推理。原始输入与搜索缓存不受污染。
- **对比 Tabs**: 同一条回答下并排对比多模型输出，复用原消息的搜索缓存，不污染线性历史与缓存前缀。
- **AI 自主上网（可零 Key）**: AI 自主决定搜索、自选关键词、打开网页，边查边答至信息足够；搜索链 Tavily → 博查 → 必应网页版自动降级，不配任何搜索 Key 也能联网。
- **供应商与模型全动态**: 18 内置供应商 + 无限自定义接口，一键拉取最新模型列表并勾选启用，不写死任何模型清单。
- **搜索结果独立缓存**: 按消息 ID 缓存，重新生成与对比时零重复联网。
- **缓存前缀稳定保障**: 确定性序列化（键序稳定），Gemini Context Cache 命中率不受编辑与对比影响。

### 多供应商与模型管理
- **18 内置供应商 + 无限自定义**: 内置 Gemini、OpenAI、DeepSeek、SiliconFlow、阿里百炼、智谱、Kimi、MiniMax、Groq、OpenRouter、xAI、Ollama、火山、阶跃、Together、Mistral、OpenCode Zen、OpenCode Go（订阅网关），并可添加任意 OpenAI 兼容接口；通道按“已启用 + 有 Key + 有可用模型”自动过滤显示。
- **双 API 路径**: Gemini 走原生 SDK（含 Context Cache），其余统一走 OpenAI 兼容路径。
- **动态模型库**: 在「模型库管理」页一键拉取各平台最新模型列表，逐供应商报告成功/失败与模型增减通知（`/models` 端点不可用时以最小 chat 请求验证 Key 有效性），按平台勾选启用；支持“自定义…”手输任意模型 ID。

### 工作区与命名
- **审计级工作区**: 互相隔离的工作区，各自独立保存通道、模型、提示词、生成参数与对话历史，可新增（复制当前配置）与删除。
- **自动命名**: 首轮对话后按内容生成简洁案例标题，可选命名通道与模型（默认关闭），也支持一键重新生成。

### 对话与附件
- **时间戳自动注入**: 发送时自动为消息追加提交时间发往 API，AI 感知每轮的时间上下文；显示层带同款后缀便于审计，重新生成时按规则复用原时间戳，编辑重发则更新为当前时间。
- **多格式附件**: 支持 `txt/md/py/json/csv/pdf/html`，PDF 经 PyMuPDF 解析并清洗文本污染，全部解析、全部物理备份，无大小截断；上传后展示解析详情（字符数/体积/类型）。
- **重新生成**: 直接重新生成 / 编辑并重新生成，编辑时附件自动重挂载，时间戳按规则复用。
- **草稿与中断恢复**: 生成中的消息双写内存与磁盘，浏览器刷新或手机端清理后不丢失；可删除整轮对话（连带清理反馈、搜索缓存与压缩区间），也可彻底销毁会话。

### 联网搜索
- **三层联网开关**: 全局联网为总开关，下挂「搜索引擎」(Tavily/博查 API) 与「AI 自主上网」两个子开关；按组合自动切换 老式搜索 / 自主上网 / 纯聊天 三种模式；搜索关键词提取提示词可在侧边栏编辑。
- **AI 自主上网**: AI 自选关键词、抓取网页正文、循环查证后作答；搜索自动降级 Tavily → 博查 → 必应网页版（无 Key 兜底）。上网记录按消息缓存，重新生成与对比时复用，编辑重发自动重搜；失败信息跨页面保留。
- **提取模型跨平台自选**: 关键词提取可用与主对话不同平台的便宜模型；AI 可主动建议搜索关键词，一键同意即追加搜索并重新生成；也支持手动多次联网。
- **Gemini 原生搜索**: 老式联网模式下 Gemini 通道使用原生 Google Search grounding（不占 Tavily/博查额度）。

### 上下文压缩与对比
- **非侵入式压缩**: 按轮次选择任意连续区间（含中间段），自定义压缩提示词、通道与模型，摘要替换进上下文而原始消息完整保留，取消即恢复；**支持多区间并存**（如同时压缩 2-4 轮与 6-8 轮），自动检测区间重叠，删除消息时自清理。
- **对比生成**: 同一条回答可并排 Tabs 对比多个模型的输出，复用原消息的搜索缓存，不污染线性历史与缓存前缀。
- **对话索引导航**: 按用户消息分轮，hash 锚点跳转不触发页面重载，压缩段分组标注。

### Gemini 缓存与 Gem 专区
- **长效缓存管理**: 可视化查看 Gemini 后台缓存（ID、剩余寿命、命中状态），支持续期与废弃；可按任意消息位置创建缓存。
- **Gem 专属区**: 固定 Prompt + 固定知识库，一键实例化并预建底层缓存；支持重置到初始态。
- **思维链显示**: DeepSeek 推理过程实时展开，思考强度（low/medium/high）可调。

### 可观测与审计
- **Token 预估**: 一键预估当前上下文消耗（Gemini 精确计数，其余 tiktoken 估算）；每条回复附带时间、模型、耗时与 Token 明细。
- **导入导出**: 导出 JSON（含完整元数据）/ Markdown；导入 JSON 覆盖恢复整个会话（含标题、提示词与参数）。
- **反馈与审计**: 👍/👎 反馈落盘，操作记入 `audit_trail.jsonl`；确定性序列化保障缓存前缀一致；首次写对话文件自动备份。
- **极客工具**: 底层 I/O 日志查看器（发往 API 的原始 payload）、Raw 源码模式、API 自动重连（指数退避、次数可配，覆盖流式中途断流）。

## 项目结构

```text
.
├── app.py                    # Streamlit 主应用入口
├── config_manager.py         # 配置与常量管理
├── provider_registry.py      # 供应商注册表、OpenAI client 工厂
├── compression_engine.py     # 上下文压缩引擎
├── file_utils.py             # 文件解析服务 (PDF/TXT/HTML)
├── token_utils.py            # Token 计算与估算 (Gemini 精确 + 全 OpenAI 兼容估算)
├── chat_utils.py             # 对话历史、工作区管理 & 共用联网搜索
├── browse_tools.py           # AI 自主上网工具集（搜索降级链/网页抓取/边查边答）
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

### 3. 启动应用

```bash
streamlit run app.py
```

Windows 用户也可直接双击 `启动沙盒.bat` 启动（脚本会自动检查 Streamlit 是否安装）。

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