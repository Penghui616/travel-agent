# 旅行规划 Agent

一个基于 Streamlit、LangChain、LangGraph、RAG、智谱大模型和高德地图 API 的多轮旅行规划 Agent。用户可以用自然语言提出旅行需求，系统会自动理解目的地、天数、偏好和特殊要求，结合本地攻略知识库、天气、景点、餐厅、酒店等工具结果生成结构化行程，并支持继续对话修改已有方案。

## 功能特点

- 多轮对话：支持首次生成行程，也支持基于已有行程继续修改。
- Query Rewriting：先优化用户输入，再进入需求解析和行程生成。
- 结构化需求解析：将自然语言转换为城市、天数、预算、偏好、交通方式等字段。
- 高德工具调用：查询天气、景点、餐厅、酒店和路线距离。
- LangChain 工具层：高德能力封装为 LangChain tools。
- LangGraph 工作流：用状态图编排首次生成、行程修改和普通追问。
- RAG 知识检索：从本地旅行攻略知识库中检索相关片段，补充路线节奏和避坑建议。
- 长期记忆：自动记录用户偏好、避雷点、出门时间、交通方式等，后续新行程自动作为软约束注入。
- 进度反馈：Streamlit 页面展示当前执行阶段。
- 耗时与 token 统计：记录每次流程的阶段耗时和模型 token 消耗。

## 技术栈

- 前端与交互：Streamlit
- LLM 框架：LangChain
- Agent 编排：LangGraph
- RAG 检索：本地 Markdown 知识库 + 智谱 Embedding + Chroma 向量数据库
- 大模型：智谱 GLM
- 地图与 POI：高德开放平台 API
- 后端语言：Python

## 项目结构

```text
travel_agent/
├── app.py                         # Streamlit 主入口
├── requirements.txt               # Python 依赖
├── data/
│   └── travel_knowledge/           # 本地 RAG 攻略知识库
├── utils/
│   ├── langgraph_workflow.py      # LangGraph 主工作流
│   ├── langchain_llm.py           # LangChain 智谱模型适配层
│   ├── langchain_tools.py         # LangChain 工具封装
│   ├── rag_retriever.py           # 本地 RAG 检索器
│   ├── travel_graph.py            # 旅行工具调度
│   ├── amap_tools.py              # 高德 API 工具
│   ├── llm_parser.py              # 旅行需求解析
│   ├── llm_query_rewriter.py      # Query Rewriting
│   ├── llm_followup.py            # 多轮修改与追问
│   ├── llm_itinerary.py           # 行程生成与后处理
│   ├── llm_planner.py             # 本地任务规划
│   ├── context_compactor.py       # 多轮对话上下文压缩
│   ├── user_memory.py             # 长期用户偏好记忆
│   ├── token_usage.py             # token 统计
│   └── config.py                  # 本地 .env / Streamlit Secrets 配置读取
└── test_amap.py                   # 高德工具测试脚本
```

## LangGraph 流程

首次生成行程：

```text
rewrite_initial
  -> parse_initial
  -> plan_trip
  -> run_tools
  -> retrieve_knowledge
  -> generate_itinerary
```

继续修改行程：

```text
rewrite_update
  -> update_request
  -> plan_trip
  -> run_tools
  -> retrieve_knowledge
  -> generate_itinerary
```

普通追问：

```text
rewrite_followup
  -> answer_followup
```

## 本地运行

1. 安装依赖：

```powershell
cd "D:\OneDrive - udmercy.edu\桌面\1\travel_agent"
python -m pip install -r requirements.txt
```

2. 新建 `.env` 文件：

```env
ZHIPU_API_KEY=你的智谱APIKey
ZHIPU_MODEL=glm-4-flash
ZHIPU_EMBEDDING_MODEL=embedding-3
ZHIPU_TIMEOUT_SECONDS=120
ZHIPU_MAX_RETRIES=1
ZHIPU_PREFER_OFFICIAL_SDK=1
AMAP_KEY=你的高德APIKey
TRAVEL_AGENT_ENABLE_DISTANCE=0
TRAVEL_AGENT_FAST_ITINERARY=1
TRAVEL_AGENT_ATTRACTION_POI_LIMIT=40
TRAVEL_AGENT_MAX_TOOL_WORKERS=2
TRAVEL_AGENT_MAX_POI_SEARCH_WORKERS=1
TRAVEL_AGENT_AMAP_MIN_REQUEST_INTERVAL=0.35
TRAVEL_AGENT_FOLLOWUP_HISTORY_LIMIT=4
TRAVEL_AGENT_CONTEXT_ITEMS_PER_DAY=5
TRAVEL_AGENT_ENABLE_MEMORY=1
```

3. 启动应用：

```powershell
streamlit run app.py
```

4. 浏览器打开：

```text
https://travel-agent-penghui.streamlit.app/
```

## Streamlit Cloud 部署

在 Streamlit Cloud 的 `Secrets` 中配置：

```toml
ZHIPU_API_KEY = "你的智谱APIKey"
ZHIPU_MODEL = "glm-4-flash"
ZHIPU_EMBEDDING_MODEL = "embedding-3"
ZHIPU_TIMEOUT_SECONDS = "120"
ZHIPU_MAX_RETRIES = "1"
ZHIPU_PREFER_OFFICIAL_SDK = "1"
AMAP_KEY = "你的高德APIKey"
TRAVEL_AGENT_ENABLE_DISTANCE = "0"
TRAVEL_AGENT_FAST_ITINERARY = "1"
TRAVEL_AGENT_ATTRACTION_POI_LIMIT = "40"
TRAVEL_AGENT_MAX_TOOL_WORKERS = "2"
TRAVEL_AGENT_MAX_POI_SEARCH_WORKERS = "1"
TRAVEL_AGENT_AMAP_MIN_REQUEST_INTERVAL = "0.35"
TRAVEL_AGENT_FOLLOWUP_HISTORY_LIMIT = "4"
TRAVEL_AGENT_CONTEXT_ITEMS_PER_DAY = "5"
TRAVEL_AGENT_ENABLE_MEMORY = "1"
```

部署时入口文件填写：

```text
app.py
```

如果应用在仓库子目录中，部署页面的 main file path 仍然应指向该仓库内的 `app.py`。

## 推荐测试用例

```text
去重庆玩三天
再加一天
我习惯下午出门
第三天轻松一点
这个行程适合带老人吗
```



## 注意事项

- 不要把 `.env` 或真实 API Key 提交到 GitHub。
- 如果 Streamlit Cloud 报密钥缺失，优先检查 Secrets 是否为合法 TOML 格式。
- 高德路线距离查询会增加耗时，可以通过 `TRAVEL_AGENT_ENABLE_DISTANCE=0` 关闭。
- 如果高德报 `CUQPS_HAS_EXCEEDED_THE_LIMIT`，说明请求太密；项目默认已做全局限速和工具降级，可适当增大 `TRAVEL_AGENT_AMAP_MIN_REQUEST_INTERVAL`。
- 默认启用 `TRAVEL_AGENT_FAST_ITINERARY=1`，最终行程用本地结构化生成器从真实 POI 组装，避免 5-7 天大 JSON 生成拖到几分钟。
- 多轮追问默认只传压缩后的行程摘要、工具摘要和最近 4 条对话，降低输入 token；可通过 `TRAVEL_AGENT_FOLLOWUP_HISTORY_LIMIT` 和 `TRAVEL_AGENT_CONTEXT_ITEMS_PER_DAY` 调整。
- 长期记忆默认保存在本地 `data/user_memory.json`，该文件已加入 `.gitignore`，不会提交到 GitHub。
- 默认优先使用智谱官方 SDK 调用模型，减少 LangChain 包装层偶发读超时；如需强制测试 LangChain ChatModel，可设置 `ZHIPU_PREFER_OFFICIAL_SDK=0`。
- 模型输出 JSON 偶尔可能格式不稳定，项目中已做基础 JSON 修复与行程后处理。
- RAG 知识库默认使用 Chroma 本地向量数据库；如果 embedding 或 Chroma 不可用，会自动退回 TF-IDF 检索，保证应用仍可运行。
- 如果云端偶发 `read operation timed out`，可以适当增大 `ZHIPU_TIMEOUT_SECONDS`，或者降低 `TRAVEL_AGENT_MAX_CONTEXT_ATTRACTIONS_CAP`、`TRAVEL_AGENT_MAX_RAG_CHUNKS` 等上下文上限。
