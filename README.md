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
AMAP_KEY=你的高德APIKey
TRAVEL_AGENT_ENABLE_DISTANCE=0
```

3. 启动应用：

```powershell
streamlit run app.py
```

4. 浏览器打开：

```text
http://localhost:8501
```

## Streamlit Cloud 部署

在 Streamlit Cloud 的 `Secrets` 中配置：

```toml
ZHIPU_API_KEY = "你的智谱APIKey"
ZHIPU_MODEL = "glm-4-flash"
ZHIPU_EMBEDDING_MODEL = "embedding-3"
AMAP_KEY = "你的高德APIKey"
TRAVEL_AGENT_ENABLE_DISTANCE = "0"
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

## 面试介绍

这个项目可以这样介绍：

> 我做了一个基于 LangChain 和 LangGraph 的旅行规划 Agent。LangChain 负责封装智谱大模型调用和高德地图工具，LangGraph 负责编排 Query Rewriting、需求解析、工具调用、行程生成、多轮修改和普通追问等节点。前端使用 Streamlit，支持实时进度展示、token 统计和多轮上下文对话。

如果强调 RAG 版本，可以这样介绍：

> 我在项目中加入了本地 RAG 知识库，使用 Markdown 存储城市攻略和通用旅行规划经验，按标题切分文档后调用智谱 Embedding 生成语义向量，并写入 Chroma 本地向量数据库。行程生成前，LangGraph 会先执行 RAG 检索召回相关片段，再把 `rag_context` 注入行程生成节点。

## 注意事项

- 不要把 `.env` 或真实 API Key 提交到 GitHub。
- 如果 Streamlit Cloud 报密钥缺失，优先检查 Secrets 是否为合法 TOML 格式。
- 高德路线距离查询会增加耗时，可以通过 `TRAVEL_AGENT_ENABLE_DISTANCE=0` 关闭。
- 模型输出 JSON 偶尔可能格式不稳定，项目中已做基础 JSON 修复与行程后处理。
- RAG 知识库默认使用 Chroma 本地向量数据库；如果 embedding 或 Chroma 不可用，会自动退回 TF-IDF 检索，保证应用仍可运行。
