# wiki_multi_hop_data_pipeline
基于wiki数据集，从单条数据开始，基于“实体-标题”的精准匹配，构建一个有向知识图谱，从图中生成并采样指定跳数的推理路径，用于合成多跳任务。

# 维基百科知识图谱与推理路径生成器——graph_spacy_R.py

本项目用于处理维基百科的文本转储数据，自动构建一个知识图谱，并从中抽取出高质量、多跳的推理路径，最终生成一个可用于训练语言模型的数据集。

## 功能特性

- **数据处理**: 从Parquet格式的维基百科文件中高效加载和采样数据。
- **并行构建图谱**:根据“实体-标题”的精准匹配关系构建知识图谱。
- **图谱缓存**: 首次构建的图会保存为GML文件，后续运行可直接加载，节省时间。
- **高质量路径生成**:
  - 生成指定**跳数**的推理路径。
  - 通过**(文章组合 + 桥梁实体组合)**的复合指纹确保路径的**唯一性**。
  - 过滤掉内部“桥梁”实体过于相似的低质量路径。
- **灵活配置**: 所有关键参数均可通过命令行进行配置，无需修改代码。

## 如何运行

通过命令行运行 `graph_spacy_R.py` 脚本。您需要指定数据路径和输出路径。

### 基本示例

以下命令将从 `/path/to/your/wiki/data` 目录中读取前5个Parquet文件，处理10万篇文章，构建图谱，并生成5万条2跳的推理路径，结果保存在 `/path/to/output`。

```bash
python graph_spacy_R.py \
    --wiki_data_path "/path/to/your/wiki/data" \
    --output_path "/path/to/output" \
    --num_files 5 \
    --num_samples 100000 \
    --num_paths_to_generate 50000 \
    --num_hops 2 \
    --gpus "0,1"
```

### 命令行参数详解

您可以通过 `--help` 查看所有可用参数：```bash
python graph_spacy_R.py --help
```

- `--wiki_data_path`: **(必需)** 维基百科Parquet文件所在的目录。
- `--output_path`: **(必需)** 输出结果（图文件、数据集）的保存目录。
- `--start_file_index`: 从文件列表的哪个索引开始读取 (默认: `0`)。
- `--num_files`: 读取文件的数量 (默认: `1`, `-1`表示全部)。
- `--num_samples`: 用于处理的文章采样数量 (默认: `100000`, `-1`表示全部)。
- `--num_paths_to_generate`: 最终生成的数据集大小 (默认: `100000`)。
- `--num_hops`: 每条推理路径的跳数 (默认: `2`)。
- `--gpus`: 指定使用的GPU ID，用逗号分隔 (默认: `"0,1,2,3"`)。
- `--processes_per_gpu`: 每个GPU上启动的进程数 (默认: `10`)。
- `--force_rebuild_graph`: 如果设置此标志，将强制重新构建图，即使已存在缓存文件。
- ...以及其他微调参数。

## 输出格式

脚本会生成两个主要文件：

1.  `knowledge_graph.gml`: 使用 `networkx` 保存的图文件，可用于后续分析。
2.  `unique_{N+1}R_{N}hops.jsonl`: 生成的数据集，其中 `N` 是跳数。这是一个 [JSON Lines](https://jsonlines.org/) 文件，每一行都是一个JSON对象，代表一条推理路径。

**单条路径的JSON结构示例 (2跳):**
```json
{
  "document1": {
    "id": "123",
    "title": "Article A",
    "text": "Text of article A, which mentions the entity 'Bridge OneTwo'."
  },
  "connection_d1_d2": {
    "bridge": "Bridge OneTwo",
    "edit_distance": 0
  },
  "document2": {
    "id": "456",
    "title": "Bridge OneTwo",
    "text": "Text of article B (title is 'Bridge OneTwo'), which mentions 'Bridge TwoThree'."
  },
  "connection_d2_d3": {
    "bridge": "Bridge TwoThree",
    "edit_distance": 0
  },
  "document3": {
    "id": "789",
    "title": "Bridge TwoThree",
    "text": "Text of article C (title is 'Bridge TwoThree')."
  }
}
```

# 知识路径证据提炼器 (Knowledge Path Refiner)——nR_deal.py

本项目是一个命令行工具，作为多阶段数据处理流水线的中间步骤。它的主要功能是**精炼**由上游路径生成器产出的知识路径，为下游的问答生成模型**准备高质量、低噪声的上下文**。

## 项目目标

大型语言模型在处理长文本时会面临上下文窗口限制、计算成本高昂以及“迷失在中间”等问题。直接将整篇维基百科文章作为上下文喂给模型，效果往往不佳。

本脚本旨在解决这一问题，通过以下方式对原始路径进行“降噪”和“聚焦”：

1.  **提取核心证据**：代替使用全文，脚本会智能地从每篇文章中提取出与推理链直接相关的段落。
2.  **控制上下文长度**：所有提取的证据片段都必须符合预设的长度限制，确保输入给下游模型的上下文简洁有效。
3.  **保证路径完整性**：如果路径中的任何一环缺失了有效的证据，整条路径将被丢弃，以确保生成数据的质量。

## 功能详解

- **动态跳数处理**：无需指定，脚本能自动处理任意长度（N-hop）的推理路径。
- **上下文感知提取**：
  - **起点**：提取与文章标题和第一个“桥梁实体”相关的段落。
  - **中间节点**：提取能连接“上一个桥梁”和“下一个桥梁”的段落，优先寻找同时包含二者的段落。
  - **终点**：提取与最后一个“桥梁实体”相关的段落。
- **严格的长度过滤**：可以自定义每个证据片段的最大Token数，所有超长片段都会被舍弃。


### 命令格式

```bash
python nR_deal.py --input-file <path_to_input.jsonl> --output-file <path_to_output.jsonl> [OPTIONS]
```

### 示例

```bash
python nR_deal.py \
    --input-file ./raw_paths/unique_4R_3hops.jsonl \
    --output-file ./refined_paths/unique_4R_data.jsonl \
    --max-tokens-per-snippet 1024
```
这个命令会读取 `unique_4R_3hops.jsonl` 文件，提取所有证据片段（每个片段不超过1024个token），并将结果保存到 `unique_4R_data.jsonl`。

### 命令行参数

- `--input-file` (必需): 输入的JSONL文件路径，包含原始的多跳知识路径。
- `--output-file` (必需): 输出的JSONL文件路径，用于保存提炼后的路径数据。
- `--max-tokens-per-snippet` (可选, 默认: `1024`): 每个提取出的证据片段允许的最大Token数（通过空格分割进行简单估算）。

## 数据格式转换示例

### 输入 (`input-file`)

```json
{
  "document1": {"id": "1", "title": "A", "text": "Full text of document A... mentions Bridge1..."},
  "connection_d1_d2": {"bridge": "Bridge1"},
  "document2": {"id": "2", "title": "Bridge1", "text": "Full text of document B... mentions Bridge1 and Bridge2..."},
  "connection_d2_d3": {"bridge": "Bridge2"},
  "document3": {"id": "3", "title": "Bridge2", "text": "Full text of document C... mentions Bridge2..."}
}
```

### 输出 (`output-file`)

处理后，会在原始数据的基础上增加 `R1`, `R2`, `R3` 字段。

```json
{
  "document1": {"id": "1", "title": "A", "text": "..."},
  "connection_d1_d2": {"bridge": "Bridge1"},
  "document2": {"id": "2", "title": "Bridge1", "text": "..."},
  "connection_d2_d3": {"bridge": "Bridge2"},
  "document3": {"id": "3", "title": "Bridge2", "text": "..."},
  "R1": ["The title is: A. Relevant paragraph from document A containing Bridge1."],
  "R2": ["Relevant paragraph from document B connecting Bridge1 and Bridge2."],
  "R3": ["Relevant paragraph from document C containing Bridge2."]
}
```


# 多跳问答数据集生成器 (逆向推理)——generate_qa.py

本项目提供了一个强大的工具，用于将结构化的“推理路径”数据自动转换为高质量的多跳问答（Multi-hop QA）数据集。它通过与大型语言模型（LLM）交互，采用一种创新的“逆向推理”流水线来生成问题链。

## 核心思想：逆向推理

传统的QA生成可能是从一个问题出发寻找答案。本项目反其道而行之：

1.  **输入**：一条已知的知识路径，例如 `文档A -> 文档B -> 文档C`。
2.  **初始化**：首先只看路径的终点（`文档B -> 文档C`），让LLM根据这个子路径生成一个最终答案（**A**）和一个只能被`文档C`回答的初始问题（**Q_initial**）。
3.  **迭代替换**：然后，程序向后移动一步，将`Q_initial`和`文档B`作为上下文，要求LLM将问题改写成一个需要`文档B`才能回答的新问题（**Q_new**）。
4.  **循环**：重复此过程，直到路径的起点。

最终，对于一条 N 跳的路径，我们会得到一个最终答案和 N+1 个问题，形成一个从复杂到简单的问题链。

## 主要功能

- **自动化QA生成**：将推理路径（JSONL格式）批量转换为结构化的QA对。
- **并行处理**：利用多进程架构，能够高效处理大规模数据集。
- **与本地LLM集成**：专为连接本地部署的OpenAI兼容API服务（如 VLLM）而设计，支持多实例负载均衡。
- **“思考过程”记录**：如果LLM支持，脚本能够捕获并保存模型在生成每一步时的详细思考过程（`<think>...</think>`），极大地便利了Prompt调试和模型行为分析。
- **健壮的错误处理**：自动分离处理成功和失败的数据，方便问题排查和任务重试。
- **高度可配置**：所有参数，包括文件路径、模型名称、API端点和生成参数，均可通过命令行灵活配置。

## 使用方法

### 1. 准备输入文件

- **推理路径文件 (`input.jsonl`)**: 每一行是一个JSON对象，包含`documentN`和`connection_dN_dN+1`等字段。
- **Prompt 模板**:
  - `prompt1_initializer_cn.md`: 用于流水线第一步的模板。
  - `prompt2_replacer_cn.md`: 用于迭代步骤的模板。

### 2. 运行脚本

通过命令行启动生成过程。

**基本示例：**

```bash
python generate_qa.py \
    --input_file /path/to/your/paths.jsonl \
    --output_path /path/to/results/successful.jsonl \
    --failed_output_path /path/to/results/failed.jsonl \
    --prompt1_file /path/to/prompts/prompt1_initializer_cn.md \
    --prompt2_file /path/to/prompts/prompt2_replacer_cn.md \
    --model_name "qwen3" \
    --num_workers 64 \
    --base_port 9668 \
    --num_endpoints 6
```

## 输出格式

脚本会生成两个 `.jsonl` 文件。

- **成功文件 (`successful.jsonl`)**: 每一行是一个包含以下字段的JSON对象：
  - `A`: (str) LLM生成的最终答案。
  - `Q_chain`: (list) 一个问题列表，每个元素是 `{"level": int, "question": str}`。
  - `context`: (dict) 原始的输入数据项。
  - `debug_info`: (list) 包含LLM在每个生成步骤的详细思考过程和原始响应。
- **失败文件 (`failed.jsonl`)**: 包含处理失败的条目，其中包含一个 `error` 字段来描述失败原因。
