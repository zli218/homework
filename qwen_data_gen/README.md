# 本地代码仓智能训练数据生成系统设计文档

## 1. 项目背景与目标

在机器学习领域，高质量的训练数据是模型性能的关键。随着企业对 AI 应用需求的增长，如何利用**本地私有代码仓库**中的业务流程和规则来微调模型变得至关重要。

本任务旨在设计并实现一套自动化系统框架，用于生成和处理训练数据，支持基于 **Qwen 2.5** 系列模型的微调。该系统需使模型具备以下能力：
1.  回答关于本地代码仓的业务流程和规则的问题。
2.  基于代码仓架构给出合理的设计方案。

## 2. 系统架构设计

本系统采用模块化的流水线架构，确保数据的自动化生成、处理和验证。

1.  **Repo Loader (仓库加载器)**: 负责从本地或远程 Git 仓库加载代码，处理文件遍历。
2.  **Code Parser (代码解析器)**: 基于 AST (抽象语法树) 对 Python 代码进行语义切分，提取类 (Class) 和函数 (Function) 作为独立的数据单元 (Chunk)，保留代码的结构完整性。
3.  **Data Generator (数据生成器)**: 核心模块，利用 Teacher Model (如 Gemini/GPT) 基于特定 Prompt 模板生成数据。
    *   集成 **RAG (检索增强)** 模块，利用 `sentence-transformers` 检索相关代码上下文，增强 Prompt 的丰富度。
    *   集成 **Critic (质量审查)** 模块，对生成结果进行自动打分和修正。
4.  **Formatter (格式化器)**: 将生成的数据清洗并转换为适配 Qwen 模型微调的 JSONL 格式。
5.  **Validator (验证器)**: 使用小参数量模型 (Qwen2.5-0.5B) 进行快速微调测试，验证数据格式和收敛性。

## 3. 训练数据集结构定义

为了满足两个核心场景的需求，我们定义了如下的数据集结构。所有数据均包含**代码原文**和**推理过程**。

### 3.1 通用字段
*   `id`: 样本唯一标识符 (UUID)。
*   `scenario`: 场景标签 (`business_logic` 或 `architecture_design`)。
*   `code_ref`: 代码来源文件路径，用于追溯。
*   `conversation`: 符合 ChatML 格式的对话列表。

### 3.2 场景 1：业务流程与规则问答 (Business Logic QA)
*   **目标**: 训练模型理解本地代码的具体业务逻辑和实现规则。
*   **User 输入**: 针对特定代码片段的业务逻辑提问（例如：“这段代码如何处理支付失败？”）。
*   **Assistant 输出**:
    1.  **相关代码 (Original Code)**: **必须包含**来自代码仓的原始代码段，作为回答的依据。
    2.  **推理过程 (Reasoning Trace)**: 详细解释如何从代码中推导出答案的逻辑步骤。
    3.  **最终答案**: 针对问题的直接回答。

**数据样本示例**:
```json
{
  "scenario": "business_logic",
  "conversation": [
    { "role": "user", "content": "请分析 allocate 函数是如何处理库存不足的情况的？" },
    { "role": "assistant", "content": "相关代码:\n```python\n# (此处为代码仓中的原始代码片段)\ndef allocate(self, line):\n    ...\n```\n\n推理过程 (Reasoning Trace):\n1. 分析 `allocate` 方法，发现它首先调用 `can_allocate` 检查库存。\n2. 如果返回 False，代码抛出 `OutOfStock` 异常。\n...\n\n最终答案:\n当库存不足时，该函数会..." }
  ]
}
```

### 3.3 场景 2：架构设计方案 (Architecture Design)
*   **目标**: 训练模型基于现有架构提出新的设计或重构方案。
*   **User 输入**: 一个假设的新需求（例如：“如何为该模块增加审计日志功能？”）。
*   **Assistant 输出**:
    1.  **相关代码**: 涉及的核心类或模块的原始代码。
    2.  **推理过程 (Reasoning Trace)**: 分析现有架构的优缺点，解释选择特定设计模式（如装饰器、观察者模式）的理由。
    3.  **设计方案**: 具体的代码修改建议或架构图描述。

## 4. 多样性与代表性保障策略

为防止模型过拟合单一模式，确保数据的泛化能力，我们采取以下措施：

1.  **🎭 多样化人设 (Persona Rotation)**:
    *   生成器会随机切换“专家人设”（如：DDD 架构师、安全审计专家、Python 性能专家）。
    *   不同人设关注代码的不同侧面（如架构师关注解耦，安全专家关注漏洞），从而生成多角度的问答数据。

2.  **🌡️ 动态温度 (Dynamic Temperature)**:
    *   在调用 Teacher Model 生成数据时，随机调整 Temperature 参数 (0.7 - 0.9)。
    *   这增加了生成文本的语言风格和表达方式的多样性。

3.  **🧠 思维链增强 (CoT Enforcement)**:
    *   强制要求输出 `Reasoning Trace`。这不仅提高了数据的含金量，也迫使 Teacher Model 进行更深层的逻辑思考，而非简单的文本补全。

4.  **🔍 RAG 上下文注入**:
    *   利用 `sentence-transformers` 检索项目中的相关代码片段。
    *   这模拟了真实开发中“查看定义”和“查找引用”的行为，使模型能理解跨文件的代码依赖关系。

## 5. 快速验证 (Validation)

为了验证生成数据的有效性，本项目包含一个轻量级微调脚本 `finetune_quick_test.py`。

*   **模型**: Qwen2.5-0.5B-Instruct (参数量小，可在普通显卡甚至 CPU 上快速运行)。
*   **方法**: LoRA (Low-Rank Adaptation)。
*   **目标**: 验证数据格式是否正确，Loss 是否正常下降，以及模型是否学会了输出“推理过程”和“相关代码”的格式。

## �️ 环境准备

*   Python 3.8+
*   有效的 Google Gemini API Key (可在 Google AI Studio 免费获取)
*   Git 环境

## 📦 安装

1.  安装项目依赖：

```bash
pip install -r requirements.txt
```

## ⚙️ 配置

1.  在项目根目录创建 `.env` 文件，填入你的 API Key 和模型名称：

```dotenv
# Google Gemini API Key
GOOGLE_API_KEY=your_api_key_here

# 模型名称 (建议使用 gemini-2.5-flash)
MODEL_NAME=gemini-2.5-flash
```

2.  (可选) 修改 `config.py` 中的配置项：
    *   `REPO_URL`: 目标代码仓库地址 (默认是一个 Python 架构示例仓库)。
    *   `MAX_SAMPLES`: 生成数据的目标数量。
    *   `OUTPUT_FILE`: 输出文件名。
    *   `RAG_ENABLED`: 是否启用 RAG 检索增强 (默认 True)。
    *   `CRITIC_ENABLED`: 是否启用自我修正机制 (默认 True)。
    *   `ENGLISH_RATIO`: 生成数据中英文样本的比例 (0.0 - 1.0, 默认为 0.3)。

## 🚀 使用指南

### 1. 生成数据

运行主程序开始生成数据。程序会自动克隆默认的 `Cosmic Python` 仓库（或你在 config 中配置的仓库），解析代码，并生成问答对。

```bash
python main.py
```

程序将执行以下步骤：
1.  克隆或更新目标代码仓库到本地临时目录。
2.  解析所有 Python 文件，基于 AST 提取函数和类级别的代码片段 (Chunks)。
3.  调用 Gemini API，根据随机人设和场景生成问答对，并**强制包含原始代码段**。
4.  结果实时保存到 `qwen_finetune_data.jsonl`。

### 2. 验证数据 (小模型 Qwen2.5-0.5B)

生成数据后，可以使用内置的微调脚本快速验证数据质量：

```bash
python finetune_quick_test.py
```

该脚本会：
*   加载 `Qwen/Qwen2.5-0.5B-Instruct` 模型 (显存占用极低)。
*   加载生成的 `qwen_finetune_data.jsonl` 数据集。
*   使用 LoRA 进行极少步数的微调训练。
*   如果 Loss 正常下降，说明数据格式有效，可以直接用于大规模训练。

## 📂 项目结构

```text
.
├── main.py                 # 主入口：负责流程编排
├── config.py               # 配置文件
├── requirements.txt        # 项目依赖
├── .env                    # 环境变量 (需自行创建)
├── finetune_quick_test.py  # 微调验证脚本
├── src/
│   ├── repo_loader.py      # Git 仓库加载器
│   ├── code_parser.py      # AST 代码解析器
│   ├── generator.py        # Gemini API 调用与 Prompt 构建
│   └── utils.py            # 工具函数 (JSONL 保存)
└── qwen_finetune_data.jsonl # 生成的数据文件 (输出)
```

## ⚠️ 注意事项

*   **API 限制**: 请注意 Google Gemini API 的速率限制 (Rate Limit)。程序内置了指数退避重试机制，但在大量生成时仍需留意。
*   **网络问题**: 国内用户可能需要配置代理才能访问 Google API 和 Hugging Face。但可以通过替换api（比如替换为qwen3.0等来解决，需要更新config里的api访问流程和env里的api key）。

    *   设置代理: `export HTTPS_PROXY=http://127.0.0.1:7890` (根据实际情况调整)
    *   HF 镜像: `export HF_ENDPOINT=https://hf-mirror.com`

## 📖 技术文档与设计理念

### 1. 为什么这样设计？(Design Rationale)

本项目的设计核心在于解决大模型在特定领域（如软件架构、复杂代码逻辑）微调时面临的**高质量数据匮乏**问题。

*   **基于 AST 的上下文提取**: 简单的文本切片 (Chunking) 往往会破坏代码的语义结构。本项目使用 Python 的 `ast` 模块解析代码，以**类 (Class)** 和 **函数 (Function)** 为单位提取片段，确保了输入给生成模型的上下文是语义完整的。
*   **多样化人设 (Persona-based Generation)**: 为了避免生成的数据千篇一律，我们引入了“角色扮演”机制。同一个代码片段，安全专家关注的是漏洞，而架构师关注的是解耦。这种策略显著增加了数据的多样性和覆盖面。
*   **RAG 检索增强 (Retrieval-Augmented Generation)**: 单个代码片段往往缺乏项目全局视角。通过引入 RAG，我们在生成 Prompt 时检索项目中的相关代码依赖，为模型提供更丰富的上下文，减少幻觉并提升回答的准确性。
*   **自我修正机制 (Self-Correction/Critic)**: LLM 生成的数据难免存在瑕疵。我们引入了一个独立的 Critic 代理，模拟代码审查流程，对生成结果进行打分和修正，确保只有高质量的数据进入最终数据集。

### 2. 技术选型思考

*   **Teacher Model: Google Gemini**: 选择 Gemini 作为教师模型是因为其在长文本理解和逻辑推理方面的优势，且 API 响应速度快，适合大规模数据生成任务。
*   **Validation Model: Qwen2.5-0.5B**: 在数据生成流水线中引入一个极小的模型进行“冒烟测试”是工程上的最佳实践。它能在几分钟内验证数据格式是否符合 ChatML 标准，以及 Loss 是否收敛，极大地降低了试错成本。
*   **GitPython & AST**: 直接操作 Git 仓库和语法树，保证了数据源的真实性和解析的准确性，避免了基于正则匹配的脆弱性。

### 3. 如何更进一步？(Future Work)

将本项目用于生产环境或更大规模的训练，可以考虑以下优化方向：

*   **多轮对话生成**: 目前主要是单轮 QA。可以扩展逻辑，让 Teacher Model 模拟用户和助手之间的多轮交互，生成更符合真实场景的对话数据。（由于免费api的token和每日次数限制，这里无法针对此问题进行更进一步的设计和实现）
*   **DPO/RLHF 数据**: 生成“好”与“坏”的回答对，用于后续的偏好对齐训练。
*   **同源蒸馏**：采用同源的更大规模大模型（如qwen3.0 api），以获得更好的效果