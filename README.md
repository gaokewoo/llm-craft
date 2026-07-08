# llm-craft

从零开始构建大型语言模型（LLM）的学习实践项目。基于 PyTorch 从底层组件逐步实现 GPT 架构，并完成预训练权重加载、文本生成、分类微调与指令微调等完整工作流。

项目参考自 Sebastian Raschka 所著《Build a Large Language Model From Scratch》，所有核心模块（注意力、前馈网络、LayerNorm、Transformer Block、GPT 模型等）均为手写实现，便于深入理解 LLM 的内部原理。

## 目录结构

```
llm-craft/
├── model/              # GPT 模型核心实现（从零搭建）
├── tools/              # 通用工具函数（生成、损失、日志、绘图）
├── data/               # 数据下载与输出产物
│   ├── tools/          #   各类数据/权重下载脚本
│   └── output/         #   下载的数据、训练产物、模型权重、图表
├── demo/               # 各组件的独立演示脚本
│   └── others/         #   注意力机制的渐进式实现与数据加载器
├── llm_example/        # 完整的端到端使用示例
│   └── gpt_example/
│       ├── gpt_model/          # 加载 OpenAI GPT-2 预训练权重
│       ├── verdict_example/    # 在《The Verdict》上从零训练 GPT
│       ├── spam_example/       # SMS 垃圾短信分类微调
│       └── instruction_example/# 指令微调（Instruction Fine-tuning）
├── test/               # 单元测试
├── pyproject.toml      # 项目配置
└── README.md
```

### `model/` — GPT 模型核心实现

从零实现的 GPT 架构，每个文件对应一个核心组件：

| 文件 | 功能 |
|------|------|
| `config.py` | GPT-2 124M 模型配置（词汇表 50257、上下文长度 1024、嵌入维度 768、12 头、12 层） |
| `gelu.py` | GELU 激活函数（tanh 近似） |
| `layer_norm.py` | 层归一化（LayerNorm） |
| `feed_forward.py` | 前馈神经网络（Linear → GELU → Linear，维度扩展 4 倍） |
| `multi_head_attention.py` | 多头因果注意力（含因果掩码与缩放点积） |
| `transformer_block.py` | Transformer Block（Pre-LN + 注意力 + 前馈 + 残差连接） |
| `gpt_model.py` | 完整 GPT 模型（词嵌入 + 位置嵌入 + N 层 Transformer + 输出头） |
| `trainer.py` | 训练循环（`train_model_simple`）、评估、采样生成、模型保存 |

### `tools/` — 通用工具

| 文件 | 功能 |
|------|------|
| `generate_text.py` | 文本生成：`generate_text_simple`（贪心）、`generate`（支持 temperature / top-k 采样）、token 与文本互转 |
| `model_tool.py` | 损失计算：`calc_loss_batch`、`calc_loss_loader`（交叉熵） |
| `log.py` | 统一日志记录器 |
| `matplot.py` | 训练损失/准确率曲线绘制（保存至 `data/output/`） |

### `data/` — 数据管理

- **`data/tools/`** — 数据与权重下载脚本：
  - `download_verdict_file.py`：下载《The Verdict》训练文本
  - `download_gpt_model.py`：下载 GPT-2 124M 预训练权重
  - `download_gpt335m_model.py`：下载 GPT-2 355M (medium) 预训练权重（指令微调用）
  - `gpt_download.py`：核心下载逻辑（通过 HuggingFace `transformers` 加载权重，无需 TensorFlow）
  - `download_spam_data.py`：下载并预处理 SMS 垃圾短信数据集
  - `download_instruction_data.py`：下载指令微调数据集
- **`data/output/`** — 存放下载的数据、训练产生的模型权重（`.pth`）、数据文件（`.json` / `.csv`）和图表（`.pdf`）

### `demo/` — 组件演示

每个文件独立演示一个组件的工作原理，适合按顺序学习：

| 文件 | 演示内容 |
|------|----------|
| `bpe_demo.py` | BPE 分词器（tiktoken / GPT-2 编码）与滑动窗口数据构造 |
| `embedding_demo.py` | 词嵌入层基本用法 |
| `embedding256_demo.py` | 256 维嵌入演示 |
| `layer_norm_demo.py` | 层归一化前后均值/方差对比 |
| `gelu_demo.py` | GELU 与 ReLU 激活函数对比 |
| `feed_forward_demo.py` | 前馈网络维度变化 |
| `multi_head_attention_demo.py` | 多头注意力计算 |
| `transformer_block_demo.py` | 单个 Transformer Block 输入输出 |
| `gpt_model_demo.py` | 完整 GPT 模型前向传播与参数统计 |
| `others/self_attention_v1.py` | 自注意力（逐元素实现） |
| `others/self_attention_v2.py` | 自注意力（矩阵实现） |
| `others/casual_attention.py` | 因果注意力（带掩码） |
| `others/compute_attn_*.py` | 注意力分数计算的逐步演示 |
| `others/multi_head_attention_wrapper.py` | 多头注意力的拼接式实现 |
| `others/dataloader_v1.py` | `GPTDatasetV1` 与 `create_dataloader_v1`（滑动窗口数据集） |

### `llm_example/` — 端到端示例

三个完整的实战案例，展示如何使用核心模块完成真实任务：

#### 1. `verdict_example/` — 从零训练 GPT

在小规模文本《The Verdict》上从随机初始化训练 GPT 模型。

- `verdict_dataset.py`：构建训练/验证 DataLoader
- `train_model_by_verdict.py`：训练并保存模型与优化器状态
- `gen_text_with_no_trained_model.py`：未训练模型的生成效果（展示损失/交叉熵概念）
- `gen_text_with_trained_model.py`：加载已训练模型生成文本（支持 top-k / temperature）

#### 2. `spam_example/` — 垃圾短信分类微调

基于预训练 GPT-2 进行分类任务微调（冻结主体 + 替换输出头）。

- `spam_dataset.py`：`SpamDataset`（分词、填充、截断）
- `classifier_model.py`：`get_classifier_model`（冻结所有层 → 替换输出头为 2 分类 → 解冻最后一个 Transformer Block 与 Final Norm）
- `classifier_tools.py`：准确率计算、分类损失、`classify_review` 推理函数
- `train_classifier_simple.py`：分类训练循环
- `train_model_by_classifier.py`：完整训练 + 评估 + 推理 + 保存

#### 3. `instruction_example/` — 指令微调

使用 Alpaca 风格指令数据集对 GPT-2 进行指令微调（SFT）。

- `instruction_tools.py`：`format_input`（构造 Instruction/Input/Response 模板）、`customized_collate_fn`（批量填充与标签掩码）
- `instruction_dataset.py`：`InstructionDataset` 与数据集划分（85% 训练 / 5% 验证 / 10% 测试）
- `train_model_by_instruction.py`：加载 GPT-2 medium (355M) → 微调 → 生成响应 → 保存结果与权重

### `test/` — 单元测试

| 文件 | 测试内容 |
|------|----------|
| `test_gelu.py` | GELU 实现与 PyTorch 内置实现对比 |
| `test_instruction_tools.py` | collate 函数的填充与标签掩码逻辑 |

## 环境要求

- Python ≥ 3.10
- PyTorch
- tiktoken
- numpy
- matplotlib
- pandas
- tqdm
- requests
- transformers（用于加载 GPT-2 预训练权重）

## 安装

```bash
cd llm-craft
pip install -e .
# 安装运行时依赖
pip install torch tiktoken numpy matplotlib pandas tqdm requests transformers
```

## 使用方法

### 第一步：下载数据与预训练权重

按需运行下载脚本（产物保存至 `data/output/`）：

```bash
# 下载《The Verdict》训练文本（从零训练用）
python -m data.tools.download_verdict_file

# 下载 GPT-2 124M 预训练权重（分类微调用）
python -m data.tools.download_gpt_model

# 下载 GPT-2 355M 预训练权重（指令微调用）
python -m data.tools.download_gpt335m_model

# 下载 SMS 垃圾短信数据集（分类任务用）
python -m data.tools.download_spam_data

# 下载指令微调数据集（指令微调用）
python -m data.tools.download_instruction_data
```

### 第二步：运行组件演示（学习阶段）

建议按以下顺序运行 `demo/` 中的脚本，逐步理解各组件：

```bash
python -m demo.bpe_demo              # 1. 分词与数据构造
python -m demo.embedding_demo        # 2. 词嵌入
python -m demo.layer_norm_demo       # 3. 层归一化
python -m demo.gelu_demo             # 4. 激活函数
python -m demo.feed_forward_demo     # 5. 前馈网络
python -m demo.multi_head_attention_demo  # 6. 多头注意力
python -m demo.transformer_block_demo     # 7. Transformer Block
python -m demo.gpt_model_demo        # 8. 完整 GPT 模型
```

### 第三步：运行端到端示例（实战阶段）

#### 示例 A：从零训练 GPT 并生成文本

```bash
# 训练（在《The Verdict》上训练 10 个 epoch，保存至 data/output/model_and_optimizer.pth）
python -m llm_example.gpt_example.verdict_example.train_model_by_verdict

# 使用未训练模型生成（对比效果）
python -m llm_example.gpt_example.verdict_example.gen_text_with_no_trained_model

# 使用已训练模型生成
python -m llm_example.gpt_example.verdict_example.gen_text_with_trained_model
```

#### 示例 B：加载预训练 GPT-2 生成文本

```bash
python -m llm_example.gpt_example.gpt_model.load_gpt_model
```

#### 示例 C：垃圾短信分类微调

```bash
python -m llm_example.gpt_example.spam_example.train_model_by_classifier
```

训练完成后会输出训练/验证/测试准确率，并对示例文本进行 spam / not spam 推理。

#### 示例 D：指令微调

```bash
python -m llm_example.gpt_example.instruction_example.train_model_by_instruction
```

加载 GPT-2 medium (355M)，在指令数据集上微调 2 个 epoch，生成测试响应并保存模型权重与结果。

### 第四步：运行测试

```bash
python -m pytest test/ -v
# 或
python -m unittest discover test
```

## GPT 模型架构说明

本项目实现的 GPT 架构与 OpenAI GPT-2 一致：

```
输入 token IDs
  │
  ├─→ Token Embedding (vocab_size × emb_dim)
  ├─→ Positional Embedding (context_length × emb_dim)
  │         │ (相加)
  │         ▼
  │   Dropout
  │         │
  │   ┌─────┴─────────────────────────────────┐
  │   │  Transformer Block × n_layers         │
  │   │   ├─ LayerNorm                         │
  │   │   ├─ MultiHeadAttention (因果掩码)     │
  │   │   ├─ Dropout + 残差连接                 │
  │   │   ├─ LayerNorm                         │
  │   │   ├─ FeedForward (GELU)                │
  │   │   └─ Dropout + 残差连接                 │
  │   └────────────────────────────────────────┘
  │         │
  │   Final LayerNorm
  │         │
  └─→ Output Head (Linear → vocab_size logits)
```

**GPT-2 124M 配置**（`model/config.py`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `vocab_size` | 50257 | 词汇表大小 |
| `context_length` | 1024 | 最大上下文长度 |
| `emb_dim` | 768 | 嵌入维度 |
| `n_heads` | 12 | 注意力头数 |
| `n_layers` | 12 | Transformer 层数 |
| `drop_rate` | 0.1 | Dropout 概率 |
| `qkv_bias` | False | QKV 是否使用偏置 |

## 学习路线建议

1. **理解数据**：运行 `demo/bpe_demo.py`，学习 BPE 分词与滑动窗口数据构造
2. **理解组件**：依次运行 `embedding` → `layer_norm` → `gelu` → `feed_forward` → `attention` → `transformer_block` 演示
3. **组装模型**：运行 `demo/gpt_model_demo.py`，查看完整模型前向传播与参数量
4. **从零训练**：运行 `verdict_example`，体验完整的训练 → 生成流程
5. **加载预训练**：运行 `load_gpt_model`，学习如何加载 OpenAI 官方 GPT-2 权重
6. **分类微调**：运行 `spam_example`，学习冻结 + 替换输出头的微调范式
7. **指令微调**：运行 `instruction_example`，学习 SFT 指令微调与 Alpaca 模板

## 关键设计说明

- **权重共享（Weight Tying）**：GPT-2 原始架构将词嵌入层作为输出层复用，`gpt_model_demo.py` 中展示了考虑权重共享后的参数量计算。
- **因果掩码**：`MultiHeadAttention` 使用上三角掩码确保当前位置只能关注历史位置。
- **标签掩码**：`customized_collate_fn` 中将填充 token 对应的 target 设为 `-100`（CrossEntropy 的忽略索引），避免填充影响损失。
- **微调策略**：分类任务采用「冻结主体 + 解冻最后一层」的策略，在保持预训练知识的同时适配下游任务。
- **预训练权重加载**：通过 HuggingFace `transformers` 库加载 GPT-2 权重并映射到本项目自实现的模型结构，无需依赖 TensorFlow。

## 常见问题

**Q: 训练时没有 GPU 怎么办？**

代码自动检测设备（`cuda` → `cpu`），CPU 也可运行但速度较慢。指令微调示例建议使用 GPU。

**Q: 下载 GPT-2 权重失败？**

`gpt_download.py` 内置主备两个下载源（OpenAI 公共存储 + Backblaze 镜像），若仍失败需检查网络。权重加载依赖 `transformers` 库，确保已安装。

**Q: `data/output/` 下的文件是什么？**

包含下载的原始数据、训练产生的模型权重（`.pth`）、处理后的数据集（`.json` / `.csv`）以及绘制的损失/准确率曲线图（`.pdf`）。该目录已在 `.gitignore` 中忽略。
