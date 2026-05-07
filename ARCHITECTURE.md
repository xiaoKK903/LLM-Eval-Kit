# LLM-Eval-Kit 架构设计

> 版本：0.2.0 | 最后更新：2026-05-07

---

## 1. 设计哲学

### 核心原则

| 原则 | 说明 | 决策依据 |
|------|------|----------|
| **极致轻量** | 不依赖 torch/transformers/flask 等重型框架 | 目标用户是个人开发者和中小团队，降低上手成本 |
| **开箱即用** | pip install + 三行代码跑评测 | 减少用户决策负担，最快看到结果 |
| **模块可替换** | 适配器/评分器/报表均为接口隔离 | 用户可按需替换任意组件 |
| **业务优先** | 评分维度针对中文客服/问答场景设计 | 不做通用学术 benchmark，做业务可落地工具 |

### 不做的事

- 不涉及多模态（纯文本评测）
- 不重复实现 benchmark 数据集（如 MMLU、C-Eval）
- 不提供可视化后台（第一版聚焦 CLI + HTML 报告）
- 不评测多轮对话（第一版只做单轮）

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                       用户入口                            │
│   CLI (llm-eval-kit eval)   │   Python API (Evaluator)    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    Core 调度层                            │
│               Evaluator (编排管线)                        │
│         load → call → score → compare → report           │
└──────┬──────────┬──────────┬──────────┬─────────────────┘
       │          │          │          │
┌──────▼──┐ ┌─────▼──────┐ ┌▼─────────┐ ┌▼───────────────┐
│ Dataset │ │ Adapters   │ │ Scorers   │ │ Reporter       │
│ Loader  │ │ (模型适配)  │ │ (评分引擎) │ │ (报表输出)      │
├─────────┤ ├────────────┤ ├──────────┤ ├────────────────┤
│ JSONL   │ │ BaseAdapter│ │BaseScorer│ │ConsoleReporter  │
│ 解析     │ │  │         │ │  │       │ │HtmlReporter    │
│ 样本验证 │ │ OpenAICompat│ │RuleScorer│ │ModelComparator │
│ 去重与   │ │  httpx     │ │LLMJudge  │ │JSON 导出       │
│ 采样     │ │  并发控制  │ │ 位置偏差  │ │                │
└─────────┘ └────────────┘ └──────────┘ └────────────────┘
                                       │
                              ┌────────▼────────┐
                              │    Utils          │
                              │  cost_calc.py    │
                              │  common.py        │
                              └─────────────────┘
```

### 数据流

```
用户输入(models, data_path)
        │
        ▼
DatasetLoader.load() ────► List[EvaluationSample]
        │
        ▼
for each model:
  adapter = OpenAICompatibleAdapter(config)
  for each sample:
    resp = await adapter.generate(sample.question)  ────► ModelResponse
    score = await scorer.score(sample, resp)         ────► ScoreResult
    cost = calculate_cost_cny(model, tokens)          ────► float
    result = EvaluationResult(sample, resp, score, cost)
        │
        ▼
ModelComparator.compare_models(results) ────► ComparisonResult
        │
        ├── ConsoleReporter.print_comparison()  ────► 终端表格
        ├── HtmlReporter.generate_report()      ────► HTML 文件
        └── _save_json()                         ────► JSON 文件
```

---

## 3. 模块详解

### 3.1 Dataset — 数据集加载

```
dataset/
└── loader.py
    ├── DatasetLoader          # 主加载器
    └── EvaluationSample       # 样本数据类
```

**输入**：JSONL 文件路径  
**输出**：`List[EvaluationSample]`

**设计要点**：
- 每行一个 JSON 对象，支持 `id`、`question`、`reference`、`expected_keywords` 字段
- 内置校验：缺少必要字段时抛出明确异常
- 支持 `max_samples` 截断，方便快速调试

### 3.2 Adapters — 模型适配器

```
adapters/
├── base.py                   # BaseAdapter 抽象基类
└── openai_compat.py          # OpenAICompatibleAdapter 实现
```

**设计模式**：适配器模式（Adapter Pattern）

**BaseAdapter 接口**：
- `generate(prompt) → ModelResponse`：核心方法，所有适配器必须实现
- `get_cost(token_usage) → float`：成本计算

**OpenAICompatibleAdapter 关键设计**：

```
┌────────────────────────────────────────────┐
│            generate(prompt)                 │
│                    │                        │
│         semaphore.acquire()                 │
│                    │                        │
│         for attempt in range(N):            │
│                    │                        │
│    ┌───────────────┼───────────────┐        │
│    │  success       │ retryable     │ fail   │
│    │  (200)        │ (429/502/504) │ (401)  │
│    │               │               │        │
│    ▼               ▼               ▼        │
│  parse       backoff*2+jitter    raise      │
│  response    → 重试               error     │
│                    │                        │
│         semaphore.release()                  │
│                    │                        │
│         return ModelResponse                 │
└────────────────────────────────────────────┘
```

**并发控制策略（双层 Semaphore）**：

| 层级 | 位置 | 作用 | 默认值 |
|------|------|------|--------|
| Adapter 层 | `OpenAICompatibleAdapter._semaphore` | 单实例并发上限 | 10 |
| Evaluator 层 | `Evaluator.run()` 的 `Semaphore` | 全局并发控制 | 5 |

**重试策略**：

| 条件 | 行为 |
|------|------|
| 200 | 立即返回 |
| 429 Too Many Requests | 指数退避重试 up to 3 次 |
| 502/503/504 | 指数退避重试 up to 3 次 |
| 超时 / 网络错误 | 指数退避重试 up to 3 次 |
| 401/403 | 立即抛出异常（不重试） |

**延迟计算公式**：
```
delay = base_delay × 2^attempt + random(0, delay × 0.3)
```

- `base_delay` = 1.0s
- attempt=0: 1.0~1.3s
- attempt=1: 2.0~2.6s
- attempt=2: 4.0~5.2s

**为什么不用 OpenAI SDK 而用 httpx**：
1. 减少依赖体积（openai SDK ≈ 300KB，httpx ≈ 200KB）
2. 更精细的连接控制（超时、重试、连接池）
3. 统一的 HTTP 接口，不绑定 OpenAI 生态

### 3.3 Scorers — 评分引擎

```
scorers/
├── base.py          # BaseScorer 基类 + ScoreResult 数据类
├── rule_scorer.py   # RuleScorer 规则评分器
└── llm_judge.py     # LLMJudgeScorer LLM 裁判评分器
```

**设计模式**：策略模式（Strategy Pattern）

#### RuleScorer — 规则评分

**四维评分算法**：

```
total_score = 0.4 × keyword_match
            + 0.2 × length_appropriateness
            + 0.2 × structure_quality
            + 0.2 × completeness
```

| 维度 | 计算公式 | 取值范围 |
|------|----------|----------|
| keyword_match | matched_keywords / total_expected_keywords | [0, 1] |
| length_appropriateness | min(len(response), len(reference)) / max(len(response), len(reference)) | [0, 1] |
| structure_quality | 1 if 包含 Markdown 结构化元素 else 0 | {0, 1} |
| completeness | 中文业务指标命中数 / 总指标数 | [0, 1] |

**中文业务完整性指标**（共 10 个）：
总结、建议、步骤、原因、注意、方案、说明、例如、对比、优势

**复杂度**：O(n) 其中 n 为回答长度，毫秒级完成。

#### LLMJudgeScorer — LLM 裁判

**核心问题**：LLM-as-Judge 存在位置偏差（position bias），即模型倾向于选择先出现的答案。

**消除策略**：双评取平均（Two-pass averaging）

```
第一轮：参考在前 + 回答在后  → score1
第二轮：回答在前 + 参考在后  → score2
最终分数 = (score1 + score2) / 2
```

**Judge Prompt 模板**：

```
你是一个专业的AI回答质量评估专家。

## 评测任务
评估AI助手的回答质量。

## 评分维度（每项 1-5 分）
1. 准确性(accuracy)：回答是否准确
2. 完整性(completeness)：是否覆盖了所有要点
3. 简洁性(conciseness)：是否简洁明了

## 输出格式
{ "accuracy": N, "completeness": N, "conciseness": N, "reasoning": "..." }
```

**JSON 解析策略**：
1. 优先 `json.loads()` 直接解析
2. 失败则用正则提取 `{...}` 内容
3. 再次失败则返回默认分数 3.0

### 3.4 Reporter — 报表输出

```
reporter/
├── models.py              # EvaluationResult 数据类
├── comparator.py          # ModelComparator 对比分析器
├── console_reporter.py    # 终端表格输出
└── html_reporter.py       # HTML 可视化报告
```

#### ModelComparator 对比分析

**四维度最优模型识别**：

| 指标 | 定义 | 决策逻辑 |
|------|------|----------|
| 综合最优 | avg_score 最高 | `max(comparisons, key=lambda x: x.avg_score)` |
| 速度最快 | avg_latency 最低 | `min(comparisons, key=lambda x: x.avg_latency)` |
| 成本最低 | cost_per_sample 最低 | `min(comparisons, key=lambda x: x.cost_per_sample)` |
| 性价比王 | score_per_cost 最高 | `max(comparisons, key=lambda x: x.score_per_cost)` |

#### HtmlReporter 输出结构

```
┌──────────────────────────────────────────────┐
│  1. 报告头：数据集名、样本数、模型数、时间戳     │
│  2. 对比总览表：模型/分数/延迟/Token/成本/成功率  │
│  3. 结论卡片：最优/最快/最便宜/性价比王           │
│  4. 样本详情：每个样本各个模型的回答 + 评分        │
│  5. 页脚                                       │
└──────────────────────────────────────────────┘
```

**样式策略**：纯内联 CSS，不依赖任何 CSS 框架，单文件可独立打开。

### 3.5 Utils — 工具函数

```
utils/
├── cost_calc.py      # 统一成本计算
└── common.py         # 通用工具函数
```

#### 成本模型

**双币种定价体系**：

| 定价表 | 单位 | 适用场景 |
|--------|------|----------|
| CNY_PRICING | ¥/百万 token | 国内模型定价惯例 |
| USD_PRICING | $/1K token | OpenAI 定价惯例 |

**价格匹配策略**：模糊前缀匹配
- 输入 `deepseek-chat` → 匹配 `deepseek-chat` key
- 输入 `qwen-turbo-xxxx` → 匹配 `qwen-turbo` key
- 无匹配 → 使用 `default` 定价

**微成本展示**：当 `cost < 0.01` 时自动切换为"分"单位
- ¥0.003 → "0.30分"
- ¥0.0005 → "0.05分"

---

## 4. 关键设计决策

### 决策 1：为什么不用 OpenAI SDK？

| 方案 | 优势 | 劣势 |
|------|------|------|
| **openai SDK** | 开箱即用、类型安全 | 增加 300KB 依赖、绑定 OpenAI 接口规范 |
| **httpx 直连** ✅ | 零额外依赖、灵活控制超时/重试/连接池 | 需要自己处理请求组装和响应解析 |

**结论**：对于轻量工具，httpx 直连更可控，且不需要 openai SDK 的流式/函数调用等高级特性。

### 决策 2：为什么做双评分引擎？

| 评分方式 | 优势 | 劣势 | 适用场景 |
|----------|------|------|----------|
| 规则评分 | 毫秒级、零成本、可解释 | 只能评估表层特征 | 大批量初筛、CI 门禁 |
| LLM Judge | 语义级理解、灵活 | 有成本、有位置偏差 | 小批量精评、趋势判断 |

**结论**：两者互补，用户按需选择。可以在同一流程中先用规则过滤，再对 Top-K 用 Judge 精评。

### 决策 3：为什么用 Semaphore 做并发控制？

| 方案 | 优势 | 劣势 |
|------|------|------|
| **Semaphore** ✅ | 精确控制并发数、不依赖第三方 | 需要手动管理 |
| asyncio.gather 无限制 | 简单 | 容易打满 API 限流 |
| 第三方限流库 | 功能丰富 | 增加依赖 |

**结论**：双层 Semaphore（Adapter 层 + Evaluator 层）提供了足够的并发控制粒度，且零依赖。

### 决策 4：位置偏差消除方案

| 方案 | 成本 | 效果 |
|------|------|------|
| **双评取平均** ✅ | 2× API 调用 | 消除大部分位置偏差 |
| 三评取中位数 | 3× API 调用 | 更好但成本高 |
| 单次 Judge | 1× API 调用 | 有偏差 |

**结论**：双评取平均是性价比最优方案，被学术界广泛使用（Zheng et al., 2023）。

---

## 5. 扩展点

### 新增模型适配器

```python
class MyAdapter(BaseAdapter):
    async def generate(self, prompt, **kwargs) -> ModelResponse:
        # 实现与自定义模型的通信
        pass
```

### 新增评分器

```python
class MyScorer(BaseScorer):
    async def score(self, question, response, reference=None, **kwargs):
        # 实现自定义评分逻辑
        return ScoreResult(total_score=0.85, details={}, reasoning="")
```

### 新增报表格式

```python
class MyReporter:
    def generate(self, comparison: ComparisonResult, **kwargs):
        # 实现自定义报表输出
        pass
```

---

## 6. 技术栈

| 组件 | 选型 | 版本 | 理由 |
|------|------|------|------|
| 语言 | Python | ≥3.8 | AI 生态首选，用户基础广 |
| HTTP | httpx | ≥0.24 | 异步原生，连接池，超时控制 |
| 分词 | jieba | ≥0.42 | 中文分词最轻量的方案 |
| 构建 | setuptools | ≥61.0 | 标准方案，无需 poetry |
| CI | GitHub Actions | — | 开源项目标准 |

**总依赖数：2 个（httpx + jieba）**

---

## 7. 限制与未来

### 当前限制

1. **仅支持单轮对话**：不支持多轮上下文评测
2. **仅支持 OpenAI 兼容接口**：不支持非标准 API（如 Claude、Gemini 原生 API）
3. **无缓存机制**：相同 prompt 会重复调用 API
4. **无并行模型评测**：多模型串行执行，非并行

### 规划路线

| 版本 | 功能 |
|------|------|
| v0.2 | 评测历史对比 + 成本效益分析 + CI |
| v0.3 | 多轮对话支持 + 缓存 |
| v0.4 | 并行多模型评测 + 更多适配器 |
| v1.0 | Web UI + 评测报告分享 |

---

## 8. 参考资料

- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", 2023
- Wang et al., "Position Bias in LLM-as-Judge", 2024
- OpenAI API Pricing: https://openai.com/api/pricing
- DeepSeek API Pricing: https://platform.deepseek.com/api-docs/pricing
- 阿里云千问 Pricing: https://help.aliyun.com/zh/model-studio/getting-started/models
