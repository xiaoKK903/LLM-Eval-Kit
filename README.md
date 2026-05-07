# LLM-Eval-Kit / LLM评测工具包

[English](#english) | [中文](#中文)

---

<a name="中文"></a>
## 🇨🇳 中文版

### 🚀 简介

LLM-Eval-Kit 是一个轻量级的LLM业务评测工具，专为工程师设计，帮助评估Prompt修改效果、模型替换影响和RAG系统效果。

### ✨ 核心特性

- 🚀 **简单易用**: 三行代码即可运行评测
- 🔌 **模型兼容**: 支持所有兼容OpenAI接口的主流模型
- 📊 **业务导向**: 专注于业务场景评测指标
- ⚡ **高效并发**: 异步并发调用，提升评测效率
- 🎯 **智能评分**: 规则评分 + LLM Judge多维度评估
- 📈 **对比报告**: 多模型对比、成本分析、选型建议

### 🏗️ 项目架构

```
llm_eval_kit/
├── adapters/           # 模型适配器
│   ├── base.py                    # 适配器基类
│   └── openai_compatible_httpx.py # OpenAI兼容适配器
├── dataset/            # 数据集处理
│   └── loader.py                  # JSONL数据集加载
├── scorers/            # 评分引擎
│   ├── base.py                    # 评分器基类
│   ├── rule_scorer.py            # 规则评分器
│   └── llm_judge.py              # LLM Judge评分器
├── reporter/           # 报告生成
│   ├── console.py                # 控制台报告
│   ├── comparator.py             # 模型对比器
│   └── models.py                 # 数据模型
├── metrics/            # 质量指标
│   └── quality_scorer.py         # 质量评分器
└── client.py           # 主客户端
```

### 🚀 快速开始

#### 安装

```bash
pip install llm-eval-kit
```

#### 基础用法

```python
import asyncio
from llm_eval_kit import EvalClient, EvaluationConfig

async def main():
    # 配置评测参数
    config = EvaluationConfig(
        data_path="data.jsonl",
        model_config={
            "base_url": "https://api.openai.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-3.5-turbo"
        }
    )
    
    # 创建客户端并运行评测
    client = EvalClient()
    results = await client.evaluate(config)
    
    # 查看结果
    print(f"评测完成: {len(results)} 条样本")

asyncio.run(main())
```

#### 多模型对比

```python
import asyncio
from llm_eval_kit.scorers import RuleScorer
from llm_eval_kit.reporter.console import ConsoleReporter

async def compare_models():
    # 初始化评分器和报告器
    scorer = RuleScorer()
    reporter = ConsoleReporter()
    
    # 运行多模型评测...
    # 生成对比报告
    reporter.print_comparison_report(results)

asyncio.run(compare_models())
```

### 📊 支持的主流模型

#### 国内模型
- **阿里云千问系列**: qwen-turbo, qwen-plus, qwen-max
- **深度求索DeepSeek**: deepseek-chat, deepseek-coder
- **智谱AI**: glm-4, glm-3-turbo
- **百度文心一言**: ernie-bot, ernie-bot-turbo
- **讯飞星火**: spark-v3, spark-v2
- **字节跳动豆包**: doubao-pro, doubao-lite

#### 国际模型
- **OpenAI**: gpt-4, gpt-3.5-turbo, gpt-4-turbo
- **Anthropic Claude**: claude-3-opus, claude-3-sonnet
- **Google Gemini**: gemini-pro, gemini-ultra
- **Meta Llama**: llama-2, llama-3

#### 开源模型
- **通义千问开源版**: qwen-7b, qwen-14b
- **ChatGLM系列**: chatglm3-6b, chatglm2-6b
- **Baichuan系列**: baichuan2-7b, baichuan2-13b
- **InternLM系列**: internlm-7b, internlm-20b

### 🎯 评测指标

#### 性能指标
- **响应延迟**: 请求到响应的总时间
- **Token使用量**: 输入/输出Token统计
- **成功率**: API调用成功率

#### 质量指标
- **规则评分**: 关键词命中率 + 长度合理性
- **LLM Judge评分**: 准确性、完整性、简洁性
- **成本效益**: 得分/成本比分析

#### 业务指标
- **回答相关性**: 与业务场景的匹配度
- **信息完整性**: 关键信息覆盖程度
- **表达规范性**: 语言表达的专业性

### 📈 示例输出

```
═══════════════════════════════════════════════════
               模型评测对比报告
═══════════════════════════════════════════════════

模型名称         平均分   延迟(s)  Token   成本(¥)  成功率
────────────────────────────────────────────────
qwen-turbo       0.82    5.1     1029    ¥0.0006   100%
deepseek-chat    0.78    7.4     1364    ¥0.0027   100%

────────────────────────────────────────────────
🏆 综合最优：qwen-turbo    （得分最高 0.82）
⚡ 速度最快：qwen-turbo   （平均 5.1s）
💰 成本最低：qwen-turbo   （¥0.0002/次）
📈 性价比王：qwen-turbo   （得分/成本比最高）
```

### 🔧 高级功能

#### 自定义评分器
```python
from llm_eval_kit.scorers import BaseScorer

class CustomScorer(BaseScorer):
    def score(self, question, response, reference):
        # 实现自定义评分逻辑
        return ScoreResult(total_score=0.9, details={}, reasoning="自定义评分")
```

#### 并发控制
```python
config = EvaluationConfig(
    data_path="data.jsonl",
    model_config=model_config,
    concurrency_per_model=3,    # 单模型并发数
    model_level_concurrency=4   # 模型级别并发数
)
```

#### 成本计算
```python
from llm_eval_kit.reporter.comparator import ModelComparator

comparator = ModelComparator({
    "custom-model": {"input": 0.5, "output": 1.0}  # 元/百万token
})
```

### 📚 文档

- [快速开始指南](docs/quickstart.md)
- [API参考文档](docs/api_reference.md)
- [高级用法示例](docs/advanced_usage.md)
- [模型适配指南](docs/model_adapter.md)

### 🤝 贡献

欢迎提交Issue和Pull Request！

### 📄 许可证

MIT License

---

<a name="english"></a>
## 🇺🇸 English Version

### 🚀 Introduction

LLM-Eval-Kit is a lightweight LLM business evaluation toolkit designed for engineers to evaluate prompt modification effects, model replacement impacts, and RAG system performance.

### ✨ Core Features

- 🚀 **Easy to Use**: Run evaluations with just 3 lines of code
- 🔌 **Model Compatibility**: Supports all OpenAI-compatible mainstream models
- 📊 **Business-Oriented**: Focuses on business scenario evaluation metrics
- ⚡ **Efficient Concurrency**: Asynchronous concurrent calls for improved efficiency
- 🎯 **Smart Scoring**: Rule-based scoring + LLM Judge multi-dimensional evaluation
- 📈 **Comparison Reports**: Multi-model comparison, cost analysis, selection recommendations

### 🏗️ Project Architecture

```
llm_eval_kit/
├── adapters/           # Model adapters
│   ├── base.py                    # Adapter base class
│   └── openai_compatible_httpx.py # OpenAI-compatible adapter
├── dataset/            # Dataset processing
│   └── loader.py                  # JSONL dataset loading
├── scorers/            # Scoring engines
│   ├── base.py                    # Scorer base class
│   ├── rule_scorer.py            # Rule-based scorer
│   └── llm_judge.py              # LLM Judge scorer
├── reporter/           # Report generation
│   ├── console.py                # Console reporter
│   ├── comparator.py             # Model comparator
│   └── models.py                 # Data models
├── metrics/            # Quality metrics
│   └── quality_scorer.py         # Quality scorer
└── client.py           # Main client
```

### 🚀 Quick Start

#### Installation

```bash
pip install llm-eval-kit
```

#### Basic Usage

```python
import asyncio
from llm_eval_kit import EvalClient, EvaluationConfig

async def main():
    # Configure evaluation parameters
    config = EvaluationConfig(
        data_path="data.jsonl",
        model_config={
            "base_url": "https://api.openai.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-3.5-turbo"
        }
    )
    
    # Create client and run evaluation
    client = EvalClient()
    results = await client.evaluate(config)
    
    # View results
    print(f"Evaluation completed: {len(results)} samples")

asyncio.run(main())
```

#### Multi-Model Comparison

```python
import asyncio
from llm_eval_kit.scorers import RuleScorer
from llm_eval_kit.reporter.console import ConsoleReporter

async def compare_models():
    # Initialize scorer and reporter
    scorer = RuleScorer()
    reporter = ConsoleReporter()
    
    # Run multi-model evaluation...
    # Generate comparison report
    reporter.print_comparison_report(results)

asyncio.run(compare_models())
```

### 📊 Supported Mainstream Models

#### Chinese Models
- **Alibaba Qwen Series**: qwen-turbo, qwen-plus, qwen-max
- **DeepSeek**: deepseek-chat, deepseek-coder
- **Zhipu AI**: glm-4, glm-3-turbo
- **Baidu ERNIE**: ernie-bot, ernie-bot-turbo
- **iFlytek Spark**: spark-v3, spark-v2
- **ByteDance Doubao**: doubao-pro, doubao-lite

#### International Models
- **OpenAI**: gpt-4, gpt-3.5-turbo, gpt-4-turbo
- **Anthropic Claude**: claude-3-opus, claude-3-sonnet
- **Google Gemini**: gemini-pro, gemini-ultra
- **Meta Llama**: llama-2, llama-3

#### Open Source Models
- **Qwen Open Source**: qwen-7b, qwen-14b
- **ChatGLM Series**: chatglm3-6b, chatglm2-6b
- **Baichuan Series**: baichuan2-7b, baichuan2-13b
- **InternLM Series**: internlm-7b, internlm-20b

### 🎯 Evaluation Metrics

#### Performance Metrics
- **Response Latency**: Total time from request to response
- **Token Usage**: Input/output token statistics
- **Success Rate**: API call success rate

#### Quality Metrics
- **Rule-based Scoring**: Keyword hit rate + length appropriateness
- **LLM Judge Scoring**: Accuracy, completeness, conciseness
- **Cost-effectiveness**: Score/cost ratio analysis

#### Business Metrics
- **Answer Relevance**: Match with business scenarios
- **Information Completeness**: Key information coverage
- **Expression Standardization**: Language professionalism

### 📈 Sample Output

```
═══════════════════════════════════════════════════
              Model Evaluation Report
═══════════════════════════════════════════════════

Model Name      Avg Score  Latency(s)  Token   Cost(¥)  Success
────────────────────────────────────────────────
qwen-turbo       0.82      5.1        1029    ¥0.0006   100%
deepseek-chat    0.78      7.4        1364    ¥0.0027   100%

────────────────────────────────────────────────
🏆 Best Overall: qwen-turbo    (Highest score 0.82)
⚡ Fastest: qwen-turbo         (Average 5.1s)
💰 Cheapest: qwen-turbo        (¥0.0002/sample)
📈 Best Value: qwen-turbo      (Highest score/cost ratio)
```

### 🔧 Advanced Features

#### Custom Scorer
```python
from llm_eval_kit.scorers import BaseScorer

class CustomScorer(BaseScorer):
    def score(self, question, response, reference):
        # Implement custom scoring logic
        return ScoreResult(total_score=0.9, details={}, reasoning="Custom scoring")
```

#### Concurrency Control
```python
config = EvaluationConfig(
    data_path="data.jsonl",
    model_config=model_config,
    concurrency_per_model=3,    # Per-model concurrency
    model_level_concurrency=4    # Model-level concurrency
)
```

#### Cost Calculation
```python
from llm_eval_kit.reporter.comparator import ModelComparator

comparator = ModelComparator({
    "custom-model": {"input": 0.5, "output": 1.0}  # ¥ per million tokens
})
```

### 📚 Documentation

- [Quick Start Guide](docs/quickstart.md)
- [API Reference](docs/api_reference.md)
- [Advanced Usage Examples](docs/advanced_usage.md)
- [Model Adapter Guide](docs/model_adapter.md)

### 🤝 Contributing

Welcome to submit Issues and Pull Requests!

### 📄 License

MIT License