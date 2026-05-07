# llm-eval-kit

轻量级 LLM 业务评测工具箱  
量化模型效果、Prompt 改动、成本耗时，不靠感觉靠数据。

## 安装

```bash
git clone https://github.com/xiaoKK903/LLM-Eval-Kit
cd LLM-Eval-Kit
pip install -e .
```

只需要 httpx 和 jieba，没有重型依赖。

## 快速开始

### 命令行一键评测

```bash
llm-eval-kit eval \
  --model deepseek-chat \
  --api-key sk-xxx \
  --data data.jsonl \
  --base-url https://api.deepseek.com/v1
```

### Python API（三行代码跑评测）

```python
import asyncio
from llm_eval_kit import Evaluator

evaluator = Evaluator()
result = asyncio.run(evaluator.run(
    models=[{"model": "deepseek-chat", "api_key": "sk-xxx", "base_url": "https://api.deepseek.com/v1"}],
    data_path="data.jsonl",
))
```

输出自动包含：综合评分、延迟、Token 消耗、成本（¥）。

## 示例

| 场景 | 文件 |
|------|------|
| 规则评分（纯本地，无需 API） | `examples/rule_scorer_demo.py` |
| 单模型评测 | `examples/basic_usage.py` |
| 多模型对比 | `examples/multi_model_compare.py` |

## 数据格式

JSONL，每行一个样本：

```jsonl
{"id": "1", "question": "退款多久到账？", "reference": "3-5个工作日", "expected_keywords": ["退款", "到账"]}
{"id": "2", "question": "怎么改密码？", "reference": "在设置页面修改", "expected_keywords": ["设置", "修改"]}
```

- `id` / `question`：必填
- `reference`：参考答案，用于评分对比
- `expected_keywords`：关键词列表，规则评分用

## 评分引擎

**规则评分**：关键词命中 + 长度合理性 + 结构质量 + 完整度，毫秒级，零外部依赖。

**LLM Judge**：用模型评模型，准确性/完整性/简洁性三维打分。  
自动消除位置偏差——每条数据正反顺序各 Judge 一次，取平均。

两种方式可在同一个评估流程中一键切换。

## 项目结构

```
llm_eval_kit/
├── adapters/          模型适配（OpenAI 兼容接口）
│   ├── base.py
│   └── openai_compat.py
├── scorers/           评分引擎
│   ├── base.py
│   ├── rule_scorer.py
│   └── llm_judge.py
├── dataset/           数据集加载
│   └── loader.py
├── reporter/          报表输出
│   ├── console_reporter.py
│   ├── comparator.py
│   └── models.py
├── core/              核心调度
│   └── evaluator.py
├── utils/             工具函数
│   ├── cost_calc.py
│   └── common.py
├── __init__.py
```

不重依赖、不用笨重框架，Python 原生 + httpx。

## 已测试模型

- DeepSeek（deepseek-chat, deepseek-coder）
- 千问（qwen-turbo, qwen-plus, qwen-max）
- OpenAI（gpt-3.5-turbo, gpt-4, gpt-4-turbo）

兼容所有 OpenAI 接口的模型，配置 `base_url` + `api_key` 即用。

## License

MIT
