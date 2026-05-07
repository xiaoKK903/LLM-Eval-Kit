# llm-eval-kit

> 轻量级 LLM 业务评测工具，帮工程师量化模型/Prompt 改动的效果差异

![报告截图](docs/images/report_demo.png)

## 它解决什么问题

工程师在做大模型应用时，经常面对这些问题：

- 换了模型，效果到底变好还是变差？
- 改了 Prompt，怎么证明这次迭代有价值？
- 几个模型同时备选，哪个性价比最高？

llm-eval-kit 用代码 + 数据回答这些问题，不靠感觉。

## 功能

- 并发评测：异步并发调用多个模型，速度提升 4 倍以上
- 规则评分：关键词命中率 + 长度合理性
- LLM Judge：用模型评模型，多维度打分，消除位置偏差
- 多模型对比：自动输出最优、最快、最便宜、性价比最高
- HTML 报告：一键生成可视化报告，方便汇报和归档

## 快速开始

### 安装

```bash
git clone https://github.com/xiaoKK903/LLM-Eval-Kit
cd LLM-Eval-Kit
pip install -e .
```

### 准备数据

新建 `data.jsonl`：

```jsonl
{"id": "001", "question": "退款多久到账？", "reference": "3-5个工作日"}
{"id": "002", "question": "怎么改密码？", "reference": "在设置页面修改"}
```

### 运行评测

```python
import json, asyncio
from llm_eval_kit.adapters.openai_compatible import OpenAICompatibleHttpxAdapter
from llm_eval_kit.scorers.rule_scorer import RuleScorer
from llm_eval_kit.reporter.comparator import ModelComparator
from llm_eval_kit.reporter.html_reporter import HtmlReporter

async def main():
    samples = [json.loads(l) for l in open("data.jsonl")]

    adapter = OpenAICompatibleHttpxAdapter(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-your-key",
        model="deepseek-chat"
    )
    evals = await adapter.batch_call([s["question"] for s in samples])

    scorer = RuleScorer()
    for e, s in zip(evals, samples):
        e.scoring_result = scorer.score(e.response, s.get("reference", ""))

    cmp = ModelComparator().compare_models({"deepseek-chat": evals})
    HtmlReporter().generate(cmp, "客服问答评测", "report.html")
    print("✅ report.html 已生成")

asyncio.run(main())
```

> 替换 `api_key` 即可直接运行。如需对比多个模型，在 `compare_models` 中传入更多结果即可。

## 支持的模型

理论上支持所有兼容 OpenAI 接口的模型。

已测试：
- DeepSeek（deepseek-chat、deepseek-coder）
- 千问（qwen-turbo、qwen-plus、qwen-max）
- OpenAI（gpt-3.5-turbo、gpt-4）

只要模型提供 OpenAI 兼容接口，配置 `base_url` 和 `api_key` 即可。

## 评分方式

### 规则评分

基于关键词命中率和长度合理性，毫秒级响应，适合大批量数据初筛。

### LLM Judge

用一个模型评测另一个模型的回答，从准确性、完整性、简洁性三个维度打分。

为消除位置偏差，每条数据 Judge 两次（参考答案在前/在后），取平均分。

## 项目结构

```
llm_eval_kit/
├── adapters/       # 模型适配器
├── dataset/        # 数据加载
├── scorers/        # 评分器
├── reporter/       # 报告生成
└── client.py       # 主入口
```

## 开发计划

- [x] 多模型并发评测
- [x] 规则评分 + LLM Judge
- [x] HTML 报告生成
- [ ] CLI 命令行工具
- [ ] 评测历史对比（版本回归）
- [ ] PyPI 发布

## License

MIT
