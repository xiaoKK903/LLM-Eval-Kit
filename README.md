# LLM-Eval-Kit

一个轻量级的LLM业务评测工具，帮助工程师评估Prompt修改效果、模型替换影响和RAG系统效果。

## 特性

- 🚀 **简单易用**: 三行代码即可运行评测
- 🔌 **模型兼容**: 支持所有兼容OpenAI接口的模型
- 📊 **业务导向**: 专注于业务场景评测指标
- ⚡ **高效并发**: 异步并发调用，提升评测效率
- 📈 **可视化报告**: 控制台表格和HTML报告

## 快速开始

### 安装

```bash
pip install llm-eval-kit
```

### 基础用法

```python
from llm_eval_kit import EvalClient

# 创建评测客户端
client = EvalClient()

# 运行评测
results = client.evaluate(
    data_path="data.jsonl",
    model_config={
        "base_url": "https://api.openai.com/v1",
        "api_key": "your-api-key",
        "model": "gpt-3.5-turbo"
    }
)

# 查看结果
print(results.summary())
```

## 文档

- [快速开始指南](docs/quickstart.md)
- [API参考](docs/api_reference.md)
- [高级用法](docs/advanced_usage.md)

## 许可证

MIT License