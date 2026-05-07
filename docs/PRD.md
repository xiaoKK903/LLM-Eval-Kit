# LLM-Eval-Kit 产品需求文档 (PRD)

> 版本：0.2.0-draft | 状态：开发中 | 最后更新：2026-05-07  
> 产品负责人：llm-eval-kit team

---

## 1. 产品概述

### 1.1 产品定位

**中文轻量业务 LLM 评测工具箱**

面向个人开发者和小型团队，提供一款极致轻量、开箱即用、可复现的 LLM 效果评测工具。不依赖重型框架，pip install 后三行代码即可完成一次评测。

### 1.2 目标用户

| 用户画像 | 典型场景 | 核心痛点 | 使用方式 |
|----------|----------|----------|----------|
| 独立开发者 | 用 API 做 AI 产品，需要对比模型效果 | benchmark 太重，人工抽测不客观 | CLI + Python API |
| 小团队技术负责人 | 技术选型、Prompt 迭代评估 | 没有标准化评估流程，靠感觉决策 | Python API |
| AI 应用创业者 | 控制成本、优化模型选择 | API 成本不可控，不知道哪个模型性价比最高 | CLI + HTML 报告 |
| 技术面试候选人 | 作品集展示 | 需要可量化的技术深度证明 | 架构文档 + 示例 |

### 1.3 核心价值

```
三行代码跑评测，不靠感觉靠数据
```

- **10 分钟**：从零到拿到第一份评测报告
- **2 个依赖**：httpx + jieba，不装 torch/transformers
- **3 个维度**：效果 + 成本 + 效率，一次评测全掌握

---

## 2. 用户故事

### 2.1 优先级分级

| 优先级 | 定义 | 对应版本 |
|--------|------|----------|
| P0 | 不可用、不可发布 | v0.1 MVP |
| P1 | 核心体验、必须做 | v0.2 |
| P2 | 体验优化、应该做 | v0.3 |
| P3 | 锦上添花、有余力做 | v1.0+ |

### 2.2 P0 用户故事（已实现）

| ID | 故事标题 | 验收条件 |
|----|----------|----------|
| US-001 | 作为开发者，我想用三行代码启动一次评测 | `Evaluator().run(models=[...], data_path=...)` 即可工作 |
| US-002 | 作为开发者，我想从 JSONL 文件加载测试数据 | 支持 `id`/`question`/`reference`/`expected_keywords` 字段 |
| US-003 | 作为开发者，我想用命令行跑评测 | `llm-eval-kit eval --model xxx --api-key xxx --data xxx` |
| US-004 | 作为开发者，我想看到评测的对比表格 | 终端输出包含：模型名、综合分、延迟、Token、成本 |
| US-005 | 作为开发者，我想看到评测的 HTML 报告 | 生成独立 HTML 文件，包含对比总览 + 样本详情 |
| US-006 | 作为开发者，我想保存评测结果为 JSON | `output_path` 参数指定 JSON 输出路径 |
| US-007 | 作为开发者，我想用规则评分（不花 API 钱） | RuleScorer 可独立使用，毫秒级返回评分 |

### 2.3 P1 用户故事（当前版本）

| ID | 故事标题 | 验收条件 | 对应模块 |
|----|----------|----------|----------|
| US-008 | 作为开发者，我想看到两次评测的差异对比 | 输入两个 JSON 结果文件，输出评分变化、回归/改进标识 | regression.py |
| US-009 | 作为开发者，我想做 API 成本效益分析 | 输入模型配置 + 日请求量，输出各模型成本对比 + 自部署方案 | cost_analysis.py |
| US-010 | 作为开发者的 Leader，我想看到系统的架构设计文档 | 有清晰的模块图、关键决策记录、扩展指南 | ARCHITECTURE.md |
| US-011 | 作为面试官，我想看到候选人的方法论沉淀 | 有评测方法论白皮书，包含算法原理、偏差消除方案 | docs/evaluation-methodology.md |
| US-012 | 作为开发者，我想用 YAML 配置管理评测参数 | 提供 config.yaml 示例，支持多环境配置 | 配置系统 |
| US-013 | 作为 CI 系统，我想自动运行评测回归 | 支持在 CI 中运行回归分析，失败时标记 | .github/workflows/ci.yml |

### 2.4 P2 用户故事（规划中）

| ID | 故事标题 | 预期版本 |
|----|----------|----------|
| US-014 | 作为开发者，我想支持多轮对话评测 | v0.3 |
| US-015 | 作为开发者，我想缓存 API 响应避免重复调用 | v0.3 |
| US-016 | 作为开发者，我想并行评测多个模型 | v0.4 |
| US-017 | 作为开发者，我想支持 Claude/Gemini 原生 API | v0.4 |

---

## 3. 功能规格

### 3.1 CLI 命令行

```
llm-eval-kit eval [OPTIONS]

选项：
  --model TEXT         模型名称（必填，可多次指定）
  --api-key TEXT       API Key（必填）
  --base-url TEXT      API 基础地址（必填）
  --data TEXT          JSONL 数据文件路径（必填）
  --max-samples INT   最大样本数（可选，默认全部）
  --concurrency INT   并发数（可选，默认 5）
  --output TEXT       JSON 结果输出路径（可选）
  --scorer TEXT        评分引擎: rule | judge（可选，默认 rule）
  --judge-model TEXT   Judge 使用的模型（可选，scorer=judge 时必填）
```

### 3.2 Python API

```python
from llm_eval_kit import Evaluator

evaluator = Evaluator()
result = await evaluator.run(
    models=[
        {"model": "deepseek-chat", "api_key": "sk-xxx", "base_url": "https://api.deepseek.com/v1"},
    ],
    data_path="./data.jsonl",
    scorer=my_scorer,          # 可选，默认 RuleScorer
    max_samples=10,            # 可选
    concurrency=5,             # 可选
    output_path="./result.json",  # 可选
)
```

### 3.3 评分 API

```python
from llm_eval_kit import RuleScorer, LLMJudgeScorer

# 规则评分（零成本、毫秒级）
scorer = RuleScorer(keyword_weight=0.4, length_weight=0.1,
                    structure_weight=0.2, completeness_weight=0.3)
result = await scorer.score("问题", "回答", "参考答案")

# LLM Judge（语义级、自动消除位置偏差）
judge = LLMJudgeScorer(judge_config={...})
result = await judge.score("问题", "回答", "参考答案")
```

### 3.4 回归分析

```python
from llm_eval_kit.reporter import RegressionReporter

diff = RegressionReporter.compare("baseline.json", "current.json")
RegressionReporter.print_diff(diff)        # 终端输出
RegressionReporter.save_diff_html(diff, "regression.html")  # HTML 报告
```

### 3.5 成本效益分析

```python
from llm_eval_kit.utils import CostBenefitAnalyzer

analyzer = CostBenefitAnalyzer()
report = analyzer.analyze(
    models=[{"model": "deepseek-chat", "avg_score": 0.85, "avg_latency": 1.2}],
    daily_requests=10000,
)
analyzer.print_report(report)
```

---

## 4. 数据模型

### 4.1 输入数据（JSONL）

```jsonl
{"id": "1", "question": "退款多久到账？", "reference": "3-5个工作日", "expected_keywords": ["退款", "到账"]}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 是 | 样本唯一标识 |
| question | string | 是 | 用户问题 |
| reference | string | 否 | 参考答案（评分对比用） |
| expected_keywords | string[] | 否 | 期望关键词列表（规则评分用） |

### 4.2 输出数据（JSON）

```json
{
  "best_overall": {"model": "deepseek-chat", "score": 0.85},
  "fastest": {"model": "qwen-turbo", "latency": 0.8},
  "cheapest": {"model": "qwen-turbo", "cost": 0.003},
  "best_value": {"model": "deepseek-chat", "ratio": 280.0},
  "models": [
    {
      "name": "deepseek-chat",
      "avg_score": 0.85,
      "avg_latency": 1.2,
      "total_tokens": 3500,
      "total_cost": 0.006,
      "cost_per_sample": 0.002,
      "score_per_cost": 425.0,
      "success_rate": 1.0
    }
  ]
}
```

### 4.3 配置模型（YAML）

```yaml
evaluation:
  name: "客服模型对比 v2"
  description: "DeepSeek vs Qwen 对比评测"
  
datasets:
  - path: "./data/train.jsonl"
    name: "训练集"
  - path: "./data/test.jsonl"
    name: "测试集"
    max_samples: 50

models:
  - model: "deepseek-chat"
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com/v1"
    name: "DeepSeek Chat"
    
  - model: "qwen-turbo"
    api_key: "${QWEN_API_KEY}"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    name: "千问 Turbo"

scorer:
  type: "rule"
  weights:
    keyword: 0.4
    length: 0.1
    structure: 0.2
    completeness: 0.3

output:
  json: "./results/report.json"
  html: "./results/report.html"
  regression_baseline: "./results/baseline.json"
```

---

## 5. 非功能需求

### 5.1 性能指标

| 指标 | 目标 | 测量方式 |
|------|------|----------|
| 评分延迟（规则） | < 10ms / sample | timeit 100 次取平均 |
| 内存占用（规则评分） | < 50MB | memory_profiler |
| CLI 启动时间 | < 1s | time python -m llm_eval_kit.cli --help |
| 并发吞吐 | 4× 串行 | 5 并发 vs 1 并行的完成时间对比 |
| JSON 导出 | < 100ms | 100 个样本输出时间 |

### 5.2 兼容性

| 维度 | 要求 |
|------|------|
| Python 版本 | ≥ 3.8 |
| 操作系统 | macOS / Linux / Windows |
| API 协议 | OpenAI 兼容接口（/v1/chat/completions） |
| 终端编码 | UTF-8（GBK 回退方案） |

### 5.3 安全性

| 项目 | 要求 |
|------|------|
| API Key 处理 | 支持环境变量，不硬编码 |
| 数据传输 | HTTPS 加密 |
| 日志安全 | 不记录 API Key |
| 用户数据 | 不收集、不上传任何数据 |

---

## 6. 发布标准

### 6.1 版本定义

| 版本 | 阶段 | 标准 |
|------|------|------|
| v0.1.x | MVP | 核心评测流程可用，有文档 |
| v0.2.x | 增长 | 回归分析 + 成本分析 + CI + 方法论 |
| v0.3.x | 成熟 | 多轮对话 + 缓存 + 更多适配器 |
| v1.0.x | 稳定 | Web UI + 报告分享 + 完善文档 |

### 6.2 v0.2 Release Checklist

- [x] Evaluator 核心管线
- [x] RuleScorer + LLMJudgeScorer
- [x] CLI 交互
- [x] HTML 报告
- [x] 回归分析（regression.py）
- [x] 成本效益分析（cost_analysis.py）
- [x] 架构文档（ARCHITECTURE.md）
- [x] 方法论文档（docs/evaluation-methodology.md）
- [x] PRD 文档（docs/PRD.md）
- [ ] YAML 配置系统
- [ ] pytest 单元测试（覆盖率 > 60%）
- [ ] GitHub Actions CI
- [ ] 所有示例可运行

---

## 7. 竞品分析

| 维度 | llm-eval-kit | LM Eval Harness | DeepEval | 人工评测 |
|------|-------------|----------------|----------|----------|
| 定位 | 中文业务评测 | 学术 benchmark | 通用评测框架 | — |
| 安装 | pip install（2 依赖） | pip install（20+ 依赖） | pip install（10+ 依赖） | 不需要 |
| 上手时间 | 10 分钟 | 1 小时+ | 30 分钟+ | 无标准化 |
| 数据格式 | JSONL | 自定义格式 | 自定义格式 | 不适用 |
| 评分方式 | 规则 + LLM Judge | benchmark 对比 | 多种 | 人工 |
| 成本计算 | ¥ + $ 双币种 | 无 | 有 | 无 |
| 回归分析 | 有 | 无 | 有 | 无 |
| 报告 | 终端 + HTML | 终端 | 平台 | 靠记忆 |
| 重型依赖 | 无 | torch + transformers | 多种 | 不适用 |

**差异化优势**：极致轻量 + 中文场景优化 + 成本效益分析

---

## 8. 路线图

### v0.2（当前，2026 Q2）

```
核心功能：回归分析、成本效益、配置系统
文档体系：架构文档、方法论文档、PRD
质量保障：单元测试、CI 集成
```

### v0.3（2026 Q3）

```
多轮对话评测
API 响应缓存
批量模式（非 asyncio 回退）
```

### v1.0（2026 Q4）

```
简单 Web UI
评测报告在线分享
Claude / Gemini 原生适配器
社区贡献指南
```

---

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| API 价格变化 | 高 | 中 | 成本数据可配置，定期更新定价表 |
| LLM Judge 质量不稳定 | 中 | 高 | 规则评分兜底，双评取平均，可配置阈值 |
| jieba 分词效果不足 | 低 | 中 | 关键词长度过滤，支持自定义词典 |
| 用户数据格式不标准 | 中 | 高 | 提供格式校验工具和示例数据 |
| 开源项目无人维护 | 中 | 高 | 自动化 CI 测试，减少手动维护成本 |

---

## 10. 附录

### A. 术语表

| 术语 | 定义 |
|------|------|
| 规则评分 | 基于关键词、长度、结构等可量化维度的自动评分 |
| LLM Judge | 使用大语言模型作为裁判进行语义级评分 |
| 位置偏差 | LLM 在对比评估中倾向于选择先出现的答案 |
| 回归分析 | 比较两次评测结果的差异，识别退化或改进 |
| 成本效益分析 | 综合评估模型效果与使用成本的经济学分析 |
| JSONL | JSON Lines 格式，每行一个独立的 JSON 对象 |

### B. 参考文档

- [架构文档](../ARCHITECTURE.md)
- [评测方法论](../docs/evaluation-methodology.md)
- [README](../README.md)
