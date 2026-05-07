"""Quick test: generate an HTML report with mock EvaluationResult data."""
import asyncio
from llm_eval_kit.reporter.models import EvaluationResult
from llm_eval_kit.reporter.html_reporter import HtmlReporter

mock_results = [
    EvaluationResult(
        sample_id="1",
        question="退款多久到账？",
        response="一般退款会在3-5个工作日内到账，具体时效取决于支付方式。",
        latency=1.2,
        token_usage={"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130},
        model="deepseek-chat",
        scoring_result={"total_score": 0.85, "reasoning": "关键词命中完整，长度合理"},
        cost=0.002,
    ),
    EvaluationResult(
        sample_id="1",
        question="退款多久到账？",
        response="退款通常在1-3个工作日内处理完成。",
        latency=0.8,
        token_usage={"prompt_tokens": 45, "completion_tokens": 40, "total_tokens": 85},
        model="qwen-turbo",
        scoring_result={"total_score": 0.62, "reasoning": "部分关键词缺失"},
        cost=0.0003,
    ),
    EvaluationResult(
        sample_id="2",
        question="怎么改密码？",
        response="您可以在设置页面找到修改密码的选项，按提示操作即可。",
        latency=0.9,
        token_usage={"prompt_tokens": 40, "completion_tokens": 50, "total_tokens": 90},
        model="deepseek-chat",
        scoring_result={"total_score": 0.78, "reasoning": "关键词匹配良好"},
        cost=0.001,
    ),
    EvaluationResult(
        sample_id="2",
        question="怎么改密码？",
        response="在个人中心点击修改密码，输入原密码和新密码即可。",
        latency=0.7,
        token_usage={"prompt_tokens": 38, "completion_tokens": 42, "total_tokens": 80},
        model="qwen-turbo",
        scoring_result={"total_score": 0.55, "reasoning": "回答较简短"},
        cost=0.0002,
    ),
]

reporter = HtmlReporter()
path = reporter.generate_report(mock_results, "客服问答测试集")
print(f"HTML report generated: {path}")
