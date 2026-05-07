"""Generate demo report for README screenshot."""
import asyncio
from llm_eval_kit.reporter.models import EvaluationResult
from llm_eval_kit.reporter.html_reporter import HtmlReporter

results = [
    EvaluationResult("1", "退款多久到账？", "一般退款会在3-5个工作日内到账，具体时效取决于支付方式。如果超过5个工作日未到账，建议联系客服查询。",
                      1.2, {"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130},
                      "deepseek-chat", {"total_score": 0.85, "reasoning": "关键词命中完整，包含退款、到账、工作日等核心信息"},
                      0.002),
    EvaluationResult("1", "退款多久到账？", "退款通常在1-3个工作日内处理完成。",
                      0.8, {"prompt_tokens": 45, "completion_tokens": 40, "total_tokens": 85},
                      "qwen-turbo", {"total_score": 0.62, "reasoning": "回答了到账时间但不够详细"},
                      0.0003),
    EvaluationResult("1", "退款多久到账？", "退款一般3-5个工作日到账。",
                      1.5, {"prompt_tokens": 48, "completion_tokens": 35, "total_tokens": 83},
                      "gpt-3.5-turbo", {"total_score": 0.58, "reasoning": "回答简短，缺乏细节说明"},
                      0.003),
    EvaluationResult("2", "怎么改密码？", "您可以在「设置→账户安全→修改密码」中找到入口，按页面提示输入原密码和新密码即可完成修改。",
                      0.9, {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100},
                      "deepseek-chat", {"total_score": 0.78, "reasoning": "路径指引清晰，包含操作步骤"},
                      0.0015),
    EvaluationResult("2", "怎么改密码？", "在个人中心点击修改密码，输入原密码和新密码即可。",
                      0.7, {"prompt_tokens": 38, "completion_tokens": 35, "total_tokens": 73},
                      "qwen-turbo", {"total_score": 0.55, "reasoning": "回答了基本步骤但缺少路径细节"},
                      0.0002),
    EvaluationResult("2", "怎么改密码？", "进入设置页面，找到修改密码选项，按提示操作。",
                      1.1, {"prompt_tokens": 42, "completion_tokens": 40, "total_tokens": 82},
                      "gpt-3.5-turbo", {"total_score": 0.52, "reasoning": "步骤描述过于笼统"},
                      0.0025),
    EvaluationResult("3", "订单未发货怎么办？", "如果订单超过承诺时间未发货，建议您：1）在订单详情页点击「催发货」按钮；2）联系在线客服查询库存状态；3）如需取消可申请退款。",
                      1.0, {"prompt_tokens": 45, "completion_tokens": 80, "total_tokens": 125},
                      "deepseek-chat", {"total_score": 0.91, "reasoning": "分步骤给出具体操作方案，完整度高"},
                      0.0018),
    EvaluationResult("3", "订单未发货怎么办？", "建议联系客服查询发货时间。",
                      0.6, {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
                      "qwen-turbo", {"total_score": 0.35, "reasoning": "回答过于简短，缺乏具体操作指引"},
                      0.00015),
    EvaluationResult("3", "订单未发货怎么办？", "您可以联系卖家咨询发货进度，或者申请退款。",
                      1.3, {"prompt_tokens": 44, "completion_tokens": 35, "total_tokens": 79},
                      "gpt-3.5-turbo", {"total_score": 0.48, "reasoning": "给出了方向但没有具体步骤"},
                      0.0022),
]

reporter = HtmlReporter()
path = reporter.generate_report(results, "客服问答评测集", "docs/images")
print(f"Report generated: {path}")
