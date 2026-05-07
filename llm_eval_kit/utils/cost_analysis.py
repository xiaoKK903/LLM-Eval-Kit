"""
Cost-benefit analysis for LLM model selection.

Compares API-based vs self-hosted model costs, generates
cost projections, and identifies cost-optimal model choices.

Usage:
    from llm_eval_kit.utils.cost_analysis import CostBenefitAnalyzer

    analyzer = CostBenefitAnalyzer()
    report = analyzer.analyze(models=[...], daily_requests=10000)
    analyzer.print_report(report)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .cost_calc import calculate_cost_cny, format_cost, CNY_PRICING


@dataclass
class ModelCostProjection:
    model_name: str
    cost_per_1k_samples: float
    cost_per_10k_samples: float
    cost_per_100k_samples: float
    daily_cost_10k: float
    monthly_cost_10k: float
    yearly_cost_10k: float
    latency_seconds: float
    avg_score: float
    score_per_cost: float


@dataclass
class SelfHostedEstimate:
    gpu_type: str
    gpu_count: int
    monthly_gpu_cost: float
    monthly_bandwidth: float
    monthly_maintenance: float
    monthly_total: float
    max_throughput_daily: int
    cost_per_1k: float
    payback_period_months: float
    break_even_daily_requests: int


@dataclass
class CostBenefitReport:
    projections: List[ModelCostProjection]
    self_hosted: Optional[SelfHostedEstimate]
    recommended_api: str
    recommended_reason: str
    break_even_analysis: str


class CostBenefitAnalyzer:
    GPU_OPTIONS = [
        {"gpu": "A100 80G", "count": 1, "monthly_rent": 3500},
        {"gpu": "A100 80G", "count": 4, "monthly_rent": 12000},
        {"gpu": "H100 80G", "count": 1, "monthly_rent": 8000},
        {"gpu": "H100 80G", "count": 8, "monthly_rent": 55000},
        {"gpu": "RTX 4090", "count": 1, "monthly_rent": 800},
        {"gpu": "RTX 4090", "count": 4, "monthly_rent": 2800},
    ]

    def analyze(
        self,
        models: List[Dict[str, Any]],
        daily_requests: int = 10000,
        avg_prompt_tokens: int = 500,
        avg_completion_tokens: int = 500,
        self_hosted: bool = True,
    ) -> CostBenefitReport:
        projections = []
        for m in models:
            name = m.get("model", "unknown")
            latency = m.get("avg_latency", 1.0)
            score = m.get("avg_score", 0.8)

            cost_1k = calculate_cost_cny(name, avg_prompt_tokens * 1000, avg_completion_tokens * 1000)
            cost_10k = cost_1k * 10
            cost_100k = cost_1k * 100

            daily_10k = calculate_cost_cny(name, avg_prompt_tokens * daily_requests, avg_completion_tokens * daily_requests)
            monthly = daily_10k * 30
            yearly = daily_10k * 365
            spc = score / daily_10k if daily_10k > 0 else 0

            projections.append(ModelCostProjection(
                model_name=name,
                cost_per_1k_samples=round(cost_1k, 4),
                cost_per_10k_samples=round(cost_10k, 4),
                cost_per_100k_samples=round(cost_100k, 4),
                daily_cost_10k=round(daily_10k, 4),
                monthly_cost_10k=round(monthly, 4),
                yearly_cost_10k=round(yearly, 4),
                latency_seconds=latency,
                avg_score=score,
                score_per_cost=round(spc, 4),
            ))

        sh = None
        if self_hosted:
            sh = self._estimate_self_hosted(daily_requests, avg_prompt_tokens, avg_completion_tokens, projections)

        cheapest = min(projections, key=lambda p: p.cost_per_1k_samples)
        best_value = max(projections, key=lambda p: p.score_per_cost)

        if best_value.model_name == cheapest.model_name:
            recommended_api = cheapest.model_name
            recommended_reason = f"成本最低且性价比最高"
        else:
            recommended_api = best_value.model_name
            recommended_reason = f"性价比最优（评分 {best_value.avg_score} / 成本 {format_cost(best_value.cost_per_1k_samples)}/千次）"

        be = self._break_even_analysis(projections, sh)

        return CostBenefitReport(
            projections=sorted(projections, key=lambda p: p.cost_per_1k_samples),
            self_hosted=sh,
            recommended_api=recommended_api,
            recommended_reason=recommended_reason,
            break_even_analysis=be,
        )

    def _estimate_self_hosted(
        self,
        daily_requests: int,
        avg_prompt_tokens: int,
        avg_completion_tokens: int,
        api_projections: List[ModelCostProjection],
    ) -> SelfHostedEstimate:
        best_opt = self.GPU_OPTIONS[0]
        min_cost = float("inf")
        for opt in self.GPU_OPTIONS:
            throughput = opt["count"] * 10000
            if throughput >= daily_requests and opt["monthly_rent"] < min_cost:
                min_cost = opt["monthly_rent"]
                best_opt = opt

        bw = 500
        maint = 500
        monthly = best_opt["monthly_rent"] + bw + maint
        max_tp = best_opt["count"] * 10000

        avg_cost_1k = sum(p.cost_per_1k_samples for p in api_projections) / len(api_projections) if api_projections else 0
        cost_1k = monthly * 1000 / (daily_requests * 30) if daily_requests > 0 else 0

        if avg_cost_1k > 0:
            payback = monthly / (avg_cost_1k * daily_requests * 30 / 1000)
        else:
            payback = 999

        be_requests = int(monthly / avg_cost_1k * 1000 / 30) if avg_cost_1k > 0 else 999999

        return SelfHostedEstimate(
            gpu_type=best_opt["gpu"],
            gpu_count=best_opt["count"],
            monthly_gpu_cost=best_opt["monthly_rent"],
            monthly_bandwidth=bw,
            monthly_maintenance=maint,
            monthly_total=monthly,
            max_throughput_daily=max_tp,
            cost_per_1k=round(cost_1k, 4),
            payback_period_months=round(payback, 1) if payback < 100 else 999,
            break_even_daily_requests=be_requests,
        )

    @staticmethod
    def _break_even_analysis(projections: List[ModelCostProjection], sh: Optional[SelfHostedEstimate]) -> str:
        if not sh:
            return "未进行自部署成本分析"

        cheapest_api = min(projections, key=lambda p: p.cost_per_1k_samples)
        parts = [
            f"自部署方案: {sh.gpu_count}×{sh.gpu_type}",
            f"月度成本: {sh.monthly_total:.0f}元（GPU {sh.monthly_gpu_cost}元 + 带宽{sh.monthly_bandwidth}元 + 运维{sh.monthly_maintenance}元）",
            f"API方案: {cheapest_api.model_name} 千次成本 {format_cost(cheapest_api.cost_per_1k_samples)}",
            f"盈亏平衡点: 日请求量 > {sh.break_even_daily_requests:,} 时自部署更划算",
        ]
        return " | ".join(parts)

    @staticmethod
    def print_report(report: CostBenefitReport):
        print("\n" + "=" * 72)
        print("  LLM-Eval-Kit 成本效益分析")
        print("=" * 72)
        print(f"  API推荐: {report.recommended_api}")
        print(f"  理由: {report.recommended_reason}")
        print("-" * 72)

        header = f"{'模型':<20} {'千次成本':>12} {'日成本':>12} {'月成本':>12} {'年成本':>12} {'性价比':>10}"
        print(header)
        print("-" * 72)
        for p in report.projections:
            print(f"{p.model_name:<20} {format_cost(p.cost_per_1k_samples):>12} {format_cost(p.daily_cost_10k):>12} {format_cost(p.monthly_cost_10k):>12} {format_cost(p.yearly_cost_10k):>12} {p.score_per_cost:>10.2f}")
        print("-" * 72)

        if report.self_hosted:
            sh = report.self_hosted
            print(f"\n  自部署方案对比:")
            print(f"    GPU: {sh.gpu_count}×{sh.gpu_type}")
            print(f"    月总成本: {sh.monthly_total:.0f}元")
            print(f"    最大日吞吐: {sh.max_throughput_daily:,} 请求")
            print(f"    千次成本: {format_cost(sh.cost_per_1k)}")
            print(f"    盈亏平衡: 日请求 > {sh.break_even_daily_requests:,}")

        print(f"\n  {report.break_even_analysis}")
        print("=" * 72 + "\n")

    @staticmethod
    def save_html_report(report: CostBenefitReport, output_path: str):
        rows = ""
        for p in report.projections:
            rows += f"""
            <tr>
                <td style="padding:10px;border-bottom:1px solid #eee;font-weight:600">{p.model_name}</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{format_cost(p.cost_per_1k_samples)}</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{format_cost(p.daily_cost_10k)}</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{format_cost(p.monthly_cost_10k)}</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{format_cost(p.yearly_cost_10k)}</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{p.score_per_cost:.2f}</td>
            </tr>"""

        sh_html = ""
        if report.self_hosted:
            sh = report.self_hosted
            sh_html = f"""
            <div style="margin-top:24px;background:white;padding:24px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                <h3 style="margin:0 0 16px">自部署方案</h3>
                <table style="width:100%;border-collapse:collapse">
                    <tr><td style="padding:8px;color:#666">GPU</td><td style="padding:8px;font-weight:600">{sh.gpu_count}×{sh.gpu_type}</td></tr>
                    <tr><td style="padding:8px;color:#666">月总成本</td><td style="padding:8px;font-weight:600">{sh.monthly_total:.0f}元</td></tr>
                    <tr><td style="padding:8px;color:#666">最大日吞吐</td><td style="padding:8px;font-weight:600">{sh.max_throughput_daily:,} 请求</td></tr>
                    <tr><td style="padding:8px;color:#666">千次成本</td><td style="padding:8px;font-weight:600">{format_cost(sh.cost_per_1k)}</td></tr>
                    <tr><td style="padding:8px;color:#666">盈亏平衡</td><td style="padding:8px;font-weight:600">日请求 &gt; {sh.break_even_daily_requests:,}</td></tr>
                </table>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>LLM 成本效益分析</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #333; }}
.header {{ background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 24px 32px; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ margin: 0 0 8px; }}
.header p {{ margin: 2px 0; opacity: 0.85; font-size: 14px; }}
.tag {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; background: #d1fae5; color: #065f46; margin-top: 8px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th {{ background: #f1f5f9; padding: 12px 10px; font-size: 13px; text-align: left; font-weight: 600; color: #475569; }}
td {{ font-size: 14px; }}
.footer {{ text-align: center; color: #94a3b8; font-size: 12px; margin-top: 32px; }}
</style>
</head>
<body>
<div class="header">
    <h1>💰 LLM 成本效益分析</h1>
    <p>推荐: <strong>{report.recommended_api}</strong> — {report.recommended_reason}</p>
</div>

<table>
<tr>
    <th>模型</th>
    <th>千次成本</th>
    <th>日成本</th>
    <th>月成本</th>
    <th>年成本</th>
    <th>性价比</th>
</tr>
{rows}
</table>

{sh_html}

<div class="footer">LLM-Eval-Kit · 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  成本效益分析报告已保存: {output_path}")
