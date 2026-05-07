"""
Cost-benefit analysis for LLM model selection.

Provides actionable recommendations for model selection based on
comprehensive cost, performance, and quality analysis.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .comparator import ModelComparison, ComparisonResult


@dataclass
class CostBenefitMetric:
    model_name: str
    avg_score: float
    avg_latency: float
    total_cost: float
    cost_per_sample: float
    score_per_cost: float
    latency_score: float
    cost_score: float
    composite_score: float
    recommendation: str
    use_case_tags: List[str]


@dataclass
class CostBenefitReport:
    metrics: List[CostBenefitMetric]
    best_quality: str
    best_latency: str
    cheapest: str
    best_value: str
    recommended_for_production: str
    recommended_for_budget: str


class CostBenefitAnalyzer:
    """Analyze cost-benefit tradeoffs across models."""

    def analyze(self, comparison: ComparisonResult) -> CostBenefitReport:
        metrics = []
        for mc in comparison.model_comparisons:
            if mc.total_cost <= 0 or mc.avg_score <= 0:
                continue

            max_cost = max(m.total_cost for m in comparison.model_comparisons if m.total_cost > 0) or 1
            max_latency = max(m.avg_latency for m in comparison.model_comparisons) or 1
            max_score = max(m.avg_score for m in comparison.model_comparisons) or 1

            cost_score = 1.0 - (mc.total_cost / max_cost) * 0.5
            latency_score = 1.0 - (mc.avg_latency / max_latency) * 0.3
            quality_score = mc.avg_score / max_score

            composite = quality_score * 0.5 + cost_score * 0.25 + latency_score * 0.25

            use_case_tags = []
            if mc.avg_score >= max_score * 0.9:
                use_case_tags.append("高质量")
            if mc.cost_per_sample <= 0.01:
                use_case_tags.append("低成本")
            if mc.avg_latency <= 2.0:
                use_case_tags.append("低延迟")
            if mc.score_per_cost > 0 and mc.score_per_cost >= max(
                (m.score_per_cost for m in comparison.model_comparisons if m.score_per_cost > 0), default=0
            ):
                use_case_tags.append("性价比之王")

            if composite >= 0.7:
                recommendation = "强烈推荐"
            elif composite >= 0.5:
                recommendation = "推荐"
            elif composite >= 0.3:
                recommendation = "可考虑"
            else:
                recommendation = "不推荐"

            metrics.append(CostBenefitMetric(
                model_name=mc.model_name,
                avg_score=mc.avg_score,
                avg_latency=mc.avg_latency,
                total_cost=mc.total_cost,
                cost_per_sample=mc.cost_per_sample,
                score_per_cost=mc.score_per_cost,
                cost_score=round(cost_score, 4),
                latency_score=round(latency_score, 4),
                composite_score=round(composite, 4),
                recommendation=recommendation,
                use_case_tags=use_case_tags,
            ))

        if not metrics:
            return CostBenefitReport(
                metrics=[], best_quality="", best_latency="",
                cheapest="", best_value="",
                recommended_for_production="", recommended_for_budget=""
            )

        metrics.sort(key=lambda x: x.composite_score, reverse=True)
        best_quality = max(metrics, key=lambda x: x.avg_score)
        best_latency_model = min(metrics, key=lambda x: x.avg_latency)
        cheapest_model = min(metrics, key=lambda x: x.total_cost)
        best_value_model = max(metrics, key=lambda x: x.score_per_cost)

        production_candidates = [m for m in metrics if m.avg_score >= 0.6 and m.composite_score >= 0.5]
        recommended_prod = max(production_candidates, key=lambda x: x.composite_score) if production_candidates else best_quality

        budget_candidates = [m for m in metrics if m.cost_per_sample <= 0.01 and m.avg_score >= 0.4]
        recommended_budget = max(budget_candidates, key=lambda x: x.composite_score) if budget_candidates else cheapest_model

        return CostBenefitReport(
            metrics=metrics,
            best_quality=best_quality.model_name,
            best_latency=best_latency_model.model_name,
            cheapest=cheapest_model.model_name,
            best_value=best_value_model.model_name,
            recommended_for_production=recommended_prod.model_name,
            recommended_for_budget=recommended_budget.model_name,
        )

    def print_report(self, report: CostBenefitReport):
        print(f"\n{'='*70}")
        print(f"  成本效益分析")
        print(f"{'='*70}\n")

        header = f"{'模型':<20} {'综合分':>8} {'延迟(秒)':>10} {'总成本(¥)':>10} {'单位成本':>10} {'性价比':>10} {'推荐等级':>10}"
        print(header)
        print("-" * 78)

        for m in report.metrics:
            print(
                f"{m.model_name:<20} "
                f"{m.avg_score:>8.4f} "
                f"{m.avg_latency:>10.4f} "
                f"{m.total_cost:>10.4f} "
                f"{m.cost_per_sample:>10.4f} "
                f"{m.score_per_cost:>10.2f} "
                f"{m.recommendation:>10}"
            )
            if m.use_case_tags:
                print(f"{'':>20} 标签: {', '.join(m.use_case_tags)}")

        print(f"\n")
        print(f"  🏆 综合最优:      {report.best_quality}")
        print(f"  ⚡ 速度最快:       {report.best_latency}")
        print(f"  💰 成本最低:       {report.cheapest}")
        print(f"  📊 性价比王:       {report.best_value}")
        print(f"  ✅ 生产推荐:       {report.recommended_for_production}")
        print(f"  🎯 预算推荐:       {report.recommended_for_budget}")
        print()

    def generate_html(self, report: CostBenefitReport, output_path: str):
        rows_html = ""
        for m in report.metrics:
            tags_html = "".join(
                f'<span style="display:inline-block;padding:2px 8px;margin:2px;border-radius:10px;font-size:11px;background:#e3f2fd;color:#1565c0">{t}</span>'
                for t in m.use_case_tags
            )
            rec_color = {"强烈推荐": "#28a745", "推荐": "#2196f3", "可考虑": "#ff9800", "不推荐": "#dc3545"}
            rows_html += f"""\
<tr>
    <td style="padding:8px;border-bottom:1px solid #eee;font-weight:500">{m.model_name}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{m.avg_score:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{m.avg_latency:.2f}s</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">¥{m.total_cost:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">¥{m.cost_per_sample:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{m.score_per_cost:.2f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{m.composite_score:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center"><span style="color:{rec_color.get(m.recommendation, '#333')};font-weight:600">{m.recommendation}</span></td>
    <td style="padding:8px;border-bottom:1px solid #eee">{tags_html}</td>
</tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>LLM 成本效益分析报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 10px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
th {{ background: #4361ee; color: #fff; padding: 12px 8px; text-align: center; font-size: 13px; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.recommendations {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.rec-item {{ padding: 16px; border-radius: 8px; }}
.rec-item h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
.footer {{ margin-top: 30px; text-align: center; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<h1>LLM 成本效益分析</h1>
<div class="recommendations">
    <div class="rec-item" style="background:#e8f5e9">
        <h3 style="color:#28a745">✅ 生产推荐</h3>
        <div style="font-size:20px;font-weight:700">{report.recommended_for_production}</div>
        <div style="font-size:13px;color:#666;margin-top:4px">综合质量 & 成本最优</div>
    </div>
    <div class="rec-item" style="background:#fff3e0">
        <h3 style="color:#ff9800">🎯 预算推荐</h3>
        <div style="font-size:20px;font-weight:700">{report.recommended_for_budget}</div>
        <div style="font-size:13px;color:#666;margin-top:4px">预算有限时的最优选择</div>
    </div>
</div>
<div class="card">
    <span style="display:inline-block;padding:4px 12px;border-radius:12px;background:#e8f5e9;color:#28a745;margin-right:8px">🏆 综合最优: {report.best_quality}</span>
    <span style="display:inline-block;padding:4px 12px;border-radius:12px;background:#e3f2fd;color:#1565c0;margin-right:8px">⚡ 速度最快: {report.best_latency}</span>
    <span style="display:inline-block;padding:4px 12px;border-radius:12px;background:#fce4ec;color:#c62828;margin-right:8px">💰 成本最低: {report.cheapest}</span>
    <span style="display:inline-block;padding:4px 12px;border-radius:12px;background:#f3e5f5;color:#7b1fa2">📊 性价比王: {report.best_value}</span>
</div>
<table>
<thead>
<tr>
    <th>模型</th>
    <th>综合分</th>
    <th>延迟</th>
    <th>总成本</th>
    <th>单样本成本</th>
    <th>性价比</th>
    <th>综合评分</th>
    <th>推荐</th>
    <th>标签</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
<div class="footer">
    <p>由 LLM-Eval-Kit 成本效益分析模块生成 | {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML 成本效益报告已保存: {output_path}")
