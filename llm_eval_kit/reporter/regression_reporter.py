"""
Regression reporter for comparing two evaluation runs.

Detects score regressions, improvements, and cost changes
between baseline and new evaluation results.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RegressionDelta:
    model_name: str
    baseline_score: float
    new_score: float
    score_delta: float
    score_delta_pct: float
    baseline_cost: float
    new_cost: float
    cost_delta: float
    regression: bool
    improvement: bool


@dataclass
class RegressionReport:
    deltas: List[RegressionDelta]
    regressions: List[RegressionDelta]
    improvements: List[RegressionDelta]
    total_baseline_cost: float
    total_new_cost: float
    avg_score_delta: float
    score_stable: bool


class RegressionReporter:
    """Compare two evaluation runs and detect regressions."""

    REGRESSION_THRESHOLD = 0.05

    def load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare(self, baseline_path: str, new_path: str) -> RegressionReport:
        baseline = self.load_json(baseline_path)
        new = self.load_json(new_path)

        baseline_models = {m["name"]: m for m in baseline.get("models", [])}
        new_models = {m["name"]: m for m in new.get("models", [])}

        all_models = set(baseline_models.keys()) | set(new_models.keys())

        deltas = []
        for model in sorted(all_models):
            bm = baseline_models.get(model)
            nm = new_models.get(model)

            if bm and nm:
                score_delta = nm["avg_score"] - bm["avg_score"]
                score_delta_pct = score_delta / bm["avg_score"] if bm["avg_score"] > 0 else 0
                cost_delta = nm["total_cost"] - bm["total_cost"]
            elif bm:
                score_delta = -bm["avg_score"]
                score_delta_pct = -1.0
                cost_delta = -bm["total_cost"]
            else:
                score_delta = nm["avg_score"]
                score_delta_pct = 1.0
                cost_delta = nm["total_cost"]

            d = RegressionDelta(
                model_name=model,
                baseline_score=bm["avg_score"] if bm else 0,
                new_score=nm["avg_score"] if nm else 0,
                score_delta=round(score_delta, 4),
                score_delta_pct=round(score_delta_pct, 4),
                baseline_cost=bm["total_cost"] if bm else 0,
                new_cost=nm["total_cost"] if nm else 0,
                cost_delta=round(cost_delta, 4),
                regression=score_delta < -self.REGRESSION_THRESHOLD,
                improvement=score_delta > self.REGRESSION_THRESHOLD,
            )
            deltas.append(d)

        regressions = [d for d in deltas if d.regression]
        improvements = [d for d in deltas if d.improvement]

        total_baseline_cost = sum(d.baseline_cost for d in deltas)
        total_new_cost = sum(d.new_cost for d in deltas)

        score_deltas = [d.score_delta for d in deltas if d.baseline_score > 0 and d.new_score > 0]
        avg_score_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0
        score_stable = abs(avg_score_delta) < self.REGRESSION_THRESHOLD

        return RegressionReport(
            deltas=deltas,
            regressions=regressions,
            improvements=improvements,
            total_baseline_cost=total_baseline_cost,
            total_new_cost=total_new_cost,
            avg_score_delta=avg_score_delta,
            score_stable=score_stable,
        )

    def print_report(self, report: RegressionReport):
        print(f"\n{'='*70}")
        print(f"  回归分析报告")
        print(f"{'='*70}\n")

        header = f"{'模型':<20} {'基线分':>8} {'新分':>8} {'Δ':>8} {'Δ%':>8} {'基线成本':>10} {'新成本':>10}"
        print(header)
        print("-" * 72)

        for d in report.deltas:
            tag = ""
            if d.regression:
                tag = " 🔴"
            elif d.improvement:
                tag = " 🟢"
            print(
                f"{d.model_name:<20} "
                f"{d.baseline_score:>8.4f} "
                f"{d.new_score:>8.4f} "
                f"{d.score_delta:>+8.4f} "
                f"{d.score_delta_pct:>+7.1%}"
                f" {d.baseline_cost:>8.4f} "
                f"{d.new_cost:>8.4f}"
                f"{tag}"
            )

        print("-" * 72)
        print(f"{'汇总':<20} {'':>8} {'':>8} {report.avg_score_delta:>+8.4f} {'':>8} "
              f"{report.total_baseline_cost:>10.4f} {report.total_new_cost:>10.4f}")

        if report.regressions:
            print(f"\n⚠️  发现 {len(report.regressions)} 个回归：")
            for d in report.regressions:
                print(f"    {d.model_name}: {d.baseline_score:.4f} → {d.new_score:.4f} ({d.score_delta_pct:+.1%})")

        if report.improvements:
            print(f"\n✅  {len(report.improvements)} 个改进：")
            for d in report.improvements:
                print(f"    {d.model_name}: {d.baseline_score:.4f} → {d.new_score:.4f} ({d.score_delta_pct:+.1%})")

        status = "✅ 分数稳定" if report.score_stable else "⚠️ 分数波动较大"
        print(f"\n{status}  |  均值变化: {report.avg_score_delta:+.4f}")
        print(f"总成本: {report.total_baseline_cost:.4f} → {report.total_new_cost:.4f}")
        print()

    def generate_html(self, report: RegressionReport, output_path: str):
        rows_html = ""
        for d in report.deltas:
            tag = ""
            if d.regression:
                tag = '<span style="color:#dc3545">🔴 回归</span>'
            elif d.improvement:
                tag = '<span style="color:#28a745">🟢 改进</span>'
            else:
                tag = '<span style="color:#6c757d">─ 持平</span>'
            rows_html += f"""\
<tr>
    <td style="padding:8px;border-bottom:1px solid #eee;font-weight:500">{d.model_name}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{d.baseline_score:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{d.new_score:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center;{'color:#dc3545' if d.regression else 'color:#28a745' if d.improvement else ''}">{d.score_delta:+.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{d.score_delta_pct:+.1%}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{d.baseline_cost:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{d.new_cost:.4f}</td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{tag}</td>
</tr>"""

        status_color = "#28a745" if report.score_stable else "#dc3545"
        status_text = "分数稳定 ✅" if report.score_stable else "分数波动较大 ⚠️"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>LLM 回归分析报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 10px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
th {{ background: #4361ee; color: #fff; padding: 12px 8px; text-align: center; font-size: 14px; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 14px; font-weight: 600; }}
.summary {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 16px 0; }}
.summary-item {{ background: #fff; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.footer {{ margin-top: 30px; text-align: center; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<h1>LLM 回归分析报告</h1>
<div class="summary">
    <div class="summary-item">
        <div style="font-size:12px;color:#999">均值变化</div>
        <div style="font-size:24px;font-weight:700;{'color:#dc3545' if report.avg_score_delta < -0.05 else 'color:#28a745' if report.avg_score_delta > 0.05 else 'color:#333'}">{report.avg_score_delta:+.4f}</div>
    </div>
    <div class="summary-item">
        <div style="font-size:12px;color:#999">基线总成本</div>
        <div style="font-size:24px;font-weight:700">¥{report.total_baseline_cost:.4f}</div>
    </div>
    <div class="summary-item">
        <div style="font-size:12px;color:#999">新总成本</div>
        <div style="font-size:24px;font-weight:700">¥{report.total_new_cost:.4f}</div>
    </div>
</div>
<div class="card">
    <span class="badge" style="background:{status_color}20;color:{status_color}">{status_text}</span>
    <span class="badge" style="background:#e3f2fd;color:#1565c0;margin-left:8px">{len(report.improvements)} 个改进</span>
    <span class="badge" style="background:#fbe9e7;color:#c62828;margin-left:8px">{len(report.regressions)} 个回归</span>
</div>
<table>
<thead>
<tr>
    <th>模型</th>
    <th>基线分</th>
    <th>新分</th>
    <th>Δ</th>
    <th>Δ%</th>
    <th>基线成本</th>
    <th>新成本</th>
    <th>状态</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
<div class="footer">
    <p>由 LLM-Eval-Kit 生成 | {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML 回归报告已保存: {output_path}")
