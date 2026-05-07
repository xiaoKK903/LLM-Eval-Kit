"""
Regression analysis reporter.

Compares two evaluation runs (baseline vs current) to detect
regressions and improvements in model performance.

Usage:
    diff = RegressionReporter.compare(baseline_path, current_path)
    RegressionReporter.print_diff(diff)
    RegressionReporter.save_diff_html(diff, "regression_report.html")
"""

import json
import sys
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SampleDiff:
    score_change: float
    latency_change: float
    cost_change: float
    old_score: float
    new_score: float
    old_latency: float
    new_latency: float
    old_cost: float
    new_cost: float
    regression: bool
    improvement: bool


@dataclass
class ModelDiff:
    model_name: str
    score_change: float
    latency_change: float
    cost_change: float
    score_per_cost_change: float
    old_avg_score: float
    new_avg_score: float
    old_avg_latency: float
    new_avg_latency: float
    old_total_cost: float
    new_total_cost: float
    old_score_per_cost: float
    new_score_per_cost: float
    sample_diffs: Dict[str, SampleDiff] = field(default_factory=dict)
    regression_count: int = 0
    improvement_count: int = 0


@dataclass
class RegressionResult:
    baseline_name: str
    current_name: str
    baseline_time: str
    current_time: str
    model_diffs: Dict[str, ModelDiff]
    total_regressions: int
    total_improvements: int
    best_change: str
    worst_change: str


class RegressionReporter:
    REGRESSION_THRESHOLD = 0.05

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_score(result: Dict[str, Any]) -> float:
        if "scoring_result" in result:
            sr = result["scoring_result"]
            return sr.get("total_score", 0.0) if isinstance(sr, dict) else (sr or 0.0)
        return result.get("avg_score", 0.0)

    @staticmethod
    def compare(
        baseline_path: str,
        current_path: str,
        regression_threshold: float = 0.05,
    ) -> RegressionResult:
        baseline = RegressionReporter._load_json(baseline_path)
        current = RegressionReporter._load_json(current_path)

        baseline_models = {m["name"]: m for m in baseline.get("models", [])}
        current_models = {m["name"]: m for m in current.get("models", [])}

        model_names = set(list(baseline_models.keys()) + list(current_models.keys()))
        model_diffs: Dict[str, ModelDiff] = {}
        total_regressions = 0
        total_improvements = 0

        for name in sorted(model_names):
            bm = baseline_models.get(name)
            cm = current_models.get(name)

            if bm and cm:
                score_ch = cm["avg_score"] - bm["avg_score"]
                lat_ch = cm["avg_latency"] - bm["avg_latency"]
                cost_ch = cm["total_cost"] - bm["total_cost"]
                spc_old = bm.get("score_per_cost", 0)
                spc_new = cm.get("score_per_cost", 0)
                spc_ch = spc_new - spc_old
            elif bm:
                score_ch = -bm["avg_score"]
                lat_ch = -bm["avg_latency"]
                cost_ch = -bm["total_cost"]
                spc_ch = -bm.get("score_per_cost", 0)
            else:
                score_ch = cm["avg_score"]
                lat_ch = cm["avg_latency"]
                cost_ch = cm["total_cost"]
                spc_ch = cm.get("score_per_cost", 0)

            old_score = bm["avg_score"] if bm else 0.0
            new_score = cm["avg_score"] if cm else 0.0
            old_lat = bm["avg_latency"] if bm else 0.0
            new_lat = cm["avg_latency"] if cm else 0.0
            old_cost = bm["total_cost"] if bm else 0.0
            new_cost = cm["total_cost"] if cm else 0.0
            old_spc = bm.get("score_per_cost", 0) if bm else 0.0
            new_spc = cm.get("score_per_cost", 0) if cm else 0.0

            if score_ch < -regression_threshold:
                total_regressions += 1
            if score_ch > regression_threshold:
                total_improvements += 1

            model_diffs[name] = ModelDiff(
                model_name=name,
                score_change=round(score_ch, 4),
                latency_change=round(lat_ch, 4),
                cost_change=round(cost_ch, 4),
                score_per_cost_change=round(spc_ch, 4),
                old_avg_score=round(old_score, 4),
                new_avg_score=round(new_score, 4),
                old_avg_latency=round(old_lat, 4),
                new_avg_latency=round(new_lat, 4),
                old_total_cost=round(old_cost, 4),
                new_total_cost=round(new_cost, 4),
                old_score_per_cost=round(old_spc, 4),
                new_score_per_cost=round(new_spc, 4),
            )

        best_change_name = max(model_diffs, key=lambda k: model_diffs[k].score_change)
        worst_change_name = min(model_diffs, key=lambda k: model_diffs[k].score_change)

        return RegressionResult(
            baseline_name=baseline_path.split("\\")[-1],
            current_name=current_path.split("\\")[-1],
            baseline_time=baseline.get("timestamp", "unknown"),
            current_time=current.get("timestamp", "unknown"),
            model_diffs=model_diffs,
            total_regressions=total_regressions,
            total_improvements=total_improvements,
            best_change=best_change_name,
            worst_change=worst_change_name,
        )

    @staticmethod
    def print_diff(result: RegressionResult):
        print("\n" + "=" * 72)
        print("  LLM-Eval-Kit 回归分析报告")
        print("=" * 72)
        print(f"  基线: {result.baseline_name}  ({result.baseline_time})")
        print(f"  当前: {result.current_name}  ({result.current_time})")
        print(f"  回归: {result.total_regressions}  |  改进: {result.total_improvements}")
        print("-" * 72)

        header = f"{'模型':<22} {'评分变化':>12} {'延迟变化':>12} {'成本变化':>12} {'性价比变化':>12}"
        print(header)
        print("-" * 72)

        for name, d in sorted(result.model_diffs.items()):
            score_str = f"{d.score_change:+.4f}"
            lat_str = f"{d.latency_change:+.4f}s"
            cost_str = f"{d.cost_change:+.4f}"
            spc_str = f"{d.score_per_cost_change:+.4f}"

            if d.score_change > 0.05:
                score_str = f"\033[32m{score_str}\033[0m"
            elif d.score_change < -0.05:
                score_str = f"\033[31m{score_str}\033[0m"

            print(f"{name:<22} {score_str:>12} {lat_str:>12} {cost_str:>12} {spc_str:>12}")

        print("-" * 72)
        if result.model_diffs:
            best = result.model_diffs[result.best_change]
            worst = result.model_diffs[result.worst_change]
            print(f"  最佳改进: {result.best_change}  ({best.score_change:+.4f})")
            print(f"  最大退化: {result.worst_change}  ({worst.score_change:+.4f})")
        print("=" * 72 + "\n")

    @staticmethod
    def save_diff_html(result: RegressionResult, output_path: str):
        rows_html = ""
        for name, d in sorted(result.model_diffs.items()):
            direction = "→"
            color = "#666"
            if d.score_change > 0.05:
                direction = "↑"
                color = "#22c55e"
            elif d.score_change < -0.05:
                direction = "↓"
                color = "#ef4444"

            rows_html += f"""
            <tr>
                <td style="padding:10px;border-bottom:1px solid #eee;font-weight:600">{name}</td>
                <td style="padding:10px;border-bottom:1px solid #eee;color:{color};font-weight:600">
                    {direction} {d.score_change:+.4f}
                </td>
                <td style="padding:10px;border-bottom:1px solid #eee">{d.old_avg_score:.4f} → {d.new_avg_score:.4f}</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{d.latency_change:+.4f}s</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{d.cost_change:+.4f}</td>
                <td style="padding:10px;border-bottom:1px solid #eee">{d.score_per_cost_change:+.4f}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>LLM 评测回归分析</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #333; }}
h1 {{ font-size: 24px; margin-bottom: 4px; }}
.header {{ background: linear-gradient(135deg, #1e293b, #334155); color: white; padding: 24px 32px; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ margin: 0 0 8px; }}
.header p {{ margin: 2px 0; opacity: 0.85; font-size: 14px; }}
.summary {{ display: flex; gap: 16px; margin-bottom: 24px; }}
.card {{ flex: 1; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
.card .num {{ font-size: 28px; font-weight: 700; }}
.card .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
.card.red .num {{ color: #ef4444; }}
.card.green .num {{ color: #22c55e; }}
.card.blue .num {{ color: #3b82f6; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th {{ background: #f1f5f9; padding: 12px 10px; font-size: 13px; text-align: left; font-weight: 600; color: #475569; }}
td {{ font-size: 14px; }}
.tag {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
.tag.green {{ background: #dcfce7; color: #166534; }}
.tag.red {{ background: #fce4ec; color: #c62828; }}
.footer {{ text-align: center; color: #94a3b8; font-size: 12px; margin-top: 32px; }}
</style>
</head>
<body>
<div class="header">
    <h1>📊 LLM 评测回归分析</h1>
    <p>基线: {result.baseline_name} ({result.baseline_time})</p>
    <p>当前: {result.current_name} ({result.current_time})</p>
</div>

<div class="summary">
    <div class="card green">
        <div class="num">{result.total_improvements}</div>
        <div class="label">改进模型</div>
    </div>
    <div class="card red">
        <div class="num">{result.total_regressions}</div>
        <div class="label">回归模型</div>
    </div>
    <div class="card blue">
        <div class="num">{len(result.model_diffs)}</div>
        <div class="label">模型总数</div>
    </div>
</div>

<table>
<tr>
    <th>模型</th>
    <th>评分变化</th>
    <th>评分（旧→新）</th>
    <th>延迟变化</th>
    <th>成本变化</th>
    <th>性价比变化</th>
</tr>
{rows_html}
</table>

<div style="margin-top:20px;background:white;padding:20px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
    <h3 style="margin:0 0 12px">结论</h3>
    <p>最佳改进: <strong>{result.best_change}</strong> ({result.model_diffs[result.best_change].score_change:+.4f})</p>
    <p>最大退化: <strong>{result.worst_change}</strong> ({result.model_diffs[result.worst_change].score_change:+.4f})</p>
</div>

<div class="footer">LLM-Eval-Kit · 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  回归分析报告已保存: {output_path}")
