"""
HTML reporter for generating standalone evaluation reports.

This module generates a self-contained HTML report file with all styles
inlined, requiring no external dependencies like Jinja2 or CSS frameworks.
"""

import os
import datetime
from typing import List, Dict, Any, Optional
from .models import EvaluationResult
from .comparator import ModelComparator, ComparisonResult, ModelComparison


class HtmlReporter:
    """Reporter for generating standalone HTML evaluation reports."""

    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        self.comparator = ModelComparator(custom_pricing)

    def generate_report(self, results: List[EvaluationResult],
                        data_name: str = "unknown",
                        output_dir: str = "reports") -> str:
        if not results:
            raise ValueError("No results to generate report from.")

        comparison = self.comparator.compare_models(results)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_safe = timestamp.replace(":", "-").replace(" ", "_")
        os.makedirs(output_dir, exist_ok=True)

        html = self._build_html(results, comparison, data_name, timestamp)

        output_path = os.path.join(output_dir, f"report_{filename_safe}.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    def _build_html(self, results: List[EvaluationResult],
                    comparison: ComparisonResult,
                    data_name: str, timestamp: str) -> str:
        header_section = self._build_header_section(comparison, data_name, timestamp)
        overview_table = self._build_overview_table(comparison)
        conclusion_cards = self._build_conclusion_cards(comparison)
        sample_details = self._build_sample_details(results)
        footer = self._build_footer(timestamp)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM 评测报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f7fa; color: #333; padding: 30px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="container">
{header_section}
{overview_table}
{conclusion_cards}
{sample_details}
{footer}
</div>
</body>
</html>"""

    def _build_header_section(self, comparison: ComparisonResult,
                               data_name: str, timestamp: str) -> str:
        model_count = len(comparison.model_comparisons)
        sample_count = sum(m.sample_count for m in comparison.model_comparisons)
        return f"""
<div style="background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; border-radius: 12px; padding: 30px 35px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(26,115,232,0.3);">
  <h1 style="font-size: 26px; font-weight: 600; margin-bottom: 8px;">LLM 评测报告</h1>
  <p style="font-size: 14px; opacity: 0.85; margin-bottom: 18px;">生成时间：{timestamp}</p>
  <div style="display: flex; gap: 30px; flex-wrap: wrap;">
    <div style="background: rgba(255,255,255,0.12); border-radius: 8px; padding: 12px 20px; text-align: center;">
      <div style="font-size: 24px; font-weight: 700;">{sample_count}</div>
      <div style="font-size: 12px; opacity: 0.8;">评测样本</div>
    </div>
    <div style="background: rgba(255,255,255,0.12); border-radius: 8px; padding: 12px 20px; text-align: center;">
      <div style="font-size: 24px; font-weight: 700;">{model_count}</div>
      <div style="font-size: 12px; opacity: 0.8;">对比模型</div>
    </div>
    <div style="background: rgba(255,255,255,0.12); border-radius: 8px; padding: 12px 20px; text-align: center;">
      <div style="font-size: 14px; font-weight: 700;">{data_name}</div>
      <div style="font-size: 12px; opacity: 0.8;">数据集</div>
    </div>
  </div>
</div>"""

    def _build_overview_table(self, comparison: ComparisonResult) -> str:
        rows_html = ""
        best_name = comparison.best_overall_model
        for mc in comparison.model_comparisons:
            is_best = mc.model_name == best_name
            bg = "#e8f0fe" if is_best else ("#fafafa" if comparison.model_comparisons.index(mc) % 2 == 0 else "#ffffff")
            bold = "font-weight: 700;" if is_best else ""
            trophy = " 🏆" if is_best else ""
            rows_html += f"""
    <tr style="background: {bg}; {bold}">
      <td style="padding: 10px 12px; border-bottom: 1px solid #e8e8e8;">{mc.model_name}{trophy}</td>
      <td style="padding: 10px 12px; border-bottom: 1px solid #e8e8e8; text-align: center;">{mc.avg_score:.2f}</td>
      <td style="padding: 10px 12px; border-bottom: 1px solid #e8e8e8; text-align: center;">{mc.avg_latency:.1f}</td>
      <td style="padding: 10px 12px; border-bottom: 1px solid #e8e8e8; text-align: center;">{mc.total_tokens}</td>
      <td style="padding: 10px 12px; border-bottom: 1px solid #e8e8e8; text-align: center;">¥{mc.total_cost:.4f}</td>
      <td style="padding: 10px 12px; border-bottom: 1px solid #e8e8e8; text-align: center;">{mc.success_rate * 100:.0f}%</td>
    </tr>"""

        return f"""
<div style="background: white; border-radius: 12px; padding: 20px 25px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
  <h2 style="font-size: 18px; font-weight: 600; margin-bottom: 15px; color: #1a73e8;">模型对比总览</h2>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <thead>
      <tr style="background: #f0f4f8;">
        <th style="padding: 10px 12px; text-align: left; font-weight: 600; color: #555; border-bottom: 2px solid #ddd;">模型名称</th>
        <th style="padding: 10px 12px; text-align: center; font-weight: 600; color: #555; border-bottom: 2px solid #ddd;">平均分</th>
        <th style="padding: 10px 12px; text-align: center; font-weight: 600; color: #555; border-bottom: 2px solid #ddd;">延迟(s)</th>
        <th style="padding: 10px 12px; text-align: center; font-weight: 600; color: #555; border-bottom: 2px solid #ddd;">Token</th>
        <th style="padding: 10px 12px; text-align: center; font-weight: 600; color: #555; border-bottom: 2px solid #ddd;">成本(¥)</th>
        <th style="padding: 10px 12px; text-align: center; font-weight: 600; color: #555; border-bottom: 2px solid #ddd;">成功率</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
  <p style="font-size: 12px; color: #999; margin-top: 10px;">🏆 标记表示综合最优模型</p>
</div>"""

    def _build_conclusion_cards(self, comparison: ComparisonResult) -> str:
        cards = [
            ("🏆", "综合最优", comparison.best_overall_model,
             f"得分最高 {comparison.best_overall_score:.2f} 分", "#fff3e0", "#e65100"),
            ("⚡", "速度最快", comparison.fastest_model,
             f"平均 {comparison.fastest_latency:.1f}s", "#e8f5e9", "#2e7d32"),
            ("💰", "成本最低", comparison.cheapest_model,
             f"¥{comparison.cheapest_cost:.4f}/次", "#e3f2fd", "#1565c0"),
            ("📈", "性价比王", comparison.best_value_model,
             "得分/成本比最高", "#f3e5f5", "#6a1b9a"),
        ]

        cards_html = ""
        for icon, title, model, desc, bg, color in cards:
            cards_html += f"""
  <div style="background: white; border-radius: 12px; padding: 18px 20px; flex: 1; min-width: 200px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid {color};">
    <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
    <div style="font-size: 13px; color: #888; margin-bottom: 4px;">{title}</div>
    <div style="font-size: 17px; font-weight: 700; color: {color}; margin-bottom: 4px;">{model}</div>
    <div style="font-size: 12px; color: #999;">{desc}</div>
  </div>"""

        return f"""
<div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 25px;">
{cards_html}
</div>"""

    def _build_sample_details(self, results: List[EvaluationResult]) -> str:
        sample_ids = sorted(set(r.sample_id for r in results))
        models = list(dict.fromkeys(r.model for r in results))

        details_html = ""
        for sid in sample_ids:
            sample_results = [r for r in results if r.sample_id == sid]
            if not sample_results:
                continue
            first = sample_results[0]
            question = first.question
            reference = first.question

            models_html = ""
            for r in sample_results:
                score_str = f"{r.scoring_result.get('total_score', 0):.2f}" if r.scoring_result else "N/A"
                reasoning = r.scoring_result.get("reasoning", "") if r.scoring_result else ""
                models_html += f"""
      <div style="border: 1px solid #e8e8e8; border-radius: 8px; padding: 12px 15px; margin-bottom: 10px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
          <span style="font-weight: 600; font-size: 14px; color: #1a73e8;">{r.model}</span>
          <span style="background: #e8f0fe; color: #1a73e8; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600;">得分: {score_str}</span>
          <span style="font-size: 12px; color: #999;">延迟: {r.latency:.1f}s</span>
        </div>
        <div style="font-size: 13px; color: #555; line-height: 1.6; white-space: pre-wrap;">{self._escape_html(r.response)}</div>
        {f'<div style="font-size: 12px; color: #888; margin-top: 4px; font-style: italic;">{self._escape_html(reasoning)}</div>' if reasoning else ''}
      </div>"""

            details_html += f"""
  <div style="background: white; border-radius: 12px; padding: 20px 25px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
      <h3 style="font-size: 15px; font-weight: 600; color: #333;">样本 #{sid}</h3>
    </div>
    <div style="background: #f9fafb; border-radius: 8px; padding: 12px 15px; margin-bottom: 12px;">
      <div style="font-size: 12px; color: #888; margin-bottom: 4px;">📝 问题</div>
      <div style="font-size: 14px; color: #333; line-height: 1.6;">{self._escape_html(question)}</div>
    </div>
    <div style="margin-bottom: 12px;">
      <div style="font-size: 12px; color: #888; margin-bottom: 4px;">🎯 参考答案</div>
      <div style="font-size: 13px; color: #666; line-height: 1.6;">{self._escape_html(reference)}</div>
    </div>
    <div>
      <div style="font-size: 12px; color: #888; margin-bottom: 8px;">🤖 模型回答</div>
{models_html}
    </div>
  </div>"""

        return f"""
<h2 style="font-size: 18px; font-weight: 600; margin-bottom: 15px; color: #1a73e8;">样本详细结果</h2>
{details_html}"""

    def _build_footer(self, timestamp: str) -> str:
        return f"""
<div style="text-align: center; padding: 20px 0 10px; font-size: 12px; color: #bbb; border-top: 1px solid #eee; margin-top: 10px;">
  LLM-Eval-Kit &copy; {datetime.datetime.now().year} &mdash; 生成于 {timestamp}
</div>"""

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters in text."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))