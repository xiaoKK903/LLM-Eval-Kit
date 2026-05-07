"""Generate an English HTML report for screenshot."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_eval_kit.reporter.models import EvaluationResult
from llm_eval_kit.scorers.rule_scorer import RuleScorer
from llm_eval_kit.reporter.comparator import ModelComparator, ComparisonResult
import datetime
import json

# Build mock data (same as generate_report.py)
scorer = RuleScorer()
test_samples = [
    {"id": "1", "question": "Explain what is Redis and its main use cases", "reference": "Redis is an in-memory data structure store used as cache, message broker, and database."},
    {"id": "2", "question": "What is the difference between TCP and UDP?", "reference": "TCP is connection-oriented with guaranteed delivery; UDP is connectionless with lower latency but no guarantee."},
    {"id": "3", "question": "How does a load balancer work?", "reference": "A load balancer distributes incoming traffic across multiple servers to ensure high availability and reliability."},
]

models_config = [
    {"name": "gpt-4", "base_response": "GPT-4 detailed response with comprehensive explanation"},
    {"name": "gpt-3.5-turbo", "base_response": "GPT-3.5 concise response covering key points"},
    {"name": "claude-3", "base_response": "Claude-3 well-structured response with examples"},
]

results = []
for sample in test_samples:
    for model_cfg in models_config:
        response_text = f"{model_cfg['base_response']} about {sample['question'][:30]}..."
        score_result = scorer.score(sample["question"], response_text, sample["reference"])
        results.append(EvaluationResult(
            sample_id=sample["id"],
            question=sample["question"],
            response=response_text,
            latency={"gpt-4": 3.2, "gpt-3.5-turbo": 1.8, "claude-3": 4.5}[model_cfg["name"]],
            token_usage={"prompt_tokens": {"gpt-4": 200, "gpt-3.5-turbo": 120, "claude-3": 280}[model_cfg["name"]],
                          "completion_tokens": {"gpt-4": 320, "gpt-3.5-turbo": 190, "claude-3": 400}[model_cfg["name"]]},
            model=model_cfg["name"],
            quality_scores={"relevance": 0.9, "completeness": 0.85},
            cost={"gpt-4": 0.015, "gpt-3.5-turbo": 0.002, "claude-3": 0.018}[model_cfg["name"]],
            scoring_result={"total_score": score_result.total_score, "reasoning": score_result.reasoning or "Meets quality standards"}
        ))

# Build comparison
comparator = ModelComparator()
comparison = comparator.compare_models(results)

# Render English HTML
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def build_html():
    header = build_header(comparison)
    table = build_table(comparison)
    cards = build_cards(comparison)
    samples = build_samples(results)
    footer = f"""<div style="text-align:center;padding:20px 0 10px;font-size:12px;color:#bbb;border-top:1px solid #eee;margin-top:10px;">LLM-Eval-Kit &copy; {datetime.datetime.now().year} &mdash; Generated at {timestamp}</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Evaluation Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif; background: #f5f7fa; color: #333; padding: 30px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="container">
{header}{table}{cards}{samples}{footer}
</div>
</body>
</html>"""

def build_header(comp):
    model_count = len(comp.model_comparisons)
    sample_count = sum(m.sample_count for m in comp.model_comparisons)
    return f"""
<div style="background:linear-gradient(135deg,#1a73e8,#0d47a1);color:white;border-radius:12px;padding:30px 35px;margin-bottom:25px;box-shadow:0 4px 15px rgba(26,115,232,0.3);">
  <h1 style="font-size:26px;font-weight:600;margin-bottom:8px;">LLM Evaluation Report</h1>
  <p style="font-size:14px;opacity:0.85;margin-bottom:18px;">Generated: {timestamp}</p>
  <div style="display:flex;gap:30px;flex-wrap:wrap;">
    <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:12px 20px;text-align:center;">
      <div style="font-size:24px;font-weight:700;">{sample_count}</div>
      <div style="font-size:12px;opacity:0.8;">Samples Evaluated</div>
    </div>
    <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:12px 20px;text-align:center;">
      <div style="font-size:24px;font-weight:700;">{model_count}</div>
      <div style="font-size:12px;opacity:0.8;">Models Compared</div>
    </div>
    <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:12px 20px;text-align:center;">
      <div style="font-size:14px;font-weight:700;">demo</div>
      <div style="font-size:12px;opacity:0.8;">Dataset</div>
    </div>
  </div>
</div>"""

def build_table(comp):
    rows = ""
    best = comp.best_overall_model
    for i, mc in enumerate(comp.model_comparisons):
        is_best = mc.model_name == best
        bg = "#e8f0fe" if is_best else ("#fafafa" if i % 2 == 0 else "#ffffff")
        bold = "font-weight:700;" if is_best else ""
        trophy = " 🏆" if is_best else ""
        rows += f"""
    <tr style="background:{bg};{bold}">
      <td style="padding:10px 12px;border-bottom:1px solid #e8e8e8;">{mc.model_name}{trophy}</td>
      <td style="padding:10px 12px;border-bottom:1px solid #e8e8e8;text-align:center;">{mc.avg_score:.2f}</td>
      <td style="padding:10px 12px;border-bottom:1px solid #e8e8e8;text-align:center;">{mc.avg_latency:.1f}</td>
      <td style="padding:10px 12px;border-bottom:1px solid #e8e8e8;text-align:center;">{mc.total_tokens}</td>
      <td style="padding:10px 12px;border-bottom:1px solid #e8e8e8;text-align:center;">${mc.total_cost:.4f}</td>
      <td style="padding:10px 12px;border-bottom:1px solid #e8e8e8;text-align:center;">{mc.success_rate*100:.0f}%</td>
    </tr>"""
    return f"""
<div style="background:white;border-radius:12px;padding:20px 25px;margin-bottom:25px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
  <h2 style="font-size:18px;font-weight:600;margin-bottom:15px;color:#1a73e8;">Model Comparison Overview</h2>
  <table style="width:100%;border-collapse:collapse;font-size:14px;">
    <thead>
      <tr style="background:#f0f4f8;">
        <th style="padding:10px 12px;text-align:left;font-weight:600;color:#555;border-bottom:2px solid #ddd;">Model</th>
        <th style="padding:10px 12px;text-align:center;font-weight:600;color:#555;border-bottom:2px solid #ddd;">Avg Score</th>
        <th style="padding:10px 12px;text-align:center;font-weight:600;color:#555;border-bottom:2px solid #ddd;">Latency(s)</th>
        <th style="padding:10px 12px;text-align:center;font-weight:600;color:#555;border-bottom:2px solid #ddd;">Tokens</th>
        <th style="padding:10px 12px;text-align:center;font-weight:600;color:#555;border-bottom:2px solid #ddd;">Cost($)</th>
        <th style="padding:10px 12px;text-align:center;font-weight:600;color:#555;border-bottom:2px solid #ddd;">Success</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <p style="font-size:12px;color:#999;margin-top:10px;">🏆 Best overall model</p>
</div>"""

def build_cards(comp):
    cards_data = [
        ("🏆", "Best Overall", comp.best_overall_model, f"Highest score: {comp.best_overall_score:.2f}", "#fff3e0", "#e65100"),
        ("⚡", "Fastest", comp.fastest_model, f"Avg {comp.fastest_latency:.1f}s", "#e8f5e9", "#2e7d32"),
        ("💰", "Cheapest", comp.cheapest_model, f"${comp.cheapest_cost:.4f}/sample", "#e3f2fd", "#1565c0"),
        ("📈", "Best Value", comp.best_value_model, "Highest score/cost ratio", "#f3e5f5", "#6a1b9a"),
    ]
    cards = ""
    for icon, title, model, desc, bg, color in cards_data:
        cards += f"""
  <div style="background:white;border-radius:12px;padding:18px 20px;flex:1;min-width:200px;box-shadow:0 2px 8px rgba(0,0,0,0.06);border-left:4px solid {color};">
    <div style="font-size:24px;margin-bottom:8px;">{icon}</div>
    <div style="font-size:13px;color:#888;margin-bottom:4px;">{title}</div>
    <div style="font-size:17px;font-weight:700;color:{color};margin-bottom:4px;">{model}</div>
    <div style="font-size:12px;color:#999;">{desc}</div>
  </div>"""
    return f"""<div style="display:flex;gap:15px;flex-wrap:wrap;margin-bottom:25px;">{cards}</div>"""

def build_samples(results_list):
    sample_ids = sorted(set(r.sample_id for r in results_list))
    details = ""
    for sid in sample_ids:
        sr = [r for r in results_list if r.sample_id == sid]
        if not sr:
            continue
        first = sr[0]
        models_html = ""
        for r in sr:
            score_str = f"{r.scoring_result.get('total_score',0):.2f}" if r.scoring_result else "N/A"
            reasoning = r.scoring_result.get("reasoning","") if r.scoring_result else ""
            escaped_resp = r.response.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            escaped_reason = reasoning.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") if reasoning else ""
            models_html += f"""
      <div style="border:1px solid #e8e8e8;border-radius:8px;padding:12px 15px;margin-bottom:10px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <span style="font-weight:600;font-size:14px;color:#1a73e8;">{r.model}</span>
          <span style="background:#e8f0fe;color:#1a73e8;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;">Score: {score_str}</span>
          <span style="font-size:12px;color:#999;">Latency: {r.latency:.1f}s</span>
        </div>
        <div style="font-size:13px;color:#555;line-height:1.6;white-space:pre-wrap;">{escaped_resp}</div>
        {f'<div style="font-size:12px;color:#888;margin-top:4px;font-style:italic;">{escaped_reason}</div>' if escaped_reason else ''}
      </div>"""
        escaped_q = first.question.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        escaped_ref = first.question.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        details += f"""
  <div style="background:white;border-radius:12px;padding:20px 25px;margin-bottom:15px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <h3 style="font-size:15px;font-weight:600;color:#333;margin-bottom:12px;">Sample #{sid}</h3>
    <div style="background:#f9fafb;border-radius:8px;padding:12px 15px;margin-bottom:12px;">
      <div style="font-size:12px;color:#888;margin-bottom:4px;">📝 Question</div>
      <div style="font-size:14px;color:#333;line-height:1.6;">{escaped_q}</div>
    </div>
    <div style="margin-bottom:12px;">
      <div style="font-size:12px;color:#888;margin-bottom:4px;">🎯 Reference Answer</div>
      <div style="font-size:13px;color:#666;line-height:1.6;">{escaped_ref}</div>
    </div>
    <div>
      <div style="font-size:12px;color:#888;margin-bottom:8px;">🤖 Model Responses</div>
{models_html}
    </div>
  </div>"""
    return f"""<h2 style="font-size:18px;font-weight:600;margin-bottom:15px;color:#1a73e8;">Sample Details</h2>{details}"""

html = build_html()
output_path = os.path.join(os.path.dirname(__file__), "..", "reports", "report_en_demo.html")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"✅ English report generated: {output_path}")
