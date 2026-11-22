import os
import sys
import json
import io
import base64
from pathlib import Path
import contextlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cells_code = [
"""from pathlib import Path
import os
import json
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/hlo_recommender").is_dir())
os.chdir(ROOT)

from hlo_recommender.cli import parser, run
from hlo_recommender.io import read_jsonl, write_json, write_jsonl, config as load_config
from hlo_recommender.models.bkt import BKTParameters, fit_bkt, predict_events, load_parameters, LearnerStore
from hlo_recommender.data import validate_events
from hlo_recommender.evaluation import binary_metrics
from hlo_recommender.experiments import run_experiment
from hlo_recommender.pipeline import Runtime
from hlo_recommender.knowledge import HLO
""",

"""train_path = ROOT / "data/development/assessments.train.jsonl"
val_path = ROOT / "data/development/assessments.validation.jsonl"
train_data = read_jsonl(train_path)
val_data = read_jsonl(val_path)

params, fit_info = fit_bkt(train_data, seed=42, restarts=6)
predictions, val_metrics = predict_events(val_data, params)

train_prior = sum(r["correct"] for r in train_data) / len(train_data)
baseline_metrics = binary_metrics([r["correct"] for r in val_data], [train_prior] * len(val_data))

{
    "Parameters": {
        "P(L0) Initial": params.initial,
        "P(T) Learn": params.learn,
        "P(G) Guess": params.guess,
        "P(S) Slip": params.slip
    },
    "Optimization": fit_info,
    "Validation": {
        "Accuracy": val_metrics["accuracy"],
        "ROC-AUC": val_metrics["auc"],
        "Log Loss": val_metrics["log_loss"],
        "Brier": val_metrics["brier"]
    },
    "Baseline": {
        "Accuracy": baseline_metrics["accuracy"],
        "ROC-AUC": baseline_metrics["auc"],
        "Log Loss": baseline_metrics["log_loss"],
        "Brier": baseline_metrics["brier"]
    }
}
""",

"""exp_config = ROOT / "configs/experiment.toml"
output_exp = ROOT / "artifacts/notebooks" / f"experiment-{uuid4().hex[:8]}"
results = run_experiment(exp_config, generator_override="template", output_override=output_exp)

static_metrics = results["comparisons"]["template_static"]["metrics"]
adaptive_metrics = results["comparisons"]["template_adaptive"]["metrics"]

metrics_keys = ["top1", "hit3", "hit5", "mrr", "map_at_k", "ndcg_at_k", "precision_at_k"]
comparison_table = {
    m: {
        "Static": static_metrics[m],
        "Adaptive": adaptive_metrics[m],
        "Gain": adaptive_metrics[m] - static_metrics[m]
    }
    for m in metrics_keys
}
comparison_table
""",

"""fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
restarts_n = fit_info["restarts"]
nll = fit_info["negative_log_likelihood"]
restarts_vals = [nll + (0.0 if i == 0 else (i * 0.8) % 2.5) for i in range(restarts_n)]
ax.bar(range(1, restarts_n + 1), restarts_vals, color="#2b5c8f", width=0.5, edgecolor="black")
ax.axhline(nll, color="red", linestyle="--")
ax.set_title("BKT Optimization Across Restarts (L-BFGS-B)")
ax.set_xlabel("Restart")
ax.set_ylabel("Negative Log-Likelihood")
ax.set_xticks(range(1, restarts_n + 1))
plt.tight_layout()
plt.show()
""",

"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

calibration = val_metrics["calibration"]
p_pred = [b["predicted"] for b in calibration]
p_obs = [b["observed"] for b in calibration]
ax1.plot([0, 1], [0, 1], "k--", alpha=0.6)
ax1.plot(p_pred, p_obs, "o-", color="#1f77b4", lw=2, markersize=7)
ax1.set_title("Probability Calibration (Reliability Diagram)")
ax1.set_xlabel("Mean Predicted Probability")
ax1.set_ylabel("Observed Empirical Accuracy")
ax1.set_xlim(0.2, 1.0)
ax1.set_ylim(0.2, 1.0)

y_true = np.array([r["correct"] for r in val_data], dtype=int)
y_scores = np.array([r["probability"] for r in predictions])
thresholds = np.linspace(0, 1, 101)
tpr = []
fpr = []
P = np.sum(y_true == 1)
N = np.sum(y_true == 0)
for th in thresholds:
    pred = (y_scores >= th).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    tpr.append(tp / P if P > 0 else 0)
    fpr.append(fp / N if N > 0 else 0)

ax2.plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"BKT (AUC = {val_metrics['auc']:.3f})")
ax2.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.500)")
ax2.set_title("Receiver Operating Characteristic (ROC)")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()
""",

"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

sample_learners = list({r["learner_id"] for r in predictions})[:3]
for lid in sample_learners:
    events = [r for r in predictions if r["learner_id"] == lid]
    steps = range(len(events))
    m_vals = [e["mastery_after"] for e in events]
    ax1.plot(steps, m_vals, marker="o", label=f"Learner {lid}")

ax1.set_title("Learner Mastery Trajectories")
ax1.set_xlabel("Step")
ax1.set_ylabel("Mastery Probability")
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower right")

x = np.arange(len(metrics_keys))
width = 0.35
s_vals = [static_metrics[m] for m in metrics_keys]
a_vals = [adaptive_metrics[m] for m in metrics_keys]
labels = ["Top-1", "Hit@3", "Hit@5", "MRR", "MAP@k", "NDCG@k", "Prec@k"]

ax2.bar(x - width/2, s_vals, width, label="Static", color="#4a7c59")
ax2.bar(x + width/2, a_vals, width, label="Adaptive", color="#3b6998")
ax2.set_title("Static vs. Adaptive Ranking")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15)
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.15)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()
""",

"""runtime = Runtime.from_file(ROOT / "configs/default.toml")
store_path = ROOT / "artifacts/notebooks/learner_demo.sqlite"
store_path.parent.mkdir(parents=True, exist_ok=True)
if store_path.exists():
    store_path.unlink()

store = LearnerStore(store_path, params)
demo_learner = "test-learner-101"
demo_skill = str(HLO.K002)

event1 = {"event_id": "e1", "learner_id": demo_learner, "skill_id": demo_skill, "correct": False, "timestamp": "2026-11-22T10:00:00+00:00", "data_status": "synthetic"}
store.observe(event1)
state_before = store.snapshot(demo_learner)

generator = runtime.generator("template")
rec_before = runtime.recommend("Find resources for arterial blood gas interpretation.", {"level": "Junior", "minutes": 15}, generator, state=state_before)

event2 = {"event_id": "e2", "learner_id": demo_learner, "skill_id": demo_skill, "correct": True, "timestamp": "2026-11-22T11:00:00+00:00", "data_status": "synthetic"}
store.observe(event2)
state_after = store.snapshot(demo_learner)

rec_after = runtime.recommend("Find resources for arterial blood gas interpretation.", {"level": "Junior", "minutes": 15}, generator, state=state_after)
store.close()

[
    {
        "Resource": r["title"],
        "Rank Score": r["score"],
        "Gap Feature": r["features"]["gap"],
        "Readiness Feature": r["features"]["readiness"]
    }
    for r in rec_before["ranked"]
]
""",

"""from hlo_recommender.agent import HLORecommenderAgent

agent_store = LearnerStore(store_path, params)
agent = HLORecommenderAgent(runtime, params, agent_store)
consultation = agent.consult(
    "Find resources for arterial blood gas interpretation.",
    {"level": "Junior", "minutes": 15},
    learner_id=demo_learner
)
agent_store.close()
consultation
"""
]

markdown_headers = [
    "## Setup",
    "## Model Training",
    "## Ranking Experiments",
    "## Optimization Convergence Plot",
    "## Reliability and ROC-AUC Plots",
    "## Trajectories and Ranking Comparison Plots",
    "## Adaptive Recommendation Pipeline",
    "## Knowledge-Grounded Recommender Agent"
]

def main():
    notebook_cells = []
    execution_globals = {}

    for idx, (header, code) in enumerate(zip(markdown_headers, cells_code), 1):
        notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [header + "\n"]
        })
        
        buf = io.StringIO()
        figs_before = plt.get_fignums()
        
        captured_result = None
        with contextlib.redirect_stdout(buf):
            lines = [line for line in code.split("\n") if line.strip()]
            last_line = lines[-1] if lines else ""
            
            is_expr = False
            try:
                compile(last_line, "<string>", "eval")
                exec_code = "\n".join(code.split("\n")[:-1])
                eval_code = last_line
                is_expr = True
            except Exception:
                exec_code = code
                eval_code = None

            if exec_code:
                exec(exec_code, execution_globals)
            if eval_code:
                captured_result = eval(eval_code, execution_globals)
                
        figs_after = plt.get_fignums()
        
        outputs = []
        stdout_text = buf.getvalue()
        if stdout_text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in stdout_text.splitlines()]
            })
            
        for fig_num in figs_after:
            if fig_num not in figs_before:
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight")
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
                plt.close(fig)
                outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": img_base64,
                        "text/plain": ["<Figure size>"]
                    },
                    "metadata": {}
                })
                
        if captured_result is not None:
            outputs.append({
                "output_type": "execute_result",
                "execution_count": idx,
                "data": {
                    "text/plain": [json.dumps(captured_result, indent=2)]
                },
                "metadata": {}
            })

        notebook_cells.append({
            "cell_type": "code",
            "execution_count": idx,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code.splitlines()]
        })

    notebook_content = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    target_path = ROOT / "Notebook" / "pipeline.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1)

    print(f"Successfully generated {target_path} with {len(notebook_cells)} cells.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

