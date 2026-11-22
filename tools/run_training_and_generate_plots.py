"""
Comprehensive training, logging, and plotting pipeline for HLO Recommender.
Trains the Bayesian Knowledge Tracing (BKT) model, logs details to log/,
and generates analytical plots for training loss, ROC-AUC, calibration,
mastery trajectories, and ranking comparisons.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local src is in path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from hlo_recommender.models.bkt import BKTParameters, ordered, predict_events
from hlo_recommender.data import validate_events
from hlo_recommender.evaluation import binary_metrics
from hlo_recommender.io import read_jsonl, write_json, write_jsonl, provenance
from hlo_recommender.experiments import run_experiment
from hlo_recommender.io import config as load_config


def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    
    logger = logging.getLogger("hlo_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def train_bkt_with_logging(train_path: Path, val_path: Path, log_dir: Path, logger, seed=42, restarts=6):
    logger.info("==================================================================")
    logger.info("           STARTING BAYESIAN KNOWLEDGE TRACING (BKT)              ")
    logger.info("==================================================================")
    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"Loading validation data from: {val_path}")
    
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    validate_events(train_rows)
    validate_events(val_rows)
    
    logger.info(f"Training events count: {len(train_rows)} from {len({r['learner_id'] for r in train_rows})} learners")
    logger.info(f"Validation events count: {len(val_rows)} from {len({r['learner_id'] for r in val_rows})} learners")
    
    # Group sequences
    sequences = {}
    for row in ordered(train_rows):
        sequences.setdefault((row['learner_id'], row['skill_id']), []).append(row['correct'])
    
    logger.info(f"Unique (learner, skill) sequence tracks: {len(sequences)}")
    
    iteration_history = []
    
    def create_objective(restart_idx):
        restart_steps = []
        def objective(values):
            params = BKTParameters(*map(float, values))
            loss = 0.0
            for outcomes in sequences.values():
                p = params.initial
                for outcome in outcomes:
                    prediction = params.predict(p)
                    loss -= float(np.log(max(1e-12, prediction if outcome else 1 - prediction)))
                    p = params.update(p, outcome)
            restart_steps.append((list(values), loss))
            return loss
        return objective, restart_steps

    rng = np.random.default_rng(seed)
    bounds = [(0.01, 0.99), (0.001, 0.4), (0.001, 0.45), (0.001, 0.4)]
    starts = [[0.2, 0.1, 0.2, 0.1]] + [rng.uniform([b[0] for b in bounds], [b[1] for b in bounds]) for _ in range(restarts - 1)]
    
    fits = []
    all_restart_traces = []
    
    logger.info(f"Running L-BFGS-B bounded optimization across {restarts} random restarts...")
    for idx, start in enumerate(starts):
        obj_func, steps = create_objective(idx)
        fit = minimize(obj_func, start, bounds=bounds, method='L-BFGS-B')
        fits.append(fit)
        all_restart_traces.append(steps)
        logger.info(
            f"  Restart #{idx+1}/{restarts} | Initial: {np.round(start, 3)} | "
            f"Loss: {fit.fun:.4f} | Converged: {fit.success} ({fit.message})"
        )
    
    finite_fits = [f for f in fits if np.isfinite(f.fun)]
    if not finite_fits:
        raise RuntimeError("All BKT optimization restarts failed.")
    best_fit = min(finite_fits, key=lambda f: f.fun)
    best_params = BKTParameters(*map(float, best_fit.x))
    
    logger.info("------------------------------------------------------------------")
    logger.info(f"OPTIMIZATION COMPLETE - Best Loss (NLL): {best_fit.fun:.4f}")
    logger.info(f"Learned BKT Parameters:")
    logger.info(f"  - Initial Mastery P(L0): {best_params.initial:.4f}")
    logger.info(f"  - Transition/Learn P(T):  {best_params.learn:.4f}")
    logger.info(f"  - Guess Probability P(G): {best_params.guess:.4f}")
    logger.info(f"  - Slip Probability P(S):  {best_params.slip:.4f}")
    logger.info("------------------------------------------------------------------")
    
    # Validation evaluation
    logger.info("Evaluating fitted model on held-out validation set...")
    val_predictions, val_metrics = predict_events(val_rows, best_params)
    
    train_prior = sum((r['correct'] for r in train_rows)) / len(train_rows)
    baseline_metrics = binary_metrics([r['correct'] for r in val_rows], [train_prior] * len(val_rows))
    
    logger.info(f"Validation Metrics:")
    logger.info(f"  - Log Loss:  {val_metrics['log_loss']:.4f} (Baseline: {baseline_metrics['log_loss']:.4f})")
    logger.info(f"  - Accuracy:  {val_metrics['accuracy']*100:.2f}% (Baseline: {baseline_metrics['accuracy']*100:.2f}%)")
    logger.info(f"  - ROC-AUC:   {val_metrics['auc']:.4f} (Baseline: {baseline_metrics['auc']:.4f})")
    logger.info(f"  - Brier Score: {val_metrics['brier']:.4f} (Baseline: {baseline_metrics['brier']:.4f})")
    
    # Save validation predictions to log
    write_jsonl(log_dir / "learner_mastery_events.jsonl", val_predictions)
    
    # Save training record
    training_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "knowledge_tracing",
        "status": "trained",
        "best_parameters": {
            "initial_mastery": best_params.initial,
            "learn_rate": best_params.learn,
            "guess_rate": best_params.guess,
            "slip_rate": best_params.slip
        },
        "optimization": {
            "negative_log_likelihood": float(best_fit.fun),
            "converged": bool(best_fit.success),
            "message": str(best_fit.message),
            "num_restarts": restarts,
            "training_events": len(train_rows),
            "training_sequences": len(sequences)
        },
        "validation_metrics": val_metrics,
        "baseline_metrics": baseline_metrics
    }
    write_json(log_dir / "training_summary.json", training_summary)
    logger.info(f"Saved training summary and mastery events to {log_dir}")
    
    return best_params, training_summary, val_predictions, all_restart_traces, val_rows


def run_and_log_experiments(project_root: Path, log_dir: Path, logger):
    logger.info("==================================================================")
    logger.info("          RUNNING ADAPTIVE RECOMMENDATION EXPERIMENT              ")
    logger.info("==================================================================")
    exp_config_path = project_root / "configs" / "experiment.toml"
    exp_results = run_experiment(exp_config_path, generator_override="template", output_override=log_dir / "experiment_run")
    
    write_json(log_dir / "evaluation_ranking.json", exp_results)
    
    static_m = exp_results["comparisons"]["template_static"]["metrics"]
    adapt_m = exp_results["comparisons"]["template_adaptive"]["metrics"]
    
    logger.info("Ranking Evaluation Comparison:")
    logger.info(f"  Metric       | Static Ranking | Adaptive Ranking | Delta")
    logger.info(f"  -------------+----------------+------------------+--------")
    for k in ["top1", "hit3", "hit5", "mrr", "map_at_k", "ndcg_at_k", "precision_at_k"]:
        s_val = static_m.get(k, 0.0)
        a_val = adapt_m.get(k, 0.0)
        delta = a_val - s_val
        logger.info(f"  {k:<12} | {s_val:14.4f} | {a_val:16.4f} | {delta:+6.4f}")
    
    return exp_results


def generate_all_plots(log_dir: Path, training_summary, val_predictions, restart_traces, val_rows, exp_results, logger):
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating visual plots in: {plots_dir}")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # Plot 1: Optimization Restarts Loss
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    restarts_losses = [steps[-1][1] for steps in restart_traces if steps]
    restart_indices = list(range(1, len(restarts_losses) + 1))
    
    bars = ax.bar(restart_indices, restarts_losses, color="#2b5c8f", width=0.55, edgecolor="black", alpha=0.85)
    best_idx = np.argmin(restarts_losses)
    bars[best_idx].set_color("#2ca02c")
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.5)
    
    ax.axhline(min(restarts_losses), color="#d62728", linestyle="--", alpha=0.8, label=f"Best NLL: {min(restarts_losses):.2f}")
    ax.set_title("BKT Optimization Loss across Random Restarts (L-BFGS-B)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Restart Number", fontsize=11)
    ax.set_ylabel("Negative Log-Likelihood (NLL)", fontsize=11)
    ax.set_xticks(restart_indices)
    ax.set_ylim(min(restarts_losses) - 10, max(restarts_losses) + 15)
    
    for bar, val in zip(bars, restarts_losses):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}", ha='center', va='bottom', fontsize=9)
    
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    p1 = plots_dir / "1_optimization_loss.png"
    fig.savefig(p1)
    plt.close(fig)
    logger.info(f"Saved: {p1.name}")

    # -------------------------------------------------------------
    # Plot 2: Model Calibration Curve (Reliability Diagram)
    # -------------------------------------------------------------
    calibration_bins = training_summary["validation_metrics"]["calibration"]
    pred_probs = [b["predicted"] for b in calibration_bins]
    obs_probs = [b["observed"] for b in calibration_bins]
    bin_counts = [b["n"] for b in calibration_bins]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
    scatter = ax.scatter(pred_probs, obs_probs, s=[np.sqrt(n)*25 for n in bin_counts], 
                         color="#1f77b4", edgecolor="black", zorder=4, label="Calibration Bins (Size ~ √N)")
    ax.plot(pred_probs, obs_probs, color="#1f77b4", linestyle="-", linewidth=2, alpha=0.85)
    
    for x, y, n in zip(pred_probs, obs_probs, bin_counts):
        ax.annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
        
    ax.set_title("Reliability Diagram (BKT Probability Calibration)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Predicted Probability P(Correct)", fontsize=11)
    ax.set_ylabel("Observed Empirical Accuracy", fontsize=11)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    p2 = plots_dir / "2_calibration_curve.png"
    fig.savefig(p2)
    plt.close(fig)
    logger.info(f"Saved: {p2.name}")

    # -------------------------------------------------------------
    # Plot 3: ROC Curve
    # -------------------------------------------------------------
    y_true = np.array([r['correct'] for r in val_rows], dtype=int)
    y_scores = np.array([r['probability'] for r in val_predictions])
    
    # Compute ROC manually or via thresholds
    thresholds = np.linspace(0, 1, 101)
    tpr_list = []
    fpr_list = []
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tpr_list.append(tp / P if P > 0 else 0)
        fpr_list.append(fp / N if N > 0 else 0)
        
    auc_score = training_summary["validation_metrics"]["auc"]
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.plot(fpr_list, tpr_list, color="#2ca02c", lw=2.5, label=f"BKT Model (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC Curve)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(frameon=True, loc="lower right", fontsize=11)
    fig.tight_layout()
    p3 = plots_dir / "3_roc_auc_curve.png"
    fig.savefig(p3)
    plt.close(fig)
    logger.info(f"Saved: {p3.name}")

    # -------------------------------------------------------------
    # Plot 4: Learner Mastery Trajectories over Time
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    
    # Pick 3 representative learners from val_predictions
    learners = list({r['learner_id'] for r in val_predictions})[:4]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for learner_id, color in zip(learners, colors):
        l_events = [r for r in val_predictions if r['learner_id'] == learner_id]
        if not l_events:
            continue
        # Sort by step/time
        steps = list(range(len(l_events)))
        mastery_vals = [e['mastery_after'] for e in l_events]
        ax.plot(steps, mastery_vals, marker='o', linewidth=2, color=color, label=f"Learner {learner_id}")
        
        # Mark correct vs incorrect
        for step, e in enumerate(l_events):
            marker = '^' if e['correct'] else 'v'
            m_color = 'green' if e['correct'] else 'red'
            ax.scatter(step, e['mastery_after'], color=m_color, marker=marker, s=50, zorder=5)

    ax.set_title("Learner Mastery Trajectories Pt(Mastery) over Assessment Events", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Assessment Sequence Step", fontsize=11)
    ax.set_ylabel("Mastery Probability P(L_t)", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    
    # Custom legend for markers
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Correct (▲)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=8, label='Incorrect (▼)')
    ]
    main_legend = ax.legend(frameon=True, loc="lower right")
    ax.add_artist(main_legend)
    ax.legend(handles=custom_lines, loc="upper left", frameon=True)
    
    fig.tight_layout()
    p4 = plots_dir / "4_mastery_trajectories.png"
    fig.savefig(p4)
    plt.close(fig)
    logger.info(f"Saved: {p4.name}")

    # -------------------------------------------------------------
    # Plot 5: Static vs Adaptive Recommendation Ranking Metrics
    # -------------------------------------------------------------
    static_m = exp_results["comparisons"]["template_static"]["metrics"]
    adapt_m = exp_results["comparisons"]["template_adaptive"]["metrics"]
    
    metrics_to_plot = ["top1", "hit3", "hit5", "mrr", "map_at_k", "ndcg_at_k", "precision_at_k"]
    labels = ["Top-1", "Hit@3", "Hit@5", "MRR", "MAP@k", "NDCG@k", "Prec@k"]
    static_vals = [static_m[k] for k in metrics_to_plot]
    adapt_vals = [adapt_m[k] for k in metrics_to_plot]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    rects1 = ax.bar(x - width/2, static_vals, width, label='Static Ranking', color='#4a7c59', alpha=0.85, edgecolor='black')
    rects2 = ax.bar(x + width/2, adapt_vals, width, label='Adaptive Ranking (HLO)', color='#3b6998', alpha=0.85, edgecolor='black')
    
    ax.set_title("Performance Comparison: Static vs. Adaptive Knowledge-Aware Ranking", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Score (0 to 1)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, loc='upper right', fontsize=11)
    
    # Values labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    fig.tight_layout()
    p5 = plots_dir / "5_ranking_metrics_comparison.png"
    fig.savefig(p5)
    plt.close(fig)
    logger.info(f"Saved: {p5.name}")

    # -------------------------------------------------------------
    # Plot 6: Integrated Research Dashboard
    # -------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Subplot (0, 0): Restarts loss
    axs[0, 0].bar(restart_indices, restarts_losses, color="#2b5c8f", width=0.55, edgecolor="black")
    axs[0, 0].axhline(min(restarts_losses), color="#d62728", linestyle="--")
    axs[0, 0].set_title("(A) BKT Multi-Restart Optimization", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("Restart #")
    axs[0, 0].set_ylabel("Negative Log-Likelihood")
    
    # Subplot (0, 1): ROC Curve
    axs[0, 1].plot(fpr_list, tpr_list, color="#2ca02c", lw=2, label=f"AUC = {auc_score:.3f}")
    axs[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axs[0, 1].set_title("(B) ROC Curve & Predictive Discrimination", fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel("False Positive Rate")
    axs[0, 1].set_ylabel("True Positive Rate")
    axs[0, 1].legend(loc="lower right")
    
    # Subplot (1, 0): Calibration
    axs[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.7)
    axs[1, 0].scatter(pred_probs, obs_probs, color="#1f77b4", edgecolor="black", s=60)
    axs[1, 0].plot(pred_probs, obs_probs, color="#1f77b4", lw=1.8)
    axs[1, 0].set_title("(C) Reliability & Calibration", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("Predicted Probability")
    axs[1, 0].set_ylabel("Observed Accuracy")
    
    # Subplot (1, 1): NDCG & Ranking gains
    axs[1, 1].bar(x - width/2, static_vals, width, label='Static', color='#4a7c59')
    axs[1, 1].bar(x + width/2, adapt_vals, width, label='Adaptive', color='#3b6998')
    axs[1, 1].set_title("(D) Adaptive Recommendation Metrics", fontsize=11, fontweight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels, fontsize=9, rotation=15)
    axs[1, 1].set_ylabel("Score")
    axs[1, 1].legend(loc="upper right")
    
    fig.suptitle("HLO Recommender: Training, Knowledge Tracing, and Adaptive Evaluation Dashboard", 
                 fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout()
    p6 = plots_dir / "dashboard_summary.png"
    fig.savefig(p6)
    plt.close(fig)
    logger.info(f"Saved: {p6.name}")
    
    logger.info("==================================================================")
    logger.info("All logs and charts successfully generated and saved to log/")
    logger.info("==================================================================")


def main():
    log_dir = project_root / "log"
    logger = setup_logger(log_dir)
    
    train_path = project_root / "data" / "development" / "assessments.train.jsonl"
    val_path = project_root / "data" / "development" / "assessments.validation.jsonl"
    
    # Step 1: Train BKT model with logging
    best_params, summary, val_preds, restart_traces, val_rows = train_bkt_with_logging(
        train_path=train_path,
        val_path=val_path,
        log_dir=log_dir,
        logger=logger,
        seed=42,
        restarts=6
    )
    
    # Step 2: Run experiment and evaluate
    exp_results = run_and_log_experiments(project_root, log_dir, logger)
    
    # Step 3: Generate visual plots
    generate_all_plots(log_dir, summary, val_preds, restart_traces, val_rows, exp_results, logger)


if __name__ == "__main__":
    main()
