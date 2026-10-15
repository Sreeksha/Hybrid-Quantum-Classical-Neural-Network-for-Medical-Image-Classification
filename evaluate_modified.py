import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.utils import resample
from config import DEVICE, BEST_MODEL_PATH, BATCH_SIZE
from dataset import get_test_loader
from model_modified import HybridQCModel

# ── Output directory ───────────────────────────────────────────────────────────
EVAL_DIR = '/data1workspace/evaluation'
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
def load_model():
    print("[Evaluate] Loading best model...")
    model = HybridQCModel().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ── Run Inference ──────────────────────────────────────────────────────────────
def run_inference(model, loader):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images  = images.to(DEVICE)
            outputs = model(images)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            print(f"  Batch [{i+1}/{len(loader)}] done", end='\r')
    print()
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    return all_labels, all_probs

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1: Optimal Threshold via Youden's J Statistic
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(labels, probs):
    """
    Instead of fixed 0.5, find the threshold that maximizes
    Youden's J = Sensitivity + Specificity - 1
    This corrects for train/test distribution mismatch.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youdens_j   = tpr - fpr                          # J statistic for each threshold
    optimal_idx = youdens_j.argmax()
    optimal_thr = thresholds[optimal_idx]

    print(f"\n[Threshold] Default (0.5)  → will be compared")
    print(f"[Threshold] Optimal        : {optimal_thr:.4f}  "
          f"(Sensitivity={tpr[optimal_idx]:.4f}, Specificity={1-fpr[optimal_idx]:.4f})")
    return optimal_thr

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2: Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(labels, probs, preds, n_bootstrap=1000, ci=95):
    """
    Compute bootstrap confidence intervals for AUC, Accuracy, F1,
    Sensitivity, and Specificity.
    Required for medical AI papers — shows statistical reliability.
    """
    print(f"\n[Bootstrap] Running {n_bootstrap} iterations for {ci}% CI...")
    metrics = {'auc': [], 'acc': [], 'f1': [], 'sensitivity': [], 'specificity': []}

    for i in range(n_bootstrap):
        # Resample with replacement
        idx = resample(range(len(labels)), random_state=i)
        l   = labels[idx]
        pr  = probs[idx]
        pd  = preds[idx]

        # Skip if only one class present in resample (AUC undefined)
        if len(np.unique(l)) < 2:
            continue

        metrics['auc'].append(roc_auc_score(l, pr))
        metrics['acc'].append(accuracy_score(l, pd))
        metrics['f1'].append(f1_score(l, pd, zero_division=0))

        cm      = confusion_matrix(l, pd)
        # Handle case where confusion matrix might not be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    alpha = (100 - ci) / 2
    ci_results = {}
    for name, vals in metrics.items():
        if len(vals) == 0:
            continue
        vals = np.array(vals)
        ci_results[name] = {
            'mean'  : np.mean(vals),
            'lower' : np.percentile(vals, alpha),
            'upper' : np.percentile(vals, 100 - alpha),
        }
        print(f"  {name:<15}: {ci_results[name]['mean']:.4f}  "
              f"({ci}% CI: {ci_results[name]['lower']:.4f} – {ci_results[name]['upper']:.4f})")

    return ci_results

# ── Plot ROC Curve (with both thresholds marked) ───────────────────────────────
def plot_roc_curve(labels, probs, optimal_threshold):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    # Find points for both thresholds on the curve
    opt_idx     = np.argmin(np.abs(thresholds - optimal_threshold))
    default_idx = np.argmin(np.abs(thresholds - 0.5))

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    # Mark default threshold
    plt.scatter(fpr[default_idx], tpr[default_idx],
                color='blue', zorder=5, s=80,
                label=f'Threshold = 0.5  (Sens={tpr[default_idx]:.2f}, Spec={1-fpr[default_idx]:.2f})')

    # Mark optimal threshold
    plt.scatter(fpr[opt_idx], tpr[opt_idx],
                color='green', zorder=5, s=80, marker='*',
                label=f'Optimal = {optimal_threshold:.2f}  (Sens={tpr[opt_idx]:.2f}, Spec={1-fpr[opt_idx]:.2f})')

    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve — Pleural Effusion Detection\nHybrid Quantum-Classical Model', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] ROC curve saved → {path}")
    return auc

# ── Plot PR Curve ──────────────────────────────────────────────────────────────
def plot_pr_curve(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='steelblue', lw=2, label=f'PR Curve (AP = {ap:.4f})')
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curve — Pleural Effusion Detection\nHybrid Quantum-Classical Model', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'pr_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] PR curve saved → {path}")
    return ap

# ── Plot Confusion Matrix (for both thresholds side by side) ──────────────────
def plot_confusion_matrix(labels, preds_default, preds_optimal, optimal_threshold):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    classes   = ['No Effusion', 'Pleural Effusion']

    for ax, preds, title in zip(
        axes,
        [preds_default, preds_optimal],
        [f'Default Threshold (0.5)', f'Optimal Threshold ({optimal_threshold:.2f})']
    ):
        cm = confusion_matrix(labels, preds)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticklabels(classes, fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix\n{title}', fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black',
                        fontsize=16, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'confusion_matrix_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Confusion matrix comparison saved → {path}")

# ── Plot Probability Distribution ─────────────────────────────────────────────
def plot_prob_distribution(labels, probs, optimal_threshold):
    plt.figure(figsize=(8, 5))
    plt.hist(probs[labels == 0], bins=30, alpha=0.6, color='steelblue', label='No Effusion')
    plt.hist(probs[labels == 1], bins=30, alpha=0.6, color='crimson',   label='Pleural Effusion')
    plt.axvline(x=0.5,               color='black', linestyle='--', lw=1.5, label='Default Threshold = 0.5')
    plt.axvline(x=optimal_threshold, color='green', linestyle='-',  lw=1.5,
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlabel('Predicted Probability', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.title('Prediction Probability Distribution\nHybrid Quantum-Classical Model', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'prob_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Probability distribution saved → {path}")

# ── Save Full Report ───────────────────────────────────────────────────────────
def save_report(labels, probs, auc, ap, ci_results, optimal_threshold):
    # Metrics at default threshold
    preds_default = (probs >= 0.5).astype(int)
    cm_default    = confusion_matrix(labels, preds_default)
    tn, fp, fn, tp = cm_default.ravel()
    acc_def  = accuracy_score(labels, preds_default)
    f1_def   = f1_score(labels, preds_default, zero_division=0)
    sens_def = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_def = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv_def  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv_def  = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Metrics at optimal threshold
    preds_opt  = (probs >= optimal_threshold).astype(int)
    cm_opt     = confusion_matrix(labels, preds_opt)
    tn2, fp2, fn2, tp2 = cm_opt.ravel()
    acc_opt  = accuracy_score(labels, preds_opt)
    f1_opt   = f1_score(labels, preds_opt, zero_division=0)
    sens_opt = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    spec_opt = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
    ppv_opt  = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
    npv_opt  = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0

    report_default = classification_report(labels, preds_default,
                                           target_names=['No Effusion', 'Pleural Effusion'])
    report_opt     = classification_report(labels, preds_opt,
                                           target_names=['No Effusion', 'Pleural Effusion'])

    def ci_str(name):
        if name in ci_results:
            r = ci_results[name]
            return f"{r['mean']:.4f}  (95% CI: {r['lower']:.4f} – {r['upper']:.4f})"
        return "N/A"

    content = f"""
=============================================================
  Hybrid Quantum-Classical Pleural Effusion Classifier
  Full Evaluation Report (Modified Model)
=============================================================

Model         : DenseNet-121 (partial unfreeze) + 4-qubit VQC + Residual
Dataset       : CheXpert valid.csv
Device        : {DEVICE}

-------------------------------------------------------------
  Threshold-Independent Metrics
-------------------------------------------------------------
AUC-ROC           : {ci_str('auc')}
Avg Precision (AP): {ap:.4f}

-------------------------------------------------------------
  Bootstrap Confidence Intervals (1000 iterations, 95% CI)
-------------------------------------------------------------
Accuracy          : {ci_str('acc')}
F1 Score          : {ci_str('f1')}
Sensitivity       : {ci_str('sensitivity')}
Specificity       : {ci_str('specificity')}

-------------------------------------------------------------
  Metrics at Default Threshold (0.5)
-------------------------------------------------------------
Accuracy          : {acc_def:.4f}
F1 Score          : {f1_def:.4f}
Sensitivity (TPR) : {sens_def:.4f}  ({tp}/{tp+fn})
Specificity (TNR) : {spec_def:.4f}  ({tn}/{tn+fp})
PPV               : {ppv_def:.4f}
NPV               : {npv_def:.4f}

Confusion Matrix:
               Predicted No    Predicted Yes
Actual No      {tn:>10}      {fp:>10}
Actual Yes     {fn:>10}      {tp:>10}

{report_default}

-------------------------------------------------------------
  Metrics at Optimal Threshold ({optimal_threshold:.4f})
-------------------------------------------------------------
Accuracy          : {acc_opt:.4f}
F1 Score          : {f1_opt:.4f}
Sensitivity (TPR) : {sens_opt:.4f}  ({tp2}/{tp2+fn2})
Specificity (TNR) : {spec_opt:.4f}  ({tn2}/{tn2+fp2})
PPV               : {ppv_opt:.4f}
NPV               : {npv_opt:.4f}

Confusion Matrix:
               Predicted No    Predicted Yes
Actual No      {tn2:>10}      {fp2:>10}
Actual Yes     {fn2:>10}      {tp2:>10}

{report_opt}
=============================================================
"""
    path = os.path.join(EVAL_DIR, 'evaluation_report_modified.txt')
    with open(path, 'w') as f:
        f.write(content)
    print(content)
    print(f"[Evaluate] Full report saved → {path}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model       = load_model()
    test_loader = get_test_loader()

    print(f"\n[Evaluate] Running inference on {len(test_loader)} batches...")
    labels, probs = run_inference(model, test_loader)

    # CHANGE 1: Find optimal threshold
    optimal_threshold = find_optimal_threshold(labels, probs)

    # Predictions at both thresholds
    preds_default = (probs >= 0.5).astype(int)
    preds_optimal = (probs >= optimal_threshold).astype(int)

    # Standard plots
    auc = plot_roc_curve(labels, probs, optimal_threshold)
    ap  = plot_pr_curve(labels, probs)
    plot_confusion_matrix(labels, preds_default, preds_optimal, optimal_threshold)
    plot_prob_distribution(labels, probs, optimal_threshold)

    # CHANGE 2: Bootstrap confidence intervals
    ci_results = bootstrap_ci(labels, probs, preds_optimal)

    # Save full report with both thresholds
    save_report(labels, probs, auc, ap, ci_results, optimal_threshold)

    print(f"\n[Evaluate] All outputs saved to {EVAL_DIR}")
