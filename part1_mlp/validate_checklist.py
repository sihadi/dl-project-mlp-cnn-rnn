"""Part 1 checklist: train MLP variants, export metrics/figures, and build a LIME explanation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

from .models import SequentialMLP, SimpleMLP
from .train import (
    _build_model,
    _get_device,
    _initialize_model,
    _inspect_model,
    _set_seed,
    run as train_run,
)
from .utils import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part1_mlp"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_lime_explanation(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    class_names: list,
    device: torch.device,
) -> Dict[str, str]:
    """Build a local tabular LIME explanation and export PNG + HTML."""
    model.eval()

    def predict_proba(samples: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(samples.astype(np.float32)).to(device)
            logits = model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )
    sample_idx = 0
    for idx in range(min(5, len(X_test))):
        probs = predict_proba(X_test[idx].reshape(1, -1))[0]
        if np.isfinite(probs).all() and probs.sum() > 0:
            sample_idx = idx
            break

    explanation = explainer.explain_instance(
        X_test[sample_idx],
        predict_proba,
        num_features=min(10, len(feature_names)),
        top_labels=len(class_names),
    )

    if explanation.top_labels:
        label = explanation.top_labels[0]
    else:
        label = int(np.argmax(predict_proba(X_test[sample_idx].reshape(1, -1))[0]))
    pairs = explanation.as_list(label=label)
    labels = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    colors = ["#10b981" if w >= 0 else "#ef4444" for w in weights]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, weights, color=colors)
    ax.set_title("LIME - Breast Cancer (local explanation)")
    ax.set_xlabel("Feature weight")
    fig.tight_layout()
    png_path = FIGURES_DIR / "lime_explanation.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    html_path = FIGURES_DIR / "lime_explanation.html"
    rows = "".join(f"<tr><td>{name}</td><td>{weight:.4f}</td></tr>" for name, weight in pairs)
    html_path.write_text(
        f"""<html><body>
<h3>LIME explanation - Breast Cancer (part1)</h3>
<img src="{png_path.name}" alt="lime">
<table border="1" cellpadding="4"><tr><th>Feature</th><th>Weight</th></tr>{rows}</table>
</body></html>""",
        encoding="utf-8",
    )
    return {"lime_png": str(png_path), "lime_html": str(html_path)}


def run(output_root: Path = OUTPUT_DIR) -> Dict[str, object]:
    """Run Part 1 training pipeline and export checklist artifacts."""
    _set_seed()
    _ensure_dirs()
    device = _get_device()

    train_summary = train_run(PROJECT_ROOT / "outputs")
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    X_train, X_test, y_train, y_test, feature_names, class_names = load_data()
    X_tr, _, _, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    best_variant = metrics.get("best_variant", "sequential")
    best_strategy = metrics.get("best_strategy", "xavier")
    model = _build_model(best_variant, input_dim=X_train.shape[1], hidden=64)
    model = _initialize_model(model, best_strategy)
    state_path = OUTPUT_DIR / "saved_models" / "best_model.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)

    lime_paths = _generate_lime_explanation(model, X_tr, X_test, feature_names, class_names, device)

    init_comparison = {
        row["strategy"]: row["val_accuracy"]
        for row in metrics.get("strategy_results", [])
        if row.get("variant") == best_variant
    }
    if init_comparison:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(init_comparison.keys())
        values = [init_comparison[n] for n in names]
        ax.bar(names, values, color=["#3b82f6", "#f59e0b", "#10b981"])
        ax.set_ylim(0, 1)
        ax.set_title("Part 1 - Initialization comparison (validation accuracy)")
        ax.set_ylabel("Accuracy")
        fig.tight_layout()
        init_png = FIGURES_DIR / "initialization_comparison.png"
        fig.savefig(init_png, dpi=150)
        plt.close(fig)
        (ANALYSIS_DIR / "initialization_comparison.json").write_text(
            json.dumps(init_comparison, indent=2), encoding="utf-8"
        )
    else:
        init_png = None

    checklist = {
        "part": "part1_mlp",
        "status": "success",
        "device": str(device),
        "best_variant": best_variant,
        "best_strategy": best_strategy,
        "metrics": metrics,
        "parameter_report": _inspect_model(model),
        "confusion_png": str(FIGURES_DIR / "confusion_matrix.png"),
        "metrics_summary_png": str(FIGURES_DIR / "metrics_summary.png"),
        "initialization_png": str(init_png) if init_png else None,
        **lime_paths,
        "train_summary": train_summary,
    }
    (OUTPUT_DIR / "checklist_summary.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    (ANALYSIS_DIR / "checklist_summary.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    return checklist


if __name__ == "__main__":
    print(run())
