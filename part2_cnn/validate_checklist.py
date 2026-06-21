from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .convolution_manual import cross_correlation_2d, output_size
from .models import FlexibleCNN, LeNetLikeCNN, SimpleCNN, SimpleMLP
from .pooling_manual import avg_pool2d, max_pool2d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part2_cnn" / "part2_cnn"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
SAVED_MODELS_DIR = OUTPUT_DIR / "saved_models"
DATA_ROOT = PROJECT_ROOT / "data"


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_dataset(limit_train: int = 4000, limit_test: int = 1000):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
    train_subset = Subset(train_set, list(range(min(limit_train, len(train_set)))))
    test_subset = Subset(test_set, list(range(min(limit_test, len(test_set)))))
    return train_subset, test_subset


def _loaders(batch_size: int = 128, limit_train: int = 4000, limit_test: int = 1000):
    train_set, test_set = _load_dataset(limit_train=limit_train, limit_test=limit_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _train(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int = 1) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_true.extend(yb.tolist())
        y_pred.extend(preds)
    return np.asarray(y_true), np.asarray(y_pred)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    labels = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Part 2 - Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_comparison(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    names = list(results.keys())
    acc = [results[name]["accuracy"] for name in names]
    f1 = [results[name]["f1_score"] for name in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("MLP vs CNN comparison on Fashion-MNIST")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_ablation(results: List[Dict[str, object]], output_path: Path) -> None:
    names = [item["name"] for item in results]
    acc = [item["accuracy"] for item in results]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(names, acc, color="#5b8ff9")
    ax.set_ylim(0, 1)
    ax.set_title("Part 2 - Architectural ablation study")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _manual_examples() -> Dict[str, object]:
    x = np.array([
        [1.0, 2.0, 0.0, 1.0],
        [0.0, 1.0, 3.0, 2.0],
        [1.0, 2.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 3.0],
    ])
    kernel = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])
    corr = cross_correlation_2d(x, kernel, stride=1, padding=0)
    max_pool = max_pool2d(x, pool_size=2, stride=2)
    avg_pool = avg_pool2d(x, pool_size=2, stride=2)
    return {
        "input": x.tolist(),
        "kernel": kernel.tolist(),
        "cross_correlation": corr.tolist(),
        "cross_correlation_shape": list(corr.shape),
        "cross_correlation_size_formula": "(n + 2p - k)/s + 1",
        "cross_correlation_output_size": output_size(4, 2, stride=1, padding=0),
        "max_pool": max_pool.tolist(),
        "avg_pool": avg_pool.tolist(),
        "pool_output_size_formula": "floor((n - pool)/stride) + 1",
        "pool_output_size": int((4 - 2) // 2 + 1),
    }


@torch.no_grad()
def _custom_vs_pytorch() -> Dict[str, object]:
    x = np.array([
        [1.0, 2.0, 0.0, 1.0],
        [0.0, 1.0, 3.0, 2.0],
        [1.0, 2.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 3.0],
    ], dtype=np.float32)
    kernel = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float32)
    custom_corr = cross_correlation_2d(x, kernel, stride=1, padding=0)
    tensor_x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
    weight = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
    torch_corr = F.conv2d(tensor_x, weight, stride=1, padding=0).squeeze(0).squeeze(0).cpu().numpy()

    pool_input = torch.tensor(x).unsqueeze(0).unsqueeze(0)
    torch_max = F.max_pool2d(pool_input, kernel_size=2, stride=2).squeeze(0).squeeze(0).cpu().numpy()
    torch_avg = F.avg_pool2d(pool_input, kernel_size=2, stride=2).squeeze(0).squeeze(0).cpu().numpy()

    return {
        "cross_correlation_max_abs_diff": float(np.max(np.abs(custom_corr - torch_corr))),
        "max_pool_max_abs_diff": float(np.max(np.abs(max_pool2d(x) - torch_max))),
        "avg_pool_max_abs_diff": float(np.max(np.abs(avg_pool2d(x) - torch_avg))),
        "torch_cross_correlation": torch_corr.tolist(),
        "torch_max_pool": torch_max.tolist(),
        "torch_avg_pool": torch_avg.tolist(),
    }


@torch.no_grad()
def _feature_maps(model: SimpleCNN, loader: DataLoader, device: torch.device, output_path: Path) -> Dict[str, object]:
    model.eval()
    xb, yb = next(iter(loader))
    sample = xb[:1].to(device)
    input_image = sample.squeeze(0).squeeze(0).cpu().numpy()
    act1 = torch.relu(model.conv1(sample)).cpu().squeeze(0)
    act2 = torch.relu(model.conv2(act1.unsqueeze(0).to(device))).cpu().squeeze(0)

    channels_to_show = min(6, act1.shape[0])
    fig, axes = plt.subplots(3, channels_to_show, figsize=(2.2 * channels_to_show, 6))
    axes = np.atleast_2d(axes)
    axes[0, 0].imshow(input_image, cmap="gray")
    axes[0, 0].set_title(f"Input (label={int(yb[0])})")
    axes[0, 0].axis("off")
    for col in range(1, channels_to_show):
        axes[0, col].axis("off")

    for idx in range(channels_to_show):
        axes[1, idx].imshow(act1[idx].numpy(), cmap="viridis")
        axes[1, idx].set_title(f"conv1 ch {idx}")
        axes[1, idx].axis("off")
        axes[2, idx].imshow(act2[idx].numpy(), cmap="magma")
        axes[2, idx].set_title(f"conv2 ch {idx}")
        axes[2, idx].axis("off")

    fig.suptitle("Feature maps from the trained CNN")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "input_shape": list(input_image.shape),
        "conv1_shape": list(act1.shape),
        "conv2_shape": list(act2.shape),
        "label": int(yb[0]),
    }


def _load_trained_cnn(device: torch.device) -> SimpleCNN:
    model = SimpleCNN(num_classes=10).to(device)
    state_path = SAVED_MODELS_DIR / "cnn.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location=device))
        return model
    return model


def _ensure_trained_reference_model(train_loader: DataLoader, device: torch.device) -> SimpleCNN:
    model = _load_trained_cnn(device)
    if (SAVED_MODELS_DIR / "cnn.pt").exists():
        return model
    trained = _train(model, train_loader, device, epochs=1)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), SAVED_MODELS_DIR / "cnn.pt")
    return trained


def _evaluate_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, epochs: int = 1):
    trained = _train(model, train_loader, device, epochs=epochs)
    y_true, y_pred = _predict(trained, test_loader, device)
    metrics = _metrics(y_true, y_pred)
    return trained, y_true, y_pred, metrics


def _ablation_experiments(train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> List[Dict[str, object]]:
    configs = [
        {"name": "baseline", "kwargs": {"num_filters": 16, "padding": 1, "stride": 1, "pool_type": "max", "use_1x1": False}},
        {"name": "no_padding", "kwargs": {"num_filters": 16, "padding": 0, "stride": 1, "pool_type": "max", "use_1x1": False}},
        {"name": "stride_2", "kwargs": {"num_filters": 16, "padding": 1, "stride": 2, "pool_type": "max", "use_1x1": False}},
        {"name": "avg_pool", "kwargs": {"num_filters": 16, "padding": 1, "stride": 1, "pool_type": "avg", "use_1x1": False}},
        {"name": "more_filters", "kwargs": {"num_filters": 32, "padding": 1, "stride": 1, "pool_type": "max", "use_1x1": False}},
        {"name": "with_1x1", "kwargs": {"num_filters": 16, "padding": 1, "stride": 1, "pool_type": "max", "use_1x1": True}},
    ]
    results: List[Dict[str, object]] = []
    for cfg in configs:
        model = FlexibleCNN(num_classes=10, **cfg["kwargs"])
        trained, y_true, y_pred, metrics = _evaluate_model(model, train_loader, test_loader, device, epochs=1)
        results.append({"name": cfg["name"], **metrics, "params": cfg["kwargs"]})
    return results


def run(output_root: Path = OUTPUT_DIR) -> Dict[str, object]:
    _set_seed()
    _ensure_dirs()
    device = _device()
    train_loader, test_loader = _loaders()

    manual = _manual_examples()
    (ANALYSIS_DIR / "manual_calculations.json").write_text(json.dumps(manual, indent=2), encoding="utf-8")
    manual_markdown = f"""# Part 2 checklist evidence

## 1. Why MLPs are weak for images
- Flattening removes spatial locality.
- Dense layers do not share weights across the image grid.
- CNNs keep local receptive fields and build hierarchical features.

## 2. Manual calculations
- Cross-correlation example output shape: {manual['cross_correlation_shape']}
- Pooling example output shape: {manual['pool_output_size']}
- Formula reminder: {manual['cross_correlation_size_formula']}

## 3-4. Custom ops vs PyTorch
See `comparison.json` for max absolute differences.

## 5. CNN architecture
`SimpleCNN` and `LeNetLikeCNN` are available in `models.py`.

## 6. Ablation study
We compare padding, stride, pooling type, number of filters, and 1x1 convolution.

## 7. Feature maps
See `figures/feature_maps.png`.

## 8. MLP vs CNN
See `figures/mlp_vs_cnn.png`.
"""
    (ANALYSIS_DIR / "report.md").write_text(manual_markdown, encoding="utf-8")

    comparison = _custom_vs_pytorch()
    (ANALYSIS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    reference_model = _ensure_trained_reference_model(train_loader, device)
    feature_stats = _feature_maps(reference_model, test_loader, device, FIGURES_DIR / "feature_maps.png")
    (ANALYSIS_DIR / "feature_maps.json").write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")

    mlp = SimpleMLP(num_classes=10)
    cnn = SimpleCNN(num_classes=10)
    _, mlp_true, mlp_pred, mlp_metrics = _evaluate_model(mlp, train_loader, test_loader, device, epochs=1)
    _, cnn_true, cnn_pred, cnn_metrics = _evaluate_model(cnn, train_loader, test_loader, device, epochs=1)
    comparison_metrics = {"MLP": mlp_metrics, "CNN": cnn_metrics}
    (ANALYSIS_DIR / "mlp_vs_cnn.json").write_text(json.dumps(comparison_metrics, indent=2), encoding="utf-8")
    _plot_comparison(comparison_metrics, FIGURES_DIR / "mlp_vs_cnn.png")

    ablation_results = _ablation_experiments(train_loader, test_loader, device)
    (ANALYSIS_DIR / "ablation.json").write_text(json.dumps(ablation_results, indent=2), encoding="utf-8")
    _plot_ablation(ablation_results, FIGURES_DIR / "ablation.png")

    baseline_model = _ensure_trained_reference_model(train_loader, device)
    baseline_true, baseline_pred = _predict(baseline_model, test_loader, device)
    baseline_metrics = _metrics(baseline_true, baseline_pred)
    _plot_confusion_matrix(baseline_true, baseline_pred, FIGURES_DIR / "confusion_matrix.png")

    summary = {
        "part": "part2_cnn",
        "status": "success",
        "device": str(device),
        "manual_examples": str(ANALYSIS_DIR / "manual_calculations.json"),
        "comparison": str(ANALYSIS_DIR / "comparison.json"),
        "report": str(ANALYSIS_DIR / "report.md"),
        "comparison_metrics": comparison_metrics,
        "ablation_results": ablation_results,
        "baseline_metrics": baseline_metrics,
        "feature_maps": str(FIGURES_DIR / "feature_maps.png"),
        "mlp_vs_cnn": str(FIGURES_DIR / "mlp_vs_cnn.png"),
        "ablation_png": str(FIGURES_DIR / "ablation.png"),
        "confusion_png": str(FIGURES_DIR / "confusion_matrix.png"),
    }
    (OUTPUT_DIR / "checklist_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run())
