import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from .models import SequentialMLP, SimpleMLP
from .utils import load_data


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _initialize_model(model: SimpleMLP, strategy: str) -> SimpleMLP:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if strategy == 'gaussian':
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif strategy == 'constant':
                torch.nn.init.constant_(module.weight, 0.01)
            elif strategy == 'xavier':
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                raise ValueError(f'Unknown initialization strategy: {strategy}')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    return model


def _build_model(variant: str, input_dim: int, hidden: int = 64) -> torch.nn.Module:
    if variant == 'custom':
        return SimpleMLP(input_dim=input_dim, hidden=hidden)
    if variant == 'sequential':
        return SequentialMLP(input_dim=input_dim, hidden=hidden)
    raise ValueError(f'Unknown model variant: {variant}')


def _train_one_model(model: SimpleMLP, train_loader: DataLoader, device: torch.device, epochs: int = 5) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


def _predict(model: SimpleMLP, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            logits = model(xb)
            preds.append(torch.softmax(logits, dim=1).cpu().argmax(axis=1).numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def _save_confusion_matrix(y_true, y_pred, class_names, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Part 1 - Confusion Matrix',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_metrics_summary(metrics: dict, output_path: Path) -> None:
    labels = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [metrics[label] for label in labels]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Part 1 - Metrics Summary')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.3f}', ha='center', va='bottom')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _inspect_model(model: torch.nn.Module) -> dict:
    named_parameters = []
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        param_info = {
            'name': name,
            'shape': list(param.shape),
            'requires_grad': bool(param.requires_grad),
            'numel': int(param.numel()),
        }
        named_parameters.append(param_info)
        total_params += int(param.numel())
        if param.requires_grad:
            trainable_params += int(param.numel())

    return {
        'named_parameters': named_parameters,
        'state_dict_keys': list(model.state_dict().keys()),
        'total_params': total_params,
        'trainable_params': trainable_params,
    }


def run(output_root: Path):
    _set_seed(42)
    device = _get_device()
    X_train, X_test, y_train, y_test, feature_names, class_names = load_data()
    out_dir = Path(output_root) / 'part1_mlp'
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_models_dir = out_dir / 'saved_models'
    figures_dir = out_dir / 'figures'
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError('Feature dimensions are inconsistent between train and test sets')

    strategies = ['gaussian', 'constant', 'xavier']
    variants = ['custom', 'sequential']
    results = []
    best_strategy = None
    best_variant = None
    best_val_accuracy = -1.0
    best_state_dict = None
    best_model_meta = {}

    for variant in variants:
        for strategy in strategies:
            model = _build_model(variant, input_dim=X_train.shape[1], hidden=64).to(device)
            model = _initialize_model(model, strategy)
            _train_one_model(model, train_loader, device, epochs=5)
            y_val_true, y_val_pred = _predict(model, val_loader, device)
            val_accuracy = float(accuracy_score(y_val_true, y_val_pred))
            model_path = saved_models_dir / f'{variant}_{strategy}.pt'
            torch.save(model.cpu().state_dict(), model_path)
            model.to(device)
            model_inspection = _inspect_model(model)
            results.append({
                'variant': variant,
                'strategy': strategy,
                'val_accuracy': val_accuracy,
                'model_path': str(model_path),
                'parameter_summary': model_inspection,
            })
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_strategy = strategy
                best_variant = variant
                best_state_dict = torch.load(model_path, map_location=device)
                best_model_meta = {
                    'variant': variant,
                    'strategy': strategy,
                    'model_path': str(model_path),
                    'parameter_summary': model_inspection,
                }

    if best_strategy is None or best_state_dict is None:
        raise RuntimeError('No best model could be selected')

    best_model = _build_model(best_variant, input_dim=X_train.shape[1], hidden=64).to(device)
    best_model.load_state_dict(best_state_dict)

    if next(best_model.parameters()).device != device:
        raise RuntimeError('Best model device mismatch')

    y_true, y_pred = _predict(best_model, test_loader, device)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    _save_confusion_matrix(y_true, y_pred, class_names, figures_dir / 'confusion_matrix.png')
    summary_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    _save_metrics_summary(summary_metrics, figures_dir / 'metrics_summary.png')

    torch.save(best_model.cpu().state_dict(), saved_models_dir / 'best_model.pt')
    (saved_models_dir / 'best_model_meta.json').write_text(json.dumps(best_model_meta, indent=2))
    (out_dir / 'parameter_report.json').write_text(json.dumps(_inspect_model(best_model), indent=2))

    metrics = {
        'part': 'part1_mlp',
        'device': str(device),
        'model_variants': variants,
        'best_strategy': best_strategy,
        'best_variant': best_variant,
        'validation_accuracy': best_val_accuracy,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'strategy_results': results,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return {'part': 'part1_mlp', 'status': 'success', 'best_strategy': best_strategy, 'best_variant': best_variant, 'accuracy': accuracy, 'output_dir': str(out_dir)}


if __name__ == '__main__':
    print(run(Path('outputs')))
