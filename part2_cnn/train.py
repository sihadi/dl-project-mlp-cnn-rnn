from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common.agents import BaseAgent, Coordinator
from .models import SimpleCNN


def _train_one_epoch(model, loader, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()


def _evaluate(model, loader, device):
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(p.argmax(axis=1))
            ys.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    return float(accuracy_score(y_true, y_pred))


def _prepare_data(root='data'):
    # Ensure images are single-channel (grayscale) to match the CNN's expected input
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=256)
    return train_loader, test_loader


def run_part(output_root: Path) -> Dict[str, object]:
    out = Path(output_root) / 'part2_cnn'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'saved_models').mkdir(exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)

    device = torch.device('cpu')
    train_loader, test_loader = _prepare_data()
    model = SimpleCNN(num_classes=10).to(device)

    # quick train (1 epoch)
    _train_one_epoch(model, train_loader, device)
    acc = _evaluate(model, test_loader, device)

    # save
    torch.save(model.state_dict(), out / 'saved_models' / 'cnn.pt')
    (out / 'metrics.json').write_text(json.dumps({'part': 'part2_cnn', 'accuracy': acc}, indent=2))

    return {'part': 'part2_cnn', 'status': 'success', 'accuracy': acc, 'output_dir': str(out)}


class AgentA(BaseAgent):
    def __init__(self):
        super().__init__('agent_a')

    def run(self, context, output_root: Path):
        return run_part(output_root)


class AgentB(BaseAgent):
    def __init__(self):
        super().__init__('agent_b')

    def run(self, context, output_root: Path):
        return run_part(output_root)


class AgentC(BaseAgent):
    def __init__(self):
        super().__init__('agent_c')

    def run(self, context, output_root: Path):
        return run_part(output_root)


def run(output_root: Path):
    agents = [AgentA(), AgentB(), AgentC()]
    coord = Coordinator('part2_cnn', agents, Path(output_root) / 'part2_cnn')
    ctx = coord.run()
    # ctx.data may be a dict returned from agent; normalize
    if isinstance(ctx.data, dict) and 'accuracy' in ctx.data:
        return {'part': 'part2_cnn', 'status': 'success', 'accuracy': ctx.data['accuracy'], 'output_dir': str(Path(output_root) / 'part2_cnn')}
    # if agent returned dictionary directly
    if isinstance(ctx, dict) and 'accuracy' in ctx:
        return ctx
    return {'part': 'part2_cnn', 'status': 'success', 'accuracy': None, 'output_dir': str(Path(output_root) / 'part2_cnn')}


if __name__ == '__main__':
    print(run(Path('outputs')))
