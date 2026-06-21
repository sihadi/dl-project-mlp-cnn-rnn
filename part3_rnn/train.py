from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.common.agents import BaseAgent, Coordinator
from .models import SimpleLSTM


class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = [self.vocab.get(w, 1) for w in self.texts[idx].split()][: self.max_len]
        toks = toks + [0] * (self.max_len - len(toks))
        return torch.tensor(toks, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def _prepare_data():
    ds = load_dataset('imdb')
    train_texts = [x['text'].lower() for x in ds['train']]
    train_labels = [int(x['label']) for x in ds['train']]
    test_texts = [x['text'].lower() for x in ds['test']]
    test_labels = [int(x['label']) for x in ds['test']]

    # build small vocab
    from collections import Counter

    counter = Counter()
    for t in train_texts:
        counter.update(t.split())
    most = [w for w, _ in counter.most_common(10000 - 3)]
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, w in enumerate(most, start=2):
        vocab[w] = i

    train_ds = SimpleTextDataset(train_texts, train_labels, vocab)
    test_ds = SimpleTextDataset(test_texts, test_labels, vocab)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    return train_loader, test_loader, len(vocab)


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


def run_part(output_root: Path) -> Dict[str, object]:
    out = Path(output_root) / 'part3_rnn'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'saved_models').mkdir(exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)

    device = torch.device('cpu')
    train_loader, test_loader, vocab_size = _prepare_data()
    model = SimpleLSTM(vocab_size=vocab_size).to(device)

    _train_one_epoch(model, train_loader, device)
    acc = _evaluate(model, test_loader, device)

    torch.save(model.state_dict(), out / 'saved_models' / 'lstm.pt')
    (out / 'metrics.json').write_text(json.dumps({'part': 'part3_rnn', 'accuracy': acc}, indent=2))
    return {'part': 'part3_rnn', 'status': 'success', 'accuracy': acc, 'output_dir': str(out)}


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
    coord = Coordinator('part3_rnn', agents, Path(output_root) / 'part3_rnn')
    ctx = coord.run()
    if isinstance(ctx, dict) and 'accuracy' in ctx:
        return ctx
    return {'part': 'part3_rnn', 'status': 'success', 'accuracy': 0.0, 'output_dir': str(Path(output_root) / 'part3_rnn')}


if __name__ == '__main__':
    print(run(Path('outputs')))
