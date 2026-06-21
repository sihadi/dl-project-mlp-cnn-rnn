from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from .models import SimpleRNN, SimpleLSTM, SimpleGRU
from .seq2seq import SeqEncoder, SeqDecoder, greedy_decode, beam_search_decode
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3_rnn" / "part3_rnn"
FIGURES = OUTPUT_DIR / "figures"
ANALYSIS = OUTPUT_DIR / "analysis"
SAVED = OUTPUT_DIR / "saved_models"
DATA_ROOT = PROJECT_ROOT / "data"


def _set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
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


def _prepare_imdb(limit_train=1000, limit_test=500):
    ds = load_dataset('imdb')
    train_texts = [x['text'].lower() for x in ds['train']][:limit_train]
    train_labels = [int(x['label']) for x in ds['train']][:limit_train]
    test_texts = [x['text'].lower() for x in ds['test']][:limit_test]
    test_labels = [int(x['label']) for x in ds['test']][:limit_test]
    from collections import Counter
    counter = Counter()
    for t in train_texts:
        counter.update(t.split())
    most = [w for w, _ in counter.most_common(2000 - 3)]
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, w in enumerate(most, start=2):
        vocab[w] = i
    train_ds = TinyTextDataset(train_texts, train_labels, vocab, max_len=100)
    test_ds = TinyTextDataset(test_texts, test_labels, vocab, max_len=100)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    return train_loader, test_loader, len(vocab), vocab


def _train_epoch(model, loader, device, optimizer, clip_norm=None, epochs: int = 1):
    """Train for a given number of epochs (small educational loop)."""
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    ys = []
    preds = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
        preds.append(p)
        ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return float(accuracy_score(y_true, y_pred))


def _perplexity_demo(vocab_size=500, seq_len=20, dataset_size=1000, epochs=1):
    # train a tiny language model (next-token prediction) on synthetic data
    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 32)
            self.rnn = nn.GRU(32, 64, batch_first=True)
            self.head = nn.Linear(64, vocab_size)

        def forward(self, x):
            emb = self.emb(x)
            out, _ = self.rnn(emb)
            logits = self.head(out)
            return logits

    device = torch.device('cpu')
    model = LM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # synthetic dataset
    data = torch.randint(2, vocab_size, (dataset_size, seq_len))
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, dataset_size, 32):
            xb = data[i:i+32]
            xb = xb.to(device)
            logits = model(xb[:, :-1])  # predict next
            loss = loss_fn(logits.reshape(-1, vocab_size), xb[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
    # compute perplexity on full data
    model.eval()
    with torch.no_grad():
        logits = model(data[:, :-1].to(device))
        loss = loss_fn(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1).to(device))
    ppl = math.exp(float(loss.item()))
    return ppl


def _seq2seq_copy_demo(vocab=100, seq_len=8, train_size=500, epochs=3):
    # synthetic copy task: input sequence -> same output
    device = torch.device('cpu')
    encoder = SeqEncoder(vocab, embed_dim=32, hidden=64).to(device)
    decoder = SeqDecoder(vocab, embed_dim=32, hidden=64).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # data
    data = torch.randint(2, vocab, (train_size, seq_len))
    for ep in range(epochs):
        for i in range(0, train_size, 32):
            xb = data[i:i+32].to(device)
            # encode
            h = encoder(xb)
            # teacher forcing: feed target tokens
            # decoder input: start token=2, then previous target
            dec_in = torch.cat([torch.full((xb.shape[0], 1), 2, dtype=torch.long), xb[:, :-1]], dim=1).to(device)
            logits = []
            hidden = h
            for t in range(seq_len):
                logit, hidden = decoder.forward_step(dec_in[:, t], hidden)
                logits.append(logit.unsqueeze(1))
            logits = torch.cat(logits, dim=1)  # (N, L, V)
            loss = loss_fn(logits.reshape(-1, vocab), xb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    # evaluate greedy and beam on few samples
    samples = data[:10]
    results = []
    for s in samples:
        h = encoder(s.unsqueeze(0).to(device))
        g = greedy_decode(decoder, start_token=2, hidden=h, max_len=seq_len)
        b = beam_search_decode(decoder, start_token=2, hidden=h, beam_width=3, max_len=seq_len)
        results.append({'input': s.tolist(), 'greedy': g, 'beam': b})
    # compute exact match rates
    greedy_match = sum(1 for i, r in enumerate(results) if r['greedy'][:seq_len] == data[i].tolist()) / len(results)
    beam_match = sum(1 for i, r in enumerate(results) if r['beam'][:seq_len] == data[i].tolist()) / len(results)
    # compute simple BLEU (cumulative BLEU-4) for each example and average
    def _ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)] if len(seq) >= n else []

    def _modified_precision(ref, hyp, n):
        ref_ngrams = _ngrams(ref, n)
        hyp_ngrams = _ngrams(hyp, n)
        ref_counts = {}
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        clipped = 0
        total = 0
        hyp_counts = {}
        for g in hyp_ngrams:
            hyp_counts[g] = hyp_counts.get(g, 0) + 1
        for g, cnt in hyp_counts.items():
            total += cnt
            clipped += min(cnt, ref_counts.get(g, 0))
        if total == 0:
            return 0.0
        # Laplace smoothing to avoid zero precision
        return (clipped + 1.0) / (total + 1.0)

    def _bleu_score(ref, hyp, max_n=4):
        precisions = []
        for n in range(1, max_n+1):
            p = _modified_precision(ref, hyp, n)
            precisions.append(p)
        # geometric mean with numerical stability
        eps = 1e-12
        geo_mean = math.exp(sum((1.0/max_n) * math.log(max(p, eps)) for p in precisions))
        # brevity penalty
        ref_len = len(ref)
        hyp_len = len(hyp)
        if hyp_len == 0:
            bp = 0.0
        elif hyp_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - float(ref_len) / float(hyp_len))
        return bp * geo_mean

    bleus = []
    for i, r in enumerate(results):
        ref = data[i].tolist()
        hyp_g = r['greedy'][:seq_len]
        hyp_b = r['beam'][:seq_len]
        b_g = _bleu_score(ref, hyp_g)
        b_b = _bleu_score(ref, hyp_b)
        r['bleu_greedy'] = b_g
        r['bleu_beam'] = b_b
        bleus.append({'greedy': b_g, 'beam': b_b})
    avg_bleu_g = sum(x['greedy'] for x in bleus) / len(bleus)
    avg_bleu_b = sum(x['beam'] for x in bleus) / len(bleus)
    return {'results': results, 'greedy_match': greedy_match, 'beam_match': beam_match, 'avg_bleu_greedy': avg_bleu_g, 'avg_bleu_beam': avg_bleu_b}


def run(output_root: Path = OUTPUT_DIR) -> Dict[str, object]:
    _set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    SAVED.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    train_loader, test_loader, vocab_size, vocab = _prepare_imdb()

    # 3 models: RNN, LSTM, GRU
    model_classes = {
        'RNN': SimpleRNN,
        'LSTM': SimpleLSTM,
        'GRU': SimpleGRU,
    }
    results = {}

    def _eval_last_k(model, loader, device, k=16):
        model.eval()
        ys = []
        preds = []
        for xb, yb in loader:
            xb = xb.to(device)
            # keep last k tokens, pad left with zeros to original length
            L = xb.shape[1]
            if k < L:
                xb_last = torch.zeros_like(xb)
                xb_last[:, -k:] = xb[:, -k:]
            else:
                xb_last = xb
            logits = model(xb_last)
            p = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
            preds.append(p)
            ys.append(yb.numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        return float(accuracy_score(y_true, y_pred))

    def _eval_first_k(model, loader, device, k=16):
        model.eval()
        ys = []
        preds = []
        for xb, yb in loader:
            xb = xb.to(device)
            L = xb.shape[1]
            if k < L:
                xb_first = torch.zeros_like(xb)
                xb_first[:, :k] = xb[:, :k]
            else:
                xb_first = xb
            logits = model(xb_first)
            p = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
            preds.append(p)
            ys.append(yb.numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        return float(accuracy_score(y_true, y_pred))

    runs_per_model = 3
    epochs_per_run = 5
    for name, cls in model_classes.items():
        accs = []
        times = []
        mem_deltas = []
        for run_i in range(runs_per_model):
            model = cls(vocab_size).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            # reproducible per-run seed
            _set_seed(42 + run_i)
            t0 = time.perf_counter()
            _train_epoch(model, train_loader, device, opt, clip_norm=None, epochs=epochs_per_run)
            t1 = time.perf_counter()
            train_time = t1 - t0
            acc_full = _eval(model, test_loader, device)
            acc_last16 = _eval_last_k(model, test_loader, device, k=16)
            acc_first16 = _eval_first_k(model, test_loader, device, k=16)
            accs.append(acc_full)
            times.append(train_time)
            mem_deltas.append(acc_full - acc_last16)
            # memory_score: ability to remember early tokens -> accuracy when only first-K are given
            mem_score = acc_first16
            mem_deltas.append(mem_score)
        # mem_deltas contains alternating memory entries (acc_full - acc_last16) and mem_score values
        mem_delta_vals = mem_deltas[0::2]
        mem_score_vals = mem_deltas[1::2]
        results[name] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'train_time_mean': float(np.mean(times)),
            'train_time_std': float(np.std(times)),
            'memory_delta_mean': float(np.mean(mem_delta_vals)) if mem_delta_vals else 0.0,
            'memory_score_mean': float(np.mean(mem_score_vals)) if mem_score_vals else 0.0,
        }
        # save last model
        torch.save(model.state_dict(), SAVED / f'{name.lower()}.pt')

    # gradient clipping demo: train same model with and without clipping on one batch and record grad norms
    model_a = SimpleRNN(vocab_size).to(device)
    model_b = SimpleRNN(vocab_size).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    # run one epoch but measure grad norm after a few steps
    grad_norms_no_clip = []
    grad_norms_clip = []
    it = iter(train_loader)
    for _ in range(5):
        xb, yb = next(it)
        xb = xb.to(device)
        yb = yb.to(device)
        # no clip
        opt_a.zero_grad()
        logits = model_a(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        total_norm = 0.0
        for p in model_a.parameters():
            if p.grad is not None:
                total_norm += float(p.grad.data.norm(2).item() ** 2)
        grad_norms_no_clip.append(math.sqrt(total_norm))
        opt_a.step()
        # with clip
        opt_b.zero_grad()
        logits = model_b(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        total_norm = 0.0
        for p in model_b.parameters():
            if p.grad is not None:
                total_norm += float(p.grad.data.norm(2).item() ** 2)
        grad_norms_clip.append(math.sqrt(total_norm))
        opt_b.step()
    # plot grad norms
    fig, ax = plt.subplots()
    ax.plot(grad_norms_no_clip, label='no_clip')
    ax.plot(grad_norms_clip, label='clip_1.0')
    ax.set_title('Gradient norm (no clip vs clip=1.0)')
    ax.legend()
    fig.savefig(FIGURES / 'grad_clip.png', dpi=180)
    plt.close(fig)

    # perplexity demo
    ppl = _perplexity_demo()

    # seq2seq demo
    # increase copy-task training for more realistic BLEU
    seq2seq_res = _seq2seq_copy_demo(vocab=200, seq_len=8, train_size=2000, epochs=20)
    (ANALYSIS / 'seq2seq_examples.json').write_text(json.dumps(seq2seq_res, indent=2), encoding='utf-8')

    # save results and plots for model comparison
    (ANALYSIS / 'models_comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    ks = [4, 8, 16, 32, 64]
    memory_curves = {}
    for model_name, cls in model_classes.items():
        path = SAVED / f'{model_name.lower()}.pt'
        model = cls(vocab_size).to(device)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
        vals = []
        for k in ks:
            try:
                vals.append(_eval_first_k(model, test_loader, device, k=k))
            except Exception:
                vals.append(_eval(model, test_loader, device))
        memory_curves[model_name] = vals
    (ANALYSIS / 'memory_curves.json').write_text(
        json.dumps({'ks': ks, 'curves': memory_curves}, indent=2), encoding='utf-8'
    )

    names = sorted(results.keys(), key=lambda n: results[n]['accuracy_mean'], reverse=True)
    stability = [results[n]['accuracy_std'] for n in names]
    performance = [results[n]['accuracy_mean'] for n in names]
    cost = [results[n]['train_time_mean'] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].bar(names, stability, color='#ff7f0e')
    axes[0].set_title('Stability (accuracy std)')
    axes[0].set_ylabel('Std')
    axes[1].bar(names, performance, color='#1f77b4')
    axes[1].set_title('Performance (mean accuracy)')
    axes[1].set_ylim(0, 1)

    markers = {'RNN': 'o', 'LSTM': 's', 'GRU': '^'}
    colors = {'RNN': '#1f77b4', 'LSTM': '#ff7f0e', 'GRU': '#2ca02c'}
    jitter = {'RNN': 0.0, 'LSTM': 0.0008, 'GRU': -0.0008}
    for model_name in names:
        vals = memory_curves.get(model_name, [0.0] * len(ks))
        vals_j = [v + jitter.get(model_name, 0.0) for v in vals]
        axes[2].plot(
            ks, vals_j, marker=markers.get(model_name, 'o'),
            label=model_name, color=colors.get(model_name, None), linewidth=2,
        )
        for x, y in zip(ks, vals_j):
            axes[2].text(x, y + 0.0006, f"{(y - jitter.get(model_name, 0.0)):.3f}", fontsize=8, ha='center')
    axes[2].set_xticks(ks)
    axes[2].set_title('Memory-decay (first-K accuracy)')
    axes[2].set_ylabel('Accuracy (first-K)')
    if memory_curves:
        ymin = max(0.0, min(min(v) for v in memory_curves.values()) - 0.01)
        axes[2].set_ylim(ymin, 1.002)
    axes[2].legend()

    axes[3].bar(names, cost, color='#d62728')
    axes[3].set_title('Computation cost (sec / epoch)')
    axes[3].set_ylabel('Seconds')
    fig.tight_layout()
    fig.savefig(FIGURES / 'models_detailed_comparison.png', dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(names, performance, color=['#1f77b4' for _ in names])
    ax2.set_ylim(0, 1)
    ax2.set_title('Models Performance (mean accuracy)')
    ax2.set_ylabel('Accuracy')
    fig2.tight_layout()
    fig2.savefig(FIGURES / 'models_comparison.png', dpi=180)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    for model_name in names:
        vals = memory_curves.get(model_name, [0.0] * len(ks))
        ax3.plot(
            ks, vals, marker=markers.get(model_name, 'o'),
            label=model_name, color=colors.get(model_name, None), linewidth=2,
        )
    ax3.set_xticks(ks)
    ax3.set_xlabel('K (first tokens)')
    ax3.set_ylabel('Accuracy on first-K tokens')
    ax3.set_title('Memory-decay (first-K accuracy)')
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(FIGURES / 'memory_decay.png', dpi=180)
    plt.close(fig3)

    summary = {
        'part': 'part3_rnn',
        'status': 'success',
        'device': str(device),
        'models': results,
        'perplexity_demo': ppl,
        'grad_clip_png': str(FIGURES / 'grad_clip.png'),
        'models_comparison_png': str(FIGURES / 'models_comparison.png'),
        'seq2seq_examples': str(ANALYSIS / 'seq2seq_examples.json'),
        'avg_bleu_greedy': seq2seq_res.get('avg_bleu_greedy', None),
        'avg_bleu_beam': seq2seq_res.get('avg_bleu_beam', None),
    }
    (OUTPUT_DIR / 'checklist_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(run())
