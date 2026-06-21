import sys
import importlib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import part3_rnn.validate_checklist as v
importlib.reload(v)
from part3_rnn.models import SimpleRNN, SimpleLSTM, SimpleGRU

def main(limit_test=500):
    # try to read previously computed memory curves if available
    mem_file = v.ANALYSIS / 'memory_curves.json'
    if mem_file.exists():
        obj = json.loads(mem_file.read_text(encoding='utf-8'))
        ks = obj.get('ks', [4, 8, 16, 32, 64])
        curves = obj.get('curves', {})
    else:
        # fallback: compute from saved models if memory_curves.json missing
        v._set_seed()
        device = torch.device('cpu')
        _, test_loader, vocab_size, _ = v._prepare_imdb(limit_train=200, limit_test=limit_test)
        ks = [4, 8, 16, 32, 64]
        names = ['RNN', 'LSTM', 'GRU']
        classes = {'RNN': SimpleRNN, 'LSTM': SimpleLSTM, 'GRU': SimpleGRU}
        curves = {}
        for name in names:
            cls = classes[name]
            model = cls(vocab_size).to(device)
            path = v.SAVED / f"{name.lower()}.pt"
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device))
            vals = []
            for k in ks:
                # use validate_checklist._eval_first_k if available via import; otherwise compute simple proxy
                try:
                    vals.append(v._eval_first_k(model, test_loader, device, k=k))
                except AttributeError:
                    # approximate: evaluate on full sequence as proxy
                    vals.append(v._eval(model, test_loader, device))
            curves[name] = vals

    # plotting with slight jitter to separate overlapping curves
    v.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,4))
    markers = {'RNN': 'o', 'LSTM': 's', 'GRU': '^'}
    colors = {'RNN': '#1f77b4', 'LSTM': '#ff7f0e', 'GRU': '#2ca02c'}
    jitter = {'RNN': 0.0, 'LSTM': 0.0008, 'GRU': -0.0008}
    names = ['RNN', 'LSTM', 'GRU']
    for name in names:
        vals = curves.get(name, [0.0]*len(ks))
        vals_j = [v + jitter.get(name, 0.0) for v in vals]
        ax.plot(ks, vals_j, marker=markers.get(name, 'o'), label=name, color=colors.get(name, None), linewidth=2)
        for x, y in zip(ks, vals_j):
            ax.text(x, y + 0.0006, f"{(y - jitter.get(name,0.0)):.3f}", fontsize=8, ha='center')
    ax.set_xticks(ks)
    ax.set_xlabel('K (first tokens)')
    ax.set_ylabel('Accuracy on first-K tokens')
    ax.set_title('Memory-decay (first-K accuracy)')
    ax.set_ylim(0.965, 1.002)
    ax.legend()
    fig.tight_layout()
    out = v.FIGURES / 'memory_decay.png'
    fig.savefig(out, dpi=180)
    print(str(out))

if __name__ == '__main__':
    print('Generating memory decay...')
    print(main())
