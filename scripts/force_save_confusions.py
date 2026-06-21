from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

WORKSPACE_ROOT = Path(r'C:/Users/lenovo/DL_Project')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_OUTPUTS = PROJECT_ROOT / 'outputs'

# imports
try:
    from part1_mlp.utils import load_data as load_data_p1
    from part1_mlp.models import SimpleMLP
except Exception:
    load_data_p1 = None

try:
    from torchvision import datasets, transforms
    from part2_cnn.models import SimpleCNN
except Exception:
    datasets = None

try:
    from part3_rnn.train import _prepare_data as prepare_p3
    from part3_rnn.models import SimpleLSTM
except Exception:
    prepare_p3 = None


def save_cm(y_true, y_pred, out_path, title='Confusion'):
    cm = confusion_matrix(y_true, y_pred)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)

# find saved models
models = list(WORKSPACE_ROOT.rglob('**/saved_models/*.pt'))
print('Found models:', models)
for m in models:
    name = m.name.lower()
    path = m
    try:
        if 'mlp' in name or 'part1' in str(m).lower():
            if load_data_p1 is None:
                print('Part1 loader missing; skipping')
            else:
                X_train, X_test, y_train, y_test, _, _ = load_data_p1()
                model = SimpleMLP(input_dim=X_train.shape[1], hidden=64)
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                with torch.no_grad():
                    logits = model(torch.from_numpy(X_test).float())
                    preds = logits.softmax(dim=1).argmax(dim=1).numpy()
                out = PROJECT_OUTPUTS / 'part1_mlp' / 'figures' / 'confusion_matrix.png'
                save_cm(y_test, preds, out, title='Part1 MLP')
        elif 'cnn' in name or 'part2' in str(m).lower():
            if datasets is None:
                print('Part2 loader missing; skipping')
            else:
                transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
                test = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
                loader = torch.utils.data.DataLoader(test, batch_size=256)
                model = SimpleCNN(num_classes=10)
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                ys, ps = [], []
                with torch.no_grad():
                    for xb, yb in loader:
                        logits = model(xb)
                        p = logits.softmax(dim=1).argmax(dim=1).numpy()
                        ps.append(p); ys.append(yb.numpy())
                y_true = np.concatenate(ys)
                y_pred = np.concatenate(ps)
                out = PROJECT_OUTPUTS / 'part2_cnn' / 'part2_cnn' / 'figures' / 'confusion_matrix.png'
                if not (PROJECT_OUTPUTS / 'part2_cnn' / 'part2_cnn').exists():
                    out = PROJECT_OUTPUTS / 'part2_cnn' / 'figures' / 'confusion_matrix.png'
                save_cm(y_true, y_pred, out, title='Part2 CNN')
        elif 'lstm' in name or 'part3' in str(m).lower():
            if prepare_p3 is None:
                print('Part3 loader missing; skipping')
            else:
                tr, te, vocab_size = prepare_p3()
                model = SimpleLSTM(vocab_size=vocab_size)
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                ys, ps = [], []
                with torch.no_grad():
                    for xb, yb in te:
                        logits = model(xb)
                        p = logits.softmax(dim=1).argmax(dim=1).numpy()
                        ps.append(p); ys.append(yb.numpy())
                y_true = np.concatenate(ys)
                y_pred = np.concatenate(ps)
                out = PROJECT_OUTPUTS / 'part3_imdb' / 'figures' / 'confusion_matrix.png'
                if not (PROJECT_OUTPUTS / 'part3_imdb').exists():
                    out = PROJECT_OUTPUTS / 'part3_rnn' / 'figures' / 'confusion_matrix.png'
                save_cm(y_true, y_pred, out, title='Part3 RNN')
    except Exception as e:
        print('Error processing', m, e)

print('Done')
