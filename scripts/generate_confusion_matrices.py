from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Part 1 - MLP (breast cancer)
try:
    from part1_mlp.utils import load_data as load_data_p1
    from part1_mlp.models import SimpleMLP
except Exception:
    load_data_p1 = None

# Part 2 - CNN (FashionMNIST)
try:
    from torchvision import datasets, transforms
    from part2_cnn.models import SimpleCNN
except Exception:
    datasets = None

# Part 3 - RNN (IMDb)
try:
    from part3_rnn.train import _prepare_data as prepare_p3
    from part3_rnn.models import SimpleLSTM
except Exception:
    prepare_p3 = None

ROOT = Path(__file__).resolve().parents[1]
outputs = ROOT / 'outputs'
WORKSPACE_ROOT = Path(r'C:/Users/lenovo/DL_Project')

def save_conf_matrix(y_true, y_pred, labels, out_png: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print('Saved', out_png)

# Part1
try:
    if load_data_p1 is not None:
        X_train, X_test, y_train, y_test, feature_names, class_names = load_data_p1()
        model = SimpleMLP(input_dim=X_train.shape[1], hidden=64)
        # search workspace for saved model
        candidates = list(WORKSPACE_ROOT.rglob('part1*/*saved_models/mlp.pt')) + list(WORKSPACE_ROOT.rglob('**/saved_models/mlp.pt'))
        state = None
        for c in candidates:
            if c.exists():
                state = c; break
        if state:
            model.load_state_dict(torch.load(state, map_location='cpu'))
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X_test).float())
                preds = logits.softmax(dim=1).argmax(dim=1).numpy()
            # save into project scaffold figures dir
            p1_dir = outputs / 'part1_mlp'
            out_png = p1_dir / 'figures' / 'confusion_matrix.png'
            save_conf_matrix(y_test, preds, labels=np.unique(y_test), out_png=out_png, title='Part1 MLP - Confusion Matrix')
        else:
            print('Part1 model not found in workspace candidates')
except Exception as e:
    print('Part1 error:', e)

# Part2
try:
    if datasets is not None:
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        test = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test, batch_size=256)
        model = SimpleCNN(num_classes=10)
        # possible saved locations
        p2_dir = outputs / 'part2_cnn'
        # search workspace for cnn.pt
        candidates = list(WORKSPACE_ROOT.rglob('**/saved_models/cnn.pt')) + [p2_dir / 'saved_models' / 'cnn.pt', p2_dir / 'part2_cnn' / 'saved_models' / 'cnn.pt']
        state = None
        for c in candidates:
            if c.exists():
                state = c; break
        if state:
            model.load_state_dict(torch.load(state, map_location='cpu'))
            model.eval()
            preds = []
            ys = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb)
                    p = torch.softmax(logits, dim=1).argmax(dim=1).numpy()
                    preds.append(p); ys.append(yb.numpy())
            y_true = np.concatenate(ys)
            y_pred = np.concatenate(preds)
            # save into project scaffold figures dir (respect nested part2_cnn layout)
            out_png = (p2_dir / 'part2_cnn' / 'figures' / 'confusion_matrix.png') if (p2_dir / 'part2_cnn').exists() else (p2_dir / 'figures' / 'confusion_matrix.png')
            save_conf_matrix(y_true, y_pred, labels=list(range(10)), out_png=out_png, title='Part2 CNN - Confusion Matrix')
        else:
            print('Part2 model not found in candidates', candidates)
except Exception as e:
    print('Part2 error:', e)

# Part3
try:
    if prepare_p3 is not None:
        train_loader, test_loader, vocab_size = prepare_p3()
        model = SimpleLSTM(vocab_size=vocab_size)
        # search workspace for lstm.pt
        p3_dir = outputs / 'part3_imdb' if (outputs / 'part3_imdb').exists() else outputs / 'part3_rnn'
        candidates = list(WORKSPACE_ROOT.rglob('**/saved_models/lstm.pt')) + [p3_dir / 'saved_models' / 'lstm.pt', outputs / 'part3_rnn' / 'part3_rnn' / 'saved_models' / 'lstm.pt']
        state = None
        for c in candidates:
            if c.exists():
                state = c; break
        if state:
            model.load_state_dict(torch.load(state, map_location='cpu'))
            model.eval()
            preds = []
            ys = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb)
                    p = torch.softmax(logits, dim=1).argmax(dim=1).numpy()
                    preds.append(p); ys.append(yb.numpy())
            y_true = np.concatenate(ys)
            y_pred = np.concatenate(preds)
            out_png = p3_dir / 'figures' / 'confusion_matrix.png'
            save_conf_matrix(y_true, y_pred, labels=[0,1], out_png=out_png, title='Part3 RNN - Confusion Matrix')
        else:
            print('Part3 model not found at', state)
except Exception as e:
    print('Part3 error:', e)

print('Done')
