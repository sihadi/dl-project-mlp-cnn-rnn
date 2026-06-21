import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from part3_rnn.train import _prepare_data
from part3_rnn.models import SimpleLSTM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_OUTPUTS = PROJECT_ROOT / 'outputs'

train_loader, test_loader, vocab_size = _prepare_data()
model = SimpleLSTM(vocab_size=vocab_size)
# find saved model
candidates = [PROJECT_OUTPUTS / 'part3_rnn' / 'saved_models' / 'lstm.pt', PROJECT_OUTPUTS / 'part3_imdb' / 'saved_models' / 'lstm.pt']
state = None
for c in candidates:
    if c.exists():
        state = c; break
if state is None:
    from pathlib import Path as P
    ws = P(r'C:/Users/lenovo/DL_Project')
    for c in ws.rglob('**/saved_models/lstm.pt'):
        state = c; break
if state is None:
    raise FileNotFoundError('lstm.pt not found')

model.load_state_dict(torch.load(state, map_location='cpu'))
model.eval()
ys, ps = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        p = logits.softmax(dim=1).argmax(dim=1).numpy()
        ps.append(p); ys.append(yb.numpy())

y_true = np.concatenate(ys)
y_pred = np.concatenate(ps)
cm = __import__('sklearn.metrics').metrics.confusion_matrix(y_true, y_pred)
out = PROJECT_OUTPUTS / 'part3_imdb' / 'figures' / 'confusion_matrix.png'
if not (PROJECT_OUTPUTS / 'part3_imdb').exists():
    out = PROJECT_OUTPUTS / 'part3_rnn' / 'figures' / 'confusion_matrix.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Part3 RNN')
plt.xlabel('Pred')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print('Saved', out)
