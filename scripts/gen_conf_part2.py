import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from part2_cnn.models import SimpleCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_OUTPUTS = PROJECT_ROOT / 'outputs'

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

test = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(test, batch_size=256)

# find saved model
candidates = [PROJECT_OUTPUTS / 'part2_cnn' / 'saved_models' / 'cnn.pt', PROJECT_OUTPUTS / 'part2_cnn' / 'part2_cnn' / 'saved_models' / 'cnn.pt']
state = None
for c in candidates:
    if c.exists():
        state = c; break
if state is None:
    # fallback: search workspace
    from pathlib import Path as P
    ws = P(r'C:/Users/lenovo/DL_Project')
    for c in ws.rglob('**/saved_models/cnn.pt'):
        state = c; break

if state is None:
    raise FileNotFoundError('cnn.pt not found')

model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load(state, map_location='cpu'))
model.eval()
ys, ps = [], []
with torch.no_grad():
    for xb, yb in loader:
        logits = model(xb)
        p = logits.softmax(dim=1).argmax(dim=1).numpy()
        ps.append(p); ys.append(yb.numpy())

y_true = np.concatenate(ys)
y_pred = np.concatenate(ps)
cm = confusion_matrix = __import__('sklearn.metrics').metrics.confusion_matrix(y_true, y_pred)
out = PROJECT_OUTPUTS / 'part2_cnn' / 'part2_cnn' / 'figures' / 'confusion_matrix.png'
if not (PROJECT_OUTPUTS / 'part2_cnn' / 'part2_cnn').exists():
    out = PROJECT_OUTPUTS / 'part2_cnn' / 'figures' / 'confusion_matrix.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(6,5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Part2 CNN')
plt.xlabel('Pred')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print('Saved', out)
