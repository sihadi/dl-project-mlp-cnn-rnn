import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from lime import lime_image
from skimage.color import gray2rgb

from .models import SimpleCNN

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'part2_cnn' / 'part2_cnn'
FIGURES = OUTPUT_DIR / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# load model
model = SimpleCNN(num_classes=10)
state_path = OUTPUT_DIR / 'saved_models' / 'cnn.pt'
if not state_path.exists():
    raise FileNotFoundError(f"Model state not found: {state_path}")
model.load_state_dict(torch.load(state_path, map_location='cpu'))
model.eval()

# prepare data (use raw dataset to get original grayscale image)
mnist = datasets.FashionMNIST('data', train=False, download=True)
img = mnist.data[0].numpy()  # shape (28,28), uint8
# create RGB by stacking grayscale
img_rgb = np.stack([img, img, img], axis=2)

# classifier function for LIME: expects list/array of images HxWx3
def classifier_fn(images):
    # images: list or array shape (N, H, W, 3)
    arr = []
    for im in images:
        # if RGB, convert to grayscale by luminosity
        if im.ndim == 3 and im.shape[2] == 3:
            gray = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = im
        # normalize to [0,1]
        gray = gray.astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        arr.append(tensor)
    batch = torch.cat(arr, dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

explainer = lime_image.LimeImageExplainer(random_state=42)
exp = explainer.explain_instance(img_rgb, classifier_fn, top_labels=5, hide_color=0, num_samples=200)
# choose top predicted label
label = exp.top_labels[0] if len(exp.top_labels) > 0 else 0
# get image and mask
temp, mask = exp.get_image_and_mask(label, positive_only=False, num_features=10, hide_rest=False)
# save overlay image
out_png = FIGURES / 'lime_explanation.png'
plt.imsave(out_png, temp.astype(np.uint8))
# save simple html embedding the PNG
out_html = FIGURES / 'lime_explanation.html'
html = f"<html><body><h3>LIME explanation - Fashion-MNIST (part2)</h3><img src=\"{out_png.name}\" alt=\"lime\"></body></html>"
out_html.write_text(html, encoding='utf-8')
print('Saved', out_png, out_html)
