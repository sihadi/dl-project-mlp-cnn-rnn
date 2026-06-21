from pathlib import Path
import shutil
archive = Path(r'C:/Users/lenovo/DL_Project/archive_dl_multi_agents_project_2026-05-25/outputs')
project_outputs = Path(r'C:/Users/lenovo/DL_Project/dl_multi_agents_project/outputs')
copies = []
# part1
src1_html = archive / 'part1_breast_cancer' / 'lime_explanation.html'
src1_png = archive / 'part1_breast_cancer' / 'lime_explanation.png'
tgt1 = project_outputs / 'part1_mlp' / 'figures'
tgt1.mkdir(parents=True, exist_ok=True)
if src1_html.exists():
    shutil.copy2(src1_html, tgt1 / src1_html.name); copies.append(tgt1 / src1_html.name)
if src1_png.exists():
    shutil.copy2(src1_png, tgt1 / src1_png.name); copies.append(tgt1 / src1_png.name)
# part3
src3_html = archive / 'part3_imdb' / 'lime_explanation.html'
tgt3 = project_outputs / 'part3_imdb' / 'figures'
tgt3.mkdir(parents=True, exist_ok=True)
if src3_html.exists():
    shutil.copy2(src3_html, tgt3 / src3_html.name); copies.append(tgt3 / src3_html.name)
# summary print
for p in copies:
    print('COPIED', p)
