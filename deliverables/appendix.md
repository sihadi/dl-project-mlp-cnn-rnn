# Annexe expérimentale

Cette annexe regroupe les métriques, tableaux comparatifs et visualisations produites par le pipeline `deliverables/run_all_and_collect.py`.

## Partie I — MLP (Breast Cancer)

| Métrique | Valeur |
|----------|--------|
| Accuracy test | 0.9474 |
| Précision | 0.9460 |
| Rappel | 0.9722 |
| F1-score | 0.9589 |
| Meilleur modèle | SequentialMLP + Xavier |
| Paramètres entraînables | ~4 130 |

**Figures** : `confusion_matrix.png`, `metrics_summary.png`, `initialization_comparison.png`, `lime_explanation_part1.png`

**Fichiers JSON** : `analysis/part1_mlp_*.json`, `checklist_summary_part1_mlp.json`

## Partie II — CNN (Fashion-MNIST)

| Modèle | Accuracy | Précision | Rappel | F1 |
|--------|----------|-----------|--------|-----|
| MLP | 0.681 | — | — | — |
| CNN | 0.760 | — | — | — |

**Vérifications manuelles** : écart maximal absolu nul entre implémentation manuelle et PyTorch (`manual_calculations.json`, `comparison.json`).

**Ablations** : `ablation.json` (padding, stride, pooling, nombre de filtres, conv 1×1).

**Figures** : `mlp_vs_cnn.png`, `ablation.png`, `feature_maps.png`, `lime_explanation_part2.png`

## Partie III — RNN / Seq2Seq

| Modèle | Accuracy moyenne | Temps moyen (s/epoch) |
|--------|------------------|------------------------|
| RNN | 0.9992 | — |
| LSTM | 1.0000 | — |
| GRU | 1.0000 | — |

**IMDb (dataset réel)** : accuracy ≈ 0.5044 (classification binaire, pipeline léger).

**Seq2Seq (tâche de copie)** : BLEU greedy ≈ 0.13–0.19 (`seq2seq_examples.json`).

**Figures** : `models_comparison.png`, `models_detailed_comparison.png`, `memory_decay.png`, `grad_clip.png`

## Commandes de reproduction

```bash
python -m pip install -r requirements.txt
python deliverables/run_all_and_collect.py
python deliverables/generate_report_html.py
python deliverables/make_pdf_from_report_and_figures.py
python package_submission.py
```

Le fichier final à remettre est `deliverables/submission_final.zip`.
