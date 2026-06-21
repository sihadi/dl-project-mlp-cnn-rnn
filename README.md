# Deep Learning Multi-Agents Project

**Auteur :** [Aya Sihadi](https://github.com/sihadi) — EMSI, Casablanca  
**Année universitaire :** 2025–2026  
**Modules couverts :** MLP / PyTorch · CNN · RNN / LSTM / GRU / Seq2Seq

## Purpose
This project studies how deep learning architectures must be adapted to the structure of the data:
- tabular data with a multilayer perceptron (`MLP`),
- image data with a convolutional neural network (`CNN`),
- sequential text data with `RNN` / `LSTM` / `GRU` and a small `Seq2Seq` module.

The goal is not only to report accuracy, but also to discuss theory, experimental choices, internal representations, and model limits in an academic style.

## Main question
How does deep learning adapt to the geometry of data, local dependencies, temporality, and representation?

## Repository structure

### Part 1 — `part1_mlp/`
Tabular classification on the Breast Cancer dataset.

- `utils.py`: loads and standardizes the dataset.
- `models.py`: defines `SimpleMLP` and `SequentialMLP`.
- `train.py`: trains several MLP variants, compares initialization strategies, selects the best model, and exports metrics, confusion matrix, and parameter reports.
- `agents/agent_a.py`, `agents/agent_b.py`, `agents/agent_c.py`: simple agent templates.

### Part 2 — `part2_cnn/`
Image classification on Fashion-MNIST.

- `models.py`: defines `SimpleCNN`, `LeNetLikeCNN`, `FlexibleCNN`, and a reference `SimpleMLP`.
- `convolution_manual.py`: manual 2D cross-correlation and output-size formulas.
- `pooling_manual.py`: manual max-pooling and average-pooling.
- `validate_checklist.py`: produces comparisons, ablations, feature maps, and metrics.
- `generate_lime.py`: creates local explainability artifacts with `LIME`.
- `train.py`: minimal training pipeline.

### Part 3 — `part3_rnn/`
Sequential text classification and generation.

- `models.py`: defines `SimpleRNN`, `SimpleLSTM`, and `SimpleGRU`.
- `seq2seq.py`: encoder-decoder architecture with greedy and beam-search decoding.
- `tokenizer.py`: lightweight tokenizer.
- `train.py`: trains the sequence models on IMDb.
- `generate_memory_decay.py`: builds the memory-decay figure.
- `validate_checklist.py`: generates the main comparison figures, memory analysis, gradient clipping plot, perplexity demo, and Seq2Seq results.

### Deliverables

- `deliverables/report.md`: scientific report.
- `deliverables/appendix.md`: technical appendix.
- `deliverables/report.html`: HTML version of the report.
- `deliverables/report_generated.pdf`: PDF version of the report.
- `deliverables/figures/`: copied figures for submission.
- `deliverables/analysis/`: collected JSON analyses.
- `deliverables/submission_notebook.ipynb`: packaging notebook.
- `deliverables/submission_notebook_executed.ipynb`: executed notebook.
- `deliverables/submission_final.zip`: final archive to submit.

## Scientific answers

### 1) Why can a well-parameterized MLP be relevant for tabular classification?
An MLP is a strong baseline for tabular data because the input is already vectorized and the discriminative information is often carried by non-linear combinations of features rather than by a spatial or temporal geometry.

It becomes relevant when:
- features are properly standardized,
- interactions are mostly global,
- the dataset is not strongly structured by locality,
- the model capacity is chosen carefully.

Its main limits are structural:
- it flattens the data representation,
- it does not exploit local neighborhoods,
- it does not encode inductive biases about geometry,
- it may learn efficiently, but not necessarily interpret the data-generating structure.

So, an MLP is useful as a baseline or a good practical solution, but it is often not the most statistically adapted model when the data have richer structure.

### 2) Why is a CNN more appropriate than an MLP for image classification?
Images have local spatial regularities. A CNN is designed to exploit them through convolution, parameter sharing, and hierarchical feature extraction.

Key ideas:
- `padding` controls border preservation and output size,
- `stride` controls spatial downsampling and information compression,
- `pooling` reduces dimensionality and increases invariance,
- depth increases abstraction and receptive field size.

A CNN therefore keeps the 2D organization of pixels, while an MLP destroys it by flattening the image. The experimental results, feature maps, and LIME explanations should be interpreted as evidence of that structural advantage.

### 3) How should deep learning adapt to tabular, image, and sequential data?
The same supervised learning paradigm must be specialized because each data type has a different geometry:

- **Tabular**: use dense layers, normalization, and feature interactions (`MLP`).
- **Image**: use local filters, shared weights, and spatial hierarchies (`CNN`).
- **Sequence**: use recurrent or gated memory mechanisms to preserve order and context (`RNN`, `LSTM`, `GRU`).

The common principle is supervised optimization, but the architecture must respect the data structure:
- geometry for images,
- dependence between variables for tabular data,
- temporal order and memory for sequences.

## How to run

Install dependencies:

```powershell
cd dl_multi_agents_project
python -m pip install -r requirements.txt
```

Run the full validation pipeline (≈15 min on CPU):

```powershell
python deliverables/run_all_and_collect.py
```

Generate reports and submission archive:

```powershell
python deliverables/generate_report_html.py
python deliverables/make_pdf_from_report_and_figures.py
python package_submission.py --skip-run
```

Or run everything in one step (re-runs validators):

```powershell
python package_submission.py
```

**Note for the evaluator** : Internet is required on first run to download Fashion-MNIST and IMDb. GPU is optional (`cuda` used automatically if available).

## Recommended submission files

For submission, the essential files are:
- `deliverables/submission_final.zip`
- `deliverables/report_generated.pdf`
- `deliverables/report.html`

Optional supporting files:
- `deliverables/submission_notebook.ipynb`
- `deliverables/submission_notebook_executed.ipynb`
- `deliverables/appendix.md`

## Notes on the report
The report should be written in a clear academic style and should interpret every figure and metric in relation to its corresponding part. In particular:
- Part 1 should emphasize the relation between feature structure and MLP performance.
- Part 2 should explain why convolutional inductive biases help image classification.
- Part 3 should discuss memory, temporal dependence, stability, and sequence generation.

