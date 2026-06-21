# Part 1 - MLP on Breast Cancer

This part covers the required MLP workflow for the PDF project.

## Fundamental concepts

- `nn.Module`: both `SimpleMLP` and `SequentialMLP` inherit from `torch.nn.Module`.
- Parameters: the learnable weights and biases are visible through `named_parameters()`.
- `state_dict()`: used to save, reload, and inspect model weights.
- Device: the training script selects `cuda` when available, otherwise `cpu`.
- Forward pass: the model computes logits from the input features.
- Backpropagation: gradients are computed with `loss.backward()` and applied with Adam.

## Data preparation

- Breast Cancer data is loaded from scikit-learn.
- Features are standardized with `StandardScaler`.
- Data is split into train, validation, and test sets.

## Two MLP versions

- `SimpleMLP`: custom class with explicit layers in `forward()`.
- `SequentialMLP`: version built with `nn.Sequential`.

## Initialization strategies

The training script tests three initializations:

- Gaussian
- Constant
- Xavier

## Inspection and evaluation

The script stores a parameter report with:

- parameter names and shapes from `named_parameters()`
- keys from `state_dict()`
- total and trainable parameter counts

It also saves:

- the best model in `saved_models/best_model.pt`
- a metrics summary image in `figures/metrics_summary.png`
- a confusion matrix in `figures/confusion_matrix.png`
