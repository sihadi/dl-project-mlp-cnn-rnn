# Part 2 — CNN / Fashion-MNIST

This part covers the CNN checklist with runnable code and generated evidence.

## Checklist coverage

1. **Why CNNs over MLPs for images**
   - Flattening destroys spatial locality.
   - CNNs use local receptive fields, weight sharing, and hierarchical feature extraction.

2. **Manual calculations**
   - See `analysis/manual_calculations.json` for worked examples.
   - Output-size formulas are included in `analysis/report.md`.

3. **Custom implementations**
   - `convolution_manual.py`: 2D cross-correlation.
   - `pooling_manual.py`: max-pooling and average-pooling.

4. **PyTorch comparison**
   - See `analysis/comparison.json` for max absolute differences against `torch.nn.functional`.

5. **CNN architecture**
   - `models.py` contains `SimpleCNN` and `LeNetLikeCNN`.

6. **Architectural study**
   - `validate_checklist.py` tests padding, stride, pooling type, number of filters, and 1×1 convolution.

7. **Feature maps**
   - `validate_checklist.py` generates `figures/feature_maps.png`.

8. **MLP vs CNN**
   - `validate_checklist.py` generates `figures/mlp_vs_cnn.png` and `analysis/mlp_vs_cnn.json`.

## Run

```bash
c:/python314/python.exe -c "import sys; sys.path.insert(0, r'C:\Users\lenovo\DL_Project\dl_multi_agents_project'); from part2_cnn.validate_checklist import run; print(run())"
```
