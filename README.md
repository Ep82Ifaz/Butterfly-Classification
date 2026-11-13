# Butterfly Image Classification (InceptionV3 Transfer Learning)

A Jupyter notebook that classifies butterfly photos into **75 species** using transfer learning with **InceptionV3** (TensorFlow/Keras). It reaches about **89% validation accuracy** on a stratified hold-out split.

The notebook is written for **Kaggle** and uses the
[Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) dataset.

## Results

| Metric | Value |
|---|---|
| Classes | 75 |
| Labeled images | 6,499 (5,199 train / 1,300 validation) |
| Validation accuracy | ~0.89 |
| Validation loss | ~0.45 |
| Unlabeled test images | 2,786 (predictions written to CSV) |

The notebook also produces training curves, a confusion matrix, and a per-class precision/recall/F1 report.

## Dataset

Expected layout (found automatically under `/kaggle/input`):

```
Training_set.csv    # filename, label
Testing_set.csv     # filename (no labels)
train/              # training images
test/               # test images
```

The test set has no labels, so evaluation uses a validation split carved out of the training data.

## Pipeline

1. **Setup**: imports, GPU check, dataset discovery and hyperparameters.
2. **Exploration**: checks that all image files exist, plots class balance, shows sample images.
3. **Split**: 80/20 train/validation, stratified by class (`seed=42`).
4. **Data pipeline**: `tf.data` loading, resize to 299×299, one-hot labels, batch size 32. Augmentation is applied to the training set only:
   - horizontal flip
   - rotation (±10%)
   - zoom (20%)
   - translation (10%)
   - contrast (20%)
5. **Model**: InceptionV3 (ImageNet weights, no top) → GlobalAveragePooling → BatchNorm → Dense(512, ReLU) → Dropout(0.4) → Dense(75, softmax). Inputs are rescaled to [-1, 1] inside the model.
6. **Phase 1, feature extraction**: base frozen, Adam at `1e-3`, up to 12 epochs.
7. **Phase 2, fine-tuning**: unfreeze the base from layer 249 onward (BatchNorm layers stay frozen), Adam at `1e-5`, up to 10 more epochs.
8. **Training controls**: `ModelCheckpoint`, `EarlyStopping` (patience 4, restores best weights) and `ReduceLROnPlateau` (factor 0.2), all monitoring `val_loss`.
9. **Evaluation**: accuracy and loss, training curves, confusion matrix, classification report.
10. **Export and inference**: saves the model and class names, provides a `predict_butterfly()` helper (top-3 predictions), and predicts the whole test set.

## Requirements

- Python 3.x
- TensorFlow 2.x (tested with 2.21)
- numpy, pandas, matplotlib, seaborn, Pillow, scikit-learn
- A GPU is strongly recommended (the notebook ran on Kaggle GPUs; each epoch takes roughly 2 minutes).

## How to run

### On Kaggle (intended)
1. Upload `image-classification.ipynb` as a new notebook.
2. Click **Add Input** and add the *butterfly-image-classification* dataset.
3. Turn on a GPU accelerator.
4. Turn **Internet on** so the ImageNet weights can download. If it is off, add the `inception_v3_..._notop.h5` weights file as an input. As a last resort the notebook trains from scratch (`ALLOW_SCRATCH = True`), which gives much worse accuracy.
5. Run all cells.

### Locally
Paths are hardcoded to `/kaggle/input` and `/kaggle/working`. Change `BASE_DIR` in the setup cell and the output paths (`/kaggle/working/...`) to local folders.

## Outputs

| File | Description |
|---|---|
| `best_phase1.weights.h5`, `best_phase2.weights.h5` | Best checkpoints from each phase |
| `inceptionv3_butterfly.keras` | Final trained model |
| `class_names.json` | Index → species label mapping |
| `test_predictions.csv` | Predicted label for each test image |

## Predicting on a single image

```python
predict_butterfly("path/to/butterfly.jpg", top_k=3)
```

This displays the image and prints the top-3 species with confidence scores.

## Key hyperparameters

| Name | Value |
|---|---|
| `IMG_SIZE` | (299, 299) |
| `BATCH_SIZE` | 32 |
| `PHASE1_EPOCHS` / `PHASE2_EPOCHS` | 12 / 10 |
| `LR_PHASE1` / `LR_PHASE2` | 1e-3 / 1e-5 |
| `FINE_TUNE_FROM` | 249 |
| `SEED` | 42 |

## Possible improvements

- Unfreeze more layers or try a stronger backbone (EfficientNet, ConvNeXt).
- Test-time augmentation.
- Cross-validation instead of a single split.
- Look at the most confused species pairs in the confusion matrix.
