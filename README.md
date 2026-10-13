# Coral Reef Benthic Point Classification

Classification of benthic point annotations from the Moorea Labeled Corals (MLC) 2010 dataset into six functional groups, comparing a convolutional neural network (CNN) and a fully connected deep neural network (DNN), both trained from scratch.

## Overview

Each quadrat photograph in MLC carries about 200 expert point annotations. For every point, the notebook extracts a 112 x 112 px patch centred on the point, resizes it to 64 x 64 px, and classifies it into one of six classes:

| Class | Source labels |
|---|---|
| CCA | CCA |
| Hard coral | Porit, P.Rus, P.Irr, Pmass, Pocill, Monti, Pavon, Mille, Acrop, Monta, Lepta, Fung, Gardin, Herpo, Favia, Cypha, Acan, Astreo, Psam, Lepto |
| Sand | Sand |
| Turf | Turf |
| Macroalgae | Macro, Lobo |
| Other | Off, Soft |

## Repository Structure

The notebook resolves all paths relative to its parent directory, so this repository must be cloned as `notebooks/` inside a project folder that also holds the dataset:

```
<project>/
├── Moorea Labeled Corals 2010/   dataset (not tracked)
├── cache/                        extracted patch arrays and split indices (generated)
├── models/                       trained models, training histories, tuner trials (generated)
├── figures/                      plots and result CSVs (generated)
└── notebooks/                    this repository
    ├── coral_classification.ipynb
    └── README.md
```

`cache/`, `models/` and `figures/` are created automatically on first run.

## Dataset

Moorea Labeled Corals 2010, collected by the Moorea Coral Reef Long Term Ecological Research site (MCR LTER) and published with Beijbom et al. (2012).

The version used here has 629 JPEG quadrat images, each with a matching `<name>.jpg.txt` annotation file in the format:

```
# Row; Col; Label
1440; 1364; CCA
773; 1505; CCA
...
```

`Row` is the pixel y-coordinate and `Col` is the pixel x-coordinate. 55 images from the original 684-image 2010 release were excluded because their JPEG files are truncated. The notebook reads only `*.jpg.txt` files, so any other files in the folder (such as `.png` images) are ignored. After label mapping, 125,171 points remain.

## Requirements

- Python 3.11
- TensorFlow 2.21 (Keras 3)
- keras-tuner
- tensorboard (required by Keras Tuner)
- opencv-python
- scikit-learn
- pandas
- numpy
- matplotlib
- jupyter

A CUDA-capable GPU is strongly recommended. The reference environment was WSL2 Ubuntu with a conda environment and an NVIDIA RTX 3050 Laptop GPU (6 GB). Native Windows builds of TensorFlow do not support GPUs after version 2.10.

## Setup

```bash
conda create -n tf-env python=3.11
conda activate tf-env
pip install "tensorflow[and-cuda]==2.21.*" keras-tuner tensorboard opencv-python scikit-learn pandas numpy matplotlib jupyter
```

Verify GPU visibility:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the pip-installed CUDA libraries are not found, add the `nvidia/*/lib` directories inside the environment's `site-packages` to `LD_LIBRARY_PATH`.

## Usage

Start Jupyter from inside `notebooks/` so that the parent directory resolves to the project root, then run all cells in order:

```bash
cd notebooks
jupyter notebook coral_classification.ipynb
```

The first run extracts the patches (about 1.5 GB in RAM) and caches them as `.npy` files in `cache/`. Later runs load from the cache. Delete the cache files to force a fresh extraction.

All settings are in the `CFG` class near the top of the notebook: paths, seed, crop and input size, class mapping, split fractions, batch size, epochs, learning rate and tuning budget.

## Method

1. **Parsing and integrity checks**: annotation files are loaded into a single DataFrame and checked for missing values, image dimensions and points outside the image bounds.
2. **Exploratory analysis**: raw label distribution, annotation overlays and image brightness.
3. **Class mapping**: label spellings are normalised and the 27 raw labels are mapped to six groups.
4. **Patch extraction**: a 112 x 112 crop with reflect padding at the borders, resized to 64 x 64 using area interpolation.
5. **Splitting**: a 70 / 15 / 15 train / validation / test split grouped by image with `GroupShuffleSplit`, so no image contributes patches to more than one split.
6. **Class balancing** (training split only): CCA is undersampled to 12,000 points, Turf and Macroalgae are oversampled to the median class count, and balanced class weights are applied to the loss.
7. **Augmentation**: random flips, rotation, brightness and contrast, applied inside the model and active only during training.
8. **Models**:
   - CNN: three blocks of two 3 x 3 convolutions with batch normalisation and ReLU, each block followed by max pooling and dropout (32, 64 and 128 filters), then a dense layer of 128 units and a softmax output.
   - DNN: flattened input, dense layers of 512, 256 and 128 units with batch normalisation, ReLU and dropout, then a softmax output.
9. **Training**: Adam with gradient clipping and sparse categorical cross-entropy. Early stopping and checkpointing monitor validation balanced accuracy, and the learning rate is reduced when validation loss plateaus.
10. **Tuning**: Keras Tuner `RandomSearch` over learning rate, layer width and dropout, with two executions per trial and validation balanced accuracy as the objective. The best configuration is retrained with the full training procedure.
11. **Evaluation**: test accuracy, macro-F1, per-class classification reports and row-normalised confusion matrices.

Balanced accuracy (the mean of per-class recall) is implemented as a custom Keras metric and is used for model selection, because the class distribution is heavily skewed towards CCA.

## Results

Test set, 18,905 patches from 95 held-out images:

| Model | Parameters | Test accuracy | Macro-F1 |
|---|---:|---:|---:|
| CNN (baseline) | 1,338,278 | 0.6387 | 0.5778 |
| CNN (tuned) | - | 0.5105 | 0.5041 |
| DNN (baseline) | 6,460,550 | 0.5465 | 0.4456 |
| DNN (tuned) | - | 0.5128 | 0.4232 |

Best hyperparameters found:

- CNN: `lr=5e-4, filters=24, dropout=0.4, dense_units=128`
- DNN: `lr=1e-4, units1=256, units2=384, dropout=0.2`

The baseline CNN performs best on both metrics while using about one fifth of the DNN's parameters. The tuned models scored below their baselines on the test set.

Figures and result tables are written to `figures/`, including `model_comparison.csv` and `tuning_before_after.csv`.

## Reproducibility

A fixed seed (42) is applied to Python, NumPy and TensorFlow, and the split indices are saved to `cache/splits.npz`. GPU kernels and oneDNN optimisations are not fully deterministic, so results may vary slightly between runs.