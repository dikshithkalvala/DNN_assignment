# Visual Story Position Classification

Deep Neural Networks and Learning Systems reassessment practical.

## Executive Summary

This project solves the reassessment task as an **image-only 5-class classification problem**. Given a single image from a five-frame story, the model predicts the image position in the narrative:

| Label | Meaning |
|---:|---|
| 1 | First story position |
| 2 | Second story position |
| 3 | Third story position |
| 4 | Fourth story position |
| 5 | Fifth story position |

I chose the **image modality** because the supplied `Sotrytelling` folder contains 1,370 images and matching XML files, but no text sentences. Since 1,370 is divisible by 5, the dataset forms **274 five-frame stories**. The split is done by story, not by individual image, so frames from the same story do not appear in both training and validation.

## Hypothesis / Pre-registered Plan

The baseline small CNN should learn visual signals related to narrative progression, such as object position, scene changes, and repeated visual structure across ordered frames. I expect **batch normalization** and **dropout** to improve validation performance by stabilizing training and reducing overfitting. I expect simply increasing model size to help only if the model still generalizes.

## Dataset Construction

The dataset is built from the project-local folder:

```text
Sotrytelling/
```
⚠️ Dataset Availability Note

The original Sotrytelling dataset folder (containing 1,370 images and XML annotations) is not included in this GitHub repository due to size constraints and GitHub file handling limitations for large multi-file datasets.

Including the full dataset would:

exceed practical repository size limits
make cloning and submission unnecessarily heavy
violate common ML project repository practices

The project is designed to load it locally using a fixed folder structure.
Therefore, only the source code, configuration files, and experimental results are provided in this repository.

Construction steps:

1. Sort image files using natural numeric order.
2. Group every five consecutive images as one story.
3. Assign labels by within-story position:
   - image 1 in story -> label 1
   - image 2 in story -> label 2
   - image 3 in story -> label 3
   - image 4 in story -> label 4
   - image 5 in story -> label 5
4. Split stories into 80% training and 20% validation.
5. Resize images to `64 x 64`.
6. Normalize images with ImageNet mean and standard deviation.

Expected dataset size:

| Split | Stories | Images |
|---|---:|---:|
| Train | 219 | 1,095 |
| Validation | 55 | 275 |
| Total | 274 | 1,370 |

Each class is balanced because every story has exactly one frame for each position.

## Model

The baseline model is a compact CNN with:

- 3 convolution blocks
- ReLU activations
- max pooling
- global average pooling
- linear classifier with output size 5
- `CrossEntropyLoss`

The code is modular:

| File | Purpose |
|---|---|
| `config.yaml` | dataset path and hyperparameters |
| `src/dataset.py` | story grouping, labels, split, dataloaders |
| `src/model.py` | CNN architecture |
| `src/train.py` | training and evaluation loops |
| `src/experiments.py` | baseline plus five controlled experiments |
| `src/visualise.py` | plots, result tables, sample predictions |

## Experiments

The practical requires five model variations. Each variation changes exactly one aspect compared with the baseline.

| Experiment | One change from baseline |
|---|---|
| baseline | reference CNN |
| dropout_030 | add dropout with probability 0.30 |
| larger_filters | increase convolution filters from `[16, 32, 64]` to `[32, 64, 128]` |
| kernel_5 | change convolution kernel size from 3 to 5 |
| batch_norm | add batch normalization after convolutions |
| four_conv_layers | add one extra convolution block |

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete practical:

```bash
python -m src.experiments --config config.yaml
```

Run a quick smoke test:

```bash
python -m src.experiments --config config.yaml --epochs 1 --experiments baseline
```

Outputs are saved in `results/`:

- `results/results_table.csv`
- `results/loss_curves.png`
- `results/accuracy_curves.png`
- `results/class_distribution.png`
- `results/sample_predictions.png`
- `results/confusion_matrix.png`
- `results/per_class_accuracy.png`
- `results/prediction_distribution.png`
- `results/validation_story_strip.png`
- `results/metrics.json`

## Results Table

The completed run produced these results with 5 epochs. The results are also saved in `results/results_table.csv`.

| Experiment | Modification | Train Loss | Validation Loss | Train Accuracy | Validation Accuracy |
|---|---|---:|---:|---:|---:|
| baseline | reference CNN | 1.6107 | 1.6099 | 0.1982 | 0.2000 |
| dropout_030 | add dropout p=0.30 | 1.6101 | 1.6097 | 0.1954 | 0.1964 |
| larger_filters | filters 16/32/64 -> 32/64/128 | 1.6097 | 1.6095 | 0.1881 | 0.2000 |
| kernel_5 | kernel size 3 -> 5 | 1.6105 | 1.6100 | 0.1936 | 0.2000 |
| batch_norm | add batch normalization | 1.6149 | 1.6134 | 0.1772 | 0.1855 |
| four_conv_layers | add fourth convolution block | 1.6099 | 1.6096 | 0.2000 | 0.2000 |

The chance baseline for five balanced classes is 20%. The models are close to chance, which is a useful finding: with a strict story-level split, the model cannot rely on seeing near-duplicate frames from the same story during training. This makes the evaluation harder but fairer.

## Analysis Questions

### 1. Which modification improved performance most?

No modification clearly improved performance. Baseline, larger filters, kernel size 5, and four convolution layers all reached 0.2000 validation accuracy. The most important conclusion is that architecture changes alone did not overcome the limited single-frame signal.

### 2. Which caused overfitting?

There is no strong overfitting pattern in the 5-epoch run because both training and validation accuracy stay near chance. Batch normalization performed worst here, with 0.1855 validation accuracy, but that is underfitting or unstable learning rather than classic overfitting.

### 3. How do you detect overfitting from the curves?

Overfitting appears when the training curve improves while the validation curve gets worse. For this task, that means training loss decreases or training accuracy increases, but validation loss rises or validation accuracy stays flat/decreases.

### 4. Did increasing model size always help?

No. The larger-filter model and the four-layer model did not improve beyond the baseline validation accuracy. This shows that more parameters do not automatically solve a weak-signal classification problem.

### 5. Why is predicting sequence position difficult?

The task is difficult because many five-frame stories are near-duplicate photographs from the same scene, where only small camera or object movements distinguish positions. A single image does not always contain clear time information. Without the surrounding frames, position prediction is much harder than ordinary object classification.


## Submission Checklist

- `experiments.ipynb` included.
- Modular source code included under `src/`.
- `config.yaml` included.
- `README.md` includes summary, method, results, and analysis.
- `results/` includes tables, plots, and sample predictions.
- Dataset size and class distribution are reported.
- Five controlled experiments are implemented.
- Loss and accuracy are computed for training and validation.
