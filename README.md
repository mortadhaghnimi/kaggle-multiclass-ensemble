# Multiclass Image Classification — 1st Place

I built this image classification system for a multiclass competition using CNNs trained from scratch.

The final ensemble reached **93.4% test accuracy and finished 1st on the leaderboard**.

No pretrained models were used.

## Result

| | |
|---|---|
| **Final Rank** | 1st Place |
| **Test Accuracy** | 93.4% |
| **Input Images** | 16×16 RGB |
| **Training** | From scratch |
| **Final Ensemble** | 3 CNNs |

The original images were only 16×16 pixels, so one of the main challenges was extracting useful visual features from very limited image information.

## Approach

I did not rely on a single model.

The final solution combined three different CNN architectures:

- WideResNet-28-10
- WideResNet-40-4
- ResNet-110

Each model learns slightly different representations, so combining their predictions gave a stronger final result than using one network alone.

## Training Pipeline

```text
16×16 RGB Images
       |
       v
Lanczos Upscaling to 32×32
       |
       v
Data Augmentation
       |
       ├── CutMix
       ├── Mixup
       ├── Random Crop
       ├── Horizontal Flip
       └── Brightness / Contrast
       |
       v
Three CNN Models
       |
       ├── WideResNet-28-10
       ├── WideResNet-40-4
       └── ResNet-110
       |
       v
Pseudo-Labeling
       |
       v
Test-Time Augmentation
       |
       v
Weighted Ensemble
       |
       v
93.4% Accuracy — 1st Place
```

## Techniques I Used

### Data augmentation

Because the images were small, overfitting was an important problem.

I used several augmentation strategies:

- CutMix
- Mixup
- random cropping
- horizontal flipping
- brightness and contrast changes

This gave the models more variation during training without requiring additional labeled data.

### Training

The models were trained using:

- SGD with Nesterov momentum
- cosine learning-rate decay
- label smoothing

I trained the architectures separately so the ensemble would have more model diversity.

### Pseudo-labeling

After the initial training, I used confident predictions from the test set as additional training examples.

I used two pseudo-labeling rounds with confidence thresholds increasing from:

```text
0.95 → 0.98
```

Only high-confidence predictions were added.

### Test-Time Augmentation

At inference time, each model predicts multiple augmented versions of the same image.

I used **15 TTA passes per model** and averaged the predictions.

### Ensemble

The final prediction combines all three CNNs.

Instead of giving every model the same importance, I searched for ensemble weights using validation performance.

```text
WideResNet-28-10
        \
WideResNet-40-4 ----> Weighted Ensemble ----> Final Prediction
        /
ResNet-110
```

## Why the Ensemble Helped

The three networks do not make exactly the same mistakes.

A model may classify an image incorrectly while the other two predict it correctly. Combining them reduced some of these individual model errors and produced a more stable final prediction.

This was especially useful with the low-resolution input images.

## Tech Stack

**Deep Learning:** TensorFlow, Keras  
**Models:** WideResNet, ResNet  
**Data:** NumPy, Pandas  
**Training:** SGD, Nesterov, Cosine Decay  
**Techniques:** CutMix, Mixup, Pseudo-Labeling, TTA, Model Ensembling  
**Environment:** Google Colab / GPU

## Repository

```text
kaggle-multiclass-ensemble/
├── notebooks/
│   └── Kaggle_Competition.ipynb
├── requirements.txt
└── README.md
```

The notebook contains the complete training and inference pipeline used for the competition.

## What I would try next

The competition is finished, but there are still several experiments I would explore to push the accuracy further:

- stronger ensemble diversity
- improved pseudo-label selection
- calibration before pseudo-labeling
- more systematic ensemble-weight optimization
- additional augmentation experiments
- error analysis by class

For me, the most useful part of this project was seeing how much performance can come from the complete training pipeline, not only from choosing a larger model.

## Author

**Mortadha Ghnimi**

M.S. Artificial Intelligence — Grand Valley State University

[GitHub](https://github.com/mortadhaghnimi) · [LinkedIn](https://www.linkedin.com/in/mortadha-ghnimi-021183252/)
