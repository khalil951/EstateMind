# CNN Model Implementations and Hyperparameters

This document documents the three CNN backbones used in the notebook benchmark and provides example implementations and hyperparameter tuning suggestions.

## ResNet50
- Backbone: `resnet50` (use `ResNet50_Weights.IMAGENET1K_V2` when available)
- Classifier head: `Dropout(p=0.2)` + `Linear(in_features -> num_classes)`
- Freeze behaviour: `freeze_backbone=True` by default (only head trained)

Example implementation (matches notebook `ResNet50Classifier`):

```python
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class ResNet50Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            weights = ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
            model = resnet50(weights=weights)
        except Exception:
            model = resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=cfg.dropout), nn.Linear(in_features, cfg.num_classes))
        self.model = model

    def forward(self, x):
        return self.model(x)
```

## EfficientNet-B0
- Backbone: `efficientnet_b0` (use `EfficientNet_B0_Weights.IMAGENET1K_V1` when available)
- Classifier head: `Dropout(p=0.2)` + `Linear(in_features -> num_classes)`

Example implementation (matches notebook `EfficientNetB0Classifier`):

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            model = efficientnet_b0(weights=weights)
        except Exception:
            model = efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout, inplace=True),
            nn.Linear(in_features, cfg.num_classes),
        )
        self.model = model

    def forward(self, x):
        return self.model(x)
```

## EfficientNetV2-S (timm)
- Backbone: `efficientnetv2_s` via `timm.create_model`
- Use `pretrained=True` when available
- `drop_rate` set to `cfg.dropout` (0.2 default)
- Create with `num_classes` matching the dataset

Example implementation (matches notebook `EfficientNetV2SClassifier`):

```python
import timm
import torch.nn as nn

class EfficientNetV2SClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            model = timm.create_model(
                'efficientnetv2_s',
                pretrained=cfg.pretrained,
                num_classes=cfg.num_classes,
                drop_rate=cfg.dropout,
            )
        except Exception:
            model = timm.create_model(
                'efficientnetv2_s',
                pretrained=False,
                num_classes=cfg.num_classes,
                drop_rate=cfg.dropout,
            )
        self.model = model

    def forward(self, x):
        return self.model(x)
```

---

## Common training defaults used in the notebook
- Epochs: 3
- Batch size: 32
- Image size: 224×224
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: CrossEntropyLoss
- Device: CUDA if available else CPU
- Freeze backbone: True (only classifier head trained)
- Pretrained backbones: True when available
- Dropout: 0.2
- Training augmentations: Resize(224,224), HorizontalFlip(p=0.5), RandomBrightnessContrast(p=0.3), ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=10, p=0.4), Normalize(imagenet_mean/std), ToTensorV2
- DataLoader `num_workers=0` in notebook; increase for local multi-core runs

## Suggested hyperparameters and ranges to tune
- Learning rate (head-only fine-tune): [1e-4, 3e-4, 1e-3]
- Learning rate (full fine-tune): [1e-5, 3e-5, 1e-4]
- Weight decay: [0.0, 1e-5, 1e-4, 1e-3]
- Batch size: [16, 32, 64] (depending on GPU memory)
- Epochs: [5, 10, 20] with early stopping on `val_f1_macro`
- Dropout (head): [0.0, 0.2, 0.5]
- Freeze backbone: [True (train head), False (fine-tune full model)]
- Optimizer: try `AdamW`, `SGD` with momentum=0.9, and `RAdam` for stability
- Learning rate scheduler: `OneCycleLR`, `CosineAnnealingWarmRestarts`, or `ReduceLROnPlateau`
- Warmup steps/epochs: small warmup for large LR (e.g., Linear warmup over 5-10% of total steps)
- Gradient clipping: max_norm in [0.0 (off), 1.0, 2.0]
- Augmentation strength: tune `RandomBrightnessContrast` magnitude and `ShiftScaleRotate` limits; consider `MixUp` or `CutMix`
- Image size: [224, 256, 320] — larger sizes may improve accuracy at higher compute cost
- Label smoothing: [0.0, 0.05, 0.1]
- Class reweighting or focal loss if class imbalance is severe
- Mixed precision training: enable `torch.cuda.amp` for faster training and larger effective batch sizes
- Number of `num_workers` for DataLoader: set to CPU cores/2 (eg. 4) for faster I/O

## Quick tuning recipes
- Head-only fine-tune: set `freeze_backbone=True`, try LR in [1e-4, 3e-4], epochs 5–10, monitor `val_f1_macro`.
- Full fine-tune: unfreeze backbone after head warmup (2–3 epochs) and switch to LR 1e-5–1e-4 with cosine scheduler.
- Large-batch recipe: increase batch size and scale LR linearly; use `OneCycleLR` with weight decay 1e-4.

---

If you want, I can:
- Insert these docstrings directly into `notebooks/notebook_images.ipynb` code cells (update notebook) — I can do that next.
- Extract the exact `in_features` for each backbone by instantiating the models in a small snippet and printing them.
- Create a small tuning YAML/Optuna config for hyperparameter search.

Which next step do you want me to take?