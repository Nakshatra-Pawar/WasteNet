# WasteNet — Transfer‑Learning‑based Classification of Nine Waste Categories

Municipal solid‑waste sorting is tedious and error‑prone. **WasteNet** tackles this problem by training a compact image‑classifier on a *small, nine‑class waste dataset* using **transfer learning**.  
Instead of training from scratch, the project fine‑tunes only the last dense layer of four large ImageNet‑pre‑trained CNN backbones while freezing earlier layers.

---

## 1&nbsp;| Project Overview
WasteNet applies aggressive data augmentation, early stopping, L2 regularisation, batch normalisation and dropout to improve generalisation and combat overfitting. It reports Precision, Recall, AUC and F1 on train/validation/test splits.

## 2&nbsp;| Dataset
* Folder‑structured image set with one directory per class  
* 80 % of each folder → **train**, 20 % → **test**  
* 20 % of the training split is further held out as **validation**  
* Images resized/padded to 224 × 224 px  
* On‑the‑fly augmentation: random crop, zoom, rotation, flip, contrast & translation

## 3&nbsp;| Methodology

| Step | Details |
|------|---------|
| **Pre‑processing** | Resize / zero‑pad, one‑hot labels |
| **Augmentation**   | Keras `ImageDataGenerator` with the ops above |
| **Backbones**      | VGG‑16 · ResNet‑50 · ResNet‑101 · EfficientNet‑B0 |
| **Classifier head**| GAP → BN → Dropout 0.2 → Dense (softmax) |
| **Training**       | 50–100 epochs, batch size 5, `Adam`, `categorical_crossentropy`, early stopping |
| **Regularisation** | L2 weight decay, BatchNorm, Dropout 0.2 |
| **Metrics**        | Precision, Recall, F1, AUC per split |

## 4&nbsp;| Repository Structure
```text
.
├── data/                 # original images (ignored or stored with Git LFS)
├── notebooks/
│   └── WasteNet.ipynb    # end‑to‑end EDA, training & evaluation
├── src/
│   ├── dataloaders.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
│   ├── checkpoints/
│   └── plots/
├── requirements.txt
└── README.md   <-- you are here
```

## 5&nbsp;| Setup
```bash
git clone https://github.com/<your‑handle>/wastenet.git
cd wastenet
python -m venv .venv && source .venv/bin/activate       # optional
pip install -r requirements.txt
```
> **Tip:** GPU acceleration (CUDA 11+) cuts training time down to ~15 min per model.

## 6&nbsp;| Training & Evaluation
```bash
# Train all four backbones
python src/train.py --config configs/all_models.yaml

# Evaluate a single checkpoint
python src/evaluate.py --weights outputs/checkpoints/resnet50_best.h5
```
Trained weights and TensorBoard logs are saved under `outputs/`.

## 7&nbsp;| Results
| Model                       | Accuracy | Macro F1 | Macro AUC |
|-----------------------------|:--------:|:--------:|:---------:|
| Baseline (CNN from scratch) | **0.24** | **0.10** | **0.62** |
| VGG-16 (+ transfer learning)| **0.64** | **0.61** | **0.92** |

### Per-class F1 scores

| Class | Baseline F1 | VGG-16 F1 |
|-------|:-----------:|:---------:|
| Cardboard            | 0.00 → **0.70** |
| Food Organics        | 0.00 → **0.70** |
| Glass                | 0.00 → **0.48** |
| Metal                | 0.01 → **0.69** |
| Misc. Trash          | 0.00 → **0.39** |
| Paper                | 0.00 → **0.76** |
| Plastic              | 0.36 → **0.65** |
| Textile Trash        | 0.00 → **0.34** |
| Vegetation           | 0.54 → **0.81** |

> **Key take-aways**
> * Transfer learning with VGG-16 lifts overall **accuracy from 24 % to 64 %** and macro-F1 from 0.10 to 0.61 (6× improvement).  
> * The biggest gains appear in classes that were virtually unlearnable by the scratch model (Cardboard, Food Organics, Paper).  
> * Remaining weak spots are **Miscellaneous Trash** and **Textile Trash**, likely due to intra-class visual diversity.  
> * A macro-AUC of **0.92** indicates strong separability across all nine categories.



## 8&nbsp;| Known Limitations & Next Steps
* Confusions between similar‑looking plastic vs glass items  
* Explore vision transformers & self‑supervised pre‑training  
* Deploy as a mobile TensorFlow‑Lite app for on‑device sorting

## 9&nbsp;| Citation
If you use this work in research, please cite the original DSCI 552 assignment prompt.

---

*Project maintained by **Nakshatra Pawar** — “Dream big. Start small. Act now.”*
