# WasteNet — Transfer‑Learning‑based Classification of Nine Waste Categories

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Keras](https://img.shields.io/badge/Keras-Transfer_Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

Municipal solid‑waste sorting is tedious and error‑prone. **WasteNet** tackles this problem by training a compact image‑classifier on a *small, nine‑class waste dataset* using **transfer learning**.  
Instead of training from scratch, the project fine‑tunes only the last dense layer of four large ImageNet‑pre‑trained CNN backbones while freezing earlier layers.

---

## 1&nbsp;| Project Overview
WasteNet follows the workflow specified in the DSCI 552 final project brief, applying aggressive data augmentation, early stopping, L2 regularisation, batch normalisation and dropout. It reports Precision, Recall, AUC and F1 scores on train/validation/test splits.

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
| Model           | Val F1 | Test F1 | Test AUC |
|-----------------|:------:|:-------:|:--------:|
| VGG‑16          | 0.88   | 0.86    | 0.97     |
| ResNet‑50       | **0.91** | **0.89** | **0.98** |
| ResNet‑101      | 0.90   | 0.88    | 0.98     |
| EfficientNet‑B0 | 0.89   | 0.87    | 0.97     |

ResNet‑50 edged out the other backbones across every metric, so its weights are provided as the default model.

## 8&nbsp;| Inference Demo
```python
from src.predict import classify
print(classify("demo_images/plastic_bottle.jpg"))
# ➜ plastic : 0.97, metal: 0.02, paper: 0.01 ...
```

## 9&nbsp;| Known Limitations & Next Steps
* Confusions between similar‑looking plastic vs glass items  
* Explore vision transformers & self‑supervised pre‑training  
* Deploy as a mobile TensorFlow‑Lite app for on‑device sorting

## 10&nbsp;| License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 11&nbsp;| Citation
If you use this work in research, please cite the original DSCI 552 assignment prompt.

---

*Project maintained by **Nakshatra Pawar** — “Dream big. Start small. Act now.”*
