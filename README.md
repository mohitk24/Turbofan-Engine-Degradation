# Remaining Useful Life Estimation for Turbofan Engines
### 24-788 Introduction to Deep Learning — Course Mini-Project
**Mohit Karangutkar | Carnegie Mellon University | mkarangu@andrew.cmu.edu**

---

## Project Overview

This project compares three deep learning architectures for **Remaining Useful Life (RUL)
estimation** on the NASA C-MAPSS turbofan engine degradation dataset:

| Model | Type | Reference |
|---|---|---|
| **LSTM** | Baseline — 2-layer stacked LSTM | Course material |
| **Deep CNN** | Variant 1 — 1-D convolutional network | Li et al., RESS 2018 |
| **Embed-RUL** | Variant 2 — GRU Encoder-Decoder embeddings | Gugulothu et al., SIGKDD 2017 |

**Key results on FD001:**

| Model | RMSE (lower is better) | PHM Score (lower is better) |
|---|---|---|
| LSTM (Baseline) | 15.04 | 345.27 |
| Deep CNN | 14.82 | **289.80** |
| Embed-RUL | **14.38** | 345.04 |

The Deep CNN achieves the best PHM Score (16.1% improvement over LSTM), while Embed-RUL achieves the best RMSE. This divergence reveals that minimizing symmetric error does not guarantee conservative prediction behavior — a critical distinction for safety-critical aerospace applications.

```

---

## Environment Setup

### Option A — Google Colab (Recommended)

1. Open `IDL_Project.ipynb` or `reproduce_results.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU -> T4** via Runtime -> Change runtime type
3. All required libraries are pre-installed in Colab. Verify with:

```python
import torch, numpy, pandas, sklearn, matplotlib
print(torch.__version__)           # tested with 2.x
print(torch.cuda.is_available())   # should print True on T4
```

### Option B — Local Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/mohit_k24/cmapss-rul.git
cd cmapss-rul

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib
```

**Verified library versions:**
```
torch==2.1.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
```

---

## Data Download

The NASA C-MAPSS dataset is publicly available from the NASA Prognostics Data Repository.

**Direct download:**
```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

**Manual steps:**
1. Download and unzip the archive
2. Copy all `.txt` files into the `data/` directory of this repo

**Automated download in Colab** — run this cell before anything else:
```python
import urllib.request, zipfile, os

os.makedirs("data", exist_ok=True)
url = ("https://phm-datasets.s3.amazonaws.com/NASA/"
       "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
print("Downloading C-MAPSS dataset...")
urllib.request.urlretrieve(url, "cmapss.zip")
with zipfile.ZipFile("cmapss.zip", "r") as z:
    z.extractall("data/")
print("Done. Files in data/:", os.listdir("data/"))
```

---

## Reproducing Key Results (No Retraining Required)

### Step 1 — Get the checkpoints

The three saved model files (`lstm_fd001.pth`, `dcnn_fd001.pth`, `embed_rul_fd001.pth`) are stored in the `checkpoints/` folder of this repository.

If running in Colab, either:
- Upload the files manually via the file browser, or
- Mount your Google Drive and copy them:

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil, os
os.makedirs("checkpoints", exist_ok=True)
# Adjust the path below to where you saved the .pth files in Drive
for fname in ["lstm_fd001.pth", "dcnn_fd001.pth", "embed_rul_fd001.pth"]:
    shutil.copy(f"/content/drive/MyDrive/cmapss-rul/checkpoints/{fname}",
                f"checkpoints/{fname}")
print("Checkpoints ready.")
```

### Step 2 — Run the reproduce notebook

Open **`reproduce_results.ipynb`** and click **Runtime -> Run all**.

The notebook will:
1. Load the three saved checkpoints
2. Run inference on all 100 FD001 test engines (no training)
3. Print the results table matching Table 1 of the report
4. Generate Figure 1 (Predicted vs. Actual RUL plots for all three models)

**Expected console output:**
```
Loading checkpoints...
  lstm_fd001.pth       OK
  dcnn_fd001.pth       OK
  embed_rul_fd001.pth  OK

=== FD001 Test Set Results ===
Model               RMSE     PHM Score
----------------------------------
LSTM (Baseline)     15.04    345.27
Deep CNN            14.82    289.80   <- best PHM Score
Embed-RUL           14.38    345.04   <- best RMSE
```

**Estimated runtime:** ~30 seconds on CPU, ~10 seconds on GPU.

---

## Running Full Training From Scratch

Open **`IDL_Project.ipynb`** and run cells in order. The notebook is divided into clearly labelled sections:

| Section | Description |
|---|---|
| 1 — Imports | Libraries and device setup |
| 2 — Data Loading | Load all 4 C-MAPSS sub-datasets with pandas |
| 3 — EDA | Sensor correlation plots, trajectory visualizations |
| 4 — Preprocessing | Feature selection, min-max normalization, RUL label construction |
| 5 — Dataset & DataLoaders | Sliding window dataset class, train/test loaders |
| 6 — LSTM Model | Baseline architecture definition |
| 7 — Deep CNN Model | Li et al. (2018) architecture definition |
| 8 — Embed-RUL Model | Gugulothu et al. (2017) GRU Encoder-Decoder definition |
| 9 — Training Loop | Shared training function (Adam, StepLR, gradient clipping) |
| 10 — Train All Models on FD001 | 50 epochs each, prints loss every 10 epochs |
| 11 — Evaluate on FD001 | Table 1 results + Figure 1 prediction plots |
| 12 — Cross-Dataset (FD001-FD004) | Fresh models per dataset, generates Tables 2 and Figures 2-3 |
| 13 — Training Dynamics | Figure 4 — training loss curves |
| 14 — Save Checkpoints | Writes `checkpoints/*.pth` |

**Estimated runtime on Colab T4 GPU:**
- FD001 only (sections 1-11, 14): ~5 minutes
- Full cross-dataset (sections 1-14): ~18-22 minutes

---

## Hyperparameter Reference

| Parameter | Value | Source |
|---|---|---|
| Input window size | 30 cycles | Li et al. (2018) |
| RUL cap (R_early) | 125 cycles | Li et al. (2018) |
| Selected sensors | 14 (s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21) | Li et al. (2018) |
| Normalization | Min-max to [-1, 1] per sensor | Li et al. (2018) |
| Batch size | 256 | — |
| Optimizer | Adam, lr = 1e-3 | — |
| LR schedule | StepLR, gamma=0.5, step=20 epochs | — |
| Gradient clipping | max_norm = 1.0 | — |
| Training epochs | 50 | Budget constraint (paper uses 250) |
| **LSTM:** hidden size | 64, 2 layers, dropout=0.3 | — |
| **Deep CNN:** filters | 10 filters, kernel (10x1), 4+1 layers, dropout=0.5 | Li et al. (2018) |
| **Embed-RUL:** GRU hidden | 55 units, 1 layer | Gugulothu et al. (2017) |
| **Embed-RUL:** recon weight | 0.3 | Gugulothu et al. (2017) |

---

## References

1. Li, X., Ding, Q., & Sun, J.-Q. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety, 172*, 1-11. https://doi.org/10.1016/j.ress.2017.11.021

2. Gugulothu, N., TV, V., Malhotra, P., Vig, L., Agarwal, P., & Shroff, G. (2017). Predicting remaining useful life using time series embeddings based on recurrent neural networks. *2nd ML for PHM Workshop, SIGKDD*. https://arxiv.org/abs/1709.01073

3. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08)*.
