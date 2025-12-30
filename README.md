# Data-driven prediction of polymer surface adhesion using high-throughput MD and hybrid network models

This repository contains a machine learning model for predicting the **Potential of Mean Force (PMF) profiles** of polymer-surface interactions using enhanced sampling data from Umbrella Sampling (US). The input to the neural network consists of **one-hot encoded polymer-surface representations** along with fractional compositions.


![Forward Model architecture](examples/fig_1.png)
---
## 📌 Project Overview
Understanding polymer–surface adhesion is crucial for designing functional nanomaterials and therapeutic systems.  
Traditionally, **PMF profiles** are computed using **umbrella sampling (US)**, but this is computationally expensive.  

Our approach:
-**Generates polymer and surface configurations** using a **2D and 1D Monte Carlo Ising model** while preserving the overall **fractional composition**.  
  This ensures physical relevance and systematic coverage of the sequence–pattern design space.
- Encodes **polymer sequences** and **surface patterns**
- Uses a **CNN–GRU–Attention hybrid model** to capture both **spatial heterogeneity** and **sequence dependence**.
- Predicts **full PMF profiles** directly, enabling rapid screening of polymer–surface combinations.

<table>
<tr>
<td><img src="examples/nps.gif" alt="snapshot" width="200"></td>
<td><img src="examples/pmf_animation.gif" alt="PMF animation" width="200"></td>
<td><img src="examples/R2.png" alt="R2 plot" width="200"></td>
</tr>
</table>
## 📂 **Repository Structure**

models/ # Neural network architectures (CNN, GRU, Transformer)  
scripts/ # Data processing, analysis, utilities

#Additional scripts
###Run the script with different options to control the preprocessing steps:
```python
python preprocess.py
###This script supports command-line flags to enable or disable specific preprocessing steps:
python prepare_data.py
|  Flag           |  Options |  Default  |  Description                        |
|-----------------|----------|-----------|-------------------------------------|
|--reshape        |   Yes/No |     Yes   | Enable/disable data reshaping       |
|--surface-augment|   Yes/No |     Yes   | Enable/disable surface augmentation |
|--polymer-augment|   Yes/No |     Yes   | Enable/disable polymer augmentation |
```
---

---

## **Installation & Setup**
### **Clone the Repository**
```bash
git clone https://github.com/siba-p/ML_surfacepoly.git
cd ML_surfacepoly

Set Up a Virtual Environment (Recommended)

python -m venv env
source env/bin/activate  

Install Dependencies

pip install -r requirements.txt
```
```python
python -m forward_predict.py \
    --checkpoint_path models/checkpoint/HybridCNN/canonical_forward_model.keras \
    --input_data data/processed/fdX_train.npy \
    --output_data models/predictions.npy \
    --target_data data/processed/fdY_train.npy \
    --NN_type HybridCNN
```
Hyperparameter tuning:
Initially a few hyperparameters are tuned through the script training_hp.py present in the folder ~models.

