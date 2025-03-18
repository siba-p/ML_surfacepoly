# 🔬 Polymer-Surface PMF Prediction using *****

This repository contains a machine learning model for predicting the **Potential of Mean Force (PMF) profiles** of polymer-surface interactions using enhanced sampling data from Umbrella Sampling (US). The input to the neural network consists of **one-hot encoded polymer-surface representations** along with fractional compositions.

---

## 📌 **Project Overview**
In molecular simulations, **PMF profiles**.  Instead of running computationally expensive **umbrella sampling (US)** simulations for each polymer-surface combination, we train a neural network to predict the **full PMF profile** from encoded polymer and surface information.

**Key Features:**
---

## 📂 **Repository Structure**

To be added.....
#scripts
Run the script with different options to control the preprocessing steps:
python preprocess.py
This script supports command-line flags to enable or disable specific preprocessing steps:
python prepare_data.py

--reshape (Enable/disable data reshaping.)
--surface-augment (Enable/disable surface augmentation)
--polymer-augment (Enable/disable polymer augmentation)

---

---

## **Installation & Setup**
### **Clone the Repository**
```bash
git clone https://github.com/yourusername/PolymerSurface-ML.git
cd PolymerSurface-ML

Set Up a Virtual Environment (Recommended)

python -m venv env
source env/bin/activate  

Install Dependencies

pip install -r requirements.txt




Hyperparameters

Modify config.yaml to fine-tune model settings:


👨‍💻 Contributing
Interested in improving the model? 

Open an issue
Submit a pull request
Suggest enhancements

