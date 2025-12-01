🧠 Multimodal Sensor Fusion for Mental Workload & Frustration Prediction

This repository contains the complete implementation of a multimodal behavioural-state prediction system that estimates mental workload (regression) and frustration levels (classification) using EEG signals, IoT behavioural data, radar micro-motion patterns, and NASA-TLX subjective scores.

The project integrates deep learning and classical machine-learning models and introduces a unified preprocessing and multimodal fusion pipeline.
This work was developed as part of an academic research study at The NorthCap University, Gurugram.

🚀 Project Overview

Human cognitive states such as workload and frustration are critical in high-demand environments like aviation, healthcare, human–computer interaction, and autonomous systems. Traditional methods such as self-reports often fail to capture real-time mental fluctuations.

This project proposes a multimodal sensor-fusion framework that integrates:

- Physiological signals (EEG)

- Behavioural signals (IoT sensors)

- Micro-motion patterns (Radar)

- Self-reported workload (NASA-TLX)

Two separate modelling pipelines are created:

- Deep Learning Models – CNN, LSTM, GRU, MultiKernel CNN, Residual CNN

- Classical Machine Learning Models – XGBoost, Random Forest, SVR, SVM

A transfer-learning inspired stacking mechanism is used where frustration predictions are passed as auxiliary features for workload regression.

🗂️ Repository Structure
📦 Multimodal-Sensor-Fusion
│
├── data/
│   ├── EEG/
│   ├── IoT/
│   ├── Radar/
│   └── NASA_TLX/
│
├── preprocessing/
│   └── preprocess_pipeline.py
│
├── models/
│   ├── cnn.py
│   ├── lstm.py
│   ├── gru.py
│   ├── multi_kernel_cnn.py
│   ├── residual_cnn.py
│   └── classical_ml.py
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── Results_Analysis.ipynb
│
├── results/
│   ├── regression_results.csv
│   ├── classification_results.csv
│   └── evaluation_plots/
│
├── README.md
└── requirements.txt


(If your project structure is different, I can update this.)

🧩 Features
✔️ Multimodal Sensor Fusion

Fuses EEG, IoT behavioural signals, radar micro-motion, and NASA-TLX scores.

✔️ End-to-End Processing

Includes merging, cleaning, alignment, encoding, scaling, and tensor conversion for DL.

✔️ Two Complete Modeling Pipelines

Deep Learning: CNN, LSTM, GRU, Residual CNN, MultiKernel CNN

Machine Learning: XGBoost, SVM, SVR, Random Forest

✔️ Stacking/Transfer Learning

Frustration predictions are used to enhance workload regression.

✔️ Comprehensive EDA

Correlation maps, scatter plots, boxplots, and merged dataset visualizations.

✔️ Model Evaluation

Regression → RMSE, MAE, R²
Classification → Accuracy, Precision, Recall, F1

🧪 Methodology
1. Data Integration

Each modality is preprocessed individually, synchronized using timestamps, and merged into a unified dataset of 210 samples × 14 features.

2. Preprocessing

Median imputation

Standard scaling

One-hot encoding

Tensor generation for DL models

Feature engineering and aggregation

3. Deep Learning Models

Residual CNN achieved the best workload regression performance:
RMSE = 4.581, MAE = 3.481, R² = 0.965

4. Machine Learning Models

SVM, XGBoost, ANN, and MultiKernel CNN achieved the best frustration classification:
Accuracy = 95.24%, F1 = 0.937

5. Evaluation & Visualization

All results are included in the results/ folder.

📊 Results Summary
Workload Regression (Best Model: Residual CNN)

R²: 0.965

RMSE: 4.581

MAE: 3.481

Frustration Classification (Top Models: SVM, XGBoost, ANN, MultiKernel CNN)

Accuracy: 0.952

Precision: 1.00

Recall: 0.882

F1-Score: 0.937

A strong positive correlation was confirmed between workload and frustration.

🛠️ Tech Stack

Python 3.10

TensorFlow / Keras

Scikit-learn

XGBoost

NumPy / Pandas

Matplotlib / Seaborn

Jupyter Notebook

📥 Installation
git clone https://github.com/YOUR_USERNAME/Multimodal-Sensor-Fusion.git
cd Multimodal-Sensor-Fusion
pip install -r requirements.txt

▶️ How to Run
1. Run preprocessing
python preprocessing/preprocess_pipeline.py

2. Train machine learning models
python models/classical_ml.py

3. Train deep learning models
python models/residual_cnn.py

4. Explore results

Open the notebook:

notebooks/Results_Analysis.ipynb

📚 Research Paper

The complete research paper for this project is available in the repository under:

/Multimodal Sensor Fusion for Mental Workload Prediction Research Paper.docx

🧭 Future Work

Future extensions may include realtime multimodal input, edge-device deployment (smartwatches/phones), addition of ECG/eye-tracking/voice signals, integration of explainable AI techniques such as SHAP, and multimodal transformer architectures for improved cross-modality reasoning.

🤝 Contributors

Mitali Dogra

Vallika Dhawan

Kritika Yadav

The NorthCap University, Gurugram
