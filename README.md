# MAGIC Gamma Telescope Classification

Machine Learning pipeline for binary classification of high-energy particle events using the MAGIC Gamma Telescope dataset from the UCI Machine Learning Repository.

---

## 📊 Dataset

**Source:**  
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository  
MAGIC Gamma Telescope Dataset  
http://archive.ics.uci.edu/ml

**Dataset Details:**
- 10 continuous numerical features
- Binary classification:
  - `1` → Gamma
  - `0` → Hadron
- 19,020 total samples

---

## 🎯 Problem Statement

Build and evaluate machine learning models to classify whether a detected particle event corresponds to a **gamma signal** or a **hadron background event**.

---

## 🔎 Data Processing Pipeline

- Dataset shuffled before splitting
- Train / Validation / Test split:
  - 60% Train
  - 20% Validation
  - 20% Test
- Feature scaling using `StandardScaler`
- Class imbalance handled using `RandomOverSampler` (applied only to training set)

---

## 🤖 Models Implemented

### 1️⃣ K-Nearest Neighbors (KNN)
- n_neighbors = 5  
- Test Accuracy: **82%**

### 2️⃣ Gaussian Naive Bayes
- Test Accuracy: **72%**

### 3️⃣ Logistic Regression
- Test Accuracy: **77%**

### 4️⃣ Support Vector Machine (SVM)
- Test Accuracy: **87%**

### 5️⃣ Neural Network (TensorFlow / Keras)

Architecture:
- Dense (ReLU)
- Dropout
- Dense (ReLU)
- Dropout
- Output (Sigmoid)

Hyperparameters tuned:
- Hidden units: 16, 32, 64
- Dropout: 0, 0.2
- Learning rate: 0.01, 0.005, 0.001
- Batch size: 32, 64, 128
- Epochs: 100

Best Neural Network Performance:
- Accuracy: **87%**
- Precision (Gamma): 0.89
- Recall (Gamma): 0.93
- F1-score (Gamma): 0.91

---

## 📈 Model Comparison

| Model | Test Accuracy |
|--------|--------------|
| Naive Bayes | 72% |
| Logistic Regression | 77% |
| KNN | 82% |
| SVM | 87% |
| Neural Network | 87% |

SVM and Neural Network achieved the best overall performance.

---

## 🛠 Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- imbalanced-learn
- TensorFlow / Keras

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/Dathwik/ml_models_magic_dataset.git
cd ml_models_magic_dataset
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch notebook:

```bash
jupyter notebook
```

---

## 📂 Repository Structure

```
ml_models_magic_dataset/
│
├── magic04.data
├── fcc_MAGIC.ipynb.ipynb
├── README.md
└── requirements.txt
```

---

## 💡 Key Takeaways

- Feature scaling significantly improves performance of distance-based models.
- Handling class imbalance improves recall on minority class.
- SVM performs strongly on structured tabular data.
- Proper hyperparameter tuning allows neural networks to match classical ML models.
