# 🚀 Fake Account Detection & Risk Analysis Dashboard

## 📌 Project Overview

This project is an **end-to-end Machine Learning system** designed to detect fake social media accounts and analyze their behavioral risk.

Unlike traditional classification systems, this project goes beyond simple prediction by integrating:

* ✅ Supervised Learning (Fake Account Classification)
* ✅ Unsupervised Learning (Behavioral Clustering)
* ✅ Anomaly Detection (Suspicious Activity Identification)
* ✅ Interactive Dashboard (Streamlit UI)

The system is capable of identifying **mass-created fake accounts**, distinguishing them from genuine users based on behavior, metadata, and activity patterns.

---

## 🎯 Objectives

* Detect fake or bot accounts using machine learning models
* Analyze user behavior patterns through clustering
* Identify suspicious or anomalous accounts
* Provide **interpretable risk insights** instead of raw predictions
* Build a **user-friendly dashboard** for real-time account analysis

---

## 📂 Dataset Description

The dataset contains features related to user profile and activity such as:

* Username characteristics (numeric ratio, patterns)
* Full name structure
* Profile metadata (profile picture, external URL, private status)
* Activity metrics (#posts, #followers, #follows)
* Description length

The dataset is already clean, but standard preprocessing and validation practices were applied for robustness.

---

## ⚙️ Project Workflow

### 🔹 1. Data Preprocessing & Feature Engineering

* Handled structured and categorical features
* Created meaningful ratios (e.g., numeric username ratio)
* Applied log transformation for skewed features
* Standardized numerical features for model compatibility

---

### 🔹 2. Exploratory Data Analysis (EDA)

#### 🟢 Supervised EDA (Target-Aware Analysis)

- **Histogram Analysis (with target `fake`)**
  - Identified strong separating features:
    - Numeric-heavy usernames → higher fake probability  
    - Low description length → strong fake indicator  
    - Low post count → suspicious behavior  

- **Correlation Heatmap**
  - Measured relationship between features and target  
  - Helped identify important predictors  

- **Boxplots**
  - Compared feature distributions across fake vs real accounts  
  - Observed spread, outliers, and class separation  

---

#### 🔵 Unsupervised EDA (Target-Free Analysis)

- **Pair Plot (Visual Inspection)**
  - Explored relationships between features  
  - Checked for natural clusters or suspicious points  
  - Confirmed dataset stability (no extreme anomalies visually)  

- **Variance Check (`df.var()`)**
  - Identified low-variance features  
  - Removed features with minimal contribution to clustering  

- **Correlation Analysis**
  - Removed highly correlated features (|corr| > 0.85)  
  - Prevented redundancy and distance distortion in clustering  

- **Skewness Checking (`df.skew()`)**
  - Found highly skewed features (e.g., #followers, #posts, #follows)  
  - Applied log transformation (`np.log1p`) for normalization  

- **Feature Scale Inspection**
  - Observed large differences in feature ranges  
  - Applied StandardScaler to ensure fair contribution  

- **PCA (Principal Component Analysis)**
  - Reduced dimensionality for visualization  
  - Confirmed presence of meaningful clusters  

👉 These steps ensured that clustering and anomaly detection operate on a **well-structured, balanced, and meaningful feature space**.

---

## 🤖 Supervised Machine Learning

Used for **fake account classification**

### Models Implemented:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* XGBoost

### 📊 Model Performance

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 96%      | 96%      |
| KNN                 | 98%      | 98%      |
| Decision Tree       | 97%      | 97%      |
| Random Forest       | 98%      | 98%      |
| XGBoost             | **99%**  | **99%**  |

### ⚖️ Model Selection

Although **XGBoost achieved the best performance**,
**KNN was used in deployment** to better understand:

* Feature scaling effects
* Distance-based learning
* Pipeline behavior

---

## 🔍 Unsupervised Machine Learning

Used for **behavioral analysis**

### Techniques Used:

* KMeans Clustering → user segmentation
* Hierarchical Clustering → pattern validation
* PCA → dimensionality reduction

### 🧠 Cluster Interpretation

| Cluster | Meaning               |
| ------- | --------------------- |
| 0       | Moderate Active Users |
| 1       | Highly Active Users   |
| 2       | Suspicious Users      |

---

## 🚨 Anomaly Detection

* Implemented using **Isolation Forest**
* Detects accounts with unusual or abnormal behavior patterns

---

## 📁 Project Structure
```
Fake_Account_Project/
│
├── app.py # Streamlit main application
│
├── src/ # Backend logic
│ ├── load_models.py
│ ├── prediction.py
│ ├── feature_engineering.py # (if you created compute_features here)
│
├── models/ # Saved ML models
│ ├── knn_pipeline.pkl
│ ├── KMeans.pkl
│ ├── isolation_model.pkl
| ├── log_model.pkl
│ ├── scaler.pkl
│ ├── pca.pkl
│
├── dataset/ # Dataset files
| ├── Instagram_fake_profile_dataset.csv # Main dataset 
│ ├── clustered_accounts.csv # Derived dataset 
│
├── notebooks/ # Jupyter notebooks
│ ├── supervised.ipynb
│ ├── unsupervised.ipynb
│
├── requirements.txt # Project dependencies
├── .gitignore # Files to ignore in Git
├── README.md # Project documentation
```
---

## 📊 Dashboard Features (Streamlit)

* 🏠 Home Page - Project Overview
* 🔍 Account Analyzer (real-time prediction)
* 📈 Behavior Insights (EDA visualizations)
* 📊 Model Performance comparison
* ⚠️ Risk Level & Anomaly Detection
* 🧠 Explainable AI (cluster-based reasoning)

---

## 🌐 Live Demo

🚀 Deployed App: https://fake-account-risk-dashboard.streamlit.app/

---

## 🌍 Real-World Applications

* Social media platforms → detect fake/bot accounts
* Fraud detection systems
* Spam account filtering
* Influencer authenticity verification
* Cybersecurity monitoring

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn
* Streamlit

---

## 🔮 Future Improvements

* Deploy using **Flask / FastAPI (API-based system)**
* Add real-time API integration
* Improve model explainability using SHAP

---

## 👨‍💻 Author

**Jash Solanki**
BTech | AI/ML Enthusiast

---

⭐ If you found this project useful, consider giving it a star!
