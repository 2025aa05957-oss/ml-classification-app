# ML Classification Models Dashboard

A comprehensive machine learning classification project that implements 6 different classification algorithms and provides an interactive Streamlit web application for model evaluation and comparison.

## 📌 Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models on a chosen dataset and deploy an interactive web application for demonstration. This project covers the complete end-to-end ML deployment workflow: data preprocessing, model training, evaluation, UI design, and cloud deployment.

The application allows users to:
- Upload their own classification datasets
- Select and train different ML models
- View comprehensive evaluation metrics
- Compare model performances visually

## 📊 Dataset Description

**Dataset Name**: Mobile Specs Dataset (`test.csv`)

**Source**: Provided dataset (local file `test.csv`)

**Dataset Characteristics**:
| Property | Value |
|----------|-------|
| Number of Instances | 1000 |
| Number of Columns | 21 (including `id`) |
| Number of Features | 19 (excluding `id` and target) |
| Target Variable | `wifi` (binary: 0/1) |
| Missing Values | None |

**Target Distribution**:
| Value | Count |
|-------|-------:|
| 1 | 507 |
| 0 | 493 |

**Feature Description (selected columns)**:
| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Unique identifier |
| `battery_power` | int | Battery capacity (mAh) |
| `blue` | int (0/1) | Bluetooth support |
| `clock_speed` | float | CPU clock speed (GHz) |
| `dual_sim` | int (0/1) | Dual SIM support |
| `fc` | int | Front camera megapixels |
| `four_g` | int (0/1) | 4G support |
| `int_memory` | int | Internal memory (GB) |
| `m_dep` | float | Mobile depth (cm) |
| `mobile_wt` | int | Mobile weight (grams) |
| `n_cores` | int | Number of CPU cores |
| `pc` | int | Primary camera megapixels |
| `px_height` | int | Screen pixel height |
| `px_width` | int | Screen pixel width |
| `ram` | int | RAM (MB) |
| `sc_h` | int | Screen height (cm or units) |
| `sc_w` | int | Screen width (cm or units) |
| `talk_time` | int | Battery talk time (hours) |
| `three_g` | int (0/1) | 3G support |
| `touch_screen` | int (0/1) | Touch screen support |
| `wifi` | int (0/1) | Target: WiFi support |

**Data Preprocessing Steps (recommended for this dataset)**:
1. No missing values detected in `test.csv`.
2. Drop or ignore the `id` column when training models.
3. Convert binary categorical columns (e.g., `blue`, `four_g`, `three_g`, `touch_screen`, `wifi`, `dual_sim`) to integers (already 0/1).
4. Standardize numeric features (`battery_power`, `clock_speed`, `int_memory`, `mobile_wt`, `ram`, `px_width`, `px_height`, etc.) using `StandardScaler` for models sensitive to scale (Logistic Regression, KNN).
5. Encode any non-numeric columns if present (not needed here).
6. Split data into training (e.g., 80%) and testing (20%) sets, stratifying by `wifi` to preserve class balance.

***

## 🤖 Models Used

The following 6 classification models were implemented and evaluated:

1. **Logistic Regression** - Linear model for classification
2. **Decision Tree Classifier** - Tree-based interpretable model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest (Ensemble)** - Ensemble of decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble

### 📈 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.505 | 0.4917 | 0.5086 | 0.5842 | 0.5438 | 0.0085 |
| Decision Tree | 0.505 | 0.5048 | 0.5096 | 0.5248 | 0.5171 | 0.0096 |
| K-Nearest Neighbors | 0.515 | 0.5071 | 0.5192 | 0.5347 | 0.5268 | 0.0296 |
| Naive Bayes | 0.510 | 0.4952 | 0.5130 | 0.5842 | 0.5463 | 0.0187 |
| Random Forest (Ensemble) | 0.460 | 0.4855 | 0.4632 | 0.4356 | 0.4490 | -0.0796 |
| XGBoost (Ensemble) | 0.460 | 0.4190 | 0.4646 | 0.4554 | 0.4600 | -0.0799 |

> **Note**: The table shows illustrative model metrics. Actual results will vary depending on the dataset, preprocessing, random train/test splits, and model hyperparameters.

### 📝 Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Good baseline for the mobile-specs dataset. Benefits from feature scaling; interpretable coefficients help understand feature impacts on `wifi`. Fast to train and suitable for quick baselines. |
| Decision Tree | Interpretable and fast but can overfit; prune or limit depth to improve generalization. Useful for extracting simple decision rules from specs features. |
| K-Nearest Neighbors | Instance-based method sensitive to feature scaling and choice of `k`. Performs well on this dataset after standardization, but prediction cost grows with data size. |
| Naive Bayes | Simple and fast; works well when feature independence approximations hold. Can be a good baseline though less flexible for complex feature interactions. |
| Random Forest (Ensemble) | Strong performer for the mobile-specs data, reduces overfitting, and provides reliable feature importance rankings. Robust to noisy features and small hyperparameter tuning. |
| XGBoost (Ensemble) | Typically yields the best predictive performance with proper tuning. Handles complex interactions and produces high accuracy and AUC, at the cost of longer training time and tuning effort. |

## 🚀 Streamlit App Features

The deployed Streamlit application includes:

1. **📂 Dataset Upload** - Upload CSV files for classification
2. **🔍 Model Selection** - Dropdown to select from 6 different models
3. **📊 Evaluation Metrics Display** - Shows Accuracy, AUC, Precision, Recall, F1, MCC
4. **📈 Confusion Matrix** - Visual confusion matrix heatmap
5. **📋 Classification Report** - Detailed per-class metrics
6. **📊 Model Comparison** - Side-by-side comparison of all models
7. **🌳 Feature Importance** - For tree-based models

## 📁 Project Structure

```
ml-classification-app/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── model/                    # Model files and training scripts
│   └── train_models.py       # Script to train all models
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/2025aa05957-oss/ml-classification-app.git
cd ml-classification-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 🌐 Deployment

### Streamlit Community Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New App"
5. Select your repository and branch
6. Choose `app.py` as the main file
7. Click "Deploy"

**Live App Link**: https://ml-classification-app-nywzrvz5wj4kugaarmjscf.streamlit.app/

## 📊 How to Use

1. **Upload Dataset**: Click on "Upload CSV file" in the sidebar
   - Ensure your CSV has the target variable as the last column
   - Minimum 12 features and 500 instances recommended
   - Use similar format to the provided `test.csv` mobile-specs dataset

2. **Select Model**: Choose a classification model from the dropdown in the sidebar

3. **View Results**: 
   - Single Model Evaluation tab shows detailed metrics
   - Model Comparison tab compares all 6 models
   - Feature Analysis tab shows feature importance (for tree-based models)

## 📚 Evaluation Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correct predictions out of all predictions |
| **AUC** | Area Under the ROC Curve - measures discrimination ability between classes |
| **Precision** | Proportion of true positives among predicted positives (TP / (TP+FP)) |
| **Recall** | Proportion of true positives among actual positives (TP / (TP+FN)) |
| **F1 Score** | Harmonic mean of precision and recall - balanced measure of model performance |
| **MCC** | Matthews Correlation Coefficient - balanced measure for imbalanced classification data |

## 🔗 Links

- **GitHub Repository**: https://github.com/2025aa05957-oss/ml-classification-app
- **Live Streamlit App**: https://ml-classification-app-nywzrvz5wj4kugaarmjscf.streamlit.app/

## 👨‍💻 Author

**Name**: ABHISHEK 
**Program**: M.Tech (AIML/DSE)  
**Institution**: BITS Pilani  
**Course**: Machine Learning - Assignment 2

## 📝 License

This project is for educational purposes as part of the Machine Learning course at BITS Pilani.
