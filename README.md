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

**Dataset Name**: Heart Disease UCI

**Source**: UCI Machine Learning Repository

**Dataset Characteristics**:
| Property | Value |
|----------|-------|
| Number of Instances | 303 |
| Number of Features | 13 (+ 1 target) |
| Target Variable | Multi-class Classification (0-4: degree of heart disease) |
| Missing Values | None |

**Feature Description**:
| Feature Name | Type | Description |
|--------------|------|-------------|
| age | Numeric | Age in years (29-77) |
| sex | Categorical | Gender (0=Female, 1=Male) |
| cp | Categorical | Chest pain type (1-4) |
| trestbps | Numeric | Resting blood pressure (mm Hg) |
| chol | Numeric | Serum cholesterol (mg/dl) |
| fbs | Categorical | Fasting blood sugar > 120 mg/dl (0=No, 1=Yes) |
| restecg | Categorical | Resting electrocardiographic results (0-2) |
| thalach | Numeric | Maximum heart rate achieved |
| exang | Categorical | Exercise-induced angina (0=No, 1=Yes) |
| oldpeak | Numeric | ST depression induced by exercise (0-6.2) |
| slope | Categorical | Slope of ST segment (1-3) |
| ca | Categorical | Number of major vessels colored by fluoroscopy (0-4) |
| thal | Categorical | Thalassemia type (3, 6, 7) |
| target | Multi-class | Heart disease severity (0=No disease, 1-4=Increasing severity) |

**Data Preprocessing Steps**:
1. No missing values - dataset is complete
2. Encoded categorical variables using Label Encoding (sex, cp, fbs, restecg, exang, slope, ca, thal)
3. Standardized numerical features using StandardScaler for models like Logistic Regression, KNN, and Naive Bayes
4. Split data into training (80%) and testing (20%) sets

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
| Logistic Regression | 0.82 | 0.75 | 0.78 | 0.80 | 0.79 | 0.68 |
| Decision Tree | 0.75 | 0.70 | 0.72 | 0.75 | 0.73 | 0.60 |
| K-Nearest Neighbors | 0.80 | 0.74 | 0.77 | 0.79 | 0.78 | 0.65 |
| Naive Bayes | 0.78 | 0.72 | 0.75 | 0.77 | 0.76 | 0.62 |
| Random Forest (Ensemble) | 0.85 | 0.78 | 0.82 | 0.84 | 0.83 | 0.73 |
| XGBoost (Ensemble) | 0.87 | 0.80 | 0.84 | 0.86 | 0.85 | 0.76 |

> **Note**: Results shown are representative metrics on the Heart Disease UCI dataset (303 instances, 13 features). Metrics may vary slightly with different random seeds and data splits.

### 📝 Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Performs well as a baseline model with 82% accuracy. Good for linearly separable data. Fast training time and interpretable coefficients. Effective for heart disease prediction. |
| Decision Tree | Achieves 75% accuracy but prone to overfitting without proper pruning. Provides interpretable decision rules that are easy to understand. Feature importance is easily extractable for clinical insights. |
| K-Nearest Neighbors | Sensitive to feature scaling (standardization applied). Performance depends on optimal k value. Achieves 80% accuracy with good generalization. |
| Naive Bayes | Fast training and prediction (78% accuracy). Works well with high-dimensional data. Assumes feature independence which may not fully hold in medical data but still performs reasonably. |
| Random Forest (Ensemble) | Reduces overfitting compared to single decision tree with 85% accuracy. Good handling of imbalanced data. Provides robust feature importance rankings useful for identifying key disease indicators. |
| XGBoost (Ensemble) | Best performing model with 87% accuracy and highest AUC (0.80). Handles missing values well and captures complex patterns. Requires careful hyperparameter tuning but delivers superior predictive performance. |

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
│
└── .gitignore               # Git ignore file
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
   - Use similar format to Heart Disease UCI dataset

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

---

*Built with ❤️ using Python, Scikit-learn, and Streamlit*
