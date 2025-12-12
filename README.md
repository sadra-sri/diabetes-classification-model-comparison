# Diabetes Classification – Machine Learning Model Comparison

This project focuses on building and comparing multiple supervised classification models
on a diabetes dataset. The goal is to evaluate how different algorithms perform on the same
preprocessed data and to understand their relative strengths and weaknesses.

## Objective

- Predict diabetes outcome based on medical features
- Compare multiple classification algorithms under a consistent preprocessing pipeline
- Evaluate model performance using standard classification metrics

## Dataset

The dataset contains medical diagnostic measurements commonly used for diabetes prediction.
Typical features include glucose level, blood pressure, BMI, age, insulin, and related indicators.

(The dataset can be obtained from public sources such as Kaggle or the UCI Machine Learning Repository.)

## Models Implemented

### 1. K-Nearest Neighbors (KNN)
- Distance-based classification algorithm
- Sensitive to feature scaling
- Useful for understanding neighborhood-based decision boundaries

Notebook: `01_knn.ipynb`

### 2. Naive Bayes
- Probabilistic classifier based on Bayes’ theorem
- Assumes conditional independence between features
- Serves as a fast and simple baseline model

Notebook: `02_naive_bayes.ipynb`

### 3. Decision Tree and Random Forest
- Tree-based models capable of capturing non-linear relationships
- Decision Tree provides interpretability
- Random Forest improves generalization by reducing overfitting

Notebook: `03_decision_tree_random_forest.ipynb`

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Key Takeaways

- Model performance varies significantly across algorithms even on the same dataset
- Tree-based models generally provide stronger generalization compared to simpler classifiers
- Simpler models such as Naive Bayes can still serve as effective baselines

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn

## How to Run

1. Clone the repository
2. Open the notebooks in Jupyter Notebook or VS Code
3. Run any notebook independently to train and evaluate the corresponding model

## Notes

This project is intended as a focused comparison of supervised classification models
and emphasizes evaluation and reasoning rather than extensive exploratory analysis.
