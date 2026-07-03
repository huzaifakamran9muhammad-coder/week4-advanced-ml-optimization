# Advanced Machine Learning Optimization Project
# Heart Disease Prediction
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# ============================================================
# Create folders
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("data/heart.csv")

print("="*60)
print("DATASET LOADED SUCCESSFULLY")
print("="*60)

print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates")
print(df.shape)

# ============================================================
# Correlation Heatmap
# ============================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("reports/correlation_heatmap.png")

plt.close()

# ============================================================
# Features and Target
# ============================================================

X = df.drop("target", axis=1)

y = df["target"]

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(

    scaler,

    "models/scaler.pkl"

)

# ============================================================
# Machine Learning Models
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(max_iter=1000),

    "Decision Tree":

        DecisionTreeClassifier(random_state=42),

    "Random Forest":

        RandomForestClassifier(random_state=42),

    "KNN":

        KNeighborsClassifier(),

    "SVM":

        SVC(probability=True, random_state=42)

}

results = []

best_model = None

best_accuracy = 0

print("\n")

print("="*60)

print("TRAINING MODELS")

print("="*60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    if hasattr(model, "predict_proba"):

        roc = roc_auc_score(

            y_test,

            model.predict_proba(X_test)[:,1]

        )

    else:

        roc = roc_auc_score(

            y_test,

            model.decision_function(X_test)

        )

    print(classification_report(

        y_test,

        prediction

    ))

    print("Accuracy :", accuracy)

    print("Precision:", precision)

    print("Recall   :", recall)

    print("F1 Score :", f1)

    print("ROC AUC  :", roc)

    results.append([

        name,

        accuracy,

        precision,

        recall,

        f1,

        roc

    ])

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model
        # ============================================================
# Model Comparison
# ============================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ]

)

print("\n")

print("="*60)

print("MODEL COMPARISON")

print("="*60)

print(results_df)

results_df.to_csv(

    "reports/model_comparison.csv",

    index=False

)

# ============================================================
# Accuracy Comparison Plot
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    data=results_df,

    x="Model",

    y="Accuracy"

)

plt.xticks(rotation=15)

plt.title("Model Accuracy Comparison")

plt.tight_layout()

plt.savefig(

    "reports/model_accuracy.png"

)

plt.close()

# ============================================================
# Hyperparameter Tuning
# ============================================================

print("\n")

print("="*60)

print("GRID SEARCH")

print("="*60)

param_grid={

    "n_estimators":[100,200,300],

    "max_depth":[5,10,20,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4]

}

grid=GridSearchCV(

    RandomForestClassifier(random_state=42),

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(

    X_train,

    y_train

)

print("Best Parameters")

print(grid.best_params_)

print("Best CV Accuracy")

print(grid.best_score_)

# ============================================================
# Random Search
# ============================================================

print("\n")

print("="*60)

print("RANDOM SEARCH")

print("="*60)

random=RandomizedSearchCV(

    RandomForestClassifier(random_state=42),

    param_distributions=param_grid,

    n_iter=10,

    cv=5,

    random_state=42,

    n_jobs=-1

)

random.fit(

    X_train,

    y_train

)

print("Best Parameters")

print(random.best_params_)

print("Best Score")

print(random.best_score_)

# ============================================================
# Tuned Model
# ============================================================

best_rf=grid.best_estimator_

prediction=best_rf.predict(

    X_test

)

print("\n")

print("="*60)

print("FINAL MODEL")

print("="*60)

print(

    classification_report(

        y_test,

        prediction

    )

)

print(

    "Final Accuracy:",

    accuracy_score(

        y_test,

        prediction

    )

)
# ============================================================
# Confusion Matrix
# ============================================================

ConfusionMatrixDisplay.from_estimator(

    best_rf,

    X_test,

    y_test

)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("reports/confusion_matrix.png")

plt.close()

# ============================================================
# ROC Curve
# ============================================================

RocCurveDisplay.from_estimator(

    best_rf,

    X_test,

    y_test

)

plt.title("ROC Curve")

plt.tight_layout()

plt.savefig("reports/roc_curve.png")

plt.close()

# ============================================================
# Feature Importance
# ============================================================

importance = pd.Series(

    best_rf.feature_importances_,

    index=X.columns

)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(

    x=importance.values,

    y=importance.index

)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("reports/feature_importance.png")

plt.close()

# ============================================================
# Cross Validation
# ============================================================

print("\n")
print("="*60)
print("CROSS VALIDATION")
print("="*60)

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

scores = cross_val_score(

    best_rf,

    X,

    y,

    cv=cv,

    scoring="accuracy"

)

print("Scores:", scores)

print("Mean Accuracy:", scores.mean())

print("Standard Deviation:", scores.std())

# ============================================================
# Save Model
# ============================================================

joblib.dump(

    best_rf,

    "models/best_model.pkl"

)

print("\nBest model saved successfully!")

print("Scaler saved successfully!")

# ============================================================
# Finished
# ============================================================

print("\n")
print("="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("\nGenerated Files:")

print("✔ models/best_model.pkl")

print("✔ models/scaler.pkl")

print("✔ reports/model_comparison.csv")

print("✔ reports/model_accuracy.png")

print("✔ reports/correlation_heatmap.png")

print("✔ reports/confusion_matrix.png")

print("✔ reports/roc_curve.png")

print("✔ reports/feature_importance.png")
