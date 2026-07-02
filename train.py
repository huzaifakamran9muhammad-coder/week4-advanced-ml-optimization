# ==============================
# Advanced Machine Learning Project
# Heart Disease Prediction
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
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
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import warnings
warnings.filterwarnings("ignore")
# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/heart.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())
# ==============================
# Data Cleaning
# ==============================

print("\nMissing Values")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates")
print(df.shape)
# ==============================
# Correlation Heatmap
# ==============================

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("reports/correlation_heatmap.png")
plt.close()
# ==============================
# Features and Target
# ==============================

X = df.drop("target", axis=1)
y = df["target"]
# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
# ==============================
# Machine Learning Models
# ==============================

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
# ==============================
# Model Training
# ==============================

results = []

best_model = None
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    print("\n==========================")
    print(name)
    print("==========================")

    print(classification_report(y_test,y_pred))

    print("Accuracy :",accuracy)
    print("Precision:",precision)
    print("Recall   :",recall)
    print("F1 Score :",f1)
    print("ROC AUC  :",roc)

    results.append([name,accuracy,precision,recall,f1,roc])

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model
      # ==============================
# Model Comparison
# ==============================

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

print("\nModel Comparison")
print(results_df)

results_df.to_csv("reports/model_comparison.csv", index=False)
# ==============================
# Accuracy Comparison Chart
# ==============================

plt.figure(figsize=(10,6))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results_df
)

plt.xticks(rotation=20)
plt.title("Model Accuracy Comparison")
plt.tight_layout()

plt.savefig("reports/model_accuracy.png")
plt.close()
# ==============================
# Grid Search
# ==============================

print("\nRunning GridSearchCV...")

param_grid = {
    "n_estimators":[100,200],
    "max_depth":[5,10,None],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2]
}

grid = GridSearchCV(

    RandomForestClassifier(random_state=42),

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(X_train,y_train)

print("\nBest Parameters")

print(grid.best_params_)

print("Best Accuracy:",grid.best_score_)
# ==============================
# Random Search
# ==============================

print("\nRunning RandomizedSearchCV...")

random = RandomizedSearchCV(

    RandomForestClassifier(random_state=42),

    param_distributions=param_grid,

    n_iter=10,

    cv=5,

    random_state=42,

    n_jobs=-1

)

random.fit(X_train,y_train)

print("\nRandom Search Best Parameters")

print(random.best_params_)

print("Best Score:",random.best_score_)
# ==============================
# Tuned Model Evaluation
# ==============================

best_rf = grid.best_estimator_

prediction = best_rf.predict(X_test)

accuracy = accuracy_score(y_test,prediction)

print("\nFinal Tuned Accuracy")

print(accuracy)

print(classification_report(y_test,prediction))
# ==============================
# Confusion Matrix
# ==============================

ConfusionMatrixDisplay.from_estimator(

    best_rf,

    X_test,

    y_test

)

plt.title("Confusion Matrix")

plt.savefig("reports/confusion_matrix.png")

plt.close()
# ==============================
# ROC Curve
# ==============================

RocCurveDisplay.from_estimator(

    best_rf,

    X_test,

    y_test

)

plt.title("ROC Curve")

plt.savefig("reports/roc_curve.png")

plt.close()
# ==============================
# Feature Importance
# ==============================

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
# ==============================
# Cross Validation
# ==============================

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

print("\nCross Validation Scores")

print(scores)

print("Mean Accuracy:",scores.mean())

print("Standard Deviation:",scores.std())
# ==============================
# Save Model
# ==============================

joblib.dump(

    best_rf,

    "models/best_model.pkl"

)

print("\nModel Saved Successfully!")
# ==============================
# Finished
# ==============================

print("\nProject Completed Successfully!")

print("Files Saved:")

print("reports/model_comparison.csv")

print("reports/model_accuracy.png")

print("reports/confusion_matrix.png")

print("reports/roc_curve.png")

print("reports/feature_importance.png")

print("models/best_model.pkl")

print("models/scaler.pkl")
