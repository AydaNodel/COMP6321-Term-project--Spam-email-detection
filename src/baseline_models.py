"""
baseline spam classification model
trains naive bayes and logistic regression classifiers on tf–idf features
and saves the trained models, vectorizer, and metrics to /src/models
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# set base paths
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "src", "data", "cleaned_spam_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "src", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# load dataset
print("loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# create tf–idf features
print("creating tf–idf features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)
X = vectorizer.fit_transform(df["text_combined"])
y = df["label"]

# split dataset
print("splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"train: {X_train.shape[0]} | test: {X_test.shape[0]}")

# train naive bayes
print("training naive bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# train logistic regression
print("training logistic regression...")
lr_model = LogisticRegression(max_iter=500, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# evaluation function
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    print(f"\n{name} results")
    print(f"accuracy:  {acc:.3f}")
    print(f"precision: {prec:.3f}")
    print(f"recall:    {rec:.3f}")
    print(f"f1-score:  {f1:.3f}")
    return acc, prec, rec, f1

# evaluate models
nb_metrics = evaluate_model("naive bayes", y_test, nb_preds)
lr_metrics = evaluate_model("logistic regression", y_test, lr_preds)

# confusion matrix plotting
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title(title)
    plt.show()

# plot matrices
plot_confusion_matrix(y_test, nb_preds, "confusion matrix – naive bayes")
plot_confusion_matrix(y_test, lr_preds, "confusion matrix – logistic regression")

# save models and vectorizer
print("saving models and vectorizer...")
nb_path = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
lr_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

joblib.dump(nb_model, nb_path)
joblib.dump(lr_model, lr_path)
joblib.dump(vectorizer, vec_path)
print("models saved to:", MODEL_DIR)

# save metrics
results = pd.DataFrame({
    "model": ["naive bayes", "logistic regression"],
    "accuracy": [nb_metrics[0], lr_metrics[0]],
    "precision": [nb_metrics[1], lr_metrics[1]],
    "recall": [nb_metrics[2], lr_metrics[2]],
    "f1-score": [nb_metrics[3], lr_metrics[3]]
})

results_path = os.path.join(MODEL_DIR, "baseline_results.csv")
results.to_csv(results_path, index=False)
print("metrics saved to:", results_path)

print("baseline training complete.")
