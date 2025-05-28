import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")


df = pd.read_csv(" ")
df = df.dropna(subset=['label'])
if df.dtypes[0] == 'object':
    df = df.drop(columns=[df.columns[0]])
df = df.fillna(df.median(numeric_only=True))

X = df.drop(columns=['label'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

knn_params = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}



def tune_model(name, model, params):
    print(f"\n🔍 Tuning {name}...")
    search = RandomizedSearchCV(
        model, params, n_iter=10, scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1, verbose=1, random_state=42
    )
    search.fit(X_train_scaled, y_train)
    return search.best_estimator_

rf_best = tune_model("Random Forest", RandomForestClassifier(random_state=42), rf_params)
knn_best = tune_model("K-Nearest Neighbors", KNeighborsClassifier(), knn_params)


def evaluate_model(name, model, X_test, y_test):
    print(f"\n📊 Evaluation for {name}")
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Scores:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


evaluate_model("Random Forest", rf_best, X_test_scaled, y_test)
evaluate_model("K-Nearest Neighbors", knn_best, X_test_scaled, y_test)
