import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# loading dataset
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "class"]
df = pd.read_csv("processed.cleveland.data", names=column_names)

# replacing missing values with median
for col in df.columns:
    if df[col].isin(['?']).any():
        median = df[df[col] != '?'][col].astype(float).median()
        df[col] = df[col].replace('?', median)
    df[col] = df[col].astype(float)

# x, y
x = df.drop("class", axis=1).values
y = df["class"].values
y = np.where(y > 0, 1, 0)

# features normalization
scaler = StandardScaler()
x = scaler.fit_transform(x)

# models
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "GaussianNB": GaussianNB(),
    "SVM": SVC(kernel='rbf', C=1),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "DecisionTree": DecisionTreeClassifier(max_depth=4),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100)
}

# k-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for name in models:
    model = models[name]
    scores = []
    for train_idx, test_idx in kfold.split(x):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    print(f"{name}: Average Accuracy = {np.mean(scores):.4f}")

# final model
final_model = GaussianNB()
final_model.fit(x, y)

# saving model
with open("mymodel.sav", "wb") as f:
    pickle.dump(final_model, f)

print("model saved successfully!")
