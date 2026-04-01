import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def detect_problem_type(y):
    if y.dtype == 'object' or len(y.unique()) < 10:
        return "classification"
    else:
        return "regression"


import pandas as pd

def preprocess_data(df, target_column):
    df = df.dropna()

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # ✅ Convert all categorical columns properly
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if classification
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    return X, y

def train_models(X, y, problem_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    results = {}

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            results[name] = (model, score)

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            results[name] = (model, score)

    # Select best model
    best_model_name = max(results, key=lambda x: results[x][1])
    best_model, best_score = results[best_model_name]

    return best_model_name, best_model, best_score


def run_pipeline(file_path, target_column):
    df = pd.read_csv(file_path)

    X, y = preprocess_data(df, target_column)
    problem_type = detect_problem_type(y)

    best_model_name, model, score = train_models(X, y, problem_type)

    predictions = model.predict(X[:5])  # sample predictions

    return {
        "problem_type": problem_type,
        "best_model": best_model_name,
        "score": round(score, 4),
        "predictions": predictions.tolist()
    }