import json
import io


def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
            },
            "body": "",
        }

    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import accuracy_score, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        body = json.loads(event.get("body") or "{}")
        csv_data = body.get("csv_data", "")
        target_column = body.get("target", "").strip()

        if not csv_data or not target_column:
            return _error(400, "Missing csv_data or target field")

        df = pd.read_csv(io.StringIO(csv_data))

        if target_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            return _error(
                400,
                f'Column "{target_column}" not found. Available columns: {available}',
            )

        df = df.dropna()
        if len(df) < 10:
            return _error(400, "Dataset too small after removing missing values (need at least 10 rows)")

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X = pd.get_dummies(X, drop_first=True)

        if y.dtype == "object":
            y = pd.factorize(y)[0]

        unique_values = len(set(y))
        problem_type = "classification" if (y.dtype == object or unique_values < 10) else "regression"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}
        if problem_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = (model, float(accuracy_score(y_test, preds)))
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = (model, float(r2_score(y_test, preds)))

        best_name = max(results, key=lambda x: results[x][1])
        best_model, best_score = results[best_name]

        sample = X.head(5)
        raw_preds = best_model.predict(sample).tolist()
        predictions = [
            round(float(p), 4) if isinstance(p, float) else int(p)
            for p in raw_preds
        ]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "problem_type": problem_type,
                    "best_model": best_name,
                    "score": round(best_score, 4),
                    "predictions": predictions,
                }
            ),
        }

    except Exception as exc:
        return _error(500, str(exc))


def _error(status, message):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }
