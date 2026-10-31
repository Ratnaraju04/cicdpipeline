import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath="iris.csv"):
    """Load dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def train_model(X_train, y_train, random_state=42):
    """Train Random Forest model."""
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    """Evaluate trained model and return accuracy score and report."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report


def run_pipeline(
    data_path="iris.csv",
    scores_path="scores.txt",
    model_path="model.joblib"
):
    """Run full pipeline: load data, train, evaluate, and save scores."""
    df = load_data(data_path)

    # Support different target column names ('species' or 'target')
    target_col = "species" if "species" in df.columns else "target"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = train_model(X_train, y_train)
    acc, report = evaluate_model(clf, X_test, y_test)

    # Save trained model
    joblib.dump(clf, model_path)

    # Save metrics to scores.txt
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Training completed successfully.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Scores written to {scores_path}")
    print(f"Model saved to {model_path}")

    return acc


if __name__ == "__main__":
    run_pipeline()
