"""
Student Performance Analytics
Complete source code for data loading, preprocessing, EDA, Random Forest
classification, evaluation, prediction, and dashboard-ready analytics.

Dataset:
UCI Student Performance Dataset
https://archive.ics.uci.edu/dataset/320/student
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def load_data():
    """Load the UCI Student Performance dataset."""
    student_performance = fetch_ucirepo(id=320)
    X = student_performance.data.features.copy()
    y = student_performance.data.targets.copy()

    df = X.copy()
    if "G3" in y.columns:
        df["G3"] = y["G3"]
    elif "G3" not in df.columns:
        raise ValueError("G3 target column was not found.")

    return df


def clean_data(df):
    """Clean duplicate rows and ensure grade columns are numeric."""
    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["G1", "G2", "G3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["G3"]).reset_index(drop=True)


def performance_category(grade):
    """Convert a 0-20 final grade into Low/Average/High."""
    if grade < 10:
        return "Low"
    if grade < 15:
        return "Average"
    return "High"


def create_target(df):
    """Create the target performance category from G3."""
    df = df.copy()
    df["Performance_Category"] = df["G3"].apply(performance_category)
    return df


def prepare_model_data(df):
    """Prepare features, target, and preprocessing definitions."""
    X = df.drop(columns=["G3", "Performance_Category"])
    y = df["Performance_Category"]

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


def build_model(preprocessor):
    """Build the preprocessing + Random Forest pipeline."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(pipeline, X_test, y_test):
    """Print and return model evaluation metrics."""
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=["Low", "Average", "High"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Low", "Average", "High"],
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Average", "High"],
    )
    disp.plot()
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.show()

    return y_pred, accuracy


def run_eda(df):
    """Generate basic project visualizations."""
    print(df[["G1", "G2", "G3", "absences", "studytime", "failures"]].describe())

    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="Performance_Category",
        order=["Low", "Average", "High"],
    )
    plt.title("Student Performance Category Distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="absences",
        y="G3",
        hue="Performance_Category",
    )
    plt.title("Absences vs Final Grade")
    plt.tight_layout()
    plt.show()


def feature_importance(pipeline):
    """Return and plot the most important transformed features."""
    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importances = pipeline.named_steps[
        "model"
    ].feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=False)

    print(importance_df.head(15))

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=importance_df.head(15),
        x="Importance",
        y="Feature",
    )
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.show()

    return importance_df


def dashboard_summary(df):
    """Create summary values for a dashboard."""
    return {
        "Total Students": len(df),
        "Average Final Grade": round(df["G3"].mean(), 2),
        "Average Absences": round(df["absences"].mean(), 2),
        "High Performers": int(
            (df["Performance_Category"] == "High").sum()
        ),
        "Average Performers": int(
            (df["Performance_Category"] == "Average").sum()
        ),
        "Low Performers": int(
            (df["Performance_Category"] == "Low").sum()
        ),
    }


def main():
    """Run the complete Student Performance Analytics workflow."""
    df = load_data()
    print("Original shape:", df.shape)

    df = clean_data(df)
    print("Cleaned shape:", df.shape)

    df = create_target(df)
    print("\nPerformance categories:")
    print(df["Performance_Category"].value_counts())

    run_eda(df)

    X_train, X_test, y_train, y_test, preprocessor = prepare_model_data(df)
    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    pipeline = build_model(preprocessor)
    pipeline.fit(X_train, y_train)

    y_pred, accuracy = evaluate_model(pipeline, X_test, y_test)

    print("\nDashboard summary:")
    print(dashboard_summary(df))

    importance_df = feature_importance(pipeline)

    print("\nExample prediction:")
    example_prediction = pipeline.predict(X_test.iloc[[0]])[0]
    print("Predicted performance:", example_prediction)

    return {
        "data": df,
        "model": pipeline,
        "predictions": y_pred,
        "accuracy": accuracy,
        "feature_importance": importance_df,
    }


if __name__ == "__main__":
    results = main()
