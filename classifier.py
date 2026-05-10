"""Spending category classifier — same pipeline as Task ML-1 notebook."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _build_models() -> dict[str, object]:
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


@dataclass
class TrainedClassifier:
    """Best model from the notebook comparison, refit on all rows for serving."""

    name: str
    holdout_accuracy: float
    model_scores: dict[str, float]
    pipeline: Pipeline
    categories: list[str]
    n_samples: int


def train_classifier(csv_path: str | Path) -> TrainedClassifier:
    path = Path(csv_path)
    df = pd.read_csv(path)
    if "description" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must have 'description' and 'category' columns")

    df = df.dropna(subset=["description", "category"])
    df["cleaned"] = df["description"].astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["cleaned"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results: dict[str, tuple[object, float]] = {}
    for name, estimator in _build_models().items():
        model = estimator
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = (model, acc)

    best_name = max(results, key=lambda k: results[k][1])
    best_acc = results[best_name][1]
    model_scores = {name: float(acc) for name, (_, acc) in results.items()}

    # Refit chosen estimator on full data for deployment predictions
    full_estimator = _build_models()[best_name]
    full_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=500,
                sublinear_tf=True,
            )),
            ("clf", full_estimator),
        ]
    )
    full_pipeline.fit(df["cleaned"], y)

    cats = sorted(y.unique().tolist())
    return TrainedClassifier(
        name=best_name,
        holdout_accuracy=float(best_acc),
        model_scores=model_scores,
        pipeline=full_pipeline,
        categories=cats,
        n_samples=len(df),
    )


def predict_category(
    trained: TrainedClassifier, description: str
) -> tuple[str, dict[str, float] | None]:
    cleaned = clean_text(description)
    pred = trained.pipeline.predict([cleaned])[0]
    proba = None
    if hasattr(trained.pipeline, "predict_proba"):
        probs = trained.pipeline.predict_proba([cleaned])[0]
        classes = trained.pipeline.classes_
        proba = {str(c): float(p) for c, p in zip(classes, probs)}
    return str(pred), proba


def predict_batch(
    trained: TrainedClassifier, descriptions: list[str]
) -> list[str]:
    """Return predicted category for each raw description."""
    cleaned = [clean_text(d) for d in descriptions]
    preds = trained.pipeline.predict(cleaned)
    return [str(p) for p in preds]
