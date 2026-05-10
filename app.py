"""
Streamlit app: Customer Spending Category Classifier (ML-1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import clean_text, predict_batch, predict_category, train_classifier

DATA_PATH = Path(__file__).resolve().parent / "data.csv"


@st.cache_resource(show_spinner="Training models on your dataset…")
def load_classifier() -> TrainedClassifier:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_PATH.name}. Place data.csv next to app.py "
            "(columns: description, category)."
        )
    return train_classifier(DATA_PATH)


def _render_prediction(trained: TrainedClassifier, description: str) -> None:
    cleaned = clean_text(description)
    category, proba = predict_category(trained, description)
    st.success(f"**Predicted category:** `{category}`")
    with st.expander("Preprocessed text (as in the notebook)", expanded=False):
        st.code(cleaned or "(empty after cleaning)", language=None)

    if proba:
        st.subheader("Class probabilities")
        prob_df = (
            pd.DataFrame([{"Category": k, "Probability": v} for k, v in proba.items()])
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        prob_df["Probability"] = prob_df["Probability"].round(4)
        st.bar_chart(prob_df.set_index("Category")["Probability"])
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
    else:
        st.info("This model does not expose class probabilities.")


def main() -> None:
    st.set_page_config(
        page_title="Spending Classifier",
        page_icon="💳",
        layout="wide",
    )

    if "desc_input" not in st.session_state:
        st.session_state.desc_input = ""

    st.title("Customer spending category classifier")
    st.caption(
        "Paste a bank or card transaction description — the app predicts one of eight "
        "categories using the same ML pipeline as **Task ML-1** (TF‑IDF + Naive Bayes / "
        "Logistic Regression / Random Forest; best model is selected on a holdout split)."
    )

    try:
        trained = load_classifier()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Could not load or train the classifier: {e}")
        return

    with st.sidebar:
        st.subheader("Model")
        st.metric("Selected model", trained.name)
        st.metric("Holdout accuracy", f"{trained.holdout_accuracy * 100:.1f}%")
        st.metric("Training examples", f"{trained.n_samples:,}")
        with st.expander("All models (holdout)"):
            score_df = pd.DataFrame(
                [
                    {"Model": k, "Accuracy %": round(v * 100, 2)}
                    for k, v in sorted(
                        trained.model_scores.items(),
                        key=lambda x: -x[1],
                    )
                ]
            )
            st.dataframe(score_df, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("**Try a sample**")
        samples = [
            "Uber ride to airport",
            "Monthly Netflix subscription",
            "Whole Foods groceries",
            "CVS pharmacy copay",
        ]
        for i, s in enumerate(samples):
            if st.button(s, key=f"sample_{i}", use_container_width=True):
                st.session_state.desc_input = s
                st.rerun()
        st.divider()
        with st.expander("About"):
            st.markdown(
                "Text is lowercased and stripped of punctuation, then converted with "
                "TF‑IDF (1–2 grams, 500 features). Three scikit-learn classifiers are "
                "compared; the winner is refit on all rows for predictions in this app."
            )

    tab_single, tab_batch = st.tabs(["Single description", "Batch CSV"])

    with tab_single:
        description = st.text_area(
            "Transaction description",
            height=100,
            placeholder='e.g. "Starbucks morning coffee" or "Shell gas station"',
            key="desc_input",
        )

        col_a, _ = st.columns([1, 2])
        with col_a:
            classify = st.button("Classify", type="primary", use_container_width=True)

        if classify:
            if not description.strip():
                st.warning("Enter a transaction description first.")
                st.session_state.pop("last_prediction", None)
            else:
                st.session_state.last_prediction = description.strip()

        last = st.session_state.get("last_prediction")
        if last:
            if description.strip() != last:
                st.warning("You edited the text — click **Classify** again to update the prediction.")
            st.caption(f"Prediction for: «{last}»")
            _render_prediction(trained, last)

    with tab_batch:
        st.markdown(
            "Upload a CSV with a **`description`** column (one transaction per row). "
            "Optional: any other columns are kept in the download."
        )
        up = st.file_uploader("CSV file", type=["csv"], key="batch_csv")
        if up is not None:
            file_token = f"{up.name}:{getattr(up, 'size', '')}"
            if st.session_state.get("batch_file_token") != file_token:
                st.session_state.batch_file_token = file_token
                st.session_state.batch_result = None

            try:
                batch_df = pd.read_csv(up)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
            else:
                col = None
                if "description" in batch_df.columns:
                    col = "description"
                elif len(batch_df.columns) >= 1:
                    col = batch_df.columns[0]
                    st.info(f"Using first column **`{col}`** as description.")
                if col is None:
                    st.error("No columns found in CSV.")
                else:
                    texts = batch_df[col].astype(str).tolist()
                    if st.button("Run batch classification", type="primary"):
                        preds = predict_batch(trained, texts)
                        out = batch_df.copy()
                        out["predicted_category"] = preds
                        st.session_state.batch_result = out
                    res = st.session_state.get("batch_result")
                    if res is not None:
                        st.dataframe(res.head(50), use_container_width=True)
                        st.caption(f"Showing first 50 of {len(res)} rows.")
                        csv_bytes = res.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results CSV",
                            data=csv_bytes,
                            file_name="classified_transactions.csv",
                            mime="text/csv",
                        )

    st.divider()
    st.subheader("Dataset snapshot")
    df_raw = pd.read_csv(DATA_PATH)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows in data.csv", len(df_raw))
    with c2:
        st.metric("Categories", df_raw["category"].nunique())
    with st.expander("Category counts"):
        counts = df_raw["category"].value_counts().sort_index()
        st.bar_chart(counts)


if __name__ == "__main__":
    main()
