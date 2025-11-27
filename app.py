"""Streamlit interface for AgroSense crop recommendations and RAG chatbot."""

from __future__ import annotations

import os
from typing import Generator, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _sync_secrets_to_env() -> None:
    """Propagate Streamlit secrets (if any) into os.environ before imports."""
    secret_keys = ("SARVAM_API_KEY", "LLAMA_CLOUD_API_KEY")
    for key in secret_keys:
        try:
            secret_val = st.secrets.get(key)
        except Exception:  # pragma: no cover - st.secrets unavailable outside Streamlit
            secret_val = None
        if secret_val and not os.getenv(key):
            os.environ[key] = secret_val


def _configure_page() -> None:
    st.set_page_config(
        page_title="AgroSense Assistant",
        page_icon="🌾",
        layout="wide",
    )
    st.title("🌾 AgroSense Assistant")
    st.caption(
        "Get crop recommendations powered by a Random Forest model and chat with your knowledge base via RAG."
    )


_sync_secrets_to_env()

from crop_predict import predict_top_3_crops  # noqa: E402
from rag import get_query_engine  # noqa: E402


@st.cache_resource(show_spinner="Booting Retrieval-Augmented Chatbot…")
def load_query_engine():
    """Instantiate and cache the RAG query engine."""
    return get_query_engine()


def stream_response_chunks(response) -> Generator[str, None, None]:
    """Yield streaming tokens from a LlamaIndex Response."""
    if hasattr(response, "response_gen") and response.response_gen is not None:
        for token in response.response_gen:
            yield token
    else:
        yield str(response)


def render_crop_tab():
    st.subheader("Crop Recommendation")
    st.markdown(
        "Provide soil nutrients and weather conditions to receive the three most suitable crops."
    )

    with st.form("crop_form"):
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
            P = st.number_input("Phosphorous (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
            K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with col2:
            temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
        submitted = st.form_submit_button("Recommend")

    if submitted:
        with st.spinner("Predicting top crops…"):
            recommendations: List[Tuple[str, float]] = predict_top_3_crops(
                N=N,
                P=P,
                K=K,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall,
            )

        if not recommendations:
            st.warning("No recommendations returned. Please verify the model file.")
            return

        first_label = recommendations[0][0]
        if first_label.lower().startswith("error"):
            st.error(first_label)
            return

        df = pd.DataFrame(recommendations, columns=["Crop", "Confidence"])
        df["Confidence (%)"] = (df["Confidence"] * 100).map(lambda v: f"{v:.2f}%")

        st.success(f"Best match: **{df.iloc[0, 0]}** with {df.iloc[0, 2]}")
        st.dataframe(df[["Crop", "Confidence (%)"]], hide_index=True, use_container_width=True)
        chart_df = df.set_index("Crop")["Confidence"]
        st.bar_chart(chart_df, height=250, use_container_width=True)


def render_chat_tab(query_engine):
    st.subheader("Knowledge-Base Chatbot")
    st.markdown(
        "Ask agronomy questions, pest remedies, or crop practices. Responses are augmented with your PDF knowledge base."
    )

    if query_engine is None:
        st.warning(
            "RAG system is not available. Ensure SARVAM and LLAMA Cloud API keys are configured and PDFs are ingested."
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hi! I am your AgroSense assistant. Ask me anything about crops or farming techniques.",
            }
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask a question about crops, soil, or pests…")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            try:
                response = query_engine.query(user_prompt)
                assistant_reply = st.write_stream(stream_response_chunks(response))
            except Exception as exc:
                assistant_reply = f"Sorry, an error occurred while querying the knowledge base: {exc}"
                st.error(assistant_reply)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})


def main():
    _configure_page()

    try:
        query_engine = load_query_engine()
    except Exception as exc:
        query_engine = None
        st.error(f"Failed to initialize the RAG query engine: {exc}")

    tab_reco, tab_chat = st.tabs(["Crop Recommendation", "Agri Chatbot"])

    with tab_reco:
        render_crop_tab()
    with tab_chat:
        render_chat_tab(query_engine)


if __name__ == "__main__":
    main()
