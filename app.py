from __future__ import annotations
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from typing import Generator, List, Tuple
from streamlit_mic_recorder import mic_recorder

# --- Environment and Page Configuration ---
load_dotenv()

def _sync_secrets_to_env() -> None:
    """Propagate Streamlit secrets to os.environ if not already set."""
    try:
        for key in ("SARVAM_API_KEY", "LLAMA_CLOUD_API_KEY"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = st.secrets[key]
    except Exception: # Covers StreamlitSecretNotFoundError and others
        pass # Silently ignore if secrets are not configured, relying on .env


_sync_secrets_to_env()

st.set_page_config(
    page_title="AgroSense Assistant",
    page_icon="🌾",
    layout="wide",
)

# --- Module Imports (after env setup) ---
from crop_predict import predict_top_3_crops
from rag import get_query_engine
from sarvam_utils import transcribe_audio, synthesize_speech

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all required session state variables."""
    defaults = {
        "chat_history": [{"role": "assistant", "content": "Hi! I am your AgroSense assistant. Ask me anything." }],
        "recommendation_history": [],
        "user_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Backend Loading ---
@st.cache_resource(show_spinner="Booting Retrieval-Augmented Chatbot…")
def load_query_engine():
    """Instantiate and cache the RAG query engine."""
    try:
        return get_query_engine()
    except Exception as e:
        st.error(f"Failed to initialize the RAG query engine: {e}")
        return None

query_engine = load_query_engine()

# --- Helper Functions ---
def stream_response_chunks(response) -> Generator[str, None, None]:
    """Yield streaming tokens from a LlamaIndex Response."""
    if hasattr(response, "response_gen") and response.response_gen is not None:
        for token in response.response_gen:
            yield token
    else:
        yield str(response)

def add_recommendation_to_chat(inputs: dict, recommendations: list):
    """Formats and adds a crop recommendation summary to the chat history."""
    summary = (
        f"Based on your inputs (N={inputs['N']}, P={inputs['P']}, K={inputs['K']}, "
        f"pH={inputs['ph']}, Temp={inputs['temperature']}°C, Humidity={inputs['humidity']}%, "
        f"Rainfall={inputs['rainfall']}mm), the top 3 recommended crops are: "
        f"{recommendations[0][0]}, {recommendations[1][0]}, and {recommendations[2][0]}."
    )
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"I have just received a crop recommendation. Here is a summary:\n{summary}"
    })

# --- UI Rendering: Tabs ---
st.title("🌾 AgroSense Assistant")
st.caption("Get crop recommendations and chat with your knowledge base via RAG.")

tab_reco, tab_chat = st.tabs(["Crop Recommendation", "Agri Chatbot"])

with tab_reco:
    st.subheader("Get Crop Recommendations")
    with st.form("crop_form"):
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0, 1.0)
            P = st.number_input("Phosphorous (P)", 0.0, 200.0, 50.0, 1.0)
            K = st.number_input("Potassium (K)", 0.0, 200.0, 50.0, 1.0)
            ph = st.number_input("pH", 0.0, 14.0, 6.5, 0.1)
        with col2:
            temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, 0.5)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, 1.0)
        
        if st.form_submit_button("Recommend"):
            with st.spinner("Predicting..."):
                inputs = {"N": N, "P": P, "K": K, "ph": ph, "temperature": temperature, "humidity": humidity, "rainfall": rainfall}
                recommendations = predict_top_3_crops(**inputs)
                
                if recommendations and not recommendations[0][0].lower().startswith("error"):
                    st.session_state.recommendation_history.append({"inputs": inputs, "recommendations": recommendations})
                    add_recommendation_to_chat(inputs, recommendations)
                    
                    df = pd.DataFrame(recommendations, columns=["Crop", "Confidence"])
                    df["Confidence (%)"] = (df["Confidence"]*100).map(lambda v: f"{v:.2f}%")
                    st.success(f"Best match: **{df.iloc[0,0]}** with {df.iloc[0,2]}")
                    st.dataframe(df[["Crop", "Confidence (%)"]], hide_index=True, use_container_width=True)
                else:
                    st.error("Could not retrieve recommendations. Please check the model.")

with tab_chat:
    st.subheader("Knowledge-Base Chatbot")

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg["content"]:
                if st.button("🔊", key=f"play_{hash(msg['content'])}", help="Read this message aloud"):
                    audio_bytes = synthesize_speech(msg["content"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
    
    # Chat Input and Audio Recorder Logic
    prompt = None
    
    # Use columns to place recorder next to the input box area
    col_input, col_mic = st.columns([4, 1])
    with col_input:
        text_input = st.chat_input("Ask a question or describe your soil...")
        if text_input:
            prompt = text_input
    with col_mic:
        st.write("") # Spacer for alignment
        st.write("") # Spacer for alignment
        audio_info = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key="recorder", format="wav")

    if audio_info and isinstance(audio_info, dict) and audio_info.get('bytes'):
        with st.spinner("Transcribing audio..."):
            prompt = transcribe_audio(audio_info['bytes'])
            if not prompt:
                st.warning("Could not understand the audio, please try again.")

    # Process the prompt if one exists
    if prompt:
        st.session_state.user_prompt = prompt
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_prompt})
        
        with st.chat_message("user"):
            st.markdown(st.session_state.user_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if query_engine:
                    response = query_engine.query(st.session_state.user_prompt)
                    assistant_reply = st.write_stream(stream_response_chunks(response))
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
                    st.session_state.user_prompt = None # Clear prompt after processing
                else:
                    st.error("Query engine is not available.")