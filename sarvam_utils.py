import os
import io
import base64
import streamlit as st
from sarvamai import SarvamAI

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

@st.cache_resource
def get_sarvam_client():
    """Initializes and returns a synchronous SarvamAI client."""
    if not SARVAM_API_KEY:
        st.error("SARVAM_API_KEY is not set. Please configure your API key.")
        return None
    return SarvamAI(api_subscription_key=SARVAM_API_KEY)

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes audio bytes to text using the synchronous Sarvam STT SDK.
    """
    client = get_sarvam_client()
    if not client:
        return ""

    try:
        # Wrap the audio bytes in a file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saarika:v2.5",
            language_code="en-IN" 
        )
        return response.transcript
    except Exception as e:
        st.error(f"Sarvam STT SDK error: {e}")
        return ""

def synthesize_speech(text: str) -> bytes | None:
    """
    Synthesizes text to speech using the synchronous Sarvam TTS SDK.
    """
    client = get_sarvam_client()
    if not client:
        return None
    
    try:
        audio = client.text_to_speech.convert(
            target_language_code="en-IN",
            text=text,
            model="bulbul:v2",
            speaker="anushka"
        )
        # The response contains a list with a single base64 encoded string
        return base64.b64decode(audio.audios[0])
    except Exception as e:
        st.error(f"Sarvam TTS SDK error: {e}")
        return None

