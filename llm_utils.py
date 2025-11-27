import os
from dotenv import load_dotenv
from llama_index.llms.sarvam import Sarvam
from llama_index.core import Settings

load_dotenv()

# Initialize LLM
llm = Sarvam(api_key=os.getenv("SARVAM_API_KEY"), model="sarvam-m", temperature=0.2)
Settings.llm = llm


ROUTER_SYSTEM = """
You are a router.
Decide whether a user query needs a crop recommendation (based on soil/weather parameters) or a general informational answer.
If it needs a crop recommendation, respond with exactly: RF
Otherwise respond with exactly: RAG
No extra words.
"""

def decide_route(user_query: str) -> str:
    """
    Decides whether to use the Crop Recommendation (RF) tool or the RAG tool.
    """
    resp = llm.complete(f"{ROUTER_SYSTEM}\n\nUser query: {user_query}")
    text = resp.text.strip().upper()
    return "RF" if "RF" in text else "RAG"
