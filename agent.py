import os
from dotenv import load_dotenv

from llama_index.llms.sarvam import Sarvam
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from crop_predict import predict_top_3_crops
from rag import get_query_engine

load_dotenv()

# Initialize Query Engine (This also configures Settings in rag.py)
print("Initializing RAG system...")
query_engine = get_query_engine()

# Ensure Settings are correct (in case rag.py didn't set them globally or we want to be sure)
if not Settings.llm:
    Settings.llm = Sarvam(api_key=os.getenv("SARVAM_API_KEY"), model="sarvam-m", temperature=0.2)
if not Settings.embed_model:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        cache_folder="./models_cache"
    )

def search_knowledge_base(query: str) -> str:
    """
    Search the agricultural knowledge base (RAG) for information.
    Use this tool to answer general farming questions, get details about crops, diseases, or farming techniques.
    
    Args:
        query (str): The user's question or search query.
    """
    response = query_engine.query(query)
    return str(response)

def recommend_crops(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float) -> str:
    """
    Predicts the best crops to grow based on soil and weather conditions.
    
    Args:
        N (float): Ratio of Nitrogen content in soil
        P (float): Ratio of Phosphorous content in soil
        K (float): Ratio of Potassium content in soil
        temperature (float): Temperature in degree Celsius
        humidity (float): Relative humidity in %
        ph (float): ph value of the soil
        rainfall (float): Rainfall in mm
        
    Returns:
        str: A formatted string listing the top 3 recommended crops with confidence scores.
    """
    top_3 = predict_top_3_crops(N, P, K, temperature, humidity, ph, rainfall)
    
    # Format output for the LLM
    result = "Top 3 recommended crops based on the provided conditions:\n"
    for crop, prob in top_3:
        result += f"- {crop} (Confidence: {prob:.2%})\n"
    return result

# Create Tools
rag_tool = FunctionTool.from_defaults(fn=search_knowledge_base)
predict_tool = FunctionTool.from_defaults(fn=recommend_crops)

# Initialize Agent
agent = ReActAgent.from_tools(
    tools=[predict_tool, rag_tool],
    llm=Settings.llm,
    verbose=True,
    context="""You are a smart farming assistant. 
    Use the 'recommend_crops' tool when the user provides soil and weather data (N, P, K, temperature, humidity, ph, rainfall).
    Use the 'search_knowledge_base' tool for general agricultural questions or to get more details about a specific crop.
    If the user asks for recommendations but hasn't provided the necessary data, ask them for it."""
)

if __name__ == "__main__":
    # Test the agent
    print("\n--- Agent Ready ---")
    response = agent.chat("what are the best conditions to grow rice")
    print(f"Agent: {response}")