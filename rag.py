import os
import glob
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sarvam import Sarvam
import chromadb

load_dotenv()


LLAMA_CLOUD_API_KEY=os.getenv("LLAMA_CLOUD_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


PDF_DIRECTORY = "./data/pdfs" 


CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "agri_knowledge_base"
MODEL_CACHE_DIR = "./models_cache"

def get_query_engine():
    if not LLAMA_CLOUD_API_KEY:
        print("Error: LLAMA_CLOUD_API_KEY is missing. Please check your .env file.")
        return

    print("--- Setting up Models ---")
    llm = Sarvam(model="sarvam-m", api_key=SARVAM_API_KEY, temperature=0.2)
    

    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        cache_folder=MODEL_CACHE_DIR
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )

    print("--- Setting up ChromaDB ---")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if data exists
    if chroma_collection.count() > 0:
        print("--- Loading existing index from ChromaDB ---")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
    else:
        print(f"--- Parsing PDFs from {PDF_DIRECTORY} ---")
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True,
            language="en"
        )
        
        pdf_files = glob.glob(os.path.join(PDF_DIRECTORY, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {PDF_DIRECTORY}.")
            return

        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor)
        documents = reader.load_data()
        print(f"Successfully parsed {len(documents)} document chunks.")

        print("--- Building Vector Index ---")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print("Index built successfully!")

    # ... existing code ...

    return index.as_query_engine(
        streaming=True,
        system_prompt="""
You are the AgroSense Assistant, an expert in agriculture.
Answer user questions based ONLY on the provided agricultural knowledge base.
If the information to answer a question is not available in the agricultural knowledge base, state that you do not have enough information to answer that specific question. Do NOT make up answers.
Avoid mentioning 'context information', 'retrieved documents', or any technical terms related to your internal processing.
Be helpful and provide general agricultural advice if appropriate, but always prioritize information from the knowledge base when available.
"""
    )

def main():
    query_engine = get_query_engine()
    if not query_engine:
        return

    print("\n--- RAG System Ready (Type 'exit' to quit) ---")
    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            break
        
        try:
            response = query_engine.query(user_query)
            print("\nAnswer: ", end="")
            response.print_response_stream()
            print("\n")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"Created directory {PDF_DIRECTORY}. Please put your PDFs there.")
    else:
        main()