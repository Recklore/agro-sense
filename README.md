# AgroSense Assistant

AgroSense Assistant is a web application built with Streamlit that provides agricultural assistance to farmers. It leverages a machine learning model for crop recommendations and a Retrieval-Augmented Generation (RAG) based chatbot for answering agricultural queries.

## Features

-   **Crop Recommendation**: Recommends the top 3 crops to grow based on soil and weather conditions (Nitrogen, Phosphorous, Potassium, pH, temperature, humidity, and rainfall).
-   **Agri Chatbot**: A conversational AI assistant that answers questions related to agriculture.
    -   **RAG-based**: The chatbot's knowledge is derived from a collection of PDF documents.
    -   **Voice-enabled**: Supports both text and audio input (microphone required).
    -   **Text-to-Speech**: Can convert its text responses into speech.

## How it Works

### Crop Recommendation

The crop recommendation system uses a pre-trained machine learning model (Random Forest) to predict the most suitable crops. The model takes soil and weather parameters as input and provides the top 3 crop recommendations with their confidence scores.

### Agri Chatbot

The Agri Chatbot is built using the Retrieval-Augmented Generation (RAG) technique. It works as follows:

1.  **Knowledge Base**: A collection of PDF documents on agriculture is stored in the `data/pdfs` directory.
2.  **Indexing**: The PDF documents are parsed and indexed into a vector database (ChromaDB). This process is done only once or when new documents are added.
3.  **Querying**: When a user asks a question, the chatbot retrieves the most relevant information from the vector database and uses a Large Language Model (LLM) from Sarvam AI to generate a comprehensive answer.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/agrosense-assistant.git
    cd agrosense-assistant
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys**:
    -   Create a `.env` file in the root directory.
    -   Add your Sarvam and LlamaParse API keys to the `.env` file:
        ```
        SARVAM_API_KEY="your-sarvam-api-key"
        LLAMA_CLOUD_API_KEY="your-llama-cloud-api-key"
        ```

5.  **Add PDF documents**:
    -   Place your agricultural PDF documents in the `data/pdfs` directory.

## Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2.  **Open the application in your browser**:
    -   The application will open in your default web browser, usually at `http://localhost:8501`.

3.  **Use the application**:
    -   **Crop Recommendation**: Navigate to the "Crop Recommendation" tab, enter the soil and weather parameters, and click "Recommend".
    -   **Agri Chatbot**: Navigate to the "Agri Chatbot" tab, type your question in the chat input, or use the microphone to ask your question.

## File Structure

```
.
├── app.py                  # Main Streamlit application
├── crop_predict.py         # Crop recommendation model prediction
├── rag.py                  # RAG pipeline for the chatbot
├── sarvam_utils.py         # Utility functions for Sarvam AI API
├── llm_utils.py            # Utility functions for LLM routing (future use)
├── requirements.txt        # Project dependencies
├── data/
│   └── pdfs/               # Directory for PDF documents
├── models/
│   └── crop_model.pkl      # Pre-trained crop recommendation model
├── chroma_db/              # Vector database for the RAG pipeline
└── ...
```

## Dependencies

The project uses the following major libraries:

-   `streamlit`: For creating the web application.
-   `pandas`, `scikit-learn`: For the crop recommendation model.
-   `llama-index`, `llama-parse`, `chromadb`: For the RAG pipeline.
-   `sarvamai`: For interacting with the Sarvam AI API.
-   `streamlit-mic-recorder`: For audio input in the chatbot.

For a full list of dependencies, see the `requirements.txt` file.
