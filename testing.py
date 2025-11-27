import os
from dotenv import load_dotenv
from sarvamai import SarvamAI  # or from sarvam import SarvamAI if that is what works for you

load_dotenv()

client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
MODEL_NAME = "sarvam-m"  # adjust if Sarvam gives you a different chat model name

# ---------------------------
# 1) REAL RF MODEL
# ---------------------------

# You: load your actual trained RandomForest once (e.g., from joblib)
# from joblib import load
# rf_model = load("rf_model.joblib")

def run_rf_recommendation(features: dict) -> dict:
    """
    features: dict with keys N,P,K,temperature,humidity,rainfall,moisture
    Return: dict like {"crop": "...", "proba": 0.87}
    """
    # Example using scikit-learn (uncomment and adapt):
    # X = [[
    #     features["N"],
    #     features["P"],
    #     features["K"],
    #     features["temperature"],
    #     features["humidity"],
    #     features["rainfall"],
    #     features["moisture"],
    # ]]
    # crop_idx = rf_model.predict(X)[0]
    # prob = max(rf_model.predict_proba(X)[0])
    # crop = idx_to_crop[crop_idx]

    # REMOVE this placeholder once you plug in:
    crop = "dummy_crop"
    prob = 0.0

    return {"crop": crop, "confidence": prob, "features": features}

# ---------------------------
# 2) REAL RAG PIPELINE
# ---------------------------

# You: plug your existing RAG here (vector DB + Sarvam or other model)

def run_rag_answer(user_query: str) -> str:
    """
    Use your existing RAG pipeline:
      - retrieve docs based on user_query
      - build context
      - call an LLM (can be Sarvam) with that context
    Return: final answer as string.
    """
    # placeholder – REPLACE with your actual RAG code
    answer = f"[RAG placeholder] Answer from your KB for: {user_query}"
    return answer

# ---------------------------
# 3) ROUTER USING SARVAM
# ---------------------------

ROUTER_SYSTEM = """
You are a router.
Decide whether a user query needs a crop recommendation (based on soil/weather parameters) or a general informational answer.
If it needs a crop recommendation using a Random Forest model, respond with exactly: RF
Otherwise respond with exactly: RAG
No extra words.
"""

def decide_route(user_query: str) -> str:
    resp = client.chat.completions(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": user_query},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    text = resp.choices[0].message.content.strip().upper()
    return "RF" if text == "RF" else "RAG"

# ---------------------------
# 4) FEATURE EXTRACTION FOR RF
# ---------------------------

def extract_features_from_query(user_query: str) -> dict | None:
    """
    You can use regex/LLM to parse N,P,K,temperature,humidity,rainfall,moisture from text.
    For now this is a stub that returns None -> forces you to ask user explicitly.
    """
    return None

def ask_missing_features(user_query: str) -> str:
    """
    Use Sarvam to ask the user exactly which values are needed.
    """
    msg = (
        "To recommend a crop I need numeric values for: "
        "N, P, K, temperature (°C), humidity (%), rainfall (mm), moisture (%). "
        "Please provide them."
    )
    return msg

# ---------------------------
# 5) ORCHESTRATION
# ---------------------------

def chat_once(user_query: str):
    route = decide_route(user_query)

    if route == "RF":
        # Try to auto-extract features (you can implement with LLM or regex)
        features = extract_features_from_query(user_query)

        if features is None:
            # Not enough info → ask user for numbers
            return ask_missing_features(user_query), "rf-need-features"

        rf_result = run_rf_recommendation(features)

        # Let Sarvam phrase the final message based on RF output
        resp = client.chat.completions(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain crop recommendations using the model output. "
                        "User is a farmer, keep it simple."
                    ),
                },
                {
                    "role": "user",
                    "content": user_query,
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Model output: crop={rf_result['crop']}, "
                        f"confidence={rf_result['confidence']}, "
                        f"features={rf_result['features']}"
                    ),
                },
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content, "rf"

    else:  # RAG
        rag_answer = run_rag_answer(user_query)

        # Optional: pass through Sarvam to polish answer
        resp = client.chat.completions(
            messages=[
                {"role": "system", "content": "You are an agriculture assistant."},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": rag_answer},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content, "rag"

# ---------------------------
# 6) CLI LOOP FOR TESTING
# ---------------------------

if __name__ == "__main__":
    while True:
        q = input("You: ")
        ans, route = chat_once(q)
        print(f"[Route: {route}]")
        print("Bot:",ans)
