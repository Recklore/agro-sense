import pickle
import pandas as pd
import os

MODEL_PATH = "./models/crop_model.pkl"

def predict_crop(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float) -> str:
    """
    Predicts the best crop to grow based on soil and weather conditions.
    
    Args:
        N (float): Ratio of Nitrogen content in soil
        P (float): Ratio of Phosphorous content in soil
        K (float): Ratio of Potassium content in soil
        temperature (float): Temperature in degree Celsius
        humidity (float): Relative humidity in %
        ph (float): ph value of the soil
        rainfall (float): Rainfall in mm
    """
    if not os.path.exists(MODEL_PATH):
        return "Error: Model file not found."
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # Create dataframe for prediction to match training input
    data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    prediction = model.predict(data)
    return str(prediction[0])

def predict_top_3_crops(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float) -> list:
    """
    Predicts the top 3 best crops to grow based on soil and weather conditions.
    
    Args:
        N (float): Ratio of Nitrogen content in soil
        P (float): Ratio of Phosphorous content in soil
        K (float): Ratio of Potassium content in soil
        temperature (float): Temperature in degree Celsius
        humidity (float): Relative humidity in %
        ph (float): ph value of the soil
        rainfall (float): Rainfall in mm
        
    Returns:
        list: A list of tuples containing (crop_name, probability) for the top 3 crops.
    """
    if not os.path.exists(MODEL_PATH):
        return [("Error: Model file not found.", 0.0)]
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # Create dataframe for prediction to match training input
    data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    # Get probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)[0]
        classes = model.classes_
        
        # Create list of (class, probability)
        class_probs = list(zip(classes, probs))
        
        # Sort by probability in descending order
        sorted_probs = sorted(class_probs, key=lambda x: x[1], reverse=True)
        
        # Return top 3
        return sorted_probs[:3]
    else:
        # Fallback if model doesn't support probabilities
        prediction = model.predict(data)
        return [(str(prediction[0]), 1.0)]
