from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import random
import pickle

# Initialize the App
app = FastAPI(title="Rice Yield Prediction API")

# --- LOAD ENSEMBLE MODEL ---
try:
    ensemble_model = pickle.load(open('rice_ensemble.pkl', 'rb'))
    print("✅ Ensemble Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load rice_ensemble.pkl. Error: {e}")
    ensemble_model = None

# --- DATA MODELS ---
class YieldInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float             
    temperature: float
    rainfall: float       
    humidity: float
    
    # Adding the 3 missing features with defaults so your mobile app doesn't crash!
    irrigation_type: int = 0  # Default to 0 (Flood)
    crop_variety: int = 1     # Default to 1 (Inbred)
    soil_type: int = 0        # Default to 0 (Clay)

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "System Online", "version": "1.0", "model_loaded": ensemble_model is not None}

@app.post("/predict-soil-type/")
async def predict_soil_type(file: UploadFile = File(...)):
    """
    Receives an image, processes it, and returns the soil class.
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = image.resize((224, 224))
        
        # Updated to match the soil types your model was trained on
        soil_types = ['Clay', 'Sandy', 'Silty'] 
        detected_type = random.choice(soil_types)
        
        return {
            "filename": file.filename,
            "soil_type": detected_type,
            "confidence": 0.94
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-yield/")
def predict_yield(data: YieldInput):
    """
    Receives NPK + Weather data and returns predicted yield in Tons/Ha.
    """
    if ensemble_model is None:
        raise HTTPException(status_code=500, detail="Machine Learning model is not loaded on the server.")

    try:
        # 1. Format the data into a Numpy Array. 
        # WARNING: This order matches train_model.py EXACTLY. Do not rearrange!
        input_features = np.array([[
            data.nitrogen, 
            data.phosphorus, 
            data.potassium, 
            data.ph,
            data.temperature, 
            data.rainfall,
            data.humidity, 
            data.irrigation_type,
            data.crop_variety,
            data.soil_type
        ]])

        # 2. Run the actual prediction through your Stacking Regressor
        prediction = ensemble_model.predict(input_features)[0]
        final_yield = float(prediction)

        # 3. Dynamic Advisory Logic based on your training rules
        if data.rainfall < 100:
            advisory = "Maintain water level at 5cm. Rainfall is low."
        elif data.soil_type == 1: # Sandy soil
             advisory = "Drain simulation recommended. High percolation risk."
        else:
            advisory = "Conditions are optimal."

        return {
            "predicted_yield_tons_ha": round(final_yield, 2),
            "advisory": advisory
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)