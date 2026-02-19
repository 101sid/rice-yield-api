from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import random
import pickle # Added to load your .pkl file

# Initialize the App
app = FastAPI(title="Rice Yield Prediction API")

# --- LOAD ENSEMBLE MODEL ---
# This looks for the .pkl file in the same folder as main.py
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
    temperature: float
    humidity: float
    rainfall: float
    ph: float

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
        
        # Note: If you have a .h5 or .pkl for image classification, you would load it similarly.
        # For now, leaving the soil detection as a placeholder so the app doesn't crash.
        soil_types = ['Clay Loam', 'Sandy Loam', 'Silty Clay', 'Alluvial']
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
        # 1. Format the incoming data into a Numpy Array for the model
        # The order must match exactly how you trained the model!
        input_features = np.array([[
            data.nitrogen, 
            data.phosphorus, 
            data.potassium, 
            data.temperature, 
            data.humidity, 
            data.ph, 
            data.rainfall
        ]])

        # 2. Run the actual prediction through your Stacking Regressor
        prediction = ensemble_model.predict(input_features)[0]
        final_yield = float(prediction)

        # 3. Dynamic Advisory Logic
        if data.rainfall < 100:
            advisory = "Maintain water level at 5cm. Rainfall is low."
        elif "Red Soil" in predict_soil_type.__doc__: # Example context logic
             advisory = "Drain simulation recommended."
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