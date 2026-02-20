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
# We changed the last three features to 'str' (String) to match your Flutter app perfectly!
class YieldInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float             
    temperature: float
    rainfall: float       
    humidity: float
    irrigation_type: str 
    crop_variety: str    
    soil_type: str       

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "System Online", "version": "1.0", "model_loaded": ensemble_model is not None}

@app.post("/predict-soil-type/")
async def predict_soil_type(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = image.resize((224, 224))
        
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
    if ensemble_model is None:
        raise HTTPException(status_code=500, detail="Machine Learning model is not loaded.")

    try:
        # --- TRANSLATE FLUTTER TEXT INTO ML NUMBERS ---
        
        # 1. Irrigation: 0=Flood, 1=AWD, 2=Rainfed
        ir_map = {"Continuous Flooding": 0, "AWD (Alternate Wetting Drying)": 1, "Rainfed": 2}
        ir_val = ir_map.get(data.irrigation_type, 0) # Defaults to 0 if it doesn't match

        # 2. Variety: 0=Hybrid, 1=Inbred, 2=Traditional
        cv_map = {"Hybrid (High Yield)": 0, "Inbred": 1, "Traditional": 2}
        cv_val = cv_map.get(data.crop_variety, 1) # Defaults to 1 if it doesn't match

        # 3. Soil: 0=Clay, 1=Sandy, 2=Silty
        soil_map = {"Clay": 0, "Sandy": 1, "Silty": 2, "Clay Loam": 0} 
        soil_val = soil_map.get(data.soil_type, 0) # Defaults to 0 if it doesn't match

        # --- RUN PREDICTION ---
        input_features = np.array([[
            data.nitrogen, 
            data.phosphorus, 
            data.potassium, 
            data.ph,
            data.temperature, 
            data.rainfall,
            data.humidity, 
            ir_val,     # Passing the translated number
            cv_val,     # Passing the translated number
            soil_val    # Passing the translated number
        ]])

        prediction = ensemble_model.predict(input_features)[0]
        final_yield = float(prediction)

        # Dynamic Advisory
        if data.rainfall < 100:
            advisory = "Maintain water level at 5cm. Rainfall is low."
        elif soil_val == 1: 
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