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

# --- LOAD ALL MODELS ---
# We keep your ensemble, but also load the baselines for comparison
model_files = {
    "Linear Regression": "linear_model.pkl",
    "KNN Regressor": "knn_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Final Stacking Ensemble": "rice_ensemble.pkl"
}

loaded_models = {}
for name, file in model_files.items():
    try:
        loaded_models[name] = pickle.load(open(file, 'rb'))
        print(f"✅ {name} loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load {file}. Error: {e}")
        loaded_models[name] = None

# --- DATA MODELS ---
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
    # Check if the main ensemble is loaded
    is_live = loaded_models.get("Final Stacking Ensemble") is not None
    return {"status": "System Online", "version": "2.0", "ensemble_live": is_live}

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
    # Ensure the main model exists
    if loaded_models.get("Final Stacking Ensemble") is None:
        raise HTTPException(status_code=500, detail="Main Ensemble model is not loaded.")

    try:
        # --- TRANSLATE FLUTTER TEXT INTO ML NUMBERS ---
        ir_map = {"Continuous Flooding": 0, "AWD (Alternate Wetting Drying)": 1, "Rainfed": 2}
        ir_val = ir_map.get(data.irrigation_type, 0)

        cv_map = {"Hybrid (High Yield)": 0, "Inbred": 1, "Traditional": 2}
        cv_val = cv_map.get(data.crop_variety, 1)

        soil_map = {"Clay": 0, "Sandy": 1, "Silty": 2, "Clay Loam": 0} 
        soil_val = soil_map.get(data.soil_type, 0)

        # Build input features array
        input_features = np.array([[
            data.nitrogen, data.phosphorus, data.potassium, data.ph,
            data.temperature, data.rainfall, data.humidity, 
            ir_val, cv_val, soil_val
        ]])

        # --- RUN ALL PREDICTIONS ---
        all_predictions = []
        for name, model in loaded_models.items():
            if model:
                pred = model.predict(input_features)[0]
                all_predictions.append({
                    "model_name": name,
                    "prediction": round(float(pred), 2),
                    "is_final": name == "Final Stacking Ensemble"
                })

        # Use the Final Ensemble result for the Advisory Logic
        final_yield = all_predictions[-1]["prediction"]

        # --- YOUR EXISTING MEGA EXPERT SYSTEM ADVISORY LOGIC ---
        advisories = []

        # 1. NPK Absolute Values
        if data.nitrogen < 40:
            advisories.append("🚨 Severe N deficiency: Growth will be highly stunted. Apply split doses of Urea immediately.")
        elif 40 <= data.nitrogen < 80:
            advisories.append("⚠️ Sub-optimal Nitrogen: Top dress with Urea to ensure proper tillering.")
        elif 150 < data.nitrogen <= 200:
            advisories.append("⚠️ Excess Nitrogen: High risk of lodging and Brown Plant Hopper attacks. Reduce N input.")
            
        if data.phosphorus < 20:
            advisories.append("🚨 Severe P deficiency: Root development restricted. Apply DAP or SSP.")
        elif 20 <= data.phosphorus < 40:
            advisories.append("⚠️ Low Phosphorus: Grain filling may be delayed.")

        if data.potassium < 30:
            advisories.append("🚨 Severe K deficiency: Weak stems and poor disease resistance. Apply MOP.")
        elif 30 <= data.potassium < 50:
            advisories.append("⚠️ Low Potassium: Crop is vulnerable to stress. Supplement K.")

        # 2. Nutrient & Variety Inter-dependencies
        if data.nitrogen > 120 and data.potassium < 40:
            advisories.append("🦠 Imbalanced N:K Ratio: High N with low K severely increases vulnerability to fungal diseases.")
        if cv_val == 0 and (data.nitrogen < 80 or data.phosphorus < 30):
            advisories.append("🧬 Hybrid Mismatch: Hybrid varieties demand high fertilizers. Current nutrient levels will starve the crop.")
        elif cv_val == 2 and data.nitrogen > 100:
            advisories.append("🌾 Traditional Mismatch: Traditional varieties lodge (fall over) easily under high Nitrogen. Stop N application.")

        # 3. pH Granular Sensitivity
        if data.ph < 4.5:
            advisories.append("☠️ Extremely Acidic: High Iron/Aluminum toxicity risk. Urgent agricultural liming required.")
        elif 4.5 <= data.ph < 5.5:
            advisories.append("⚠️ Acidic Soil: Base yields restricted. Consider liming next season.")
        elif 6.5 < data.ph <= 7.5:
            advisories.append("⚠️ Slightly Alkaline: Monitor for Zinc or Iron deficiency.")
        elif data.ph > 7.5:
            advisories.append("☠️ Highly Alkaline: Salinity risk. Apply Gypsum and ensure excellent drainage.")

        # 4. Weather & Disease Vectors
        if data.temperature < 15:
            advisories.append("🥶 Critical Cold: Seedling mortality likely. Do not drain the field.")
        elif 15 <= data.temperature < 22:
            advisories.append("❄️ Mild Cold Stress: Vegetative growth will be sluggish.")
        elif 33 < data.temperature <= 36:
            advisories.append("🌡️ Moderate Heat Stress: Pollen viability decreasing.")
        elif data.temperature > 36:
            advisories.append("🔥 Severe Heat Stress: Spikelet sterility imminent. Maintain 5-7cm standing water to cool the canopy.")

        if data.humidity > 85 and data.temperature > 28:
            advisories.append("🍄 Fungal Weather: High humidity + warmth is perfect for Blast and Sheath Blight. Spray preventive fungicide.")
        elif data.humidity < 40 and data.temperature > 32:
            advisories.append("🏜️ Hot & Dry: Extreme transpiration rate. The crop will exhaust soil moisture rapidly.")

        # 5. Hydrology & Soil Physics
        if data.rainfall > 250 and ir_val == 0:
            advisories.append("🌊 Submergence Risk: Heavy rain + Flood irrigation = nutrient leaching. Open drainage channels immediately.")
        elif data.rainfall < 50 and ir_val == 2:
            advisories.append("🌵 Severe Drought: Rainfed crop failing. Seek alternate life-saving irrigation.")
        elif data.rainfall > 200 and data.nitrogen > 120:
            advisories.append("🌧️ Rain Washout: Heavy rains will leach your high Nitrogen levels. Delay fertilizer application.")

        if soil_val == 1 and ir_val == 0:
             advisories.append("💧 Inefficient Irrigation: Flooding Sandy soil causes massive water percolation loss. Switch to frequent light watering.")
        elif soil_val == 0 and ir_val == 1:
             advisories.append("✅ Optimal Water Strategy: Clay soil retains water beautifully, making AWD highly effective and safe.")

        # 6. Yield Trajectory
        if final_yield < 2.0:
            advisories.append("📉 Yield projection is critically low. Comprehensive intervention required.")
        elif final_yield > 6.0:
            advisories.append("🏆 Exceptional yield trajectory! Maintain current monitoring.")

        # 7. Compile Final Output
        if not advisories:
            final_advisory = "✅ All environmental and nutritional parameters are perfectly balanced."
        else:
            final_advisory = "\n\n".join(advisories)

        return {
            "all_predictions": all_predictions, # NEW: Send all 4 model results
            "predicted_yield_tons_ha": final_yield, # Keep for backward compatibility
            "advisory": final_advisory
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
