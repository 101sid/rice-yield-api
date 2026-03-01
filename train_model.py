import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. GENERATE ADVANCED SYNTHETIC DATA ---
# We simulate the complex interactions described in Source 12
n_samples = 6000 # Increased dataset size
np.random.seed(42)

data = {
    'nitrogen': np.random.uniform(50, 200, n_samples),
    'phosphorus': np.random.uniform(20, 90, n_samples),
    'potassium': np.random.uniform(20, 90, n_samples),
    'ph': np.random.uniform(4.0, 9.0, n_samples),
    'temperature': np.random.uniform(20, 42, n_samples),
    'rainfall': np.random.uniform(0, 400, n_samples),
    'humidity': np.random.uniform(30, 95, n_samples),
    'irrigation_type': np.random.choice([0, 1, 2], n_samples), # 0=Flood, 1=AWD, 2=Rainfed
    'crop_variety': np.random.choice([0, 1, 2], n_samples),    # 0=Hybrid, 1=Inbred, 2=Traditional
    'soil_type': np.random.choice([0, 1, 2], n_samples)        # 0=Clay, 1=Sandy, 2=Silty
}
df = pd.DataFrame(data)

# --- 2. DEFINE THE "GROUND TRUTH" RULES [Based on Research Doc] ---
def calculate_research_grade_yield(row):
    y = 4.0 # Base yield
    
    # Interaction: Nitrogen Use Efficiency depends on Water
    n_efficiency = 1.0
    if row['irrigation_type'] == 2 and row['rainfall'] < 100:
        n_efficiency = 0.4 # N is wasted in dry soil
    
    # N response curve
    effective_n = row['nitrogen'] * n_efficiency
    if effective_n < 80: y -= 1.2
    elif effective_n > 150: y += 0.6
    
    # Interaction: Sandy Soil + AWD
    if row['soil_type'] == 1 and row['irrigation_type'] == 1:
        if row['rainfall'] < 80:
            y -= 1.8 # Severe penalty
            
    # Heat Stress Threshold
    if row['temperature'] > 35:
        y -= (row['temperature'] - 35) * 0.45
        
    # Genotype Resilience
    if row['crop_variety'] == 0: # Hybrid
        y += 1.5
        if row['temperature'] > 37: y -= 0.5 # Hybrid sensitive to extreme heat
    elif row['crop_variety'] == 2: # Traditional
        y -= 0.4
        if row['temperature'] > 37: y += 0.2 # Resilient bonus
        
    # Random environmental noise
    y += np.random.normal(0, 0.15)
    return max(0, y)

df['yield'] = df.apply(calculate_research_grade_yield, axis=1)

# --- 3. BUILD THE ADVANCED STACKING ENSEMBLE (OPTIMIZED FOR CLOUD) ---
print("Initializing Upgraded & Optimized Stacking Ensemble...")

# Level 0: Base Learners (Reduced estimators and depth to shrink file size)
estimators = [
    ('rf', RandomForestRegressor(n_estimators=40, max_depth=12, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=40, max_depth=12, random_state=42)),
    ('knn', KNeighborsRegressor(n_neighbors=5))
]

# Level 1: Meta Learner (Non-Linear)
final_estimator = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    passthrough=False 
)

# --- 4. TRAIN AND EVALUATE ---
X = df.drop('yield', axis=1)
y = df['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Ensemble (this may take a few seconds)...")
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Ensemble Trained! R^2 Score: {score:.4f}")
print("This model integrates RF, Extra Trees, KNN, and Gradient Boosting (Optimized for size).")

# --- 5. SAVE ---
with open('rice_ensemble.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved as 'rice_ensemble.pkl' - Ready for Render!")