from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Insurance Response Prediction API",
    description="API for predicting if customer will buy vehicle insurance using XGBoost",
    version="1.0.0"
)

# Load model
MODEL_PATH = 'models/xgb_insurance_response_model.joblib'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Input validation model (10 features)
class InsuranceInput(BaseModel):
    Age: int
    Gender: str
    Driving_License: int
    Region_Code: int
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: int
    Vintage: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Gender": "Male",
                "Driving_License": 1,
                "Region_Code": 28,
                "Previously_Insured": 0,
                "Vehicle_Age": "1-2 Year",
                "Vehicle_Damage": "Yes",
                "Annual_Premium": 30000,
                "Policy_Sales_Channel": 26,
                "Vintage": 150
            }
        }

# Response model
class InsurancePrediction(BaseModel):
    predicted_response: int
    probability: float
    status: str

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Vehicle Insurance Response Prediction API",
        "status": "active",
        "model": "XGBoost",
        "endpoints": {
            "/predict": "POST - Predict if customer will buy insurance",
            "/health": "GET - Check API health"
        }
    }

# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Predict endpoint
@app.post("/predict", response_model=InsurancePrediction)
def predict_charges(input_data: InsuranceInput):
    """
    Predict if customer will buy vehicle insurance
    """
    try:
        # Encode categorical variables
        vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
        vehicle_age_encoded = vehicle_age_map.get(input_data.Vehicle_Age, 0)
        vehicle_damage_encoded = 1 if input_data.Vehicle_Damage.lower() == 'yes' else 0
        gender_encoded = 1 if input_data.Gender.lower() == 'male' else 0
        
        # Feature engineering (same as training)
        Age_Vehicle_Age = input_data.Age * vehicle_age_encoded
        Age_Premium = input_data.Age * input_data.Annual_Premium
        Age_Previously_Insured = input_data.Age * input_data.Previously_Insured
        Previously_Insured_Vehicle_Damage = input_data.Previously_Insured * vehicle_damage_encoded
        Premium_Per_Vehicle_Age = input_data.Annual_Premium / (vehicle_age_encoded + 1)
        Premium_Per_Vintage = input_data.Annual_Premium / (input_data.Vintage + 1)
        
        # Age bucket
        if input_data.Age <= 25: Age_Bucket = 0
        elif input_data.Age <= 35: Age_Bucket = 1
        elif input_data.Age <= 45: Age_Bucket = 2
        elif input_data.Age <= 55: Age_Bucket = 3
        else: Age_Bucket = 4
        
        High_Premium = 1 if input_data.Annual_Premium > 30000 else 0
        Young_Driver = 1 if input_data.Age < 30 else 0
        Old_Vehicle = 1 if vehicle_age_encoded == 2 else 0
        Age_Premium_VehicleDamage = input_data.Age * input_data.Annual_Premium * vehicle_damage_encoded
        
        # Approximate risk scores
        Region_Risk_Score = 0.12
        Channel_Risk_Score = 0.12
        
        # Additional features
        Age_Vintage = input_data.Age * input_data.Vintage
        Age_Policy_Channel = input_data.Age * input_data.Policy_Sales_Channel
        Premium_Squared = input_data.Annual_Premium ** 2
        Premium_Log = np.log1p(input_data.Annual_Premium)
        Vintage_Log = np.log1p(input_data.Vintage)
        Vintage_Premium = input_data.Vintage * input_data.Annual_Premium
        Premium_Per_Age = input_data.Annual_Premium / (input_data.Age + 1)
        Age_Squared = input_data.Age ** 2
        
        # Prepare input DataFrame with ALL 31 features (NUMERIC ONLY)
        input_df = pd.DataFrame([{
            'Age': input_data.Age,
            'Gender': gender_encoded,
            'Driving_License': input_data.Driving_License,
            'Region_Code': input_data.Region_Code,
            'Previously_Insured': input_data.Previously_Insured,
            'Vehicle_Age': vehicle_age_encoded,
            'Vehicle_Damage': vehicle_damage_encoded,
            'Annual_Premium': input_data.Annual_Premium,
            'Policy_Sales_Channel': input_data.Policy_Sales_Channel,
            'Vintage': input_data.Vintage,
            'Age_Vehicle_Age': Age_Vehicle_Age,
            'Age_Premium': Age_Premium,
            'Age_Previously_Insured': Age_Previously_Insured,
            'Previously_Insured_Vehicle_Damage': Previously_Insured_Vehicle_Damage,
            'Premium_Per_Vehicle_Age': Premium_Per_Vehicle_Age,
            'Premium_Per_Vintage': Premium_Per_Vintage,
            'Age_Bucket': Age_Bucket,
            'High_Premium': High_Premium,
            'Young_Driver': Young_Driver,
            'Old_Vehicle': Old_Vehicle,
            'Age_Premium_VehicleDamage': Age_Premium_VehicleDamage,
            'Region_Risk_Score': Region_Risk_Score,
            'Channel_Risk_Score': Channel_Risk_Score,
            'Age_Vintage': Age_Vintage,
            'Age_Policy_Channel': Age_Policy_Channel,
            'Premium_Squared': Premium_Squared,
            'Premium_Log': Premium_Log,
            'Vintage_Log': Vintage_Log,
            'Vintage_Premium': Vintage_Premium,
            'Premium_Per_Age': Premium_Per_Age,
            'Age_Squared': Age_Squared
        }])
        
        # Make prediction
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        # Status
        status = "Will Buy" if pred == 1 else "Will Not Buy"
        
        return InsurancePrediction(
            predicted_response=int(pred),
            probability=float(prob),
            status=status
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)