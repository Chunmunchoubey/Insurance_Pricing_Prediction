from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import pandas as pd
import os
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Insurance Response Prediction API",
    description="API for predicting if customer will buy vehicle insurance",
    version="1.0.0"
)

# Load model - Vehicle Insurance Model
MODEL_PATH = 'models/xgb_insurance_response_calibrated.joblib'

print(f"📁 Current directory: {os.getcwd()}")
print(f"📁 Looking for: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    if os.path.exists('models'):
        print("📂 models directory contents:")
        for file in os.listdir('models'):
            print(f"   - {file}")
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()
    raise

# Input model - Vehicle Insurance features
class InsuranceInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
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
    )
    
    Age: int = Field(..., ge=18, le=85, description="Age between 18-85")
    Gender: str = Field(..., description="Male or Female")
    Driving_License: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Region_Code: int = Field(..., ge=0, le=52, description="Region code 0-52")
    Previously_Insured: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Vehicle_Age: str = Field(..., description="< 1 Year, 1-2 Year, > 2 Years")
    Vehicle_Damage: str = Field(..., description="Yes or No")
    Annual_Premium: float = Field(..., ge=1000, le=600000, description="Annual premium amount")
    Policy_Sales_Channel: int = Field(..., ge=1, le=163, description="Sales channel 1-163")
    Vintage: int = Field(..., ge=10, le=300, description="Vintage days 10-300")

# Response model
class InsurancePrediction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_response": 1,
                "probability": 0.85,
                "status": "Will Buy",
                "threshold_used": 0.715
            }
        }
    )
    
    predicted_response: int = Field(..., description="1=Will Buy, 0=Will Not Buy")
    probability: float = Field(..., description="Probability of buying (0-1)")
    status: str = Field(..., description="Will Buy or Will Not Buy")
    threshold_used: float = Field(..., description="Threshold used for prediction")

@app.get("/")
def read_root():
    return {
        "message": "Vehicle Insurance Response Prediction API",
        "status": "active",
        "model": "XGBoost + SMOTE + Calibration",
        "endpoints": {
            "/predict": "POST - Predict if customer will buy insurance",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "current_directory": os.getcwd()
    }

@app.post("/predict", response_model=InsurancePrediction)
def predict_insurance(input_data: InsuranceInput):
    try:
        print(f"\n📥 Received request:")
        print(f"   Age: {input_data.Age}")
        print(f"   Gender: {input_data.Gender}")
        print(f"   Driving_License: {input_data.Driving_License}")
        print(f"   Region_Code: {input_data.Region_Code}")
        print(f"   Previously_Insured: {input_data.Previously_Insured}")
        print(f"   Vehicle_Age: {input_data.Vehicle_Age}")
        print(f"   Vehicle_Damage: {input_data.Vehicle_Damage}")
        print(f"   Annual_Premium: {input_data.Annual_Premium}")
        print(f"   Policy_Sales_Channel: {input_data.Policy_Sales_Channel}")
        print(f"   Vintage: {input_data.Vintage}")
        
        # Input validation
        if input_data.Gender not in ['Male', 'Female', 'male', 'female']:
            raise HTTPException(status_code=400, detail="Gender must be 'Male' or 'Female'")
        
        if input_data.Vehicle_Age not in ['< 1 Year', '1-2 Year', '> 2 Years', '< 1 year', '1-2 year', '> 2 years']:
            raise HTTPException(status_code=400, detail="Vehicle_Age must be '< 1 Year', '1-2 Year', or '> 2 Years'")
        
        if input_data.Vehicle_Damage not in ['Yes', 'No', 'yes', 'no']:
            raise HTTPException(status_code=400, detail="Vehicle_Damage must be 'Yes' or 'No'")
        
        # Encode categorical variables (same as training)
        gender_encoded = 1 if input_data.Gender.lower() == 'male' else 0
        vehicle_age_map = {'< 1 year': 0, '1-2 year': 1, '> 2 years': 2}
        vehicle_age_encoded = vehicle_age_map.get(input_data.Vehicle_Age.lower(), 0)
        vehicle_damage_encoded = 1 if input_data.Vehicle_Damage.lower() == 'yes' else 0
        
        # Feature engineering (same as training)
        Age_Vehicle_Age = input_data.Age * vehicle_age_encoded
        Premium_Per_Vehicle_Age = input_data.Annual_Premium / (vehicle_age_encoded + 1)
        Previously_Insured_Vehicle_Damage = input_data.Previously_Insured * vehicle_damage_encoded
        Age_Premium = input_data.Age * input_data.Annual_Premium
        Age_Previously_Insured = input_data.Age * input_data.Previously_Insured
        Premium_Per_Vintage = input_data.Annual_Premium / (input_data.Vintage + 1)
        
        # Region and Channel risk scores (approximate)
        region_risk_approx = 0.12
        channel_risk_approx = 0.12
        
        # Age bucket
        if input_data.Age <= 25:
            Age_Bucket = 0
        elif input_data.Age <= 35:
            Age_Bucket = 1
        elif input_data.Age <= 45:
            Age_Bucket = 2
        elif input_data.Age <= 55:
            Age_Bucket = 3
        else:
            Age_Bucket = 4
        
        High_Premium = 1 if input_data.Annual_Premium > 30000 else 0
        Young_Driver = 1 if input_data.Age < 30 else 0
        Old_Vehicle = 1 if vehicle_age_encoded == 2 else 0
        Age_Premium_VehicleDamage = input_data.Age * input_data.Annual_Premium * vehicle_damage_encoded
        
        # New features
        Premium_Per_Age = input_data.Annual_Premium / (input_data.Age + 1)
        Age_Policy_Channel = input_data.Age * input_data.Policy_Sales_Channel
        Vintage_Premium = input_data.Vintage * input_data.Annual_Premium
        Age_Vintage = input_data.Age * input_data.Vintage
        Premium_Log = np.log1p(input_data.Annual_Premium)
        Vintage_Log = np.log1p(input_data.Vintage)
        Age_Squared = input_data.Age ** 2
        Premium_Squared = input_data.Annual_Premium ** 2
        
        # Create DataFrame with ALL features
        input_df = pd.DataFrame([[
            input_data.Age,
            gender_encoded,
            input_data.Driving_License,
            input_data.Region_Code,
            input_data.Previously_Insured,
            vehicle_age_encoded,
            vehicle_damage_encoded,
            input_data.Annual_Premium,
            input_data.Policy_Sales_Channel,
            input_data.Vintage,
            Age_Vehicle_Age,
            Premium_Per_Vehicle_Age,
            Previously_Insured_Vehicle_Damage,
            Age_Premium,
            Age_Previously_Insured,
            Premium_Per_Vintage,
            region_risk_approx,
            channel_risk_approx,
            Age_Bucket,
            High_Premium,
            Young_Driver,
            Old_Vehicle,
            Age_Premium_VehicleDamage,
            Premium_Per_Age,
            Age_Policy_Channel,
            Vintage_Premium,
            Age_Vintage,
            Premium_Log,
            Vintage_Log,
            Age_Squared,
            Premium_Squared
        ]], columns=[
            'Age', 'Gender', 'Driving_License', 'Region_Code', 'Previously_Insured',
            'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage',
            'Age_Vehicle_Age', 'Premium_Per_Vehicle_Age', 'Previously_Insured_Vehicle_Damage',
            'Age_Premium', 'Age_Previously_Insured', 'Premium_Per_Vintage',
            'Region_Risk_Score', 'Channel_Risk_Score', 'Age_Bucket',
            'High_Premium', 'Young_Driver', 'Old_Vehicle', 'Age_Premium_VehicleDamage',
            'Premium_Per_Age', 'Age_Policy_Channel', 'Vintage_Premium', 'Age_Vintage',
            'Premium_Log', 'Vintage_Log', 'Age_Squared', 'Premium_Squared'
        ])
        
        print(f"📊 Input DataFrame shape: {input_df.shape}")
        print(f"📊 Columns: {input_df.columns.tolist()}")
        
        # Make prediction
        pred_proba = model.predict_proba(input_df)[0][1]
        threshold = 0.715  # Best threshold from training
        pred = 1 if pred_proba >= threshold else 0
        
        print(f"✅ Prediction: {pred} (Probability: {pred_proba:.2%}, Threshold: {threshold:.3f})")
        
        return InsurancePrediction(
            predicted_response=pred,
            probability=float(pred_proba),
            status="Will Buy" if pred == 1 else "Will Not Buy",
            threshold_used=threshold
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)