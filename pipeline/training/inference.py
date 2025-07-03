import json
import os
import pickle
import numpy as np
import pandas as pd

def model_fn(model_dir):
    """Load the model and features for inference."""
    # Load model
    model_path = os.path.join(model_dir, 'xgboost-model')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
   
    # Load features
    feature_path = os.path.join(model_dir, 'features.pkl')
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
   
    return {'model': model, 'features': features}

def input_fn(request_body, request_content_type):
    """Parse input data for prediction."""
    if request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
       
        # Handle both single instance and batch
        if isinstance(data, dict):
            # Single instance
            df = pd.DataFrame([data])
        else:
            # Batch of instances
            df = pd.DataFrame(data['instances'])
       
        # Convert datetime column if present
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
           
            # Extract datetime features
            df['hour'] = df['datetime'].dt.hour
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['day_of_year'] = df['datetime'].dt.dayofyear
       
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions using the loaded model."""
    model = model_dict['model']
    features = model_dict['features']
   
    # Filter to include only the features the model expects
    available_features = [f for f in features if f in input_data.columns]
   
    if len(available_features) < len(features):
        missing = [f for f in features if f not in input_data.columns]
        print(f"Warning: Missing {len(missing)} features for prediction: {missing[:5]}...")
   
    # Make predictions
    predictions = model.predict(input_data[available_features])
   
    return predictions

def output_fn(predictions, response_content_type):
    """Format predictions for response."""
    if response_content_type == 'application/json':
        return json.dumps({'predictions': predictions.tolist()})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
