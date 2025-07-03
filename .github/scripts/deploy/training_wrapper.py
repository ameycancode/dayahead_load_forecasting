#!/usr/bin/env python
# training_wrapper.py
import argparse
import json
import logging
import os
import datetime
import pickle
import sys
import traceback
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.getcwd())

import boto3
import pandas as pd
import numpy as np


# Set up logging
def set_up_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
   
    # Remove existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
   
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
   
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
   
    logger.addHandler(console_handler)
   
    return logger


def get_training_config_from_env():
    """Get training configuration from environment variables and config file"""
   
    # Get basic training parameters from environment
    # customer_profile = os.environ.get('CUSTOMER_PROFILE', 'RES')
    # customer_segment = os.environ.get('CUSTOMER_SEGMENT', 'SOLAR')
    # customer_segment_key = f"{customer_profile}_{customer_segment}"
   
    config = {
        # 'customer_profile': customer_profile,
        # 'customer_segment': customer_segment,
        'customer_segment_key': os.environ.get('SM_HP_CUSTOMER_SEGMENT', 'RES_SOLAR'),
        'feature_selection_method': os.environ.get('SM_HP_FEATURE_SELECTION_METHOD', 'consensus'),
        'feature_count': int(os.environ.get('SM_HP_FEATURE_COUNT', 40)),
        'correlation_threshold': float(os.environ.get('SM_HP_CORRELATION_THRESHOLD', 85)) / 100,
        'hpo_method': os.environ.get('SM_HP_HPO_METHOD', 'optuna'),
        'hpo_max_evals': int(os.environ.get('SM_HP_HPO_MAX_EVALS', 5)),
        'cv_folds': int(os.environ.get('SM_HP_CV_FOLDS', 5)),
        'cv_gap_days': int(os.environ.get('SM_HP_CV_GAP_DAYS', 14)),
        'enable_multi_model': os.environ.get('SM_HP_ENABLE_MULTI_MODEL', 'false').lower() == 'true',
    }
   
    print("Training configuration from environment:")
    for key, value in config.items():
        print(f"  {key}: {value}")
   
    return config

def get_customer_specific_evaluation_config(customer_segment_key):
    """Get customer-specific evaluation periods and metric weights from config"""
   
    # Import the config module
    try:
        # Try to import from configs package
        from configs import config
       
        # Get customer-specific evaluation periods
        periods_to_evaluate = config.SEGMENT_EVALUATION_PERIODS.get(
            customer_segment_key,
            config.SEGMENT_EVALUATION_PERIODS.get("RES_SOLAR", {})
        )
       
        # Get customer-specific metric weights
        segment_metric_weights = config.SEGMENT_METRIC_WEIGHTS.get(
            customer_segment_key,
            config.SEGMENT_METRIC_WEIGHTS.get("RES_SOLAR", {})
        )
       
        # Get priority metrics for this segment
        priority_metrics = config.PRIORITY_METRICS.get(
            customer_segment_key,
            config.PRIORITY_METRICS.get("RES_SOLAR", [])
        )
       
        # Convert segment metric weights to HPO-compatible format
        # The config has period-specific weights, but HPO needs metric-specific weights
        hpo_metric_weights = convert_segment_weights_to_hpo_weights(
            segment_metric_weights,
            periods_to_evaluate,
            customer_segment_key
        )
       
        print(f"Customer-specific evaluation config for {customer_segment_key}:")
        print(f"  Evaluation periods: {list(periods_to_evaluate.keys())}")
        print(f"  Metric weights: {list(hpo_metric_weights.keys())}")
        print(f"  Priority metrics: {priority_metrics}")
       
        return {
            'periods_to_evaluate': periods_to_evaluate,
            'hpo_metric_weights': hpo_metric_weights,
            'segment_metric_weights': segment_metric_weights,
            'priority_metrics': priority_metrics
        }
       
    except ImportError as e:
        print(f"Warning: Could not import config module: {e}")
        print("Using default evaluation configuration")
        return get_default_evaluation_config()

def convert_segment_weights_to_hpo_weights(segment_weights, periods, customer_segment_key):
    """Convert segment-specific weights to HPO-compatible metric weights"""
   
    # Base metric weights that work for all segments
    base_weights = {
        "rmse_overall": 0.2,
        "mape_overall": 0.15,
        "r2_overall": 0.05
    }
   
    # Add period-specific weights based on customer segment configuration
    period_weights = {}
    remaining_weight = 0.6  # Reserve 60% for period-specific metrics
   
    if segment_weights:
        total_segment_weight = sum(segment_weights.values())
       
        for period, weight in segment_weights.items():
            # Normalize the weight and create RMSE and MAPE metrics for each period
            normalized_weight = (weight / total_segment_weight) * remaining_weight
           
            # Split weight between RMSE and MAPE for each period
            period_weights[f"rmse_{period}"] = normalized_weight * 0.6  # 60% to RMSE
            period_weights[f"mape_{period}"] = normalized_weight * 0.4  # 40% to MAPE
   
    # Combine base weights with period-specific weights
    hpo_weights = {**base_weights, **period_weights}
   
    # Ensure weights sum to 1.0
    total_weight = sum(hpo_weights.values())
    if total_weight > 0:
        hpo_weights = {k: v/total_weight for k, v in hpo_weights.items()}
   
    return hpo_weights

def get_default_evaluation_config():
    """Fallback evaluation configuration if config module is not available"""
   
    return {
        'periods_to_evaluate': {
            "solar_peak": (11, 14),
            "evening_ramp": (16, 20),
            "peak_demand": (17, 22),
            "morning_rise": (6, 9)
        },
        'hpo_metric_weights': {
            "rmse_overall": 0.25,
            "mape_overall": 0.15,
            "wape_overall": 0.15,
            "trans_weighted_error": 0.20,
            "rmse_evening_ramp": 0.1,
            "mape_evening_ramp": 0.1,
            "r2_overall": 0.05
        },
        'segment_metric_weights': {},
        'priority_metrics': []
    }

# Install required packages
def install_requirements():
    logger = logging.getLogger()
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "shap", "matplotlib", "seaborn"])
        logger.info("Additional packages installed successfully")
    except Exception as e:
        logger.warning(f"Warning: Could not install additional packages: {str(e)}")


# Setup Python path and import modules
def setup_modules():
    """Set up proper Python module structure in the container."""
    logger = logging.getLogger()
   
    try:
        # Current directory where code is located
        code_dir = '/opt/ml/code'
       
        # Create the module structure
        configs_dir = os.path.join(os.getcwd(), 'configs')
        pipeline_dir = os.path.join(os.getcwd(), 'pipeline')
        training_dir = os.path.join(pipeline_dir, 'training')
       
        # Create directories
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(pipeline_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)
       
        # Create __init__.py files
        with open(os.path.join(configs_dir, '__init__.py'), 'w') as f:
            f.write('# Configuration package\n')

        with open(os.path.join(pipeline_dir, '__init__.py'), 'w') as f:
            f.write('# Pipeline package\n')
       
        with open(os.path.join(training_dir, '__init__.py'), 'w') as f:
            f.write('# Training modules\n')
       
        # Copy module files from code directory to package structure
        module_files = {
            'config.py': os.path.join(configs_dir, 'config.py'),
            'model.py': os.path.join(training_dir, 'model.py'),
            'feature_selection.py': os.path.join(training_dir, 'feature_selection.py'),
            'hyperparameter_optimization.py': os.path.join(training_dir, 'hyperparameter_optimization.py'),
            'evaluation.py': os.path.join(training_dir, 'evaluation.py'),
            'visualization.py': os.path.join(training_dir, 'visualization.py'),
        }
       
        for source_file, dest_path in module_files.items():
            src_path = os.path.join(code_dir, source_file)
            if os.path.exists(src_path):
                with open(src_path, 'r') as src, open(dest_path, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Copied {source_file} to {dest_path}")
            else:
                logger.warning(f"Source file not found: {src_path}")
       
        # Add parent directory to Python path to allow imports
        sys.path.insert(0, os.getcwd())
        logger.info(f"Added {os.getcwd()} to Python path")
       
        # List the directories to verify setup
        logger.info(f"Contents of {os.getcwd()}: {os.listdir(os.getcwd())}")
        if os.path.exists(pipeline_dir):
            logger.info(f"Contents of {pipeline_dir}: {os.listdir(pipeline_dir)}")
        if os.path.exists(training_dir):
            logger.info(f"Contents of {training_dir}: {os.listdir(training_dir)}")
        if os.path.exists(configs_dir):
            logger.info(f"Contents of {configs_dir}: {os.listdir(configs_dir)}")
           
        logger.info("Module setup completed successfully")
        return True
       
    except Exception as e:
        logger.error(f"Error setting up modules: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def upload_to_s3(local_path, bucket, prefix, filename):
    """Upload a file to S3."""
    logger = logging.getLogger()
    s3_client = boto3.client('s3')
    s3_key = f"{prefix}/models/{filename}"
   
    try:
        s3_client.upload_file(local_path, bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
        return f"s3://{bucket}/{s3_key}"
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return None


def save_model_for_sagemaker(model, features, model_dir):
    """
    Save the model for SageMaker deployment following SageMaker's best practices.
   
    Args:
        model: The XGBoost model (either XGBRegressor or Booster)
        features: List of selected features
        model_dir: Directory to save files
    """
    import os
    import pickle
    import xgboost as xgb
    import tarfile
    from xgboost.sklearn import XGBModel  # Base class for XGBRegressor/XGBClassifier
    import logging
   
    logger = logging.getLogger()
   
    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)
   
    try:
        # 1. Save the model in the SageMaker XGBoost container's expected format
        # SageMaker XGBoost containers prefer pickle for sklearn wrapper models
        if hasattr(model, 'get_booster'):  # Check if it's a sklearn wrapper
            logger.info("Saving scikit-learn XGBoost model with pickle")
            model_file = os.path.join(model_dir, 'xgboost-model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        else:  # It's a native booster
            logger.info("Saving native XGBoost model")
            model_file = os.path.join(model_dir, 'xgboost-model.json')
            model.save_model(model_file)  # Save in JSON format for better compatibility
           
        # 2. Save features list
        features_file = os.path.join(model_dir, 'features.pkl')
        with open(features_file, 'wb') as f:
            pickle.dump(features, f)
           
        # 3. Create a custom inference script that can handle both model types
        inference_script = """
import json
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def model_fn(model_dir):
    \"\"\"Load the model and features for inference.\"\"\"
    logger.info(f"Loading model from {model_dir}")
   
    # List all files in the model directory to help with debugging
    logger.info(f"Files in model dir: {os.listdir(model_dir)}")
   
    # Try loading with pickle first - for scikit-learn wrapper
    try:
        model_path = os.path.join(model_dir, 'xgboost-model.pkl')
        if os.path.exists(model_path):
            logger.info(f"Loading scikit-learn model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model type: {type(model)}")
            # Use the model's predict method directly
        else:
            logger.info(f"File not found: {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        logger.info(f"Failed to load sklearn model: {str(e)}")
       
        # Try loading native XGBoost model
        try:
            model_path = os.path.join(model_dir, 'xgboost-model.json')
            if os.path.exists(model_path):
                logger.info(f"Loading native XGBoost model from {model_path}")
                model = xgb.Booster()
                model.load_model(model_path)
            else:
                logger.info(f"File not found: {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e2:
            logger.error(f"Failed to load XGBoost model: {str(e2)}")
            raise RuntimeError(f"Could not load model: {str(e)} | {str(e2)}")
   
    # Load features
    try:
        feature_path = os.path.join(model_dir, 'features.pkl')
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
        logger.info(f"Loaded {len(features)} features: {features[:5]}")
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        # Continue without features - we'll handle this in predict_fn
        features = []
   
    return {'model': model, 'features': features}

def input_fn(request_body, request_content_type='application/json'):
    \"\"\"Parse input data for prediction.\"\"\"
    logger.info(f"Processing input with content type: {request_content_type}")
   
    if request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
        logger.info(f"Received data type: {type(data)}")
       
        # Handle both single instance and batch
        if isinstance(data, dict) and not "instances" in data:
            # Single instance
            logger.info("Single instance mode")
            df = pd.DataFrame([data])
        elif isinstance(data, dict) and "instances" in data:
            # Batch with instances key
            logger.info(f"Batch mode with {len(data['instances'])} instances")
            df = pd.DataFrame(data['instances'])
        else:
            # Assume it's a list of instances
            logger.info(f"List mode with {len(data)} instances")
            df = pd.DataFrame(data)
       
        logger.info(f"Created DataFrame with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
       
        # Convert datetime column if present
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
           
            # Extract datetime features
            df['hour'] = df['datetime'].dt.hour
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['day_of_year'] = df['datetime'].dt.dayofyear
            logger.info("Processed datetime features")
       
        return df
    else:
        logger.error(f"Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    \"\"\"Make predictions using the loaded model.\"\"\"
    model = model_dict['model']
    features = model_dict.get('features', [])
   
    logger.info(f"Making predictions with data shape: {input_data.shape}")
   
    try:
        # If we have features, filter the input data
        if features:
            # Filter to include only the features the model expects
            available_features = [f for f in features if f in input_data.columns]
           
            if len(available_features) < len(features):
                missing = [f for f in features if f not in input_data.columns]
                logger.warning(f"Missing {len(missing)} features for prediction. First few: {missing[:5]}")
           
            logger.info(f"Using {len(available_features)} features for prediction")
            prediction_data = input_data[available_features]
        else:
            # No features list - use all columns
            logger.info("No feature list provided, using all columns")
            prediction_data = input_data
       
        # Check if it's a scikit-learn model or a native booster
        if hasattr(model, 'predict'):
            logger.info("Using scikit-learn predict method")
            predictions = model.predict(prediction_data)
        else:
            # It's a native booster
            logger.info("Using XGBoost Booster predict method")
            dmatrix = xgb.DMatrix(prediction_data)
            predictions = model.predict(dmatrix)
       
        logger.info(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        logger.info(f"First few predictions: {predictions[:5]}")
       
        return predictions
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def output_fn(predictions, response_content_type='application/json'):
    \"\"\"Format predictions for response.\"\"\"
    logger.info(f"Formatting output with content type: {response_content_type}")
   
    if response_content_type == 'application/json':
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        elif isinstance(predictions, list):
            predictions_list = predictions
        else:
            predictions_list = [predictions]
           
        logger.info(f"Returning {len(predictions_list)} predictions")
        return json.dumps({'predictions': predictions_list})
    else:
        logger.error(f"Unsupported content type: {response_content_type}")
        raise ValueError(f"Unsupported content type: {response_content_type}")
"""
        inference_path = os.path.join(model_dir, 'inference.py')
        with open(inference_path, 'w') as f:
            f.write(inference_script)
       
        # 4. Create setup.py file for the package
        setup_py = """
from setuptools import setup, find_packages

setup(
    name="inference",
    version="1.0.0",
    packages=find_packages(include=["."]),
    py_modules=["inference"],
    install_requires=[
        "pandas",
        "numpy",
        "xgboost"
    ]
)
"""
        setup_path = os.path.join(model_dir, 'setup.py')
        with open(setup_path, 'w') as f:
            f.write(setup_py)
           
        # 5. Create a tarball with all the files
        tarball_path = os.path.join(model_dir, 'model.tar.gz')
        with tarfile.open(tarball_path, 'w:gz') as tar:
            # Add model file with the correct name
            if hasattr(model, 'get_booster'):
                tar.add(model_file, arcname='xgboost-model.pkl')
            else:
                tar.add(model_file, arcname='xgboost-model.json')
               
            tar.add(features_file, arcname='features.pkl')
            tar.add(inference_path, arcname='inference.py')
            tar.add(setup_path, arcname='setup.py')
           
        logger.info(f"Model tarball created at {tarball_path}")
       
        # Return the tarball path
        return tarball_path
       
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def save_model_as_tarball(model, features, output_path):
    """
    Save the model and features as a tarball that SageMaker can deploy.
   
    Args:
        model: The model (either scikit-learn XGBRegressor or XGBoost Booster)
        features: List of selected features
        output_path: Path to save the model tarball
    """
    import os
    import tarfile
    import tempfile
    import pickle
    import xgboost as xgb
    from xgboost.sklearn import XGBRegressor, XGBClassifier
   
    logger = logging.getLogger()
    logger.info(f"Saving model artifacts as tarball to {output_path}")
   
    # Create a temporary directory to store model files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Determine the type of model and save accordingly
        model_path = os.path.join(tmp_dir, 'xgboost-model')
       
        # Check if it's a scikit-learn wrapper or a native Booster
        if isinstance(model, (XGBRegressor, XGBClassifier)):
            logger.info("Saving scikit-learn XGBoost model")
            # Get the underlying booster
            booster = model.get_booster()
            # Save in binary format (not JSON)
            booster.save_model(model_path)
        elif isinstance(model, xgb.Booster):
            logger.info("Saving native XGBoost Booster model")
            # Save in binary format (not JSON)
            model.save_model(model_path)
        else:
            logger.info(f"Unknown model type {type(model)}, using pickle")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
       
        # Save the features
        features_path = os.path.join(tmp_dir, 'features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
       
        # Create inference.py script in the temp directory
        inference_script = """
import json
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

def model_fn(model_dir):
    \"\"\"Load the model and features for inference.\"\"\"
    import xgboost as xgb
    import pickle
    import os
   
    # Try loading with native XGBoost first
    try:
        model_path = os.path.join(model_dir, 'xgboost-model')
        model = xgb.Booster()
       
        # Try loading in binary format
        model.load_model(model_path)
        print("Successfully loaded XGBoost model in binary format")
    except Exception as e:
        print(f"Error loading with native XGBoost: {str(e)}")
       
        try:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
           
            # Check if it's a scikit-learn wrapper
            if hasattr(model, 'get_booster'):
                model = model.get_booster()
                print("Loaded scikit-learn XGBoost model and extracted booster")
           
            print("Successfully loaded model with pickle")
        except Exception as e2:
            print(f"Error loading with pickle: {str(e2)}")
            raise RuntimeError(f"Could not load model: {str(e)} | {str(e2)}")
   
    # Load features
    try:
        feature_path = os.path.join(model_dir, 'features.pkl')
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
        print(f"Successfully loaded features: {len(features)} features")
    except Exception as e:
        print(f"Error loading features: {str(e)}")
        raise
   
    return {'model': model, 'features': features}


def input_fn(request_body, request_content_type):
    \"\"\"Parse input data for prediction.\"\"\"
    if request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
       
        # Handle both single instance and batch
        if isinstance(data, dict):
            # Single instance
            df = pd.DataFrame([data])
        elif 'instances' in data:
            # Batch mode with 'instances' key
            df = pd.DataFrame(data['instances'])
        else:
            # Regular batch mode
            df = pd.DataFrame(data)
       
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
    \"\"\"Make predictions using the loaded model.\"\"\"
    model = model_dict['model']
    features = model_dict['features']
   
    # Filter to include only the features the model expects
    available_features = [f for f in features if f in input_data.columns]
   
    if len(available_features) < len(features):
        missing = [f for f in features if f not in input_data.columns]
        print(f"Warning: Missing {len(missing)} features for prediction")
   
    # Convert DataFrame to DMatrix for XGBoost prediction
    dmatrix = xgb.DMatrix(input_data[available_features])
   
    # Make predictions
    predictions = model.predict(dmatrix)
   
    return predictions

def output_fn(predictions, response_content_type):
    \"\"\"Format predictions for response.\"\"\"
    if response_content_type == 'application/json':
        return json.dumps({'predictions': predictions.tolist()})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
"""
       
        inference_path = os.path.join(tmp_dir, 'inference.py')
        with open(inference_path, 'w') as f:
            f.write(inference_script)
           
        # Create setup.py file for the package
        setup_py_content = """
from setuptools import setup, find_packages

setup(
    name="inference",
    version="1.0.0",
    packages=find_packages(include=["."]),
    py_modules=["inference"],
    install_requires=[
        "pandas",
        "numpy",
        "xgboost"
    ]
)
"""
        setup_path = os.path.join(tmp_dir, 'setup.py')
        with open(setup_path, 'w') as f:
            f.write(setup_py_content)
       
        # Create a tarball containing all the files
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(model_path, arcname='xgboost-model')
            tar.add(features_path, arcname='features.pkl')
            tar.add(inference_path, arcname='inference.py')
            tar.add(setup_path, arcname='setup.py')
       
        logger.info(f"Model tarball created successfully at {output_path}")


# Copy needed modules for inference
def copy_inference_modules():
    logger = logging.getLogger()
   
    # Source directory for module files
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/processing/input/model')
   
    # Copy training modules for inference
    inference_modules = ['model.py', 'evaluation.py', 'feature_selection.py', 'visualization.py']
   
    for module in inference_modules:
        src_path = f'pipeline/training/{module}'
        dst_path = os.path.join(model_dir, module)
       
        if os.path.exists(src_path):
            with open(src_path, 'r') as src:
                with open(dst_path, 'w') as dst:
                    dst.write(src.read())
            logger.info(f"Copied {module} to model directory for inference")
   
    # Create inference.py script in model directory
    inference_script = """
import json
import os
import pickle
import numpy as np
import pandas as pd

def model_fn(model_dir):
    \"\"\"Load the model and features for inference.\"\"\"
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
    \"\"\"Parse input data for prediction.\"\"\"
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
    \"\"\"Make predictions using the loaded model.\"\"\"
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
    \"\"\"Format predictions for response.\"\"\"
    if response_content_type == 'application/json':
        return json.dumps({'predictions': predictions.tolist()})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
"""
   
    inference_path = os.path.join(model_dir, 'inference.py')
    with open(inference_path, 'w') as f:
        f.write(inference_script)
   
    logger.info("Created inference.py in model directory")


def get_csv_file_path(directory, target_filename="train.csv"):
    """
    Find a specific CSV file in a directory, defaulting to standard filenames.
   
    Args:
        directory: Directory path to search in
        target_filename: Target filename to look for (default: "train.csv")
       
    Returns:
        Full path to the file if found, None otherwise
    """
    logger = logging.getLogger()
   
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return None
   
    # List all files in directory
    all_files = os.listdir(directory)
    logger.info(f"Found {len(all_files)} files in {directory}: {all_files}")
   
    # First, look for exact match
    if target_filename in all_files:
        file_path = os.path.join(directory, target_filename)
        logger.info(f"Found exact match: {file_path}")
        return file_path
   
    # Then, look for case-insensitive match
    for file in all_files:
        if file.lower() == target_filename.lower():
            file_path = os.path.join(directory, file)
            logger.info(f"Found case-insensitive match: {file_path}")
            return file_path
   
    # Finally, look for any CSV file matching the pattern (without case sensitivity)
    target_base = target_filename.split('.')[0].lower()
    for file in all_files:
        if file.lower().endswith('.csv') and target_base in file.lower():
            file_path = os.path.join(directory, file)
            logger.info(f"Found partial match: {file_path}")
            return file_path
   
    logger.warning(f"No matching file found for {target_filename} in {directory}")
    return None


def save_results_to_csv(train_df, test_df, model, features, target, output_dir, run_id):
    """Save prediction results to CSV files in S3 bucket."""
    try:
        import pandas as pd
        import boto3

        logger = logging.getLogger()
       
        # Get S3 information
        bucket = os.environ.get('SM_HP_S3_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-data')
        prefix = os.environ.get('SM_HP_S3_PREFIX', 'res-load-forecasting')
       
        # Create predictions
        train_df = train_df.copy()
        test_df = test_df.copy()
       
        train_df['prediction'] = model.predict(train_df[features])
        test_df['prediction'] = model.predict(test_df[features])
       
        # Calculate errors
        train_df['error'] = train_df[target] - train_df['prediction']
        train_df['abs_error'] = abs(train_df['error'])
        train_df['percent_error'] = 100 * train_df['error'] / train_df[target].replace(0, 1e-10)
        train_df['abs_percent_error'] = abs(train_df['percent_error'])
       
        test_df['error'] = test_df[target] - test_df['prediction']
        test_df['abs_error'] = abs(test_df['error'])
        test_df['percent_error'] = 100 * test_df['error'] / test_df[target].replace(0, 1e-10)
        test_df['abs_percent_error'] = abs(test_df['percent_error'])
       
        # Save to local CSV files
        train_csv_path = os.path.join(output_dir, 'train_predictions.csv')
        test_csv_path = os.path.join(output_dir, 'test_predictions.csv')
       
        # Select only essential columns to keep file sizes reasonable
        columns_to_save = ['datetime', target, 'prediction', 'error', 'abs_error',
                          'percent_error', 'abs_percent_error']
        if 'hour' in train_df.columns:
            columns_to_save.append('hour')
        if 'dayofweek' in train_df.columns:
            columns_to_save.append('dayofweek')
       
        # Keep only columns that exist
        train_columns = [col for col in columns_to_save if col in train_df.columns]
        test_columns = [col for col in columns_to_save if col in test_df.columns]
       
        train_df[train_columns].to_csv(train_csv_path, index=False)
        test_df[test_columns].to_csv(test_csv_path, index=False)
       
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.upload_file(train_csv_path, bucket, f"{prefix}/output/run_{run_id}/train_predictions.csv")
        s3_client.upload_file(test_csv_path, bucket, f"{prefix}/output/run_{run_id}/test_predictions.csv")
       
        logger.info(f"Saved train and test predictions to S3 bucket {bucket}/{prefix}/output/run_{run_id}/")
       
    except Exception as e:
        logger.error(f"Error saving prediction results to CSV: {str(e)}")


# Main function for training
def train():
    """Training entry point expected by the SageMaker XGBoost container"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
   
    logger.info("Starting energy load forecasting training")

    # Setup proper module structure
    if not setup_modules():
        logger.error("Failed to set up module structure - aborting training")
        sys.exit(1)

    # Get training configuration from environment
    training_config = get_training_config_from_env()
   
    # Get customer-specific evaluation configuration
    eval_config = get_customer_specific_evaluation_config(training_config['customer_segment_key'])


    # Install additional requirements
    install_requirements()
   
    # Get data paths from environment variables
    train_path = os.environ.get('SM_CHANNEL_TRAIN')
    validation_path = os.environ.get('SM_CHANNEL_VALIDATION')
    test_path = os.environ.get('SM_CHANNEL_TEST')
    model_dir = os.environ.get('SM_MODEL_DIR')
   
    logger.info(f"Train data path: {train_path}")
    logger.info(f"Validation data path: {validation_path}")
    logger.info(f"Test data path: {test_path}")
    logger.info(f"Model directory: {model_dir}")
   
    try:
        from pipeline.training.feature_selection import select_features
        from pipeline.training.model import train_model, cross_validate_model, save_model
        from pipeline.training.hyperparameter_optimization import optimize_hyperparameters
        from pipeline.training.evaluation import evaluate_model

        # Load the data
        train_file = get_csv_file_path(train_path, "train.csv")
        val_file = get_csv_file_path(validation_path, "validation.csv")
        test_file = get_csv_file_path(test_path, "test.csv")
       
        # Check if all files were found
        if not train_file or not val_file or not test_file:
            missing_files = []
            if not train_file: missing_files.append("train.csv")
            if not val_file: missing_files.append("validation.csv")
            if not test_file: missing_files.append("test.csv")
           
            error_msg = f"Could not find required data files: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
       
        # Load data files
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
       
        # Convert datetime columns
        datetime_column = 'datetime'
        datetime_weather_column = 'datetime_weather'
        date_column = 'date'
        target = 'lossadjustedload'
       
        for df in [train_df, val_df, test_df]:
            if datetime_column in df.columns:
                df[datetime_column] = pd.to_datetime(df[datetime_column])
            if datetime_weather_column in df.columns:
                df[datetime_weather_column] = pd.to_datetime(df[datetime_weather_column])
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
       
        logger.info(f"Loaded {len(train_df)} training samples")
        logger.info(f"Loaded {len(val_df)} validation samples")
        logger.info(f"Loaded {len(test_df)} test samples")
       
        # Combine train and validation for feature selection
        combined_df = pd.concat([train_df, val_df], ignore_index=True)

        # Run feature selection
        logger.info(f"Running feature selection with {training_config['feature_selection_method']} method")
        selected_features, feature_metadata = select_features(
            combined_df,
            target=target,
            method=training_config['feature_selection_method'],
            n_features=training_config['feature_count'],
            correlation_threshold=training_config['correlation_threshold']
        )
       
        logger.info(f"Selected {len(selected_features)} features")
       
        # Run hyperparameter optimization
        logger.info(f"Running hyperparameter optimization with {training_config['hpo_method']}")
        logger.info(f"Using customer-specific evaluation periods: {list(eval_config['periods_to_evaluate'].keys())}")
        logger.info(f"Using customer-specific metric weights for {training_config['customer_segment_key']}")


        #  Generate a run ID for this training run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Generated run ID: {run_id}")
       
        best_params, optimization_results = optimize_hyperparameters(
            combined_df,
            selected_features,
            target=target,
            date_column=datetime_column,
            n_trials=training_config['hpo_max_evals'],
            n_splits=training_config['cv_folds'],
            initial_train_months=17,
            validation_months=6,
            step_months=3,
            custom_periods=eval_config['periods_to_evaluate'],  # Customer-specific periods
            metric_weights=eval_config['hpo_metric_weights'],   # Customer-specific weights
            output_dir=model_dir,  #  Save plots here
            run_id=run_id,          #  Use run ID for S3 paths
        )
       
        logger.info("Hyperparameter optimization completed")
        logger.info(f"Best parameters: {best_params}")
       
        # Train final model
        logger.info("Training final model")
        final_model, train_metrics = train_model(
            combined_df,
            selected_features,
            target=target,
            date_column=datetime_column,
            model_params=best_params,
            output_dir=model_dir,
            customer_segment=training_config['customer_segment_key'],
            run_id=run_id,
            create_plots=True
        )
       
        # Evaluate on test data
        logger.info("Evaluating model on test data")
        evaluation_results = evaluate_model(
            final_model,
            test_df,
            selected_features,
            target=target,
            date_column=datetime_column,
            output_dir=model_dir,  # Save plots to model dir
            customer_segment=training_config['customer_segment_key'],
            periods=eval_config['periods_to_evaluate'],
            create_plots=True,
            prefix="final_model",
            run_id=run_id
        )

        # Log customer-specific evaluation results
        logger.info(f"Model evaluation completed for {training_config['customer_segment_key']}")
        logger.info(f"Priority metrics performance:")
        for metric in eval_config['priority_metrics']:
            if metric in evaluation_results:
                logger.info(f"  {metric}: {evaluation_results[metric]:.4f}")

        # Save prediction results as CSV
        save_results_to_csv(
            train_df=combined_df,
            test_df=test_df,
            model=final_model,
            features=selected_features,
            target=target,
            output_dir=model_dir,
            run_id=run_id
        )
       
        # # Save model and metadata to model_dir
        # model_tarball_path = os.path.join(model_dir, 'model.tar.gz')
        # save_model_as_tarball(final_model, selected_features, model_tarball_path)

        # model_path = os.path.join(model_dir, 'xgboost-model')
        # final_model.save_model(model_path)

        # Save model and metadata to model_dir
        model_tarball_path = save_model_for_sagemaker(final_model, selected_features, model_dir)

        # Still save individual files for backward compatibility
        if hasattr(final_model, 'get_booster'):
            # It's a scikit-learn wrapper
            model_path = os.path.join(model_dir, 'xgboost-model')
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)
        else:
            # It's a native booster
            model_path = os.path.join(model_dir, 'xgboost-model')
            final_model.save_model(model_path)

        # Save feature list
        feature_path = os.path.join(model_dir, 'features.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(selected_features, f)
           
        # Save evaluation results
        eval_path = os.path.join(model_dir, 'evaluation.json')
        with open(eval_path, 'w') as f:
            # Convert numpy values to Python types
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                else:
                    return obj
           
            json.dump(convert_numpy(evaluation_results), f, indent=2)
       
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Features saved to {feature_path}")
        logger.info(f"Evaluation results saved to {eval_path}")
        logger.info(f"SageMaker model tarball saved to {model_tarball_path}")

        # Now upload to S3
        # Get bucket and prefix from environment variables or default values
        bucket = os.environ.get('SM_HP_S3_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-data')
        prefix = os.environ.get('SM_HP_S3_PREFIX', 'res-load-forecasting')
        # run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
       
        # Upload model
        s3_model_path = upload_to_s3(model_path, bucket, prefix, f"run_{run_id}/xgboost-model")
       
        # Upload features
        s3_features_path = upload_to_s3(feature_path, bucket, prefix, f"run_{run_id}/features.pkl")
       
        # Upload evaluation results
        s3_eval_path = upload_to_s3(eval_path, bucket, prefix, f"run_{run_id}/evaluation.json")

        # Upload model tarball
        s3_model_tarball_path = upload_to_s3(model_tarball_path, bucket, prefix, f"run_{run_id}/model.tar.gz")
       
        # Upload a model-info.json file with paths to all artifacts
        model_info = {
            "model_path": s3_model_path,
            "model_tarball_path": s3_model_tarball_path,
            "features_path": s3_features_path,
            "evaluation_path": s3_eval_path,
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id
        }
       
        model_info_path = os.path.join(model_dir, 'model-info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
       
        # Upload model info
        s3_model_info_path = upload_to_s3(model_info_path, bucket, prefix, f"run_{run_id}/model-info.json")
       
        logger.info(f"Model and artifacts uploaded to S3:")
        logger.info(f"- Model: {s3_model_path}")
        logger.info(f"- Model tarball: {s3_model_tarball_path}")
        logger.info(f"- Features: {s3_features_path}")
        logger.info(f"- Evaluation: {s3_eval_path}")
        logger.info(f"- Model Info: {s3_model_info_path}")
       
        # Return metrics as dictionary
        return {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "feature_count": len(selected_features),
            "rmse": evaluation_results["metrics"]["rmse"],
            "mape": evaluation_results["metrics"]["mape"],
            "smape": evaluation_results["metrics"]["smape"],
            "wape": evaluation_results["metrics"]["wape"],
            "r2": evaluation_results["metrics"]["r2"]
        }
       
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    train()
