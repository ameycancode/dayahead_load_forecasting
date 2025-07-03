"""
Utility functions for energy load forecasting.
Contains helper functions used across the application.
"""
import logging
import os
import pickle
from typing import List, Optional

import boto3
import pandas as pd

def setup_logging(log_dir: Optional[str] = None, log_level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
   
    Args:
        log_dir: Directory for log files (if None, log to console only)
        log_level: Logging level (defaults to INFO)
       
    Returns:
        Configured logger instance
    """
    if log_level is None:
        log_level = 'INFO'
       
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
   
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
   
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
   
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
   
    # File handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'forecast.log'))
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
   
    return logger


def get_model_features(run_id: Optional[str] = None, config=None) -> List[str]:
    """
    Get the list of features used by the model from S3.
   
    Args:
        run_id: The run ID of the model (if None, uses latest)
       
    Returns:
        List of feature names
    """
    try:
        s3_client = boto3.client('s3')
       
        # Determine features path based on run_id
        if run_id:
            features_key = f"{config.get('S3_PREFIX')}/models/{run_id}/features.pkl"
        else:
            features_key = f"{config.get('S3_PREFIX')}/models/features.pkl"
       
        logger.info(f"Loading model features from s3://{config.get('S3_BUCKET')}/{features_key}")
       
        try:
            # Download features file from S3
            response = s3_client.get_object(Bucket=config.get('S3_BUCKET'), Key=features_key)
            features = pickle.loads(response['Body'].read())
           
            logger.info(f"Loaded {len(features)} features from model artifacts")
            return features
           
        except Exception as e:
            logger.warning(f"Failed to load features from {features_key}: {e}")
           
            # Try to load from the models directory (without run_id)
            try:
                features_key = f"{config.get('S3_PREFIX')}/models/features.pkl"
                response = s3_client.get_object(Bucket=config.get('S3_BUCKET'), Key=features_key)
                features = pickle.loads(response['Body'].read())
               
                logger.info(f"Loaded {len(features)} features from fallback location")
                return features
               
            except Exception as e2:
                logger.error(f"Failed to load features from fallback location: {e2}")
                # Return common features as fallback
                return get_common_features(config)
   
    except Exception as e:
        logger.error(f"Error loading model features: {str(e)}")
        # Return common features as fallback
        return get_common_features(config)


def get_common_features(config=None) -> List[str]:
    """
    Get a list of commonly used features as fallback.
   
    Returns:
        List of common feature names
    """
    logger.warning("Using fallback common features")
   
    # Include basic time features
    features = [
        'hour', 'dayofweek', 'month', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend',
        'is_holiday', 'is_solar_window', 'is_solar_peak', 'is_evening_ramp'
    ]
   
    # Include weather features
    features.extend([
        'temperature_2m', 'apparent_temperature', 'cloudcover',
        'direct_radiation', 'diffuse_radiation', 'total_radiation',
        'windspeed_10m', 'relativehumidity_2m'
    ])
   
    # Include lag features
    lag_days = config.get('DEFAULT_LAG_DAYS')
    for lag_day in lag_days:
        lag_hours = lag_day * 24
        features.append(f'lossadjustedload_lag_{lag_hours}h')
        features.append(f'loadlal_lag_{lag_hours}h')
        features.append(f'genlal_lag_{lag_hours}h')
   
    # Include same hour/day of week and moving average features
    features.extend([
        'lossadjustedload_same_hour_dow_7d',
        'loadlal_same_hour_dow_7d',
        'genlal_same_hour_dow_7d',
        'lossadjustedload_ma_7d'
    ])
   
    return features


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.
   
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
       
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        logger.error("DataFrame is empty")
        return False
   
    missing_columns = [col for col in required_columns if col not in df.columns]
   
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
   
    return True


def get_s3_file_list(prefix: str, suffix: Optional[str] = None) -> List[str]:
    """
    Get a list of files in an S3 location.
   
    Args:
        prefix: S3 prefix to list
        suffix: Optional file suffix to filter by
       
    Returns:
        List of S3 keys
    """
    try:
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
       
        result = []
        for page in paginator.paginate(Bucket=config.get('S3_BUCKET'), Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if suffix is None or obj['Key'].endswith(suffix):
                        result.append(obj['Key'])
       
        return result
   
    except Exception as e:
        logger.error(f"Error listing S3 files: {e}")
        return []


# Initialize logger
logger = setup_logging()
