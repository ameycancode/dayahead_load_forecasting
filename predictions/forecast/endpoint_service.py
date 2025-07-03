"""
SageMaker endpoint service module for energy load forecasting.
Handles invoking SageMaker endpoints and saving predictions.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Union

import boto3
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


def invoke_sagemaker_endpoint(endpoint_name: str, inference_df: pd.DataFrame) -> List[float]:
    """
    Invoke SageMaker endpoint with the inference dataframe.
   
    Args:
        endpoint_name: Name of the SageMaker endpoint
        inference_df: DataFrame with features for inference
       
    Returns:
        Array of predictions
    """
    try:
        # Convert dataframe to JSON format
        instances = inference_df.to_dict(orient='records')
        payload = {"instances": instances}
       
        # Create SageMaker runtime client
        runtime_client = boto3.client('sagemaker-runtime')
       
        # Invoke endpoint
        logger.info(f"Invoking SageMaker endpoint: {endpoint_name}")
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
       
        # Parse response
        result = json.loads(response['Body'].read().decode())
        predictions = result.get('predictions', [])
       
        logger.info(f"Received predictions for {len(predictions)} hours")
        return predictions
   
    except Exception as e:
        logger.error(f"Error invoking SageMaker endpoint: {e}")
        raise


def save_predictions(predictions: List[float], forecast_date: Union[str, datetime, pd.Timestamp]) -> str:
    """
    Save predictions to S3.
   
    Args:
        predictions: Array of predictions
        forecast_date: Date of the forecast
       
    Returns:
        S3 URI of saved predictions
    """
    try:
        # Convert forecast_date to datetime if needed
        if isinstance(forecast_date, str):
            forecast_date = pd.to_datetime(forecast_date)
       
        # Create DataFrame with predictions
        forecast_hours = pd.date_range(
            start=forecast_date.replace(hour=0, minute=0, second=0),
            periods=24,
            freq='H'
        )
       
        pred_df = pd.DataFrame({
            'datetime': forecast_hours,
            'forecast': predictions
        })
       
        # Add date and hour columns
        pred_df['date'] = pred_df['datetime'].dt.date
        pred_df['hour'] = pred_df['datetime'].dt.hour
       
        # Format the date for the S3 key
        date_str = forecast_date.strftime('%Y-%m-%d')
       
        # Create CSV string
        csv_data = pred_df.to_csv(index=False)
       
        # Save to S3
        s3_client = boto3.client('s3')
        predictions_key = f"{config.S3_PREFIX}/predictions/{date_str}_forecast.csv"
       
        s3_client.put_object(
            Bucket=config.S3_BUCKET,
            Key=predictions_key,
            Body=csv_data
        )
       
        s3_uri = f"s3://{config.S3_BUCKET}/{predictions_key}"
        logger.info(f"Saved predictions to {s3_uri}")
       
        return s3_uri
   
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise


def get_endpoint_metrics(endpoint_name: str) -> Dict:
    """
    Get metrics for a SageMaker endpoint.
   
    Args:
        endpoint_name: Name of the SageMaker endpoint
       
    Returns:
        Dictionary with endpoint metrics
    """
    try:
        # Create SageMaker client
        sagemaker_client = boto3.client('sagemaker')
       
        # Get endpoint description
        response = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
       
        # Extract metrics
        metrics = {
            'endpoint_status': response['EndpointStatus'],
            'created_time': response['CreationTime'],
            'last_modified_time': response['LastModifiedTime'],
            'production_variants': []
        }
       
        # Extract production variant details
        for variant in response['ProductionVariants']:
            metrics['production_variants'].append({
                'variant_name': variant['VariantName'],
                'current_instance_count': variant['CurrentInstanceCount'],
                'instance_type': variant['InstanceType']
            })
       
        return metrics
   
    except Exception as e:
        logger.error(f"Error getting endpoint metrics: {e}")
        return {
            'endpoint_status': 'Error',
            'error': str(e)
        }
