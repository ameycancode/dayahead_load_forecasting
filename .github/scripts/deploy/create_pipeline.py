#!/usr/bin/env python3
"""
SageMaker Pipeline Creation Script for Energy Load Forecasting
"""
import os
import sys
import boto3
import sagemaker
import traceback

try:
    from pipeline.orchestration.pipeline import (
        create_preprocessing_pipeline,
        create_training_pipeline,
        create_complete_pipeline
    )
    print("✅ Successfully imported pipeline functions")
except ImportError as e:
    print(f"❌ Failed to import pipeline functions: {e}")
    print("Python path:")
    for path in sys.path:
        print(f"  {path}")
    sys.exit(1)

def get_pipeline_parameters_from_env():
    """Get pipeline parameters from environment variables"""
   
    try:
        return {
            'customer_profile': os.environ['CUSTOMER_PROFILE'],
            'customer_segment': os.environ['CUSTOMER_SEGMENT'],
            'bucket': os.environ['S3_BUCKET'],
            'prefix': os.environ['S3_PREFIX'],
            'role': os.environ['SAGEMAKER_ROLE_ARN'],
            'pipeline_name': os.environ['PIPELINE_NAME'],
            'pipeline_type': os.environ.get('PIPELINE_TYPE', 'complete'),
            'days_delay': int(os.environ['DAYS_DELAY']),
            'use_reduced_features': os.environ['USE_REDUCED_FEATURES'].lower() == 'true',
            'meter_threshold': int(os.environ['METER_THRESHOLD']),
            'use_weather': os.environ['USE_WEATHER'].lower() == 'true',
            'use_solar': os.environ['USE_SOLAR'].lower() == 'true',
            'use_cache': os.environ['USE_CACHE'].lower() == 'true',
            'weather_cache': os.environ['WEATHER_CACHE'].lower() == 'true',
            'feature_selection_method': os.environ['FEATURE_SEL_METHOD'],
            'feature_count': int(os.environ['FEATURE_COUNT']),
            'correlation_threshold': float(os.environ['CORRELATION_THRESHOLD']) / 100,
            'hpo_method': os.environ['HPO_METHOD'],
            'hpo_max_evals': int(os.environ['HPO_MAX_EVALS']),
            'cv_folds': int(os.environ['CV_FOLDS']),
            'cv_gap_days': int(os.environ['CV_GAP_DAYS']),
            'enable_multi_model': os.environ['ENABLE_MULTI_MODEL'].lower() == 'true',
            'deploy_model': os.environ.get('DEPLOY_MODEL', 'true').lower() == 'true',
            'endpoint_name': os.environ.get('ENDPOINT_NAME'),
        }
    except KeyError as e:
        print(f"❌ Missing required environment variable: {e}")
        print("Available environment variables:")
        for key, value in os.environ.items():
            if any(k in key for k in ['CUSTOMER', 'PIPELINE', 'S3_', 'DAYS', 'FEATURE', 'HPO']):
                print(f"  {key}: {value}")
        raise

if __name__ == "__main__":
    print('=== PIPELINE CREATION SCRIPT STARTED ===')
   
    # Get parameters from environment variables instead of argument parsing
    params = get_pipeline_parameters_from_env()
   
    print(f"✓ Environment variables loaded successfully:")
    print(f"  Customer: {params['customer_profile']}-{params['customer_segment']}")
    print(f"  Pipeline: {params['pipeline_name']} ({params['pipeline_type']})")
    print(f"  S3: s3://{params['bucket']}/{params['prefix']}")
    print(f"  Training: {params['hpo_method']} HPO, {params['hpo_max_evals']} evals")

    try:    
        # Create pipeline using parameters
        if params['pipeline_type'] == "preprocessing":
            pipeline = create_preprocessing_pipeline(**params)
        elif params['pipeline_type'] == "training":
            pipeline = create_training_pipeline(**params)
        else:  # complete
            pipeline = create_complete_pipeline(**params)
       
        print('✓ Pipeline created successfully')
    except Exception as e:
        print(f"❌ Error creating pipeline: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
