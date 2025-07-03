#!/usr/bin/env python3
"""
SageMaker Pipeline Execution Script for Energy Load Forecasting
"""
import os
import sys
import traceback

try:
    from pipeline.orchestration.pipeline import (
        execute_preprocessing_pipeline,
        execute_training_pipeline,
        execute_complete_pipeline
    )
    print(' Successfully imported pipeline execution functions')
except ImportError as e:
    print(f' Failed to import pipeline execution functions: {e}')
    print('Python path:')
    for path in sys.path:
        print(f'  {path}')
    print('Available directories:')
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                print(f'{subindent}{file}')
    sys.exit(1)


def get_execution_parameters_from_env():
    """Get execution parameters from environment variables"""
   
    try:
        # Get basic parameters
        customer_profile = os.environ['CUSTOMER_PROFILE']
        customer_segment = os.environ['CUSTOMER_SEGMENT']
        profile_segment = f'{customer_profile}_{customer_segment}'
       
        params = {
            'customer_profile': customer_profile,
            'customer_segment': customer_segment,
            'profile_segment': profile_segment,
            'bucket': os.environ['S3_BUCKET'],
            'prefix': os.environ['S3_PREFIX'],
            'role': os.environ['SAGEMAKER_ROLE_ARN'],
            'pipeline_name': os.environ['PIPELINE_NAME'],
            'pipeline_type': os.environ['PIPELINE_TYPE'],
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
       
        return params
       
    except KeyError as e:
        print(f"❌ Missing required environment variable: {e}")
        raise

if __name__ == "__main__":
    print('=== PIPELINE EXECUTION SCRIPT STARTED ===')
   
    # Get parameters from environment variables
    params = get_execution_parameters_from_env()
   
    print(f"✓ Environment variables parsed successfully")
    print(f"  Pipeline Name: {params['pipeline_name']}")
    print(f"  Pipeline Type: {params['pipeline_type']}")
    print(f"  Customer: {params['customer_profile']}-{params['customer_segment']}")
    print(f"  S3 Bucket: {params['bucket']}")
    print(f"  S3 Prefix: {params['prefix']}")

    # Execute appropriate pipeline based on type
    try:
        print(f'Executing {params["pipeline_type"]} pipeline...')
       
        if params['pipeline_type'] == 'preprocessing':
            print('Calling execute_preprocessing_pipeline...')
            execution_arn = execute_preprocessing_pipeline(
                pipeline_name=params['pipeline_name'],
                parameters={
                    'CustomerProfile': params['customer_profile'],
                    'CustomerSegment': params['profile_segment'],
                    'DaysDelay': params['days_delay'],
                    'UseReducedFeatures': params['use_reduced_features'],
                    'MeterThreshold': params['meter_threshold'],
                    'UseCache': params['use_cache'],
                    'UseWeather': params['use_weather'],
                    'UseSolar': params['use_solar'],
                    'WeatherCache': params['weather_cache'],
                    'QueryLimit': -1
                }
            )
        elif params['pipeline_type'] == 'training':
            print('Calling execute_training_pipeline...')
            execution_arn = execute_training_pipeline(
                pipeline_name=params['pipeline_name'],
                parameters={
                    'FeatureSelectionMethod': params['feature_selection_method'],
                    'FeatureCount': params['feature_count'],
                    'CorrelationThreshold': int(params['correlation_threshold'] * 100),
                    'HPOMethod': params['hpo_method'],
                    'HPOMaxEvals': params['hpo_max_evals'],
                    'CVFolds': params['cv_folds'],
                    'CVGapDays': params['cv_gap_days'],
                    'EnableMultiModel': params['enable_multi_model'],
                    'ModelName': f'energy-forecasting-{os.environ["ENV_NAME"]}-{os.environ["GITHUB_RUN_ID"]}'
                }
            )
        else:  # Complete pipeline
            print('Calling execute_complete_pipeline...')
            execution_arn = execute_complete_pipeline(
                pipeline_name=params['pipeline_name'],
                parameters={
                    'CustomerProfile': params['customer_profile'],
                    'CustomerSegment': params['profile_segment'],
                    'DaysDelay': params['days_delay'],
                    'UseReducedFeatures': params['use_reduced_features'],
                    'MeterThreshold': params['meter_threshold'],
                    'UseCache': params['use_cache'],
                    'UseWeather': params['use_weather'],
                    'UseSolar': params['use_solar'],
                    'WeatherCache': params['weather_cache'],
                    'FeatureSelectionMethod': params['feature_selection_method'],
                    'FeatureCount': params['feature_count'],
                    'CorrelationThreshold': int(params['correlation_threshold'] * 100),
                    'HPOMethod': params['hpo_method'],
                    'HPOMaxEvals': params['hpo_max_evals'],
                    'CVFolds': params['cv_folds'],
                    'CVGapDays': params['cv_gap_days'],
                    'EnableMultiModel': params['enable_multi_model'],
                    'ModelName': f'energy-forecasting-{os.environ["ENV_NAME"]}-{os.environ["GITHUB_RUN_ID"]}'
                }
            )

        print(f'✓ Pipeline execution started successfully')
        print(f'EXECUTION_ARN={execution_arn}')

    except Exception as e:
        print(f'❌ Error executing pipeline: {str(e)}')
        traceback.print_exc()
        sys.exit(1)

print('=== PIPELINE EXECUTION SCRIPT COMPLETED ===')
