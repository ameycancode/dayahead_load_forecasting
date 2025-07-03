"""
Enhanced Lambda function for energy load forecasting with delete/recreate endpoint management.
This function recreates endpoints before prediction and deletes them after completion for maximum cost savings.
Integrates with existing Redshift Data API and maintains backward compatibility with Athena.
"""
import os
import io
import csv
import json
import logging
import time
from datetime import datetime, timedelta

import boto3
import pandas as pd

from forecast.data_preparation import create_forecast_dataframe
from forecast.feature_engineering import add_weather_features, add_solar_features, add_solar_ratios
from forecast.feature_engineering import create_weather_solar_interactions  
from forecast.weather_service import fetch_weather_data
from forecast.endpoint_service import invoke_sagemaker_endpoint
from forecast.utils import setup_logging, get_model_features

# Set up logging
logger = setup_logging()

# Constants for endpoint recreation and deletion
ENDPOINT_RECREATION_MAX_WAIT = 900   # 15 minutes max wait for endpoint recreation
ENDPOINT_DELETION_MAX_WAIT = 300     # 5 minutes max wait for endpoint deletion
ENDPOINT_CHECK_INTERVAL = 30         # Check endpoint status every 30 seconds
ENDPOINT_READY_BUFFER = 60           # Wait 60 seconds after InService before using

def parse_time_period(time_str):
    """Parse compact time period format like "6,9" to tuple (6, 9)"""
    if not time_str:
        return None
    parts = time_str.split(',')
    return (int(parts[0]), int(parts[1])) if len(parts) == 2 else None

def parse_array_string(array_str):
    """Parse compact array format like "14,21,28,35" to list [14, 21, 28, 35]"""
    if not array_str:
        return []
    return [int(x.strip()) for x in array_str.split(',') if x.strip()]

def parse_weather_variables(weather_str):
    """Parse compact weather variables format"""
    if not weather_str:
        return []
    return [x.strip() for x in weather_str.split(',') if x.strip()]

def load_lambda_configuration():
    """
    Load and parse Lambda configuration from environment variables.
    This replaces the config.py import entirely.
    """
    config = {
        # Core identification
        'CUSTOMER_PROFILE': os.environ.get('CUSTOMER_PROFILE', 'RES'),
        'CUSTOMER_SEGMENT': os.environ.get('CUSTOMER_SEGMENT', 'SOLAR'),
        'ENVIRONMENT': os.environ.get('ENV_NAME', 'dev'),
        'ENV_NAME': os.environ.get('ENV_NAME', 'dev'),
        
        # AWS Configuration
        'S3_BUCKET': os.environ.get('S3_BUCKET'),
        'S3_PREFIX': os.environ.get('S3_PREFIX'),
        'ENDPOINT_NAME': os.environ.get('ENDPOINT_NAME'),
        
        # ENHANCED: Delete/Recreate Endpoint Management Configuration
        'ENABLE_ENDPOINT_AUTO_MANAGEMENT': os.environ.get('ENABLE_ENDPOINT_AUTO_MANAGEMENT', 'true').lower() == 'true',
        'ENABLE_ENDPOINT_DELETE_RECREATE': os.environ.get('ENABLE_ENDPOINT_DELETE_RECREATE', 'true').lower() == 'true',
        'DELETE_ENDPOINT_AFTER_PREDICTION': os.environ.get('DELETE_ENDPOINT_AFTER_PREDICTION', 'true').lower() == 'true',
        'ENDPOINT_RECREATION_TIMEOUT': int(os.environ.get('ENDPOINT_RECREATION_TIMEOUT', str(ENDPOINT_RECREATION_MAX_WAIT))),
        'ENDPOINT_DELETION_TIMEOUT': int(os.environ.get('ENDPOINT_DELETION_TIMEOUT', str(ENDPOINT_DELETION_MAX_WAIT))),
        'ENDPOINT_READY_BUFFER_TIME': int(os.environ.get('ENDPOINT_READY_BUFFER_TIME', str(ENDPOINT_READY_BUFFER))),
        'ENDPOINT_CONFIG_S3_PREFIX': os.environ.get('ENDPOINT_CONFIG_S3_PREFIX', 'endpoint-configs'),
        
        # Database Configuration
        'DATABASE_TYPE': os.environ.get('DATABASE_TYPE', 'redshift'),
        'OUTPUT_METHOD': os.environ.get('OUTPUT_METHOD', 'redshift'),
        
        # Data Processing
        'LOAD_PROFILE': os.environ.get('LOAD_PROFILE'),
        'SUBMISSION_TYPE_FINAL': os.environ.get('SUBMISSION_TYPE_FINAL', 'Final'),
        'SUBMISSION_TYPE_INITIAL': os.environ.get('SUBMISSION_TYPE_INITIAL', 'Initial'),
        'FINAL_SUBMISSION_DELAY': int(os.environ.get('FINAL_SUBMISSION_DELAY', '48')),
        'INITIAL_SUBMISSION_DELAY': int(os.environ.get('INITIAL_SUBMISSION_DELAY', '14')),
        'DATA_DELAY_DAYS': int(os.environ.get('INITIAL_SUBMISSION_DELAY', '14')),
        'RATE_GROUP_FILTER_CLAUSE': os.environ.get('RATE_GROUP_FILTER_CLAUSE', ''),
        
        # Model Configuration
        'MODEL_VERSION': os.environ.get('MODEL_VERSION', 'latest'),
        'RUN_ID': os.environ.get('RUN_ID'),
        'RUN_USER': os.environ.get('RUN_USER', 'system'),
        
        # Parse compact formats
        'MORNING_PEAK_HOURS': parse_time_period(os.environ.get('MORNING_PEAK_HOURS')),
        'SOLAR_PERIOD_HOURS': parse_time_period(os.environ.get('SOLAR_PERIOD_HOURS')),
        'EVENING_RAMP_HOURS': parse_time_period(os.environ.get('EVENING_RAMP_HOURS')),
        'EVENING_PEAK_HOURS': parse_time_period(os.environ.get('EVENING_PEAK_HOURS')),
        'DEFAULT_LAG_DAYS': parse_array_string(os.environ.get('DEFAULT_LAG_DAYS')),
        'WEATHER_VARIABLES': parse_weather_variables(os.environ.get('WEATHER_VARIABLES')),
        
        # Location
        'DEFAULT_LATITUDE': float(os.environ.get('DEFAULT_LATITUDE', '32.7157')),
        'DEFAULT_LONGITUDE': float(os.environ.get('DEFAULT_LONGITUDE', '-117.1611')),
        'DEFAULT_TIMEZONE': os.environ.get('DEFAULT_TIMEZONE', 'America/Los_Angeles'),
    }
    
    # Add database-specific configuration
    if config['DATABASE_TYPE'] == 'redshift':
        config.update({
            'REDSHIFT_CLUSTER_IDENTIFIER': os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER'),
            'REDSHIFT_DATABASE': os.environ.get('REDSHIFT_DATABASE'),
            'REDSHIFT_DB_USER': os.environ.get('REDSHIFT_DB_USER'),
            'REDSHIFT_REGION': os.environ.get('REDSHIFT_REGION'),
            'REDSHIFT_INPUT_SCHEMA': os.environ.get('REDSHIFT_INPUT_SCHEMA'),
            'REDSHIFT_INPUT_TABLE': os.environ.get('REDSHIFT_INPUT_TABLE'),
            'REDSHIFT_OUTPUT_SCHEMA': os.environ.get('REDSHIFT_OUTPUT_SCHEMA'),
            'REDSHIFT_OUTPUT_TABLE': os.environ.get('REDSHIFT_OUTPUT_TABLE'),
            'REDSHIFT_BI_SCHEMA': os.environ.get('REDSHIFT_BI_SCHEMA'),
            'REDSHIFT_BI_VIEW': os.environ.get('REDSHIFT_BI_VIEW'),
            'REDSHIFT_IAM_ROLE': os.environ.get('REDSHIFT_IAM_ROLE'),
        })
    elif config['DATABASE_TYPE'] == 'athena':
        config.update({
            'ATHENA_DATABASE': os.environ.get('ATHENA_DATABASE'),
            'ATHENA_TABLE': os.environ.get('ATHENA_TABLE'),
            'ATHENA_STAGING_DIR': os.environ.get('ATHENA_STAGING_DIR'),
            'ATHENA_RESULTS_LOCATION': os.environ.get('ATHENA_RESULTS_LOCATION'),
            'ATHENA_DATA_LOCATION': os.environ.get('ATHENA_DATA_LOCATION'),
            'ATHENA_REGION': os.environ.get('REDSHIFT_REGION'),
        })
        
        # Build full table names
        input_schema = config.get('REDSHIFT_INPUT_SCHEMA')
        input_table = config.get('REDSHIFT_INPUT_TABLE')
        output_schema = config.get('REDSHIFT_OUTPUT_SCHEMA')
        output_table = config.get('REDSHIFT_OUTPUT_TABLE')
        
        if input_schema and input_table:
            config['INPUT_FULL_TABLE_NAME'] = f"{input_schema}.{input_table}"
        if output_schema and output_table:
            config['OUTPUT_FULL_TABLE_NAME'] = f"{output_schema}.{output_table}"
    
    return config

class EndpointRecreationManager:
    """
    Manages SageMaker endpoint lifecycle using delete/recreate approach for maximum cost optimization.
    Recreates endpoints from stored configuration and deletes them after prediction completion.
    """
    
    def __init__(self, region_name='us-west-2'):
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.region_name = region_name
        
    def get_endpoint_status(self, endpoint_name):
        """Get current endpoint status"""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            return response['EndpointStatus']
        except self.sagemaker_client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ValidationException':
                logger.info(f"Endpoint {endpoint_name} does not exist")
                return 'NotFound'
            else:
                raise e
    
    def load_endpoint_configuration(self, endpoint_name, config):
        """
        Load stored endpoint configuration from S3 for recreation
        """
        try:
            logger.info(f"Loading endpoint configuration for recreation: {endpoint_name}")
            
            bucket = config.get('S3_BUCKET')
            s3_prefix = config.get('S3_PREFIX', '')
            endpoint_config_s3_prefix = config.get('ENDPOINT_CONFIG_S3_PREFIX', 'endpoint-configs')
            
            if not bucket:
                logger.error("S3 bucket not configured - cannot load endpoint configuration")
                return None
            
            # Try primary configuration location first
            base_key = f"{s3_prefix}/{endpoint_config_s3_prefix}" if s3_prefix else endpoint_config_s3_prefix
            primary_key = f"{base_key}/{endpoint_name}_config.json"
            
            try:
                logger.info(f"Attempting to load config from: s3://{bucket}/{primary_key}")
                response = self.s3_client.get_object(Bucket=bucket, Key=primary_key)
                config_data = json.loads(response['Body'].read().decode('utf-8'))
                
                logger.info(f" Successfully loaded endpoint configuration from primary location")
                return config_data
                
            except self.s3_client.exceptions.NoSuchKey:
                logger.warning(f"Primary config not found, trying customer-specific location...")
                
                # Try customer-specific location
                customer_key = f"{base_key}/customers/{config.get('CUSTOMER_PROFILE', 'unknown')}-{config.get('CUSTOMER_SEGMENT', 'unknown')}/{endpoint_name}_config.json"
                
                try:
                    logger.info(f"Attempting to load config from: s3://{bucket}/{customer_key}")
                    response = self.s3_client.get_object(Bucket=bucket, Key=customer_key)
                    config_data = json.loads(response['Body'].read().decode('utf-8'))
                    
                    logger.info(f" Successfully loaded endpoint configuration from customer location")
                    return config_data
                    
                except self.s3_client.exceptions.NoSuchKey:
                    logger.error(f" Endpoint configuration not found in either location")
                    logger.error(f"   Primary: s3://{bucket}/{primary_key}")
                    logger.error(f"   Customer: s3://{bucket}/{customer_key}")
                    return None
            
        except Exception as e:
            logger.error(f"Error loading endpoint configuration: {str(e)}")
            return None
    
    def recreate_endpoint(self, endpoint_name, config):
        """
        Recreate endpoint from stored configuration for cost-optimized predictions
        """
        logger.info(f"=== RECREATING ENDPOINT FOR PREDICTIONS: {endpoint_name} ===")
        
        try:
            # Step 1: Load stored configuration
            endpoint_config = self.load_endpoint_configuration(endpoint_name, config)
            
            if not endpoint_config:
                logger.error(f" Cannot recreate endpoint - configuration not found")
                return False
            
            logger.info(f" Endpoint configuration loaded - proceeding with recreation")
            
            # Step 2: Recreate model if needed
            model_name = endpoint_config['model_name']
            model_config = endpoint_config['model_config']
            
            # Check if model exists
            try:
                self.sagemaker_client.describe_model(ModelName=model_name)
                logger.info(f" Model {model_name} already exists")
            except self.sagemaker_client.exceptions.ClientError as e:
                if 'ValidationException' in str(e):
                    logger.info(f" Recreating model: {model_name}")
                    
                    # Recreate model from stored configuration
                    self.sagemaker_client.create_model(
                        ModelName=model_name,
                        ExecutionRoleArn=model_config['execution_role_arn'],
                        PrimaryContainer=model_config['primary_container'],
                        Tags=model_config.get('tags', [])
                    )
                    
                    logger.info(f" Model recreated: {model_name}")
                else:
                    raise e
            
            # Step 3: Recreate endpoint configuration if needed
            endpoint_config_name = endpoint_config['endpoint_config_name']
            stored_endpoint_config = endpoint_config['endpoint_config']
            
            # Check if endpoint config exists
            try:
                self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
                logger.info(f" Endpoint config {endpoint_config_name} already exists")
            except self.sagemaker_client.exceptions.ClientError as e:
                if 'ValidationException' in str(e):
                    logger.info(f" Recreating endpoint config: {endpoint_config_name}")
                    
                    # Recreate endpoint configuration from stored data
                    self.sagemaker_client.create_endpoint_config(
                        EndpointConfigName=endpoint_config_name,
                        ProductionVariants=stored_endpoint_config['production_variants'],
                        Tags=stored_endpoint_config.get('tags', [])
                    )
                    
                    logger.info(f" Endpoint config recreated: {endpoint_config_name}")
                else:
                    raise e
            
            # Step 4: Recreate endpoint
            logger.info(f" Recreating endpoint: {endpoint_name}")
            
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=[
                    {'Key': 'Recreated', 'Value': 'true'},
                    {'Key': 'CostOptimized', 'Value': 'true'},
                    {'Key': 'RecreatedAt', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                    {'Key': 'OriginalRunId', 'Value': endpoint_config.get('run_id', 'unknown')}
                ]
            )
            
            logger.info(f" Endpoint recreation initiated: {endpoint_name}")
            
            # Step 5: Wait for endpoint to be ready
            return self._wait_for_endpoint_ready(endpoint_name, config)
            
        except Exception as e:
            logger.error(f" Error recreating endpoint {endpoint_name}: {str(e)}")
            return False
    
    def _wait_for_endpoint_ready(self, endpoint_name, config):
        """Wait for recreated endpoint to become InService"""
        max_wait = config.get('ENDPOINT_RECREATION_TIMEOUT', ENDPOINT_RECREATION_MAX_WAIT)
        buffer_time = config.get('ENDPOINT_READY_BUFFER_TIME', ENDPOINT_READY_BUFFER)
        
        logger.info(f"Waiting for recreated endpoint to be ready (max {max_wait}s)...")
        
        waited = 0
        while waited < max_wait:
            try:
                status = self.get_endpoint_status(endpoint_name)
                logger.info(f"Endpoint status after {waited}s: {status}")
                
                if status == 'InService':
                    logger.info(f" Endpoint is InService! Waiting {buffer_time}s buffer...")
                    time.sleep(buffer_time)  # Buffer to ensure endpoint is fully ready
                    logger.info(" Recreated endpoint ready for predictions")
                    return True
                    
                elif status in ['Failed', 'RollingBack']:
                    logger.error(f" Endpoint recreation failed: {status}")
                    return False
                
                time.sleep(ENDPOINT_CHECK_INTERVAL)
                waited += ENDPOINT_CHECK_INTERVAL
                
            except Exception as e:
                logger.error(f"Error checking recreated endpoint status: {str(e)}")
                time.sleep(ENDPOINT_CHECK_INTERVAL)
                waited += ENDPOINT_CHECK_INTERVAL
        
        logger.error(f" Recreated endpoint did not become ready within {max_wait}s")
        return False
    
    def delete_endpoint_after_prediction(self, endpoint_name, config):
        """
        Delete endpoint after successful prediction for maximum cost optimization
        """
        logger.info(f"=== DELETING ENDPOINT AFTER PREDICTION: {endpoint_name} ===")
        
        try:
            current_status = self.get_endpoint_status(endpoint_name)
            
            if current_status == 'NotFound':
                logger.info(" Endpoint already deleted")
                return True
                
            elif current_status != 'InService':
                logger.warning(f"Endpoint not in expected state for deletion: {current_status}")
                # Proceed with deletion anyway for cost optimization
            
            logger.info(f" Deleting endpoint for cost optimization: {endpoint_name}")
            
            # Delete the endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            logger.info(f" Endpoint deletion initiated")
            
            # Wait for deletion to complete (optional)
            if config.get('WAIT_FOR_ENDPOINT_DELETION', True):
                return self._wait_for_endpoint_deleted(endpoint_name, config)
            
            return True
            
        except Exception as e:
            logger.error(f" Error deleting endpoint {endpoint_name}: {str(e)}")
            return False
    
    def _wait_for_endpoint_deleted(self, endpoint_name, config):
        """Wait for endpoint to be deleted"""
        max_wait = config.get('ENDPOINT_DELETION_TIMEOUT', ENDPOINT_DELETION_MAX_WAIT)
        
        logger.info(f"Waiting for endpoint to be deleted (max {max_wait}s)...")
        
        waited = 0
        while waited < max_wait:
            try:
                status = self.get_endpoint_status(endpoint_name)
                
                if status == 'NotFound':
                    logger.info(" Endpoint successfully deleted")
                    return True
                elif status == 'Deleting':
                    logger.info(f" Endpoint deleting... (waited {waited}s)")
                else:
                    logger.info(f"Endpoint status: {status} (waited {waited}s)")
                
                time.sleep(ENDPOINT_CHECK_INTERVAL)
                waited += ENDPOINT_CHECK_INTERVAL
                
            except Exception as e:
                logger.error(f"Error checking deletion status: {str(e)}")
                time.sleep(ENDPOINT_CHECK_INTERVAL)
                waited += ENDPOINT_CHECK_INTERVAL
        
        logger.warning(f"Endpoint deletion did not complete within {max_wait}s - may still be deleting")
        return True  # Don't fail the whole process if deletion is slow

def lambda_handler(event, context):
    """
    Enhanced Lambda handler with delete/recreate endpoint management for maximum cost optimization.
    """
    # Timeout logging
    timeout_seconds = context.get_remaining_time_in_millis()/1000
    logger.info(f"=== LAMDBA TIMEOUT CONFIGURATION ===")
    logger.info(f"Lamdba timeout configured: {timeout_seconds} seconds")
    logger.info(f"Function name: {context.function_name}")
    logger.info(f"Memory size: {context.memory_limit_in_mb} MB")
    
    endpoint_manager = None
    endpoint_recreated_by_lambda = False
    
    try:
        # Load configuration
        config = load_lambda_configuration()

        logger.info("=== ENHANCED ENERGY LOAD FORECASTING LAMBDA - DELETE/RECREATE APPROACH ===")
        logger.info(f"Lambda function: {context.function_name}")
        logger.info(f"Request ID: {context.aws_request_id}")
        logger.info(f"Delete/recreate management: {config.get('ENABLE_ENDPOINT_DELETE_RECREATE', False)}")
        
        # Extract parameters
        endpoint_name = event.get('endpoint_name', config.get('ENDPOINT_NAME', ''))
        load_profile = event.get('load_profile', config.get('CUSTOMER_PROFILE', 'RES'))
        customer_segment = event.get('customer_segment', config.get('CUSTOMER_SEGMENT', 'solar'))
        model_version = event.get('model_version', config.get('MODEL_VERSION', 'latest'))
        run_user = event.get('run_user', config.get('RUN_USER', 'system'))
        test_invocation = event.get('test_invocation', False)
        
        # Generate run_id if not provided
        run_id = event.get('run_id', config.get('RUN_ID'))
        if not run_id:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine forecast date
        forecast_date_str = event.get('forecast_date')
        if forecast_date_str:
            forecast_date = pd.to_datetime(forecast_date_str)
        else:
            forecast_date = pd.to_datetime(datetime.now() + timedelta(days=1))
        
        # Log comprehensive configuration status
        logger.info(f" Configuration loaded successfully:")
        logger.info(f"   S3_BUCKET: {config.get('S3_BUCKET')}")
        logger.info(f"   S3_PREFIX: {config.get('S3_PREFIX')}")
        logger.info(f"   Database type: {config.get('DATABASE_TYPE', 'NOT SET')}")
        logger.info(f"   Output method: {config.get('OUTPUT_METHOD', 'NOT SET')}")
        logger.info(f"   Customer profile: {load_profile}")
        logger.info(f"Processing forecast for {load_profile}-{customer_segment} on {forecast_date.strftime('%Y-%m-%d')}")

        logger.info(f"   Rate group filter: {config.get('RATE_GROUP_FILTER_CLAUSE', 'NOT SET')}")
        logger.info(f"   Submission delays: Initial={config.get('INITIAL_SUBMISSION_DELAY')}, Final={config.get('FINAL_SUBMISSION_DELAY')}")
        logger.info(f"   Weather variables: {len(config.get('WEATHER_VARIABLES', []))} configured")

        TEST_INVOCATION_BOOL = False
        if isinstance(test_invocation, str):
            TEST_INVOCATION_BOOL = (test_invocation.lower() == 'true')
        else:
            TEST_INVOCATION_BOOL = bool(test_invocation)
        
        # Initialize endpoint recreation manager if delete/recreate is enabled
        if config.get('ENABLE_ENDPOINT_DELETE_RECREATE', True) and not TEST_INVOCATION_BOOL:
            endpoint_manager = EndpointRecreationManager(region_name=config.get('REDSHIFT_REGION', 'us-west-2'))
            
            logger.info("=== ENDPOINT DELETE/RECREATE MANAGEMENT ENABLED ===")
            
            # Check if endpoint exists
            current_status = endpoint_manager.get_endpoint_status(endpoint_name)
            logger.info(f"Current endpoint status: {current_status}")
            
            if current_status == 'NotFound':
                logger.info("Endpoint not found - recreating from stored configuration...")
                
                endpoint_ready = endpoint_manager.recreate_endpoint(endpoint_name, config)
                
                if not endpoint_ready:
                    error_msg = f"Failed to recreate endpoint {endpoint_name}"
                    logger.error(error_msg)
                    return error_response(error_msg, run_id)
                
                endpoint_recreated_by_lambda = True
                logger.info(" Endpoint recreated and ready - proceeding with predictions")
                
            elif current_status == 'InService':
                logger.info(" Endpoint already InService - proceeding with predictions")
                
            else:
                error_msg = f"Endpoint in unexpected state: {current_status}"
                logger.error(error_msg)
                return error_response(error_msg, run_id)
                
        else:
            logger.info("Delete/recreate management disabled or test invocation - using existing endpoint state")

        # Log invocation details
        logger.info(f" Forecast Lambda invoked:")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Endpoint: {endpoint_name}")
        logger.info(f"   Forecast date: {forecast_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Load profile: {load_profile}")
        logger.info(f"   Customer segment: {customer_segment}")
        logger.info(f"   Model version: {model_version}")
        logger.info(f"   Run user: {run_user}")
        logger.info(f"   Test invocation: {test_invocation}")
        
        # Generate predictions
        logger.info("=== GENERATING PREDICTIONS ===")
        predictions = generate_predictions(
            config,
            endpoint_name,
            forecast_date,
            load_profile,
            customer_segment,
            run_id
        )
        
        if not predictions:
            logger.error("No predictions generated")
            return error_response("Failed to generate predictions", run_id)
        
        logger.info(f" Generated {len(predictions)} hourly predictions")

        # Save predictions (existing logic)
        output_method = config.get('OUTPUT_METHOD', 'redshift')
        logger.info(f"Saving predictions using {output_method} method...")

        if output_method == 'redshift':
            logger.info("Saving predictions to Redshift...")
            environment_name = config.get('ENVIRONMENT', 'dev')
            
            # S3 staging
            s3_result = save_predictions_to_s3_staging(
                environment_name,
                predictions,
                load_profile,
                customer_segment,
                forecast_date.strftime('%Y-%m-%d'),
                run_id,
                model_version,
                run_user,
                config
            )
            
            # Get Redshift configuration for response config.get('REDSHIFT_OUTPUT_SCHEMA', 'edp_forecasting_dev')
            output_schema = config.get('REDSHIFT_OUTPUT_SCHEMA', 'edp_forecasting_dev')
            output_table = config.get('REDSHIFT_OUTPUT_TABLE', 'dayahead_load_forecasts')
            bi_schema = config.get('REDSHIFT_BI_SCHEMA', 'edp_bi_dev')
            bi_view = config.get('REDSHIFT_BI_VIEW', 'vw_dayahead_load_forecasts')
            
            records_inserted = save_predictions_to_redshift_direct_insert(
                predictions,
                load_profile,
                customer_segment,
                forecast_date.strftime('%Y-%m-%d'),
                run_id,
                model_version,
                run_user,
                config
            )
            logger.info(f" Inserted {records_inserted} records directly to Redshift")

            response_body = {
                'message': f'Successfully processed {len(predictions)} predictions with delete/recreate endpoint management',
                'success': True,
                'run_id': run_id,
                'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                'load_profile': load_profile,
                'customer_segment': customer_segment,
                'endpoint': endpoint_name,
                'model_version': model_version,
                'run_user': run_user,
                'records_inserted': records_inserted,
                's3_location': s3_result,
                'predictions_count': len(predictions),
                'redshift_table': f"{output_schema}.{output_table}",
                'bi_view': f"{bi_schema}.{bi_view}",
                'database_type': 'redshift',
                'endpoint_delete_recreate_enabled': config.get('ENABLE_ENDPOINT_DELETE_RECREATE', False),
                'endpoint_recreated_by_lambda': endpoint_recreated_by_lambda,
                'test_invocation': test_invocation,
                'rate_group_filter_used': config.get('RATE_GROUP_FILTER_CLAUSE', 'NOT SET')
            }
            
        else:
            # Legacy Athena approach (unchanged)
            logger.info("Using legacy Athena approach...")
            athena_config = load_athena_config(config)
            
            s3_location = save_predictions_to_athena_s3(
                predictions, athena_config, load_profile, customer_segment,
                forecast_date.strftime('%Y-%m-%d'), run_id, model_version, run_user
            )
            
            record_count = "test_skipped"
            if not test_invocation:
                try:
                    record_count = verify_predictions_in_athena(
                        athena_config, load_profile, customer_segment, forecast_date.strftime('%Y-%m-%d')
                    )
                except Exception as e:
                    logger.warning(f"Could not verify predictions in Athena: {str(e)}")
                    record_count = "verification_failed"
            
            response_body = {
                'success': True,
                'run_id': run_id,
                'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                'load_profile': load_profile,
                'customer_segment': customer_segment,
                'endpoint': endpoint_name,
                'model_version': model_version,
                'run_user': run_user,
                's3_location': s3_location,
                'predictions_count': len(predictions),
                'database_type': 'athena',
                'athena_database': athena_config.get('database'),
                'athena_table': athena_config.get('table'),
                'athena_verification': record_count,
                'endpoint_delete_recreate_enabled': config.get('ENABLE_ENDPOINT_DELETE_RECREATE', False),
                'endpoint_recreated_by_lambda': endpoint_recreated_by_lambda,
                'test_invocation': test_invocation,
                'rate_group_filter_used': config.get('RATE_GROUP_FILTER_CLAUSE', 'NOT SET')
            }

        # Delete endpoint after successful prediction for maximum cost optimization
        if (endpoint_manager and 
            config.get('ENABLE_ENDPOINT_DELETE_RECREATE', True) and 
            config.get('DELETE_ENDPOINT_AFTER_PREDICTION', True) and 
            not TEST_INVOCATION_BOOL):
            
            logger.info("=== DELETING ENDPOINT AFTER SUCCESSFUL PREDICTION FOR COST OPTIMIZATION ===")
            
            try:
                endpoint_deleted = endpoint_manager.delete_endpoint_after_prediction(endpoint_name, config)
                
                if endpoint_deleted:
                    logger.info(" Endpoint successfully deleted - maximum cost optimization achieved")
                    response_body['endpoint_deleted_after_prediction'] = True
                    response_body['cost_optimization_status'] = 'maximum_savings_achieved'
                else:
                    logger.warning(" Endpoint deletion failed or incomplete")
                    response_body['endpoint_deleted_after_prediction'] = False
                    response_body['cost_optimization_status'] = 'deletion_failed'
                    
            except Exception as delete_e:
                logger.error(f"Error deleting endpoint (prediction succeeded): {str(delete_e)}")
                response_body['endpoint_deletion_error'] = str(delete_e)
                response_body['endpoint_deleted_after_prediction'] = False
                response_body['cost_optimization_status'] = 'deletion_error'
                # Don't fail the whole process if endpoint deletion fails
        
        return {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }

    except Exception as e:
        logger.error(f"Error in enhanced lambda_handler: {str(e)}", exc_info=True)
        
        # Try to delete endpoint even if prediction failed (cost optimization)
        if (endpoint_manager and 
            config.get('ENABLE_ENDPOINT_DELETE_RECREATE', True) and 
            config.get('DELETE_ENDPOINT_AFTER_PREDICTION', True) and 
            not test_invocation):
            
            logger.info("=== DELETING ENDPOINT AFTER PREDICTION FAILURE FOR COST OPTIMIZATION ===")
            try:
                endpoint_manager.delete_endpoint_after_prediction(endpoint_name, config)
                logger.info(" Endpoint deleted after prediction failure")
            except Exception as delete_e:
                logger.error(f"Error deleting endpoint after prediction failure: {str(delete_e)}")
        
        return error_response(str(e), run_id if 'run_id' in locals() else None)

# Keep all existing functions unchanged (same as previous implementation)
def save_predictions_to_redshift_direct_insert(predictions, load_profile, customer_segment,
                                             forecast_date, run_id, model_version, run_user, config):
    """Direct INSERT for small datasets (perfect for 24 rows)"""
    try:
        region = config.get('REDSHIFT_REGION', 'us-west-2')
        redshift_data_client = boto3.client('redshift-data', region_name=region)
        
        cluster_identifier = config.get('REDSHIFT_CLUSTER_IDENTIFIER')
        database = config.get('REDSHIFT_DATABASE')
        db_user = config.get('REDSHIFT_DB_USER')
        output_schema = config.get('REDSHIFT_OUTPUT_SCHEMA')
        output_table = config.get('REDSHIFT_OUTPUT_TABLE')
        
        logger.info(f"Inserting {len(predictions)} records to {output_schema}.{output_table}")
        
        values_clause = build_values_clause_for_predictions(
            predictions, load_profile, customer_segment, forecast_date,
            run_id, model_version, run_user
        )
        
        insert_sql = f"""
        INSERT INTO {output_schema}.{output_table} 
        (forecast_datetime, predicted_lossadjustedload, run_id, model_version, 
         run_user, created_at, load_profile, load_segment, year, month, day)
        VALUES {values_clause}
        """
        
        logger.info("Executing direct INSERT statement...")
        
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=insert_sql
        )
        
        query_id = response['Id']
        logger.info(f"INSERT submitted with query ID: {query_id}")
        
        wait_for_query_completion_simple(redshift_data_client, query_id)
        
        logger.info(f" Successfully inserted {len(predictions)} records via direct INSERT")
        return len(predictions)
        
    except Exception as e:
        logger.error(f"Error with direct INSERT: {str(e)}")
        raise

def build_values_clause_for_predictions(predictions, load_profile, customer_segment, forecast_date,
                                       run_id, model_version, run_user):
    """Build VALUES clause for the 24 prediction records"""
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date_obj = datetime.strptime(forecast_date, '%Y-%m-%d')
    
    values_list = []
    
    for pred in predictions:
        forecast_datetime = pred['forecast_datetime'].strftime('%Y-%m-%d %H:%M:%S')
        predicted_load = round(pred['predicted_lossadjustedload'], 4)
        
        value_tuple = f"""(
            '{forecast_datetime}', 
            {predicted_load}, 
            '{run_id}', 
            '{model_version}', 
            '{run_user}', 
            '{created_at}', 
            '{load_profile}', 
            '{customer_segment}', 
            {date_obj.year}, 
            {date_obj.month}, 
            {date_obj.day}
        )"""
        
        values_list.append(value_tuple.strip())
    
    return ",\n".join(values_list)

def wait_for_query_completion_simple(redshift_data_client, query_id, max_wait_time=60):
    """Simple wait function for quick INSERT operations"""
    start_time = time.time()

    logger.info(f"=== WAITING FOR REDSHIFT QUERY COMPLETION ===")
    logger.info(f"Query ID: {query_id}")
    logger.info(f"Max wait time: {max_wait_time} seconds")
    logger.info(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    while time.time() - start_time < max_wait_time:
        try:
            elapsed = time.time() - start_time
            logger.info(f"Checking query status... (elapsed: {elapsed:.1f}s)")

            response = redshift_data_client.describe_statement(Id=query_id)
            status = response['Status']

            elapsed = time.time() - start_time
            logger.info(f"Query status: {status} (elapsed: {elapsed:.1f}s)")
            
            if status == 'FINISHED':
                duration = time.time() - start_time
                logger.info(f"Query {query_id} completed in {duration:.2f}s")
                return True
            elif status in ['FAILED', 'ABORTED']:
                error_msg = response.get('Error', 'Unknown error')
                logger.error(f"Query {query_id} failed: {error_msg}")
                raise Exception(f'Query failed: {error_msg}')
            
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error checking query status: {str(e)}")
            raise
    
    raise Exception(f'Query timed out after {max_wait_time} seconds')

def save_predictions_to_s3_staging(environment_name, predictions, load_profile, customer_segment,
                                  forecast_date, run_id, model_version, run_user, config):
    """Save predictions to S3 staging area for Redshift COPY"""
    try:
        date_obj = datetime.strptime(forecast_date, '%Y-%m-%d')
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        bucket = config.get('S3_BUCKET')
        base_prefix = config.get('S3_STAGING_PREFIX', 'redshift-staging')
        
        df_data = []
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for pred in predictions:
            df_data.append({
                'forecast_datetime': pred['forecast_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_lossadjustedload': round(pred['predicted_lossadjustedload'], 4),
                'run_id': run_id,
                'model_version': model_version,
                'run_user': run_user,
                'created_at': created_at,
                'load_profile': load_profile,
                'load_segment': customer_segment,
                'year': int(year),
                'month': int(month),
                'day': int(day)
            })
        
        df = pd.DataFrame(df_data)
        
        s3_key = f"{base_prefix}/{environment_name}/forecasts/load_profile={load_profile}/load_segment={customer_segment}/year={year}/month={month}/day={day}/predictions_{run_id}_{datetime.now().strftime('%H%M%S')}.csv"
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, 
                 date_format='%Y-%m-%d %H:%M:%S',
                 quoting=csv.QUOTE_MINIMAL)
        
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv',
            Metadata={
                'load_profile': load_profile,
                'load_segment': customer_segment,
                'forecast_date': forecast_date,
                'run_id': run_id,
                'record_count': str(len(predictions)),
                'created_at': created_at
            }
        )
        
        s3_location = f"s3://{bucket}/{s3_key}"
        logger.info(f" Predictions saved to S3 staging: {s3_location}")
        
        return s3_location
        
    except Exception as e:
        logger.error(f"Error saving to S3 staging: {str(e)}")
        raise

def generate_predictions(config, endpoint_name, forecast_date, load_profile, customer_segment, run_id):
    """Generate hourly predictions for the forecast date"""
    try:
        inference_df = create_inference_dataframe(config, load_profile, forecast_date, run_id)
        
        if inference_df.empty:
            logger.error("Failed to create inference dataframe")
            return []
        
        predictions = []
        
        for hour in range(24):
            try:
                forecast_datetime = forecast_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                if len(inference_df) >= hour + 1:
                    hour_data = inference_df.iloc[hour:hour+1]
                else:
                    hour_data = inference_df.iloc[-1:].copy()
                
                prediction_result = invoke_sagemaker_endpoint(endpoint_name, hour_data)
                
                if isinstance(prediction_result, list) and len(prediction_result) > 0:
                    predicted_load = float(prediction_result[0])
                elif isinstance(prediction_result, dict):
                    predicted_load = float(prediction_result.get('predictions', [0])[0])
                else:
                    predicted_load = float(prediction_result)
                
                predictions.append({
                    'forecast_datetime': forecast_datetime,
                    'predicted_lossadjustedload': predicted_load
                })
                
            except Exception as hour_e:
                logger.warning(f"Error predicting for hour {hour}: {str(hour_e)}")
                predictions.append({
                    'forecast_datetime': forecast_date.replace(hour=hour, minute=0, second=0, microsecond=0),
                    'predicted_lossadjustedload': 0.0
                })
        
        logger.info(f"Generated {len(predictions)} predictions successfully")
        return predictions
        
    except Exception as e:
        logger.error(f"Error in generate_predictions: {str(e)}", exc_info=True)
        return []

def create_inference_dataframe(config, load_profile, forecast_date, run_id):
    """Create inference dataframe for the forecast"""
    try:
        forecast_date = pd.to_datetime(forecast_date)
        
        logger.info(f"Creating forecast dataframe for {forecast_date.strftime('%Y-%m-%d')}")
        logger.info(f"Using load profile: {load_profile}")
        logger.info(f"Current rate group filter: {config.get('RATE_GROUP_FILTER_CLAUSE', 'NOT SET')}")
        
        forecast_df = create_forecast_dataframe(forecast_date, config, forecast_delay_days=14)
        
        if forecast_df is None or forecast_df.empty:
            logger.error("create_forecast_dataframe returned empty dataframe")
            return pd.DataFrame()
        
        logger.info("Fetching weather data")
        weather_forecast = fetch_weather_data(forecast_date, config)
        
        logger.info("Adding weather features")
        forecast_df = add_weather_features(forecast_df, weather_forecast)

        logger.info("Creating weather-solar interactions")
        forecast_df = create_weather_solar_interactions(forecast_df)

        logger.info("Adding solar features")
        forecast_df = add_solar_features(forecast_df)
        forecast_df = add_solar_ratios(forecast_df, forecast_delay_days=14)
        
        logger.info("Getting model features")
        model_features = get_model_features(run_id, config)
        
        missing_features = [f for f in model_features if f not in forecast_df.columns]
        if missing_features:
            logger.warning(f"Adding {len(missing_features)} missing features with zeros")
            for feature in missing_features:
                forecast_df[feature] = 0
        
        inference_df = forecast_df[model_features]
        
        nan_cols = inference_df.columns[inference_df.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Filling NaN values in columns: {nan_cols}")
            inference_df = inference_df.fillna(0)
        
        logger.info(f"Created inference dataframe: {len(inference_df)} rows, {len(model_features)} features")
        logger.info(f"Successfully used rate group filter: {config.get('RATE_GROUP_FILTER_CLAUSE', 'NOT SET')}")
        
        return inference_df
        
    except Exception as e:
        logger.error(f"Error creating inference dataframe: {str(e)}", exc_info=True)
        return pd.DataFrame()

def load_athena_config(config):
    """Load Athena configuration from environment variables (legacy)"""
    if all(key in config for key in ['ATHENA_DATABASE', 'ATHENA_TABLE']):
        return {
            'database': config['ATHENA_DATABASE'],
            'table': config['ATHENA_TABLE'],
            'results_location': config.get('ATHENA_RESULTS_LOCATION', ''),
            'data_location': config.get('ATHENA_DATA_LOCATION', ''),
            'table_full_name': f"{config['ATHENA_DATABASE']}.{config['ATHENA_TABLE']}"
        }
    
    try:
        s3_client = boto3.client('s3')
        bucket = config.get('S3_BUCKET', '')
        env_name = config.get('ENV_NAME', 'dev')
        config_key = f'athena-config/{env_name}/config.json'
        
        logger.info(f"Loading Athena config from s3://{bucket}/{config_key}")
        
        response = s3_client.get_object(Bucket=bucket, Key=config_key)
        config_data = json.loads(response['Body'].read().decode('utf-8'))
        
        return {
            'database': config_data['athena_database'],
            'table': config_data['athena_table'],
            'results_location': config_data.get('results_location', ''),
            'data_location': config_data.get('data_location', ''),
            'table_full_name': config_data.get('table_full_name', f"{config_data['athena_database']}.{config_data['athena_table']}")
        }
        
    except Exception as e:
        logger.error(f"Could not load Athena config: {str(e)}")
        raise Exception("Athena configuration not found")

def save_predictions_to_athena_s3(predictions, athena_config, load_profile, customer_segment,
                                  forecast_date, run_id, model_version, run_user):
    """Save predictions to S3 in Athena-compatible format (legacy)"""
    try:
        date_obj = datetime.strptime(forecast_date, '%Y-%m-%d')
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        data_location = athena_config.get('data_location', '')
        if not data_location:
            raise Exception("Athena data location not configured")
        
        bucket = data_location.replace('s3://', '').split('/')[0]
        base_prefix = '/'.join(data_location.replace(f's3://{bucket}/', '').split('/'))
        if base_prefix.endswith('/'):
            base_prefix = base_prefix[:-1]
        
        df_data = []
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for pred in predictions:
            df_data.append({
                'forecast_datetime': pred['forecast_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_lossadjustedload': round(pred['predicted_lossadjustedload'], 2),
                'run_id': run_id,
                'model_version': model_version,
                'run_user': run_user,
                'created_at': created_at,
                'load_profile': load_profile,
                'load_segment': customer_segment,
                'year': int(year),
                'month': int(month),
                'day': int(day)
            })
        
        df = pd.DataFrame(df_data)
        
        s3_key = f"{base_prefix}/load_profile={load_profile}/load_segment={customer_segment}/year={year}/month={month}/day={day}/predictions_lp_{load_profile}_{customer_segment}_for_{forecast_date.replace('-', '')}_run_{run_id}.csv"
        
        csv_string = df.to_csv(index=False)
        
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=csv_string,
            ContentType='text/csv',
            Metadata={
                'load_profile': load_profile,
                'load_segment': customer_segment,
                'forecast_date': forecast_date,
                'run_id': run_id,
                'model_version': model_version,
                'predictions_count': str(len(predictions)),
                'created_at': created_at
            }
        )
        
        s3_location = f"s3://{bucket}/{s3_key}"
        logger.info(f" Predictions saved to Athena-compatible location: {s3_location}")
        
        return s3_location
        
    except Exception as e:
        logger.error(f"Error saving predictions to Athena S3: {str(e)}", exc_info=True)
        raise

def verify_predictions_in_athena(athena_config, load_profile, customer_segment, forecast_date):
    """Verify predictions in Athena (legacy)"""
    try:
        date_obj = datetime.strptime(forecast_date, '%Y-%m-%d')
        
        athena_client = boto3.client('athena')
        
        query = f"""
        SELECT COUNT(*) as record_count
        FROM {athena_config['table_full_name']}
        WHERE load_profile = '{load_profile}'
          AND load_segment = '{customer_segment}'
          AND year = {date_obj.year}
          AND month = {date_obj.month}
          AND day = {date_obj.day}
        """
        
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': athena_config['database']},
            ResultConfiguration={'OutputLocation': athena_config['results_location']}
        )
        
        query_execution_id = response['QueryExecutionId']
        logger.info(f"Verification query started: {query_execution_id}")
        
        max_wait = 30
        waited = 0
        
        while waited < max_wait:
            status_response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            status = status_response['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
                rows = results.get('ResultSet', {}).get('Rows', [])
                
                if len(rows) > 1:
                    count = rows[1]['Data'][0].get('VarCharValue', '0')
                    logger.info(f" Athena verification successful: {count} records found")
                    return count
                else:
                    logger.warning("Athena verification: No data returned")
                    return "0"
                    
            elif status in ['FAILED', 'CANCELLED']:
                error_msg = status_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                logger.warning(f"Athena verification failed: {error_msg}")
                return "verification_failed"
            
            time.sleep(2)
            waited += 2
        
        logger.warning("Athena verification timed out")
        return "verification_timeout"
        
    except Exception as e:
        logger.warning(f"Could not verify data in Athena: {str(e)}")
        return "verification_error"

def error_response(message, run_id=None):
    """Create standardized error response"""
    return {
        'statusCode': 500,
        'body': json.dumps({
            'success': False,
            'run_id': run_id,
            'error': message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database_type': 'redshift',
            'endpoint_delete_recreate_enabled': True
        })
    }
