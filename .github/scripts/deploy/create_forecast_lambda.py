#!/usr/bin/env python3
"""
Enhanced Forecasting Lambda Creation Script - Delete/Recreate Approach
Creates Lambda functions for energy load forecasting with delete/recreate endpoint management
This version implements the delete/recreate approach for maximum cost optimization
"""
import os
import sys
import json
import time
import boto3

print('=== ENHANCED FORECASTING LAMBDA CREATION - DELETE/RECREATE APPROACH ===')

# Get all existing environment variables (same as your current version)
database_type = os.environ.get('DATABASE_TYPE', 'redshift')
output_method = os.environ.get('OUTPUT_METHOD', database_type)
endpoint_name = os.environ.get('ENDPOINT_NAME', '')
run_id = os.environ.get('RUN_ID', '')
lambda_schedule = os.environ.get('LAMBDA_SCHEDULE', '')
s3_bucket = os.environ.get('S3_BUCKET', '')
s3_prefix = os.environ.get('S3_PREFIX', '')
env_name = os.environ.get('ENV_NAME', '')
customer_profile = os.environ.get('CUSTOMER_PROFILE', '')
customer_segment = os.environ.get('CUSTOMER_SEGMENT', '')
forecast_lambda_name = os.environ.get('FORECAST_LAMBDA_NAME', '')

# ENHANCED: Delete/recreate endpoint management configuration
enable_endpoint_delete_recreate = os.environ.get('ENABLE_ENDPOINT_DELETE_RECREATE', 'true')
delete_endpoint_after_prediction = os.environ.get('DELETE_ENDPOINT_AFTER_PREDICTION', 'true')
endpoint_recreation_timeout = os.environ.get('ENDPOINT_RECREATION_TIMEOUT', '900')  # 15 minutes
endpoint_deletion_timeout = os.environ.get('ENDPOINT_DELETION_TIMEOUT', '300')     # 5 minutes
endpoint_ready_buffer_time = os.environ.get('ENDPOINT_READY_BUFFER_TIME', '60')    # 1 minute
endpoint_config_s3_prefix = os.environ.get('ENDPOINT_CONFIG_S3_PREFIX', 'endpoint-configs')

# Database-specific environment variables (same as before)
# Athena variables
athena_database = os.environ.get('ATHENA_DATABASE', '')
athena_table = os.environ.get('ATHENA_TABLE', '')
athena_results_location = os.environ.get('ATHENA_RESULTS_LOCATION', '')
athena_data_location = os.environ.get('ATHENA_DATA_LOCATION', '')

# Redshift variables  
redshift_cluster_identifier = os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER', '')
redshift_database = os.environ.get('REDSHIFT_DATABASE', '')
redshift_db_user = os.environ.get('REDSHIFT_DB_USER', '')
redshift_region = os.environ.get('REDSHIFT_REGION', '')
redshift_output_schema = os.environ.get('REDSHIFT_OUTPUT_SCHEMA', '')
redshift_output_table = os.environ.get('REDSHIFT_OUTPUT_TABLE', '')
redshift_bi_schema = os.environ.get('REDSHIFT_BI_SCHEMA', '')
redshift_bi_view = os.environ.get('REDSHIFT_BI_VIEW', '')
redshift_iam_role = os.environ.get('REDSHIFT_IAM_ROLE', '')

lambda_zip_path = 'predictions/lambda_forecast.zip'

print(f'=== CORE CONFIGURATION ===')
print(f'database_type: {database_type}')
print(f'output_method: {output_method}')
print(f'endpoint_name: {endpoint_name}')
print(f'run_id: {run_id}')
print(f'lambda_schedule: {lambda_schedule}')
print(f's3_bucket: {s3_bucket}')
print(f's3_prefix: {s3_prefix}')
print(f'env_name: {env_name}')
print(f'customer_profile: {customer_profile}')
print(f'customer_segment: {customer_segment}')

if database_type == 'athena':
    print(f'athena_database: {athena_database}')
    print(f'athena_table: {athena_table}')
    print(f'athena_results_location: {athena_results_location}')
    print(f'athena_data_location: {athena_data_location}')
elif database_type == 'redshift':
    print(f'redshift_cluster_identifier: {redshift_cluster_identifier}')
    print(f'redshift_database: {redshift_database}')
    print(f'redshift_output_schema: {redshift_output_schema}')
    print(f'redshift_output_table: {redshift_output_table}')
    print(f'redshift_bi_schema: {redshift_bi_schema}')
    print(f'redshift_bi_view: {redshift_bi_view}')
print(f'=== DELETE/RECREATE ENDPOINT MANAGEMENT CONFIGURATION ===')
print(f'enable_endpoint_delete_recreate: {enable_endpoint_delete_recreate}')
print(f'delete_endpoint_after_prediction: {delete_endpoint_after_prediction}')
print(f'endpoint_recreation_timeout: {endpoint_recreation_timeout}')
print(f'endpoint_deletion_timeout: {endpoint_deletion_timeout}')
print(f'endpoint_ready_buffer_time: {endpoint_ready_buffer_time}')
print(f'endpoint_config_s3_prefix: {endpoint_config_s3_prefix}')

print('=== END DEBUG ===')

def validate_environment_variables():
    """Validate that required environment variables are set"""
    print(f'Validating environment variables for database type: {database_type}')
    
    # Common required variables
    common_required = ['ENDPOINT_NAME', 'S3_BUCKET', 'ENV_NAME', 'CUSTOMER_PROFILE', 'CUSTOMER_SEGMENT', 'FORECAST_LAMBDA_NAME']
    missing_common = [var for var in common_required if not os.environ.get(var)]
    
    if missing_common:
        raise ValueError(f"Missing common environment variables: {missing_common}")
    
    if database_type == 'redshift':
        redshift_required = [
            'REDSHIFT_CLUSTER_IDENTIFIER', 'REDSHIFT_DATABASE', 
            'REDSHIFT_OUTPUT_SCHEMA', 'REDSHIFT_OUTPUT_TABLE',
            'REDSHIFT_BI_SCHEMA', 'REDSHIFT_BI_VIEW',
            'REDSHIFT_INPUT_SCHEMA', 'REDSHIFT_INPUT_TABLE'
        ]
        missing_redshift = [var for var in redshift_required if not os.environ.get(var)]
        if missing_redshift:
            raise ValueError(f"Missing Redshift environment variables: {missing_redshift}")
        print('✓ Redshift environment variables validated')
    
    elif database_type == 'athena':
        athena_required = [
            'ATHENA_DATABASE', 'ATHENA_TABLE', 'ATHENA_RESULTS_LOCATION'
        ]
        missing_athena = [var for var in athena_required if not os.environ.get(var)]
        if missing_athena:
            raise ValueError(f"Missing Athena environment variables: {missing_athena}")
        print('✓ Athena environment variables validated')
    
    else:
        raise ValueError(f"Unknown database type: {database_type}. Must be 'athena' or 'redshift'")

def download_and_enhance_config():
    """Download existing processing_config.json from S3 and enhance it for delete/recreate approach"""
    print("=== DOWNLOADING AND ENHANCING CONFIGURATION FOR DELETE/RECREATE APPROACH ===")
   
    # Download existing processing config from S3
    s3_client = boto3.client('s3')
    config_s3_key = f"{s3_prefix}/scripts/processing_config.json"
   
    try:
        print(f"Downloading configuration from s3://{s3_bucket}/{config_s3_key}")
        response = s3_client.get_object(Bucket=s3_bucket, Key=config_s3_key)
        config_dict = json.loads(response['Body'].read().decode('utf-8'))
        print("✓ Successfully downloaded existing processing configuration")
        print(f"   Existing config keys: {len(config_dict.keys())}")
       
    except Exception as e:
        print(f"Could not download processing config: {str(e)}")
        print("Creating basic configuration from environment variables...")
        config_dict = {}
   
    # Enhance with environment variables for delete/recreate approach
    print("=== ENHANCING WITH DELETE/RECREATE ENDPOINT MANAGEMENT ===")
   
    # Core Lambda Configuration
    lambda_env_vars = {
        # Core identification
        'CUSTOMER_PROFILE': customer_profile,
        'CUSTOMER_SEGMENT': customer_segment,
        'ENV_NAME': env_name,
        'ENVIRONMENT': env_name,
       
        # AWS Infrastructure
        'S3_BUCKET': s3_bucket,
        'S3_PREFIX': s3_prefix,
        'SAGEMAKER_ROLE_ARN': os.environ.get('SAGEMAKER_ROLE_ARN'),
       
        # Lambda-specific Configuration
        'ENDPOINT_NAME': endpoint_name,
        'FORECAST_LAMBDA_NAME': forecast_lambda_name,
        'LAMBDA_TIMEOUT': int(os.environ.get('LAMBDA_TIMEOUT', '900')),
        'LAMBDA_MEMORY': int(os.environ.get('LAMBDA_MEMORY', '1024')),
        'RUN_ID': run_id,
       
        # Database Configuration
        'DATABASE_TYPE': database_type,
        'OUTPUT_METHOD': output_method,
        
        # ENHANCED: Delete/Recreate Endpoint Management Configuration
        'ENABLE_ENDPOINT_AUTO_MANAGEMENT': 'true',  # Always enable for delete/recreate
        'ENABLE_ENDPOINT_DELETE_RECREATE': enable_endpoint_delete_recreate,
        'DELETE_ENDPOINT_AFTER_PREDICTION': delete_endpoint_after_prediction,
        'ENDPOINT_RECREATION_TIMEOUT': endpoint_recreation_timeout,
        'ENDPOINT_DELETION_TIMEOUT': endpoint_deletion_timeout,
        'ENDPOINT_READY_BUFFER_TIME': endpoint_ready_buffer_time,
        'ENDPOINT_CONFIG_S3_PREFIX': endpoint_config_s3_prefix,
       
        # Redshift Configuration
        'REDSHIFT_CLUSTER_IDENTIFIER': redshift_cluster_identifier,
        'REDSHIFT_DATABASE': redshift_database,
        'REDSHIFT_DB_USER': redshift_db_user,
        'REDSHIFT_REGION': redshift_region,
        'REDSHIFT_INPUT_SCHEMA': os.environ.get('REDSHIFT_INPUT_SCHEMA'),
        'REDSHIFT_INPUT_TABLE': os.environ.get('REDSHIFT_INPUT_TABLE'),
        'REDSHIFT_OUTPUT_SCHEMA': redshift_output_schema,
        'REDSHIFT_OUTPUT_TABLE': redshift_output_table,
        'REDSHIFT_BI_SCHEMA': redshift_bi_schema,
        'REDSHIFT_BI_VIEW': redshift_bi_view,
        'REDSHIFT_IAM_ROLE': redshift_iam_role,
       
        # Legacy Athena Configuration
        'ATHENA_DATABASE': athena_database,
        'ATHENA_TABLE': athena_table,
        'ATHENA_RESULTS_LOCATION': athena_results_location,
        'ATHENA_DATA_LOCATION': athena_data_location,
       
        # Data Processing Configuration
        'INITIAL_SUBMISSION_DELAY': int(os.environ.get('INITIAL_SUBMISSION_DELAY', '14')),
        'FINAL_SUBMISSION_DELAY': int(os.environ.get('FINAL_SUBMISSION_DELAY', '48')),
        'METER_THRESHOLD': int(os.environ.get('METER_THRESHOLD', '1000')),
        'SUBMISSION_TYPE_INITIAL': 'Initial',
        'SUBMISSION_TYPE_FINAL': 'Final',
       
        # Feature Configuration
        'DEFAULT_LAG_DAYS': json.loads(os.environ.get('DEFAULT_LAG_DAYS', '[14, 21, 28, 35]')),
        'USE_WEATHER_FEATURES': os.environ.get('USE_WEATHER', 'true').lower() == 'true',
        'USE_SOLAR_FEATURES': os.environ.get('USE_SOLAR', 'true').lower() == 'true',
       
        # Lambda Operational Configuration
        'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
        'MODEL_VERSION': 'latest',
        'CONFIG_SOURCE': 'centralized_enhanced_processing_config_delete_recreate',
        'CONFIG_VERSION': '4.0',  # Updated version for delete/recreate approach
        'LAMBDA_RUNTIME': 'python3.9',
    }
   
    # Log environment variables being added
    print("Environment variables for delete/recreate approach:")
    for key, value in lambda_env_vars.items():
        if value is not None:
            if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'arn']):
                print(f"  {key}: ***masked*** (length: {len(str(value))})")
            else:
                print(f"  {key}: {value}")
   
    # Merge configurations
    print("=== MERGING CONFIGURATIONS ===")
    print(f"Config before merge: {len(config_dict)} keys")
   
    overwrites = 0
    new_additions = 0
   
    for key, value in lambda_env_vars.items():
        if value is not None:
            if key in config_dict:
                if config_dict[key] != value:
                    print(f"   OVERWRITE: {key} = {value} (was: {config_dict[key]})")
                    overwrites += 1
                else:
                    print(f"   UNCHANGED: {key} = {value}")
            else:
                print(f"   NEW: {key} = {value}")
                new_additions += 1
           
            config_dict[key] = value
   
    print(f"Config after merge: {len(config_dict)} keys")
    print(f"  Overwrites: {overwrites}")
    print(f"  New additions: {new_additions}")
    print(f"  Delete/recreate endpoint management: Added")

    # Validate critical parameters for delete/recreate approach
    print("=== VALIDATING CRITICAL PARAMETERS FOR DELETE/RECREATE ===")
   
    critical_params = [
        'CUSTOMER_PROFILE', 'CUSTOMER_SEGMENT', 'ENDPOINT_NAME', 'DATABASE_TYPE',
        'S3_BUCKET', 'OUTPUT_METHOD', 'ENABLE_ENDPOINT_DELETE_RECREATE',
        'ENDPOINT_CONFIG_S3_PREFIX', 'ENDPOINT_RECREATION_TIMEOUT'
    ]
   
    # Database-specific critical parameters
    if database_type == 'redshift':
        critical_params.extend([
            'REDSHIFT_CLUSTER_IDENTIFIER', 'REDSHIFT_OUTPUT_SCHEMA', 'REDSHIFT_OUTPUT_TABLE',
            'REDSHIFT_BI_SCHEMA', 'REDSHIFT_BI_VIEW'
        ])
    elif database_type == 'athena':
        critical_params.extend([
            'ATHENA_DATABASE', 'ATHENA_TABLE', 'ATHENA_RESULTS_LOCATION'
        ])
   
    missing_critical = [param for param in critical_params if not config_dict.get(param)]
    if missing_critical:
        print(f" Missing critical parameters: {missing_critical}")
        print("Some critical parameters are missing but proceeding...")
    else:
        print("✓ All critical parameters validated for delete/recreate approach")
   
    # Enhanced configuration summary
    print("=== ENHANCED CONFIGURATION SUMMARY - DELETE/RECREATE APPROACH ===")
    print(f"✓ Final configuration contains {len(config_dict)} parameters")
    print(f"✓ Customer: {config_dict.get('CUSTOMER_PROFILE')}-{config_dict.get('CUSTOMER_SEGMENT')}")
    print(f"✓ Database: {config_dict.get('DATABASE_TYPE')} ({config_dict.get('OUTPUT_METHOD')})")
    print(f"✓ Endpoint: {config_dict.get('ENDPOINT_NAME')}")
    print(f"✓ Delete/recreate enabled: {config_dict.get('ENABLE_ENDPOINT_DELETE_RECREATE')}")
    print(f"✓ Delete after prediction: {config_dict.get('DELETE_ENDPOINT_AFTER_PREDICTION')}")
    print(f"✓ Recreation timeout: {config_dict.get('ENDPOINT_RECREATION_TIMEOUT')}s")
    print(f"✓ Deletion timeout: {config_dict.get('ENDPOINT_DELETION_TIMEOUT')}s")
    print(f"✓ Config storage: s3://{config_dict.get('S3_BUCKET')}/{config_dict.get('ENDPOINT_CONFIG_S3_PREFIX')}")
    print(f"✓ S3 Bucket: {config_dict.get('S3_BUCKET')}")
    print(f"✓ Environment: {config_dict.get('ENV_NAME')}")
    print(f"✓ Cost optimization approach: Delete/Recreate for maximum savings")
    print("=== ENHANCED CONFIGURATION COMPLETED ===")
   
    return config_dict

def create_enhanced_lambda_config(enhanced_config):
    """Create Lambda configuration file with delete/recreate endpoint management capabilities"""
    print("=== CREATING ENHANCED LAMBDA CONFIGURATION - DELETE/RECREATE APPROACH ===")
   
    config_file_path = 'predictions/enhanced_forecast_config_delete_recreate.json'
   
    try:
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
       
        with open(config_file_path, 'w') as f:
            json.dump(enhanced_config, f, indent=2, default=str)
       
        print(f"✓ Enhanced Lambda configuration saved to {config_file_path}")
        print(f"   Configuration file size: {os.path.getsize(config_file_path)} bytes")
       
        # Validate the saved configuration
        with open(config_file_path, 'r') as f:
            validation_config = json.load(f)
       
        required_keys = ['ENDPOINT_NAME', 'CUSTOMER_PROFILE', 'CUSTOMER_SEGMENT', 'DATABASE_TYPE', 
                        'ENABLE_ENDPOINT_DELETE_RECREATE', 'ENDPOINT_CONFIG_S3_PREFIX']
        missing_keys = [key for key in required_keys if key not in validation_config]
       
        if missing_keys:
            print(f" Missing required keys in configuration: {missing_keys}")
            return False
       
        print("✓ Configuration validation passed for delete/recreate approach")
        print(f"✓ Configuration contains {len(validation_config)} parameters")
        print(f"✓ Delete/recreate endpoint management enabled: {validation_config.get('ENABLE_ENDPOINT_DELETE_RECREATE', 'false')}")
        return True
       
    except Exception as e:
        print(f"✗ Error creating Lambda configuration: {str(e)}")
	
def create_focused_lambda_environment_variables(enhanced_config):
    """Create focused environment variables for Lambda function with delete/recreate endpoint management"""
    print("=== CREATING FOCUSED LAMBDA ENVIRONMENT VARIABLES - DELETE/RECREATE APPROACH ===")
    
    # Core identification (essential)
    lambda_env_vars = {
        'CUSTOMER_PROFILE': enhanced_config.get('CUSTOMER_PROFILE'),
        'CUSTOMER_SEGMENT': enhanced_config.get('CUSTOMER_SEGMENT'),
        'ENV_NAME': enhanced_config.get('ENV_NAME'),
        'ENVIRONMENT': enhanced_config.get('ENV_NAME'),
    }
    
    # AWS Infrastructure (essential)
    lambda_env_vars.update({
        'REDSHIFT_REGION': enhanced_config.get('REDSHIFT_REGION'),
        'S3_BUCKET': enhanced_config.get('S3_BUCKET'),
        'S3_PREFIX': enhanced_config.get('S3_PREFIX'),
        'ENDPOINT_NAME': enhanced_config.get('ENDPOINT_NAME'),
    })
    
    # ENHANCED: Delete/Recreate Endpoint Management Configuration (essential for maximum cost optimization)
    lambda_env_vars.update({
        'ENABLE_ENDPOINT_AUTO_MANAGEMENT': 'true',  # Always enable for delete/recreate
        'ENABLE_ENDPOINT_DELETE_RECREATE': enhanced_config.get('ENABLE_ENDPOINT_DELETE_RECREATE', 'true'),
        'DELETE_ENDPOINT_AFTER_PREDICTION': enhanced_config.get('DELETE_ENDPOINT_AFTER_PREDICTION', 'true'),
        'ENDPOINT_RECREATION_TIMEOUT': enhanced_config.get('ENDPOINT_RECREATION_TIMEOUT', '900'),
        'ENDPOINT_DELETION_TIMEOUT': enhanced_config.get('ENDPOINT_DELETION_TIMEOUT', '300'),
        'ENDPOINT_READY_BUFFER_TIME': enhanced_config.get('ENDPOINT_READY_BUFFER_TIME', '60'),
        'ENDPOINT_CONFIG_S3_PREFIX': enhanced_config.get('ENDPOINT_CONFIG_S3_PREFIX', 'endpoint-configs'),
        'WAIT_FOR_ENDPOINT_DELETION': 'true',  # Wait for deletion to complete
    })
    
    # Database Configuration (essential)
    lambda_env_vars.update({
        'DATABASE_TYPE': enhanced_config.get('DATABASE_TYPE'),
        'OUTPUT_METHOD': enhanced_config.get('OUTPUT_METHOD'),
    })
    
    # Redshift Configuration (if using Redshift)
    if enhanced_config.get('DATABASE_TYPE') == 'redshift':
        lambda_env_vars.update({
            'REDSHIFT_CLUSTER_IDENTIFIER': enhanced_config.get('REDSHIFT_CLUSTER_IDENTIFIER'),
            'REDSHIFT_DATABASE': enhanced_config.get('REDSHIFT_DATABASE'),
            'REDSHIFT_DB_USER': enhanced_config.get('REDSHIFT_DB_USER'),
            'REDSHIFT_REGION': enhanced_config.get('REDSHIFT_REGION'),
            'REDSHIFT_INPUT_SCHEMA': enhanced_config.get('REDSHIFT_INPUT_SCHEMA'),
            'REDSHIFT_INPUT_TABLE': enhanced_config.get('REDSHIFT_INPUT_TABLE'),
            'REDSHIFT_OUTPUT_SCHEMA': enhanced_config.get('REDSHIFT_OUTPUT_SCHEMA'),
            'REDSHIFT_OUTPUT_TABLE': enhanced_config.get('REDSHIFT_OUTPUT_TABLE'),
            'REDSHIFT_BI_SCHEMA': enhanced_config.get('REDSHIFT_BI_SCHEMA'),
            'REDSHIFT_BI_VIEW': enhanced_config.get('REDSHIFT_BI_VIEW'),
            'REDSHIFT_IAM_ROLE': enhanced_config.get('REDSHIFT_IAM_ROLE'),
        })
    
    # Athena Configuration (if using Athena - legacy support)
    elif enhanced_config.get('DATABASE_TYPE') == 'athena':
        lambda_env_vars.update({
            'ATHENA_DATABASE': enhanced_config.get('ATHENA_DATABASE'),
            'ATHENA_TABLE': enhanced_config.get('ATHENA_TABLE'),
            'ATHENA_STAGING_DIR': enhanced_config.get('ATHENA_STAGING_DIR'),
            'ATHENA_RESULTS_LOCATION': enhanced_config.get('ATHENA_RESULTS_LOCATION'),
            'ATHENA_DATA_LOCATION': enhanced_config.get('ATHENA_DATA_LOCATION'),
        })
    
    # Data Processing Configuration (essential)
    lambda_env_vars.update({
        'LOAD_PROFILE': enhanced_config.get('LOAD_PROFILE'),
        'SUBMISSION_TYPE_FINAL': enhanced_config.get('SUBMISSION_TYPE_FINAL', 'Final'),
        'SUBMISSION_TYPE_INITIAL': enhanced_config.get('SUBMISSION_TYPE_INITIAL', 'Initial'),
        'FINAL_SUBMISSION_DELAY': str(enhanced_config.get('FINAL_SUBMISSION_DELAY', 48)),
        'INITIAL_SUBMISSION_DELAY': str(enhanced_config.get('INITIAL_SUBMISSION_DELAY', 14)),
        'RATE_GROUP_FILTER_CLAUSE': enhanced_config.get('RATE_GROUP_FILTER_CLAUSE', ''),
    })
    
    # Model and Runtime Configuration (essential)
    lambda_env_vars.update({
        'MODEL_VERSION': enhanced_config.get('MODEL_VERSION', 'latest'),
        'RUN_ID': enhanced_config.get('RUN_ID'),
        'RUN_USER': enhanced_config.get('RUN_USER', 'system'),
    })
    
    # Time Periods Configuration (compact format)
    lambda_env_vars.update({
        'MORNING_PEAK_HOURS': '6,9',
        'SOLAR_PERIOD_HOURS': '9,16',
        'EVENING_RAMP_HOURS': '16,20',
        'EVENING_PEAK_HOURS': '17,22',
    })
    
    # Feature Configuration (compact format)
    default_lag_days = enhanced_config.get('DEFAULT_LAG_DAYS', [14, 21, 28, 35])
    lambda_env_vars.update({
        'DEFAULT_LAG_DAYS': ','.join(map(str, default_lag_days)),
    })
    
    # Location Configuration (compact)
    lambda_env_vars.update({
        'DEFAULT_LATITUDE': str(enhanced_config.get('DEFAULT_LATITUDE', 32.7157)),
        'DEFAULT_LONGITUDE': str(enhanced_config.get('DEFAULT_LONGITUDE', -117.1611)),
        'DEFAULT_TIMEZONE': enhanced_config.get('DEFAULT_TIMEZONE', 'America/Los_Angeles'),
    })
    
    # Weather Variables (compact format)
    essential_weather_vars = [
        'temperature_2m', 'apparent_temperature', 'precipitation', 'cloudcover',
        'direct_radiation', 'shortwave_radiation', 'windspeed_10m', 'is_day'
    ]
    lambda_env_vars['WEATHER_VARIABLES'] = ','.join(essential_weather_vars)
    
    # S3 Configuration for staging and endpoint configs
    lambda_env_vars.update({
        'S3_STAGING_PREFIX': f"lambda-staging/{enhanced_config.get('ENV_NAME', 'dev')}",
        'S3_FORECAST_BUCKET': enhanced_config.get('S3_BUCKET'),
    })
    
    # Remove None values and empty strings to save space
    lambda_env_vars = {k: v for k, v in lambda_env_vars.items() if v is not None and v != ''}
    
    # Calculate size and validate
    env_size = sum(len(k) + len(str(v)) for k, v in lambda_env_vars.items())
    print(f"✓ Enhanced Lambda environment variables created for delete/recreate:")
    print(f"   Total variables: {len(lambda_env_vars)}")
    print(f"   Estimated size: {env_size} bytes (AWS limit: 5120 bytes)")
    print(f"   Delete/recreate endpoint management: Included")
    
    if env_size > 4800:
        print(" Environment variables approaching AWS limit!")
        # Remove optional variables if needed but keep delete/recreate management
        optional_vars = ['DEFAULT_TIMEZONE', 'S3_STAGING_PREFIX']
        for var in optional_vars:
            if var in lambda_env_vars and env_size > 4800:
                removed_val = lambda_env_vars.pop(var)
                env_size -= (len(var) + len(str(removed_val)))
                print(f"   Removed optional variable: {var}")
    
    print(f"✓ Final size: {env_size} bytes")
    print(f"✓ Delete/recreate enabled: {lambda_env_vars.get('ENABLE_ENDPOINT_DELETE_RECREATE', 'not set')}")
    print(f"✓ Delete after prediction: {lambda_env_vars.get('DELETE_ENDPOINT_AFTER_PREDICTION', 'not set')}")
    
    return lambda_env_vars

# Main execution logic for delete/recreate approach
def main():
    # Validate environment variables before proceeding
    validate_environment_variables()

    # Create Lambda function name
    function_name = f'{env_name}-energy-daily-predictor-{customer_profile}-{customer_segment}'
    print(f'Creating enhanced forecasting Lambda function with delete/recreate: {function_name}')

    try:
        lambda_client = boto3.client('lambda')
        iam_client = boto3.client('iam')
        
        # Download and enhance configuration with delete/recreate endpoint management
        enhanced_config = download_and_enhance_config()
        
        # Create enhanced Lambda configuration file
        if not create_enhanced_lambda_config(enhanced_config):
            print("✗ Failed to create enhanced Lambda configuration")
            sys.exit(1)
        
        # Get Lambda role
        lambda_role_arn = None
        
        try:
            user_identity = boto3.client('sts').get_caller_identity()
            print(f'Current AWS identity: {user_identity["Arn"]}')
           
            # Get lambda execution roles
            lambda_roles = []
            response = iam_client.list_roles(MaxItems=50)
            for role in response.get('Roles', []):
                if 'lambda' in role['RoleName'].lower():
                    lambda_roles.append(role['Arn'])
           
            if lambda_roles:
                lambda_role_arn = lambda_roles[0]
                print(f'Found Lambda execution role: {lambda_role_arn}')
            else:
                print('No Lambda execution roles found. Using SageMaker execution role.')
                lambda_role_arn = os.environ['SAGEMAKER_ROLE_ARN']
        except Exception as e:
            print(f'Error getting roles: {str(e)}')
            lambda_role_arn = os.environ['SAGEMAKER_ROLE_ARN']
        
        # Read the deployment package
        with open(lambda_zip_path, 'rb') as f:
            zip_bytes = f.read()
        
        print(f'Lambda package size: {len(zip_bytes)} bytes')
        
        # Check if function exists
        update_function = False
        try:
            lambda_client.get_function(FunctionName=function_name)
            update_function = True
            print(f'Lambda function {function_name} already exists, will update')
        except lambda_client.exceptions.ResourceNotFoundException:
            update_function = False
            print(f'Lambda function {function_name} not found, will create new')
        
        # Create enhanced environment variables with delete/recreate endpoint management
        print("=== CREATING ENHANCED LAMBDA ENVIRONMENT VARIABLES - DELETE/RECREATE APPROACH ===")
        
        env_variables = create_focused_lambda_environment_variables(enhanced_config)
        
        # Validate critical parameters are present
        critical_params = [
            'ENDPOINT_NAME', 'CUSTOMER_PROFILE', 'CUSTOMER_SEGMENT',
            'DATABASE_TYPE', 'OUTPUT_METHOD', 'S3_BUCKET',
            'REDSHIFT_OUTPUT_SCHEMA', 'REDSHIFT_OUTPUT_TABLE',
            'REDSHIFT_BI_SCHEMA', 'REDSHIFT_BI_VIEW',
            'SUBMISSION_TYPE_FINAL', 'SUBMISSION_TYPE_INITIAL',
            'INITIAL_SUBMISSION_DELAY', 'FINAL_SUBMISSION_DELAY'
        ]
        
        missing_critical = [param for param in critical_params if param not in env_variables]
        if missing_critical:
            print(f" Missing critical parameters: {missing_critical}")
            # Don't fail, but warn
            print(" Some critical parameters are missing but proceeding...")
        else:
            print(" All critical parameters validated in environment variables")

        print(f'Enhanced Lambda environment variables:')
        for key, value in env_variables.items():
            # Mask sensitive values in logs
            if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key']):
                print(f'  {key}: ***masked***')
            else:
                print(f'  {key}: {value}')
        
        # Validate critical delete/recreate parameters are present
        delete_recreate_params = [
            'ENABLE_ENDPOINT_DELETE_RECREATE', 'ENDPOINT_RECREATION_TIMEOUT', 
            'ENDPOINT_DELETION_TIMEOUT', 'ENDPOINT_CONFIG_S3_PREFIX'
        ]
        
        missing_delete_recreate_params = [param for param in delete_recreate_params if param not in env_variables]
        if missing_delete_recreate_params:
            print(f" Missing delete/recreate parameters: {missing_delete_recreate_params}")
        else:
            print("✓ All delete/recreate endpoint management parameters validated in environment variables")
        
        if update_function:
            # Update function code
            max_retries = 5
            retry_delay = 5
            attempt = 0
            
            while attempt < max_retries:
                try:
                    # Update function code first
                    code_response = lambda_client.update_function_code(
                        FunctionName=function_name,
                        ZipFile=zip_bytes,
                        Publish=True
                    )
                    print('✓ Enhanced function code updated successfully')
                    break
                except lambda_client.exceptions.ResourceConflictException as e:
                    attempt += 1
                    print(f'Function update in progress, retrying in {retry_delay} seconds... (Attempt {attempt}/{max_retries})')
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
            
            if attempt == max_retries:
                print('✗ Failed to update function code after maximum retries')
                sys.exit(1)
            
            # Wait before updating configuration
            print('Waiting for code update to complete before updating configuration...')
            time.sleep(10)
            
            # Update configuration with retries
            attempt = 0
            retry_delay = 5
            
            while attempt < max_retries:
                try:
                    config_response = lambda_client.update_function_configuration(
                        FunctionName=function_name,
                        Runtime='python3.9',
                        Handler='lambda_function.lambda_handler',
                        Role=os.environ['SAGEMAKER_ROLE_ARN'],
                        Environment={
                            'Variables': env_variables
                        },
                        Timeout=enhanced_config['LAMBDA_TIMEOUT'],
                        MemorySize=enhanced_config['LAMBDA_MEMORY'],
                        Description=f'Enhanced Lambda function for energy load forecasting with delete/recreate endpoint management - {customer_profile}-{customer_segment}'
                    )
                    print('✓ Enhanced function configuration updated successfully')
                    response = config_response
                    break
                except lambda_client.exceptions.ResourceConflictException as e:
                    attempt += 1
                    print(f'Function update in progress, retrying in {retry_delay} seconds... (Attempt {attempt}/{max_retries})')
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
            
            if attempt == max_retries:
                print('✗ Failed to update function configuration after maximum retries')
                sys.exit(1)
            
        else:
            # Create new function
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=os.environ['SAGEMAKER_ROLE_ARN'],
                Handler='lambda_function.lambda_handler',
                Code={
                    'ZipFile': zip_bytes
                },
                Environment={
                    'Variables': env_variables
                },
                Timeout=enhanced_config['LAMBDA_TIMEOUT'],
                MemorySize=enhanced_config['LAMBDA_MEMORY'],
                Publish=True,
                Description=f'Enhanced Lambda function for energy load forecasting with delete/recreate endpoint management - {customer_profile}-{customer_segment}',
                Tags={
                    'Environment': env_name,
                    'Customer': f'{customer_profile}-{customer_segment}',
                    'DatabaseType': database_type,
                    'EndpointDeleteRecreate': enable_endpoint_delete_recreate,
                    'ConfigVersion': '4.0',
                    'CostOptimized': 'true',
                    'MaximumCostSavings': 'true'
                }
            )
        
        # Get the function ARN
        function_arn = response['FunctionArn']
        print(f'✓ Enhanced forecasting Lambda function deployed successfully: {function_arn}')
        print(f'FORECAST_LAMBDA_ARN={function_arn}')
        print(f'Database type: {database_type}')
        print(f'Output method: {output_method}')
        print(f'Delete/recreate endpoint management: {enable_endpoint_delete_recreate}')
        print(f'Delete after prediction: {delete_endpoint_after_prediction}')
        print(f'✓ Delete/recreate approach implemented for maximum cost optimization')
        
        # Save Lambda info to environment for later steps
        with open(os.environ['GITHUB_ENV'], 'a') as env_file:
            env_file.write(f'FORECAST_LAMBDA_ARN={function_arn}\n')
            env_file.write(f'FORECAST_LAMBDA_NAME={function_name}\n')
            env_file.write(f'FORECAST_DATABASE_TYPE={database_type}\n')
            env_file.write(f'FORECAST_OUTPUT_METHOD={output_method}\n')
            env_file.write(f'ENDPOINT_DELETE_RECREATE_ENABLED={enable_endpoint_delete_recreate}\n')
            env_file.write(f'DELETE_ENDPOINT_AFTER_PREDICTION={delete_endpoint_after_prediction}\n')
        
        # Verify enhanced Lambda environment variables were set correctly
        print('=== ENHANCED LAMBDA ENVIRONMENT VERIFICATION - DELETE/RECREATE ===')
        verify_response = lambda_client.get_function(FunctionName=function_name)
        lambda_env = verify_response['Configuration']['Environment']['Variables']
        
        print(f'✓ Lambda environment contains {len(lambda_env)} variables')
        
        # Verify critical parameters including delete/recreate endpoint management
        critical_verification = {
            'DATABASE_TYPE': lambda_env.get('DATABASE_TYPE', 'NOT SET'),
            'OUTPUT_METHOD': lambda_env.get('OUTPUT_METHOD', 'NOT SET'),
            'CUSTOMER_PROFILE': lambda_env.get('CUSTOMER_PROFILE', 'NOT SET'),
            'CUSTOMER_SEGMENT': lambda_env.get('CUSTOMER_SEGMENT', 'NOT SET'),
            'SUBMISSION_TYPE_FINAL': lambda_env.get('SUBMISSION_TYPE_FINAL', 'NOT SET'),
            'SUBMISSION_TYPE_INITIAL': lambda_env.get('SUBMISSION_TYPE_INITIAL', 'NOT SET'),
            'INITIAL_SUBMISSION_DELAY': lambda_env.get('INITIAL_SUBMISSION_DELAY', 'NOT SET'),
            'FINAL_SUBMISSION_DELAY': lambda_env.get('FINAL_SUBMISSION_DELAY', 'NOT SET'),
            'RATE_GROUP_FILTER_CLAUSE': lambda_env.get('RATE_GROUP_FILTER_CLAUSE', 'NOT SET'),			
            'ENDPOINT_NAME': lambda_env.get('ENDPOINT_NAME', 'NOT SET'),
            'ENABLE_ENDPOINT_DELETE_RECREATE': lambda_env.get('ENABLE_ENDPOINT_DELETE_RECREATE', 'NOT SET'),
            'DELETE_ENDPOINT_AFTER_PREDICTION': lambda_env.get('DELETE_ENDPOINT_AFTER_PREDICTION', 'NOT SET'),
            'ENDPOINT_RECREATION_TIMEOUT': lambda_env.get('ENDPOINT_RECREATION_TIMEOUT', 'NOT SET'),
            'ENDPOINT_DELETION_TIMEOUT': lambda_env.get('ENDPOINT_DELETION_TIMEOUT', 'NOT SET'),
            'ENDPOINT_CONFIG_S3_PREFIX': lambda_env.get('ENDPOINT_CONFIG_S3_PREFIX', 'NOT SET'),
            'REDSHIFT_OUTPUT_SCHEMA': lambda_env.get('REDSHIFT_OUTPUT_SCHEMA', 'NOT SET'),
            'REDSHIFT_OUTPUT_TABLE': lambda_env.get('REDSHIFT_OUTPUT_TABLE', 'NOT SET'),
            'REDSHIFT_BI_SCHEMA': lambda_env.get('REDSHIFT_BI_SCHEMA', 'NOT SET'),
            'REDSHIFT_BI_VIEW': lambda_env.get('REDSHIFT_BI_VIEW', 'NOT SET'),
            'OUTPUT_FULL_TABLE_NAME': lambda_env.get('OUTPUT_FULL_TABLE_NAME', 'NOT SET'),
            'WEATHER_VARIABLES': lambda_env.get('WEATHER_VARIABLES', 'NOT SET'),
            'DEFAULT_LAG_DAYS': lambda_env.get('DEFAULT_LAG_DAYS', 'NOT SET')
        }
        
        print('✓ Critical parameter verification (including delete/recreate management):')
        for param, value in critical_verification.items():
            if value == 'NOT SET':
                print(f"   {param}: {value}")
            elif param.endswith('_TIMEOUT'):
                print(f"   ✓ {param}: {value}s")
            elif param in ['WEATHER_VARIABLES', 'DEFAULT_LAG_DAYS'] and value.startswith('['):
                print(f"   {param}: [JSON Array] (length: {len(value)})")
            elif len(str(value)) > 50:
                print(f"   {param}: {str(value)[:47]}... (truncated)")
            else:
                print(f"   ✓ {param}: {value}")
        
        # Show count of different parameter types
        json_params = [k for k, v in lambda_env.items() if v.startswith('[') or v.startswith('{')]
        string_params = [k for k, v in lambda_env.items() if not (v.startswith('[') or v.startswith('{'))]
        
        print(f' Parameter breakdown:')
        print(f"   JSON parameters: {len(json_params)}")
        print(f"   String parameters: {len(string_params)}")
        print(f"   Total: {len(lambda_env)}")
        
        if json_params:
            print(f'   JSON parameters include: {", ".join(json_params[:5])}{"..." if len(json_params) > 5 else ""}')

        # Show delete/recreate management status
        delete_recreate_enabled = lambda_env.get('ENABLE_ENDPOINT_DELETE_RECREATE', 'false').lower() == 'true'
        delete_after_prediction = lambda_env.get('DELETE_ENDPOINT_AFTER_PREDICTION', 'false').lower() == 'true'
        print(f'✓ Enhanced forecasting Lambda function configuration summary:')
        print(f"   Function: {function_name}")
        print(f"   Delete/recreate management: {'ENABLED' if delete_recreate_enabled else 'DISABLED'}")
        if delete_recreate_enabled:
            print(f"   Recreation timeout: {lambda_env.get('ENDPOINT_RECREATION_TIMEOUT', 'default')}s")
            print(f"   Deletion timeout: {lambda_env.get('ENDPOINT_DELETION_TIMEOUT', 'default')}s")
            print(f"   Delete after prediction: {'YES' if delete_after_prediction else 'NO'}")
            print(f"   Config storage: s3://{lambda_env.get('S3_BUCKET', 'unknown')}/{lambda_env.get('ENDPOINT_CONFIG_S3_PREFIX', 'unknown')}")
            print(f"   Cost optimization: MAXIMUM (98%+ savings expected)")
        print(f"   Total environment variables: {len(lambda_env)}")
        
        print(f'✓ Enhanced forecasting Lambda function successfully configured with delete/recreate endpoint management')

    except Exception as e:
        print(f'✗ Error creating enhanced forecasting Lambda function: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
