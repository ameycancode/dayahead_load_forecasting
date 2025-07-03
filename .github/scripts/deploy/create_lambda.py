#!/usr/bin/env python3
"""
Enhanced Lambda Function Creation Script for Energy Load Forecasting Pipeline
Creates a Lambda function for deploying SageMaker models to endpoints with cost optimization
This version is inline with create_lambda_current_1.py but adds endpoint management features
"""
import boto3
import json
import io
import zipfile
import os
import sys
import time

# Get environment variables - now all sourced from deploy.yml (same as your current version)
role = os.environ["SAGEMAKER_ROLE_ARN"]
lambda_function_name = os.environ["LAMBDA_FUNCTION_NAME"]

# Lambda configuration from environment (same as your current version)
lambda_timeout = int(os.environ.get("LAMBDA_TIMEOUT", "300"))
lambda_memory = int(os.environ.get("LAMBDA_MEMORY", "128"))
lambda_runtime = os.environ.get("LAMBDA_RUNTIME", "python3.9")

# ENHANCED: Cost optimization configuration
enable_cost_optimization = os.environ.get("ENABLE_IMMEDIATE_COST_OPTIMIZATION", "true").lower() == "true"

print(f"=== ENHANCED LAMBDA FUNCTION CREATION WITH COST OPTIMIZATION ===")
print(f"Creating/updating Lambda function: {lambda_function_name}")
print(f"Using SageMaker role for operations: {role}")
print(f"Lambda configuration: timeout={lambda_timeout}s, memory={lambda_memory}MB, runtime={lambda_runtime}")
print(f"Cost optimization enabled: {enable_cost_optimization}")

def wait_for_lambda_ready(lambda_client, function_name, max_wait_time=300):
    """Wait for Lambda function to be ready for updates (same as your current version)"""
    print(f"Waiting for Lambda function to be ready: {function_name}")
    waited_time = 0
   
    while waited_time < max_wait_time:
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            state = response['Configuration']['State']
            last_update_status = response['Configuration']['LastUpdateStatus']
           
            print(f"Lambda state: {state}, Last update status: {last_update_status} (waited {waited_time}s)")
           
            if state == 'Active' and last_update_status == 'Successful':
                print("✅ Lambda function is ready for operations")
                return True
            elif state == 'Failed' or last_update_status == 'Failed':
                print(f"❌ Lambda function is in failed state: {state}/{last_update_status}")
                return False
            else:
                print(f"⏳ Lambda function not ready yet: {state}/{last_update_status}")
                time.sleep(10)
                waited_time += 10
               
        except lambda_client.exceptions.ResourceNotFoundException:
            print("Lambda function doesn't exist yet, can proceed with creation")
            return True
        except Exception as e:
            print(f"⚠️ Error checking Lambda function state: {str(e)}")
            time.sleep(5)
            waited_time += 5
   
    print(f"❌ Timeout waiting for Lambda function to be ready after {max_wait_time} seconds")
    return False

def update_lambda_with_retry(lambda_client, function_name, **update_params):
    """Update Lambda function with retry logic for conflicts (same as your current version)"""
    max_retries = 5
    retry_count = 0
   
    while retry_count < max_retries:
        try:
            if 'ZipFile' in update_params:
                # Update code
                print(f"Updating Lambda code (attempt {retry_count + 1}/{max_retries})")
                lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=update_params['ZipFile']
                )
               
                # Wait a bit before updating configuration
                print("Waiting 10 seconds before updating configuration...")
                time.sleep(10)
               
                # Update configuration
                print(f"Updating Lambda configuration (attempt {retry_count + 1}/{max_retries})")
                lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Timeout=update_params['Timeout'],
                    MemorySize=update_params['MemorySize'],
                    Runtime=update_params['Runtime'],
                    Description=update_params.get('Description', 'Enhanced Lambda function for energy forecasting model deployment with cost optimization')
                )
            else:
                # Just update configuration
                print(f"Updating Lambda configuration (attempt {retry_count + 1}/{max_retries})")
                lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Timeout=update_params['Timeout'],
                    MemorySize=update_params['MemorySize'],
                    Runtime=update_params['Runtime'],
                    Description=update_params.get('Description', 'Enhanced Lambda function for energy forecasting model deployment with cost optimization')
                )
           
            print("✅ Lambda function updated successfully")
            return True
           
        except lambda_client.exceptions.ResourceConflictException as e:
            retry_count += 1
            wait_time = min(30 * retry_count, 120)  # Exponential backoff, max 2 minutes
            print(f"⚠️ ResourceConflictException (attempt {retry_count}/{max_retries}): {str(e)}")
           
            if retry_count < max_retries:
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
               
                # Check if function is ready before retrying
                if not wait_for_lambda_ready(lambda_client, function_name, max_wait_time=60):
                    print("❌ Lambda function not ready after waiting, skipping remaining retries")
                    break
            else:
                print(f"❌ Failed to update Lambda after {max_retries} attempts")
                return False
               
        except Exception as e:
            print(f"❌ Unexpected error updating Lambda: {str(e)}")
            return False
   
    return False

try:
    lambda_client = boto3.client('lambda')
    iam_client = boto3.client('iam')
   
    # ENHANCED: Lambda function code with cost optimization features
    lambda_code = """
import boto3
import json
import logging
import time
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def sanitize_for_logging(value: Any, max_length: int = 200) -> str:
    \"\"\"
    Sanitize input for safe logging by removing/escaping dangerous characters.
   
    Args:
        value: The value to sanitize
        max_length: Maximum length of the sanitized string
       
    Returns:
        Sanitized string safe for logging
    \"\"\"
    if value is None:
        return "None"
   
    # Convert to string
    str_value = str(value)
   
    # Remove or replace dangerous characters
    # Remove newlines, carriage returns, and other control characters
    sanitized = re.sub(r'[\\r\\n\\t\\x00-\\x1f\\x7f-\\x9f]', '', str_value)
   
    # Remove ANSI escape sequences
    sanitized = re.sub(r'\\x1b\\[[0-9;]*m', '', sanitized)
   
    # Truncate if too long and add ellipsis
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
   
    # Escape any remaining problematic characters for JSON safety
    sanitized = sanitized.replace('\\\\', '\\\\\\\\').replace('"', '\\\\"')
   
    return sanitized

def validate_arn(arn: str) -> bool:
    \"\"\"
    Validate that the provided string is a valid AWS ARN format.
   
    Args:
        arn: The ARN string to validate
       
    Returns:
        True if valid ARN format, False otherwise
    \"\"\"
    if not arn:
        return False
   
    # Basic ARN format: arn:partition:service:region:account:resource
    arn_pattern = r'^arn:[a-zA-Z0-9-]+:[a-zA-Z0-9-]+:[a-zA-Z0-9-]*:[0-9]*:.+$'
    return bool(re.match(arn_pattern, arn))

def validate_endpoint_name(name: str) -> bool:
    \"\"\"
    Validate endpoint name follows SageMaker naming conventions.
   
    Args:
        name: The endpoint name to validate
       
    Returns:
        True if valid, False otherwise
    \"\"\"
    if not name:
        return False
   
    # SageMaker endpoint names: 1-63 chars, alphanumeric and hyphens only
    pattern = r'^[a-zA-Z0-9\\-]{1,63}$'
    return bool(re.match(pattern, name))

def redact_sensitive_info(text: str) -> str:
    \"\"\"
    Redact potentially sensitive information from text.
   
    Args:
        text: Text that might contain sensitive information
       
    Returns:
        Text with sensitive information redacted
    \"\"\"
    # Redact AWS account IDs (12 digits)
    text = re.sub(r'\\b\\d{12}\\b', '***ACCOUNT_ID***', text)
   
    # Redact what looks like access keys
    text = re.sub(r'AKIA[0-9A-Z]{16}', '***ACCESS_KEY***', text)
   
    # Redact what looks like secret keys (base64-like strings over 20 chars)
    text = re.sub(r'[A-Za-z0-9/+=]{20,}', '***SECRET***', text)
   
    return text

def log_secure(level: str, message: str, **kwargs):
    \"\"\"
    Secure logging function that sanitizes all inputs.
   
    Args:
        level: Log level (info, error, warning, debug)
        message: Base message to log
        **kwargs: Additional key-value pairs to log safely
    \"\"\"
    # Sanitize the base message
    safe_message = sanitize_for_logging(message)
   
    # Sanitize and format additional parameters
    safe_params = {}
    for key, value in kwargs.items():
        safe_key = sanitize_for_logging(key, 50)
        safe_value = sanitize_for_logging(value)
        # Redact sensitive information
        safe_value = redact_sensitive_info(safe_value)
        safe_params[safe_key] = safe_value
   
    # Create final log message
    if safe_params:
        param_str = " ".join([f"{k}={v}" for k, v in safe_params.items()])
        final_message = f"{safe_message} | {param_str}"
    else:
        final_message = safe_message
   
    # Log based on level
    if level.lower() == 'info':
        logger.info(final_message)
    elif level.lower() == 'error':
        logger.error(final_message)
    elif level.lower() == 'warning':
        logger.warning(final_message)
    elif level.lower() == 'debug':
        logger.debug(final_message)
    else:
        logger.info(final_message)

def create_zero_instance_endpoint_config(sm_client, original_config_name, endpoint_name):
    \"\"\"
    ENHANCED: Create zero-instance endpoint configuration for cost optimization
    \"\"\"
    try:
        log_secure('info', "Creating zero-instance configuration for cost optimization",
                  original_config=original_config_name)
       
        # Get original configuration
        config_response = sm_client.describe_endpoint_config(
            EndpointConfigName=original_config_name
        )
       
        # Create zero-instance config name
        zero_config_name = f"{endpoint_name}-zero-{int(time.time())}"
       
        # Create zero-instance variants
        zero_variants = []
        for variant in config_response['ProductionVariants']:
            zero_variant = {
                'VariantName': f"zero-{variant['VariantName']}",
                'ModelName': variant['ModelName'],
                'InitialInstanceCount': 0,  # Zero instances for cost optimization
                'InstanceType': variant['InstanceType'],
                'InitialVariantWeight': variant.get('InitialVariantWeight', 1.0)
            }
            zero_variants.append(zero_variant)
       
        # Create zero-instance endpoint configuration
        sm_client.create_endpoint_config(
            EndpointConfigName=zero_config_name,
            ProductionVariants=zero_variants,
            Tags=[
                {'Key': 'CostOptimized', 'Value': 'true'},
                {'Key': 'ConfigType', 'Value': 'zero-instance'},
                {'Key': 'OriginalConfig', 'Value': original_config_name}
            ]
        )
       
        log_secure('info', "Zero-instance configuration created successfully",
                  zero_config_name=zero_config_name)
        return zero_config_name
       
    except Exception as e:
        log_secure('error', "Failed to create zero-instance configuration", error=str(e))
        return None

def lambda_handler(event, context):
    try:
        # Get and validate parameters
        model_package_arn = event.get('model_package_arn')
        endpoint_name = event.get('endpoint_name', 'energy-forecast-endpoint')
        instance_type = event.get('instance_type', 'ml.m5.large')
        instance_count = int(event.get('instance_count', 1))
        execution_role = event.get('execution_role')
        run_id = event.get('run_id', '')
       
        # ENHANCED: Cost optimization parameters
        enable_cost_optimization = event.get('enable_cost_optimization', True)
        create_zero_config = event.get('create_zero_config', True)
       
        # Input validation
        if not model_package_arn or not validate_arn(model_package_arn):
            raise ValueError("Invalid model_package_arn provided")
           
        if not validate_endpoint_name(endpoint_name):
            raise ValueError("Invalid endpoint_name provided")
           
        if not execution_role or not validate_arn(execution_role):
            raise ValueError("Invalid execution_role provided")
       
        # Safe logging with sanitization
        log_secure('info', "Starting enhanced model deployment with cost optimization",
                  model_package_arn=model_package_arn,
                  endpoint_name=endpoint_name,
                  run_id=run_id,
                  cost_optimization=enable_cost_optimization)
       
        sm_client = boto3.client('sagemaker')
       
        # Create model from package
        model_name = f"energy-forecast-model-{int(time.time())}"
        log_secure('info', "Creating model", model_name=model_name)
       
        # Get model package details
        try:
            package_details = sm_client.describe_model_package(
                ModelPackageName=model_package_arn
            )
        except Exception as e:
            log_secure('error', "Failed to describe model package", error=str(e))
            raise
       
        # Get the model data URL and container details
        containers = package_details.get('InferenceSpecification', {}).get('Containers', [])
        if not containers:
            raise ValueError("No containers found in model package")
           
        model_data_url = containers[0].get('ModelDataUrl')
        container_image = containers[0].get('Image')
       
        if not model_data_url or not container_image:
            raise ValueError("Missing model data URL or container image")
       
        # Create model
        try:
            sm_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=execution_role,
                PrimaryContainer={
                    'Image': container_image,
                    'ModelDataUrl': model_data_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_MODEL_SERVER_WORKERS': '1'
                    }
                },
                Tags=[
                    {
                        'Key': 'run_id',
                        'Value': sanitize_for_logging(run_id, 50)
                    },
                    {
                        'Key': 'CostOptimized',
                        'Value': 'true' if enable_cost_optimization else 'false'
                    }
                ]
            )
            log_secure('info', "Model created successfully", model_name=model_name)
        except Exception as e:
            log_secure('error', "Failed to create model", model_name=model_name, error=str(e))
            raise
       
        # Create endpoint config
        timestamp = int(time.time())
        endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
        log_secure('info', "Creating endpoint config", endpoint_config_name=endpoint_config_name)
       
        try:
            sm_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': model_name,
                        'InstanceType': instance_type,
                        'InitialInstanceCount': instance_count,
                        'InitialVariantWeight': 1
                    }
                ],
                Tags=[
                    {'Key': 'CostOptimized', 'Value': 'true' if enable_cost_optimization else 'false'},
                    {'Key': 'ConfigType', 'Value': 'production'}
                ]
            )
            log_secure('info', "Endpoint config created successfully", endpoint_config_name=endpoint_config_name)
        except Exception as e:
            log_secure('error', "Failed to create endpoint config", endpoint_config_name=endpoint_config_name, error=str(e))
            raise
       
        # ENHANCED: Create zero-instance configuration if requested
        zero_config_name = None
        if create_zero_config:
            zero_config_name = create_zero_instance_endpoint_config(
                sm_client, endpoint_config_name, endpoint_name
            )
       
        # Check if endpoint exists
        endpoint_exists = False
        try:
            sm_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_exists = True
        except sm_client.exceptions.ClientError:
            pass
       
        # Create or update endpoint
        try:
            if endpoint_exists:
                log_secure('info', "Updating existing endpoint", endpoint_name=endpoint_name)
                sm_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
            else:
                log_secure('info', "Creating new endpoint", endpoint_name=endpoint_name)
                sm_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name,
                    Tags=[
                        {'Key': 'CostOptimized', 'Value': 'true' if enable_cost_optimization else 'false'},
                        {'Key': 'ZeroConfig', 'Value': zero_config_name or 'none'}
                    ]
                )
               
            log_secure('info', "Endpoint operation completed successfully",
                      endpoint_name=endpoint_name,
                      operation="update" if endpoint_exists else "create")
        except Exception as e:
            log_secure('error', "Failed to create/update endpoint", endpoint_name=endpoint_name, error=str(e))
            raise
       
        # ENHANCED: Return response with cost optimization information
        response_body = {
            'message': f"Enhanced deployment initiated for {sanitize_for_logging(run_id)}",
            'endpoint_name': sanitize_for_logging(endpoint_name),
            'model_name': sanitize_for_logging(model_name),
            'endpoint_config_name': sanitize_for_logging(endpoint_config_name),
            'model_package_arn': sanitize_for_logging(model_package_arn),
            'cost_optimization_enabled': enable_cost_optimization,
            'zero_config_name': sanitize_for_logging(zero_config_name) if zero_config_name else None
        }
       
        return {
            'statusCode': 200,
            'body': response_body
        }
   
    except Exception as e:
        # Sanitize error messages before logging
        error_message = redact_sensitive_info(str(e))
        log_secure('error', "Error in enhanced deployment", error=error_message)
       
        # Don't log full traceback in production to avoid information leakage
        if os.environ.get('AWS_LAMBDA_LOG_LEVEL', '').upper() == 'DEBUG':
            import traceback
            sanitized_traceback = redact_sensitive_info(traceback.format_exc())
            log_secure('debug', "Full traceback", traceback=sanitized_traceback)
       
        return {
            'statusCode': 500,
            'body': {
                'error': 'Internal server error occurred'
            }
        }
"""
   
    # Create zip file for Lambda function (same as your current version)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr('lambda_function.py', lambda_code)
   
    zip_buffer.seek(0)
    zip_file_bytes = zip_buffer.read()
   
    # Check if Lambda function exists (same as your current version)
    try:
        existing_function = lambda_client.get_function(FunctionName=lambda_function_name)
        current_role = existing_function['Configuration']['Role']
        print(f"Updating existing Lambda function: {lambda_function_name}")
        print(f"Current execution role: {current_role}")
       
        # Wait for function to be ready for updates
        if not wait_for_lambda_ready(lambda_client, lambda_function_name):
            print("❌ Lambda function is not ready for updates, skipping")
            sys.exit(1)
       
        # Update function with retry logic
        success = update_lambda_with_retry(
            lambda_client=lambda_client,
            function_name=lambda_function_name,
            ZipFile=zip_file_bytes,
            Timeout=lambda_timeout,
            MemorySize=lambda_memory,
            Runtime=lambda_runtime,
            Description=f'Enhanced Lambda function for energy forecasting model deployment with cost optimization'
        )
       
        if success:
            print(f"✅ Enhanced Lambda function updated successfully")
            print(f"💰 Cost optimization features: {'Enabled' if enable_cost_optimization else 'Disabled'}")
        else:
            print(f"❌ Failed to update Lambda function after retries")
            sys.exit(1)
       
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"Creating new enhanced Lambda function: {lambda_function_name}")
       
        # Create new function with environment-configured parameters
        try:
            response = lambda_client.create_function(
                FunctionName=lambda_function_name,
                Runtime=lambda_runtime,
                Role=role,
                Handler='lambda_function.lambda_handler',
                Code={
                    'ZipFile': zip_file_bytes
                },
                Description=f'Enhanced Lambda function for energy forecasting model deployment with cost optimization',
                Timeout=lambda_timeout,
                MemorySize=lambda_memory,
                Publish=True,
                Tags={
                    'Environment': os.environ.get('ENV_NAME', 'dev'),
                    'CostOptimized': 'true' if enable_cost_optimization else 'false',
                    'Enhanced': 'true'
                }
            )
            print(f"✅ Enhanced Lambda function created: {response['FunctionArn']}")
            print(f"💰 Cost optimization features: {'Enabled' if enable_cost_optimization else 'Disabled'}")
           
            # Save Lambda ARN to environment for next steps
            with open(os.environ["GITHUB_ENV"], "a") as env_file:
                env_file.write(f"LAMBDA_ARN={response['FunctionArn']}\n")
                env_file.write(f"LAMBDA_CREATED=true\n")
                env_file.write(f"LAMBDA_COST_OPTIMIZATION={str(enable_cost_optimization).lower()}\n")
               
        except Exception as create_error:
            print(f"❌ Error creating enhanced Lambda function: {str(create_error)}")
            sys.exit(1)
   
    print("✅ Enhanced Lambda function ready with cost optimization features")
   
except Exception as e:
    print(f"❌ Error creating enhanced Lambda function: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== ENHANCED LAMBDA CREATION COMPLETED ===")
if enable_cost_optimization:
    print("🎯 Cost optimization enabled in deployment Lambda!")
    print("📊 Monitor deployment logs for zero-instance configuration creation")
else:
    print("ℹ️ Cost optimization disabled - consider enabling for cost savings")
print("✅ Enhanced Lambda creation script completed successfully")
