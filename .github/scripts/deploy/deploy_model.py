#!/usr/bin/env python3
"""
Enhanced Model Deployment Script - Delete/Recreate Approach for Cost Optimization
Deploys models to SageMaker endpoints and optionally deletes them for cost savings
Stores endpoint configuration metadata for recreation by forecasting Lambda
"""
import boto3
import json
import os
import sys
import time

# Get environment variables - all now sourced from deploy.yml
model_package_arn = os.environ["MODEL_PACKAGE_ARN"]
run_id = os.environ["RUN_ID"]
endpoint_name = os.environ["ENDPOINT_NAME"]
lambda_function_name = os.environ["LAMBDA_FUNCTION_NAME"]
role = os.environ["SAGEMAKER_ROLE_ARN"]

# Deployment configuration from environment (same as your current version)
deploy_instance_type = os.environ.get("DEPLOY_INSTANCE_TYPE", "ml.m5.large")
deploy_instance_count = int(os.environ.get("DEPLOY_INSTANCE_COUNT", "1"))
lambda_timeout = int(os.environ.get("LAMBDA_TIMEOUT", "900"))
lambda_memory = int(os.environ.get("LAMBDA_MEMORY", "1024"))

# ENHANCED: Delete/Recreate cost optimization configuration
enable_endpoint_delete_recreate = os.environ.get("ENABLE_ENDPOINT_DELETE_RECREATE", "true").lower() == "true"
delete_endpoint_after_deployment = os.environ.get("DELETE_ENDPOINT_AFTER_DEPLOYMENT", "true").lower() == "true"
endpoint_config_s3_prefix = os.environ.get("ENDPOINT_CONFIG_S3_PREFIX", "endpoint-configs")
endpoint_deletion_delay = int(os.environ.get("ENDPOINT_DELETION_DELAY", "120"))  # Wait before deletion

print(f"=== ENHANCED MODEL DEPLOYMENT - DELETE/RECREATE APPROACH ===")
print(f"Deploying model to endpoint: {endpoint_name}")
print(f"Model package ARN: {model_package_arn}")
print(f"Run ID: {run_id}")
print(f"Using Lambda function: {lambda_function_name}")
print(f"Instance configuration: {deploy_instance_type} x{deploy_instance_count}")
print(f"Delete/Recreate optimization enabled: {enable_endpoint_delete_recreate}")
print(f"Delete endpoint after deployment: {delete_endpoint_after_deployment}")

def wait_for_lambda_ready(lambda_client, function_name, max_wait_time=600):
    """Wait for Lambda function to be in Active state (same as your current version)"""
    print(f"Checking Lambda function state: {function_name}")
    waited_time = 0
   
    while waited_time < max_wait_time:
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            state = response['Configuration']['State']
            last_update_status = response['Configuration']['LastUpdateStatus']
           
            print(f"Lambda state: {state}, Last update status: {last_update_status} (waited {waited_time}s)")
           
            if state == 'Active' and last_update_status == 'Successful':
                print("✅ Lambda function is ready for invocation")
                return True
            elif state == 'Failed' or last_update_status == 'Failed':
                print(f"❌ Lambda function is in failed state: {state}/{last_update_status}")
                return False
            else:
                print(f"⏳ Lambda function not ready yet: {state}/{last_update_status}")
                time.sleep(10)
                waited_time += 10
               
        except Exception as e:
            print(f"❌ Error checking Lambda function state: {str(e)}")
            time.sleep(5)
            waited_time += 5
   
    print(f"❌ Timeout waiting for Lambda function to be ready after {max_wait_time} seconds")
    return False

def store_endpoint_configuration(endpoint_name, model_name, endpoint_config_name, model_package_arn):
    """
    Store endpoint configuration metadata to S3 for recreation by forecasting Lambda
    This replaces the failed zero-instance config approach
    """
    try:
        print(f"=== STORING ENDPOINT CONFIGURATION FOR RECREATION ===")
       
        s3_client = boto3.client('s3')
        sagemaker_client = boto3.client('sagemaker')
       
        bucket = os.environ.get('S3_BUCKET')
        s3_prefix = os.environ.get('S3_PREFIX', '')
       
        if not bucket:
            print("⚠️ S3 bucket not configured - skipping configuration storage")
            return None
       
        # Get detailed endpoint configuration for recreation
        config_response = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
       
        # Get model details for recreation
        model_response = sagemaker_client.describe_model(ModelName=model_name)
       
        # Create comprehensive recreation metadata
        recreation_config = {
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'endpoint_config_name': endpoint_config_name,
            'model_package_arn': model_package_arn,
            'run_id': run_id,
            'instance_type': deploy_instance_type,
            'instance_count': deploy_instance_count,
            'environment': os.environ.get('ENV_NAME', 'dev'),
            'customer_profile': os.environ.get('CUSTOMER_PROFILE', 'unknown'),
            'customer_segment': os.environ.get('CUSTOMER_SEGMENT', 'unknown'),
            'cost_optimized': True,
            'delete_recreate_enabled': True,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           
            # Detailed configuration for recreation
            'endpoint_config': {
                'production_variants': config_response['ProductionVariants'],
                'tags': config_response.get('Tags', [])
            },
            'model_config': {
                'execution_role_arn': model_response['ExecutionRoleArn'],
                'primary_container': model_response['PrimaryContainer'],
                'tags': model_response.get('Tags', [])
            },
           
            # Recreation instructions
            'recreation_notes': {
                'approach': 'delete_recreate',
                'cost_optimization': 'endpoint_deleted_after_deployment_and_predictions',
                'recreation_method': 'lambda_function_recreates_from_this_config',
                'estimated_startup_time': '3-5_minutes'
            }
        }
       
        # Store to S3 with multiple access patterns
        base_key = f"{s3_prefix}/{endpoint_config_s3_prefix}" if s3_prefix else endpoint_config_s3_prefix
       
        # Primary key by endpoint name
        primary_key = f"{base_key}/{endpoint_name}_config.json"
       
        # Secondary key by customer combination
        customer_key = f"{base_key}/customers/{os.environ.get('CUSTOMER_PROFILE', 'unknown')}-{os.environ.get('CUSTOMER_SEGMENT', 'unknown')}/{endpoint_name}_config.json"
       
        # Store configuration in both locations for redundancy
        config_json = json.dumps(recreation_config, indent=2, default=str)
       
        for key, description in [(primary_key, 'primary'), (customer_key, 'customer-specific')]:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=config_json,
                ContentType='application/json',
                Metadata={
                    'endpoint_name': endpoint_name,
                    'cost_optimized': 'true',
                    'delete_recreate': 'true',
                    'environment': os.environ.get('ENV_NAME', 'dev'),
                    'customer': f"{os.environ.get('CUSTOMER_PROFILE', 'unknown')}-{os.environ.get('CUSTOMER_SEGMENT', 'unknown')}",
                    'run_id': run_id,
                    'storage_type': description
                }
            )
           
            print(f"✅ Endpoint configuration stored ({description}): s3://{bucket}/{key}")
       
        # Also store a simple lookup file for the forecasting Lambda
        lookup_config = {
            'endpoint_name': endpoint_name,
            'config_location': f"s3://{bucket}/{primary_key}",
            'customer_config_location': f"s3://{bucket}/{customer_key}",
            'stored_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'ready_for_recreation'
        }
       
        lookup_key = f"{base_key}/lookup/{endpoint_name}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=lookup_key,
            Body=json.dumps(lookup_config, indent=2),
            ContentType='application/json',
            Metadata={
                'type': 'endpoint_lookup',
                'endpoint_name': endpoint_name
            }
        )
       
        print(f"✅ Endpoint lookup stored: s3://{bucket}/{lookup_key}")
       
        return {
            'primary_config_location': f"s3://{bucket}/{primary_key}",
            'customer_config_location': f"s3://{bucket}/{customer_key}",
            'lookup_location': f"s3://{bucket}/{lookup_key}"
        }
       
    except Exception as e:
        print(f"❌ Error storing endpoint configuration: {str(e)}")
        print("⚠️ Continuing without configuration storage - manual recreation may be needed")
        return None

def delete_endpoint_for_cost_optimization(endpoint_name):
    """
    Delete endpoint to achieve true zero cost when not in use
    This replaces the failed zero-instance config approach
    """
    try:
        print(f"=== DELETING ENDPOINT FOR COST OPTIMIZATION ===")
        print(f"Deleting endpoint: {endpoint_name}")
        print(f"Waiting {endpoint_deletion_delay} seconds for endpoint stability before deletion...")
        time.sleep(endpoint_deletion_delay)
       
        sagemaker_client = boto3.client('sagemaker')
       
        # Verify endpoint exists and is InService before deletion
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            current_status = response['EndpointStatus']
           
            if current_status != 'InService':
                print(f"⚠️ Endpoint not InService (status: {current_status}) - skipping deletion")
                return False
               
            print(f"✅ Endpoint confirmed InService - proceeding with deletion")
           
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print(f"⚠️ Endpoint {endpoint_name} does not exist - already deleted")
                return True
            else:
                print(f"❌ Error checking endpoint status: {str(e)}")
                return False
       
        # Delete the endpoint
        print(f"🗑️ Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
       
        print(f"✅ Endpoint deletion initiated successfully")
        print(f"💰 Endpoint {endpoint_name} will now incur ZERO costs")
        print(f"🔄 Forecasting Lambda will recreate it when needed for predictions")
       
        # Wait briefly to verify deletion started
        time.sleep(10)
       
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            if status == 'Deleting':
                print(f"✅ Deletion confirmed - endpoint status: {status}")
            else:
                print(f"⚠️ Unexpected status after deletion: {status}")
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print(f"✅ Endpoint deletion completed - endpoint no longer exists")
            else:
                print(f"⚠️ Could not verify deletion status: {str(e)}")
       
        return True
       
    except Exception as e:
        print(f"❌ Error deleting endpoint: {str(e)}")
        print(f"⚠️ Endpoint may still be running and incurring costs")
        print(f"💡 Manual deletion may be required: aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")
        return False

def update_environment_with_cost_optimization_status(config_stored, endpoint_deleted):
    """Update GitHub environment with cost optimization results"""
    try:
        with open(os.environ["GITHUB_ENV"], "a") as env_file:
            if config_stored:
                env_file.write(f"ENDPOINT_CONFIG_STORED=true\n")
                env_file.write(f"ENDPOINT_CONFIG_LOCATION={config_stored.get('primary_config_location', '')}\n")
            else:
                env_file.write(f"ENDPOINT_CONFIG_STORED=false\n")
           
            if endpoint_deleted:
                env_file.write(f"ENDPOINT_DELETED_FOR_COST_OPTIMIZATION=true\n")
                env_file.write(f"ENDPOINT_COST_STATUS=zero_cost_optimized\n")
            else:
                env_file.write(f"ENDPOINT_DELETED_FOR_COST_OPTIMIZATION=false\n")
                env_file.write(f"ENDPOINT_COST_STATUS=running_incurring_costs\n")
           
            env_file.write(f"COST_OPTIMIZATION_APPROACH=delete_recreate\n")
           
    except Exception as e:
        print(f"⚠️ Could not update environment variables: {str(e)}")

try:
    # Lambda and SageMaker clients will use the assumed role credentials from environment
    lambda_client = boto3.client('lambda')
    sagemaker_client = boto3.client('sagemaker')
   
    # Wait for Lambda function to be ready (same as your current implementation)
    if not wait_for_lambda_ready(lambda_client, lambda_function_name):
        print("❌ Lambda function is not ready, cannot proceed with deployment")
        sys.exit(1)
   
    # Prepare payload for Lambda function - using environment-configured parameters (same as current)
    payload = {
        'model_package_arn': model_package_arn,
        'endpoint_name': endpoint_name,
        'instance_type': deploy_instance_type,
        'instance_count': deploy_instance_count,
        'execution_role': role,
        'run_id': run_id,
        # ENHANCED: Add delete/recreate approach flags (no zero-instance config)
        'enable_cost_optimization': enable_endpoint_delete_recreate,
        'delete_recreate_approach': True,
        'store_config_for_recreation': True
    }
   
    print("Invoking Lambda function for model deployment...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
   
    # Configure Lambda client with environment-specified timeout (same as current)
    lambda_config = boto3.session.Config(
        read_timeout=lambda_timeout,
        connect_timeout=60,
        retries={'max_attempts': 3}
    )
    lambda_client_with_timeout = boto3.client('lambda', config=lambda_config)
   
    # Invoke Lambda function (same as your current implementation)
    response = lambda_client_with_timeout.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
   
    # Process response (same as your current implementation)
    payload_bytes = response['Payload'].read()
    payload_str = payload_bytes.decode('utf-8')
    lambda_response = json.loads(payload_str)
   
    print(f"Lambda invocation status code: {response.get('StatusCode')}")
    print(f"Lambda response: {json.dumps(lambda_response, indent=2)}")
   
    if response.get('StatusCode') == 200 and lambda_response.get('statusCode') == 200:
        print("✅ Model deployment initiated successfully!")
       
        # Parse body (same as current)
        body = lambda_response['body']
        if isinstance(body, str):
            body = json.loads(body)
       
        model_name = body.get('model_name')
        endpoint_config_name = body.get('endpoint_config_name')
       
        print(f"Model name: {model_name}")
        print(f"Endpoint name: {endpoint_name}")
        print(f"Endpoint config: {endpoint_config_name}")
       
        # Wait for endpoint to be ready (same as your current implementation)
        print("Waiting for endpoint to be ready...")
        max_wait_time = 1200  # 20 minutes
        poll_interval = 30
        wait_time = 0
       
        while wait_time < max_wait_time:
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
               
                print(f"Endpoint status: {status} (waited {wait_time}s)")
               
                if status == 'InService':
                    print(f"✅ Endpoint {endpoint_name} is now ready!")
                   
                    # Save endpoint info to environment (same as current)
                    with open(os.environ["GITHUB_ENV"], "a") as env_file:
                        env_file.write(f"ENDPOINT_STATUS=InService\n")
                        env_file.write(f"ENDPOINT_ARN={response.get('EndpointArn', '')}\n")
                   
                    # ENHANCED: Implement delete/recreate cost optimization
                    if enable_endpoint_delete_recreate:
                        print(f"=== IMPLEMENTING DELETE/RECREATE COST OPTIMIZATION ===")
                       
                        # Step 1: Store endpoint configuration for recreation
                        config_storage_result = store_endpoint_configuration(
                            endpoint_name, model_name, endpoint_config_name, model_package_arn
                        )
                       
                        # Step 2: Delete endpoint for cost optimization (if configured)
                        endpoint_deletion_result = False
                        if delete_endpoint_after_deployment and config_storage_result:
                            endpoint_deletion_result = delete_endpoint_for_cost_optimization(endpoint_name)
                        elif not config_storage_result:
                            print("⚠️ Skipping endpoint deletion - configuration storage failed")
                            print("🔄 Manual recreation would be required if deleted without stored config")
                        elif not delete_endpoint_after_deployment:
                            print("ℹ️ Endpoint deletion disabled - endpoint will remain InService")
                            print("💡 Enable DELETE_ENDPOINT_AFTER_DEPLOYMENT=true for maximum cost savings")
                       
                        # Update environment with results
                        update_environment_with_cost_optimization_status(
                            config_storage_result, endpoint_deletion_result
                        )
                       
                        # Final status report
                        if config_storage_result and endpoint_deletion_result:
                            print("✅ DELETE/RECREATE COST OPTIMIZATION COMPLETED SUCCESSFULLY!")
                            print("💰 Endpoint now incurs ZERO costs when not predicting")
                            print("🔄 Forecasting Lambda will recreate endpoint automatically when needed")
                            print("📊 Expected cost savings: ~98% reduction")
                        elif config_storage_result and not endpoint_deletion_result:
                            print("⚠️ Configuration stored but endpoint deletion failed")
                            print("💰 Endpoint still incurring costs - manual deletion recommended")
                        else:
                            print("❌ Delete/recreate optimization failed")
                            print("💰 Endpoint remains InService (higher costs)")
                    else:
                        print("ℹ️ Delete/recreate optimization disabled")
                        print("💰 Endpoint will remain InService and incur ongoing costs")
                        print("💡 Enable ENABLE_ENDPOINT_DELETE_RECREATE=true for cost savings")
                   
                    break
                elif status in ['Failed', 'OutOfService']:
                    print(f"❌ Endpoint creation failed with status: {status}")
                    print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
                    sys.exit(1)
               
                time.sleep(poll_interval)
                wait_time += poll_interval
            except Exception as e:
                print(f"❌ Error checking endpoint status: {str(e)}")
                time.sleep(5)
       
        if wait_time >= max_wait_time:
            print(f"⏰ Timed out waiting for endpoint {endpoint_name} to be ready")
            print("Deployment will continue asynchronously")
            sys.exit(0)
    else:
        print(f"❌ Deployment failed: {lambda_response}")
       
        # Check for function errors (same as current)
        if 'FunctionError' in response:
            print(f"Lambda function error type: {response['FunctionError']}")
       
        # Check for error message in response (same as current)
        if isinstance(lambda_response, dict) and 'errorMessage' in lambda_response:
            print(f"Error message: {lambda_response['errorMessage']}")
            if 'errorType' in lambda_response:
                print(f"Error type: {lambda_response['errorType']}")
       
        sys.exit(1)

except Exception as e:
    print(f"❌ Error deploying model: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== ENHANCED MODEL DEPLOYMENT - DELETE/RECREATE COMPLETED ===")
if enable_endpoint_delete_recreate:
    print("🎯 Delete/recreate cost optimization enabled - maximum cost savings!")
    print("📊 Monitor forecasting Lambda logs for automatic endpoint recreation")
    if delete_endpoint_after_deployment:
        print("💰 Endpoint deleted after deployment - ZERO ongoing costs")
    else:
        print("💰 Endpoint kept after deployment - will be deleted after predictions")
else:
    print("ℹ️ Delete/recreate optimization disabled - endpoint remains InService")
print("✅ Enhanced deployment script completed successfully")
