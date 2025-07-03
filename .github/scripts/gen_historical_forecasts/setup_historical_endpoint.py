import boto3
import json
import time
import os
import sys
from datetime import datetime

def get_endpoint_status(sagemaker_client, endpoint_name):
    """Get current endpoint status"""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except sagemaker_client.exceptions.ClientError as e:
        if 'ValidationException' in str(e):
            return 'NotFound'
        raise e

def load_endpoint_configuration_from_s3(s3_client, endpoint_name, s3_bucket, s3_prefix):
    """Load stored endpoint configuration from S3"""
    try:
        print(f"Loading endpoint configuration for: {endpoint_name}")
       
        endpoint_config_prefix = os.environ.get('ENDPOINT_CONFIG_S3_PREFIX', 'endpoint-configs')
        base_key = f"{s3_prefix}/{endpoint_config_prefix}" if s3_prefix else endpoint_config_prefix
        primary_key = f"{base_key}/{endpoint_name}_config.json"
       
        print(f"Attempting to load config from: s3://{s3_bucket}/{primary_key}")
       
        response = s3_client.get_object(Bucket=s3_bucket, Key=primary_key)
        config_data = json.loads(response['Body'].read().decode('utf-8'))
       
        print("✅ Successfully loaded endpoint configuration from S3")
        return config_data
       
    except s3_client.exceptions.NoSuchKey:
        print(f"❌ Endpoint configuration not found: s3://{s3_bucket}/{primary_key}")
        return None
    except Exception as e:
        print(f"❌ Error loading endpoint configuration: {str(e)}")
        return None

def recreate_endpoint_components(sagemaker_client, config_data):
    """Recreate model and endpoint configuration if needed"""
    try:
        model_name = config_data['model_name']
        endpoint_config_name = config_data['endpoint_config_name']
        model_config = config_data['model_config']
        stored_endpoint_config = config_data['endpoint_config']
       
        # Recreate model if needed
        try:
            sagemaker_client.describe_model(ModelName=model_name)
            print(f"✅ Model {model_name} already exists")
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print(f"🔄 Recreating model: {model_name}")
               
                sagemaker_client.create_model(
                    ModelName=model_name,
                    ExecutionRoleArn=model_config['execution_role_arn'],
                    PrimaryContainer=model_config['primary_container'],
                    Tags=model_config.get('tags', [])
                )
                print(f"✅ Model recreated: {model_name}")
            else:
                raise e
       
        # Recreate endpoint configuration if needed
        try:
            sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            print(f"✅ Endpoint config {endpoint_config_name} already exists")
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print(f"🔄 Recreating endpoint config: {endpoint_config_name}")
               
                sagemaker_client.create_endpoint_config(
                    EndpointConfigName=endpoint_config_name,
                    ProductionVariants=stored_endpoint_config['production_variants'],
                    Tags=stored_endpoint_config.get('tags', [])
                )
                print(f"✅ Endpoint config recreated: {endpoint_config_name}")
            else:
                raise e
       
        return True
       
    except Exception as e:
        print(f"❌ Error recreating endpoint components: {str(e)}")
        return False

def create_endpoint_for_historical(sagemaker_client, endpoint_name, endpoint_config_name):
    """Create endpoint for historical predictions"""
    try:
        print(f"🔄 Creating endpoint for historical predictions: {endpoint_name}")
       
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=[
                {'Key': 'Purpose', 'Value': 'historical_forecasting'},
                {'Key': 'CostOptimized', 'Value': 'true'},
                {'Key': 'CreatedAt', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                {'Key': 'Environment', 'Value': os.environ.get('ENVIRONMENT', 'dev')}
            ]
        )
       
        print(f"✅ Endpoint creation initiated: {endpoint_name}")
        return True
       
    except Exception as e:
        print(f"❌ Error creating endpoint: {str(e)}")
        return False

def wait_for_endpoint_ready(sagemaker_client, endpoint_name, max_wait_seconds=900):
    """Wait for endpoint to become InService"""
    print(f"Waiting for endpoint to be ready (max {max_wait_seconds}s)...")
   
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            status = get_endpoint_status(sagemaker_client, endpoint_name)
            elapsed = time.time() - start_time
           
            print(f"Endpoint status: {status} (elapsed: {elapsed:.0f}s)")
           
            if status == 'InService':
                buffer_time = int(os.environ.get('ENDPOINT_READY_BUFFER_TIME', '60'))
                print(f"✅ Endpoint is InService! Waiting {buffer_time}s buffer...")
                time.sleep(buffer_time)
                print("✅ Endpoint ready for historical predictions")
                return True
               
            elif status in ['Failed', 'RollingBack']:
                print(f"❌ Endpoint creation failed: {status}")
                return False
           
            time.sleep(30)
           
        except Exception as e:
            print(f"Error checking endpoint status: {str(e)}")
            time.sleep(30)
   
    print(f"❌ Endpoint did not become ready within {max_wait_seconds}s")
    return False

def main():
    """Main function to setup endpoint for historical predictions"""
    try:
        endpoint_name = os.environ['ENDPOINT_NAME']
        s3_bucket = os.environ['S3_BUCKET']
        s3_prefix = os.environ.get('S3_PREFIX', '')
       
        print(f"Setting up endpoint for historical predictions: {endpoint_name}")
       
        # Initialize AWS clients
        sagemaker_client = boto3.client('sagemaker')
        s3_client = boto3.client('s3')
       
        # Check current endpoint status
        current_status = get_endpoint_status(sagemaker_client, endpoint_name)
        print(f"Current endpoint status: {current_status}")
       
        if current_status == 'InService':
            print("✅ Endpoint already InService - ready for historical predictions")
            return True
       
        elif current_status == 'NotFound':
            print("Endpoint not found - recreating for historical predictions...")
           
            # Load configuration from S3
            config_data = load_endpoint_configuration_from_s3(
                s3_client, endpoint_name, s3_bucket, s3_prefix
            )
           
            if not config_data:
                print("❌ Cannot recreate endpoint - configuration not found in S3")
                return False
           
            # Recreate model and endpoint configuration
            if not recreate_endpoint_components(sagemaker_client, config_data):
                print("❌ Failed to recreate endpoint components")
                return False
           
            # Create endpoint
            if not create_endpoint_for_historical(
                sagemaker_client, endpoint_name, config_data['endpoint_config_name']
            ):
                print("❌ Failed to create endpoint")
                return False
           
            # Wait for endpoint to be ready
            if not wait_for_endpoint_ready(sagemaker_client, endpoint_name):
                print("❌ Endpoint did not become ready")
                return False
           
            print("✅ Endpoint successfully created and ready for historical predictions")
            return True
       
        else:
            print(f"❌ Endpoint in unexpected state: {current_status}")
            return False
           
    except Exception as e:
        print(f"❌ Error setting up endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
