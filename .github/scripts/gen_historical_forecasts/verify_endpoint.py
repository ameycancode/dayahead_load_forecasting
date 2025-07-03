#!/usr/bin/env python3
"""
SageMaker Endpoint Verification Script

This script verifies that the required SageMaker endpoint exists and is in service
before attempting to generate historical predictions.
"""

import boto3
import os
import sys
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def construct_endpoint_name(environment, profile, segment):
    """Construct endpoint name following the naming convention"""
    endpoint_name = f"{environment}-energy-ml-endpoint-{profile}-{segment}"
    return endpoint_name


def verify_endpoint_status(sagemaker_client, endpoint_name, max_wait_minutes=10):
    """Verify endpoint exists and is InService, with optional waiting"""
    logger.info(f"Checking endpoint: {endpoint_name}")
   
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = response['EndpointStatus']
       
        logger.info(f"Initial endpoint status: {endpoint_status}")
       
        if endpoint_status == "InService":
            logger.info("✅ Endpoint is InService and ready for predictions")
            return True, endpoint_status
           
        elif endpoint_status in ["Creating", "Updating"]:
            logger.info(f"⏳ Endpoint is {endpoint_status}, waiting for it to become InService...")
           
            # Wait up to max_wait_minutes for endpoint to become ready
            max_attempts = max_wait_minutes * 2  # Check every 30 seconds
           
            for attempt in range(1, max_attempts + 1):
                time.sleep(30)
               
                try:
                    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                    endpoint_status = response['EndpointStatus']
                   
                    logger.info(f"Attempt {attempt}/{max_attempts}: Endpoint status: {endpoint_status}")
                   
                    if endpoint_status == "InService":
                        logger.info("✅ Endpoint is now InService")
                        return True, endpoint_status
                    elif endpoint_status == "Failed":
                        logger.error("❌ Endpoint deployment failed")
                        return False, endpoint_status
                       
                except Exception as e:
                    logger.warning(f"Error checking endpoint status on attempt {attempt}: {str(e)}")
           
            logger.error(f"⏰ Timeout waiting for endpoint to become ready after {max_wait_minutes} minutes")
            return False, endpoint_status
           
        elif endpoint_status == "Failed":
            logger.error("❌ Endpoint is in Failed state")
            return False, endpoint_status
           
        else:
            logger.error(f"❌ Endpoint in unexpected state: {endpoint_status}")
            return False, endpoint_status
           
    except sagemaker_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            logger.error(f"❌ Endpoint not found: {endpoint_name}")
            return False, "NotFound"
        else:
            logger.error(f"❌ Error checking endpoint: {str(e)}")
            return False, "Error"
    except Exception as e:
        logger.error(f"❌ Unexpected error checking endpoint: {str(e)}")
        return False, "Error"


def get_endpoint_info(sagemaker_client, endpoint_name):
    """Get additional endpoint information for logging"""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
       
        endpoint_config_name = response.get('EndpointConfigName', 'Unknown')
        creation_time = response.get('CreationTime', 'Unknown')
        last_modified_time = response.get('LastModifiedTime', 'Unknown')
       
        logger.info(f"Endpoint details:")
        logger.info(f"  Config: {endpoint_config_name}")
        logger.info(f"  Created: {creation_time}")
        logger.info(f"  Last Modified: {last_modified_time}")
       
        return {
            'config_name': endpoint_config_name,
            'creation_time': str(creation_time),
            'last_modified_time': str(last_modified_time)
        }
       
    except Exception as e:
        logger.warning(f"Could not retrieve additional endpoint info: {str(e)}")
        return {}


def main():
    """Main endpoint verification function"""
    try:
        logger.info("Starting SageMaker endpoint verification...")
       
        # Get inputs from environment variables
        environment = os.environ.get('ENVIRONMENT')
        profile = os.environ.get('PROFILE')
        segment = os.environ.get('SEGMENT')
        aws_region = os.environ.get('AWS_REGION')
       
        logger.info(f"Environment: {environment}")
        logger.info(f"Profile: {profile}")
        logger.info(f"Segment: {segment}")
        logger.info(f"AWS Region: {aws_region}")
       
        # Validate required inputs
        if not all([environment, profile, segment, aws_region]):
            raise ValueError("ENVIRONMENT, PROFILE, SEGMENT, and AWS_REGION are required")
       
        # Construct endpoint name
        endpoint_name = construct_endpoint_name(environment, profile, segment)
        logger.info(f"Target endpoint: {endpoint_name}")
       
        # Initialize SageMaker client
        sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
       
        # Verify endpoint status
        is_ready, status = verify_endpoint_status(sagemaker_client, endpoint_name)
       
        if is_ready:
            logger.info("✅ Endpoint verification successful")
           
            # Get additional endpoint information
            endpoint_info = get_endpoint_info(sagemaker_client, endpoint_name)
           
            # Set GitHub outputs for success
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"endpoint_ready=true\n")
                f.write(f"endpoint_name={endpoint_name}\n")
                f.write(f"endpoint_status={status}\n")
           
        else:
            logger.error("❌ Endpoint verification failed")
           
            # Set GitHub outputs for failure
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"endpoint_ready=false\n")
                f.write(f"endpoint_name={endpoint_name}\n")
                f.write(f"endpoint_status={status}\n")
           
            # Provide troubleshooting information
            logger.info("Troubleshooting information:")
            logger.info(f"  1. Verify the main deployment workflow completed successfully")
            logger.info(f"  2. Check SageMaker console for endpoint: {endpoint_name}")
            logger.info(f"  3. Ensure the combination {profile}-{segment} was deployed")
            logger.info(f"  4. Check AWS permissions for SageMaker operations")
           
            sys.exit(1)
       
        logger.info("Endpoint verification completed successfully")
       
    except Exception as e:
        logger.error(f"❌ Endpoint verification failed: {str(e)}")
       
        # Set GitHub outputs for error
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"endpoint_ready=false\n")
            f.write(f"endpoint_name=\n")
            f.write(f"endpoint_status=Error\n")
       
        sys.exit(1)


if __name__ == "__main__":
    main()
