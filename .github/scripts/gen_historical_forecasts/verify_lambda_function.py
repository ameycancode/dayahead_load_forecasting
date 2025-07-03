#!/usr/bin/env python3
"""
Lambda Function Verification Script

This script verifies that the required Lambda function exists and is ready
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


def construct_lambda_function_name(environment, profile, segment):
    """Construct Lambda function name following the naming convention"""
    lambda_function_name = f"{environment}-energy-daily-predictor-{profile}-{segment}"
    return lambda_function_name


def verify_lambda_function_status(lambda_client, function_name, max_wait_minutes=5):
    """Verify Lambda function exists and is ready"""
    logger.info(f"Checking Lambda function: {function_name}")
   
    try:
        response = lambda_client.get_function_configuration(FunctionName=function_name)
       
        function_state = response.get('State', 'Unknown')
        last_update_status = response.get('LastUpdateStatus', 'Unknown')
       
        logger.info(f"Function state: {function_state}")
        logger.info(f"Last update status: {last_update_status}")
       
        if function_state == 'Active' and last_update_status == 'Successful':
            logger.info("✅ Lambda function is ready for invocation")
            return True, function_state
           
        elif function_state in ['Pending', 'Inactive'] or last_update_status in ['InProgress']:
            logger.info(f"⏳ Lambda function is updating: State={function_state}, Update={last_update_status}")
            logger.info("Waiting for function to become ready...")
           
            # Wait up to max_wait_minutes for function to become ready
            max_attempts = max_wait_minutes * 4  # Check every 15 seconds
           
            for attempt in range(1, max_attempts + 1):
                time.sleep(15)
               
                try:
                    response = lambda_client.get_function_configuration(FunctionName=function_name)
                    function_state = response.get('State', 'Unknown')
                    last_update_status = response.get('LastUpdateStatus', 'Unknown')
                   
                    logger.info(f"Attempt {attempt}/{max_attempts}: State={function_state}, Update={last_update_status}")
                   
                    if function_state == 'Active' and last_update_status == 'Successful':
                        logger.info("✅ Lambda function is now ready")
                        return True, function_state
                    elif last_update_status == 'Failed':
                        logger.error("❌ Lambda function update failed")
                        return False, function_state
                       
                except Exception as e:
                    logger.warning(f"Error checking function status on attempt {attempt}: {str(e)}")
           
            logger.error(f"⏰ Timeout waiting for Lambda function to become ready after {max_wait_minutes} minutes")
            return False, function_state
           
        elif last_update_status == 'Failed':
            logger.error("❌ Lambda function is in Failed state")
            return False, function_state
           
        else:
            logger.warning(f"⚠️ Lambda function in unexpected state: {function_state}, Update: {last_update_status}")
            logger.warning("Proceeding anyway - function may still work")
            return True, function_state
           
    except lambda_client.exceptions.ResourceNotFoundException:
        logger.error(f"❌ Lambda function not found: {function_name}")
        return False, "NotFound"
    except Exception as e:
        logger.error(f"❌ Error checking Lambda function: {str(e)}")
        return False, "Error"


def get_lambda_function_info(lambda_client, function_name):
    """Get additional Lambda function information for logging"""
    try:
        response = lambda_client.get_function_configuration(FunctionName=function_name)


        runtime = response.get('Runtime', 'Unknown')
        timeout = response.get('Timeout', 'Unknown')
        memory_size = response.get('MemorySize', 'Unknown')
        last_modified = response.get('LastModified', 'Unknown')
       
        logger.info(f"Lambda function details:")
        logger.info(f"  Runtime: {runtime}")
        logger.info(f"  Timeout: {timeout} seconds")
        logger.info(f"  Memory: {memory_size} MB")
        logger.info(f"  Last Modified: {last_modified}")
       
        return {
            'runtime': runtime,
            'timeout': timeout,
            'memory_size': memory_size,
            'last_modified': str(last_modified)
        }
       
    except Exception as e:
        logger.warning(f"Could not retrieve additional Lambda function info: {str(e)}")
        return {}


def main():
    """Main Lambda function verification function"""
    try:
        logger.info("Starting Lambda function verification...")
       
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
       
        # Construct Lambda function name
        function_name = construct_lambda_function_name(environment, profile, segment)
        logger.info(f"Target Lambda function: {function_name}")
       
        # Initialize Lambda client
        lambda_client = boto3.client('lambda', region_name=aws_region)
       
        # Verify Lambda function status
        is_ready, status = verify_lambda_function_status(lambda_client, function_name)
       
        if is_ready:
            logger.info("✅ Lambda function verification successful")
           
            # Get additional function information
            function_info = get_lambda_function_info(lambda_client, function_name)
           
            # Set GitHub outputs for success
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"lambda_function_ready=true\n")
                f.write(f"lambda_function_name={function_name}\n")
                f.write(f"lambda_function_status={status}\n")
           
        else:
            logger.error("❌ Lambda function verification failed")
           
            # Set GitHub outputs for failure
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"lambda_function_ready=false\n")
                f.write(f"lambda_function_name={function_name}\n")
                f.write(f"lambda_function_status={status}\n")
           
            # Provide troubleshooting information
            logger.info("Troubleshooting information:")
            logger.info(f"  1. Verify the main deployment workflow completed successfully")
            logger.info(f"  2. Check Lambda console for function: {function_name}")
            logger.info(f"  3. Ensure the combination {profile}-{segment} was deployed")
            logger.info(f"  4. Check AWS permissions for Lambda operations")
            logger.info(f"  5. Verify the forecasting Lambda was created in deployment")
           
            sys.exit(1)
       
        logger.info("Lambda function verification completed successfully")
       
    except Exception as e:
        logger.error(f"❌ Lambda function verification failed: {str(e)}")
       
        # Set GitHub outputs for error
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"lambda_function_ready=false\n")
            f.write(f"lambda_function_name=\n")
            f.write(f"lambda_function_status=Error\n")
       
        sys.exit(1)


if __name__ == "__main__":
    main()
