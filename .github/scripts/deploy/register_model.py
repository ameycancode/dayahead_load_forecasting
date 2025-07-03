#!/usr/bin/env python3
"""
Model Registration Script for Energy Load Forecasting Pipeline
"""
import boto3
import json
import os
import sys
from datetime import datetime

# Get environment variables - now all sourced from deploy.yml
customer_profile = os.environ["CUSTOMER_PROFILE"]
customer_segment = os.environ["CUSTOMER_SEGMENT"]
bucket = os.environ["S3_BUCKET"]
prefix = os.environ["S3_PREFIX"]
run_id = os.environ["RUN_ID"]
role = os.environ["SAGEMAKER_ROLE_ARN"]

# NEW: Get model framework configuration from environment
model_framework_version = os.environ.get("MODEL_FRAMEWORK_VERSION", "1.7-1")
model_python_version = os.environ.get("MODEL_PYTHON_VERSION", "py3")

print(f"Registering model from run_id: {run_id}")
print(f"Using framework version: {model_framework_version}")
print(f"Using Python version: {model_python_version}")

try:
    # Find the model file in the run_id directory
    s3_client = boto3.client('s3')
   
    # Check if the model file exists in this run
    model_key = f"{prefix}/models/{run_id}/model.tar.gz"
    try:
        s3_client.head_object(Bucket=bucket, Key=model_key)
        print(f"Found model file: {model_key}")
    except Exception as e:
        print(f"Model file not found in {model_key}: {str(e)}")
        sys.exit(1)
   
    # Get the model S3 URI
    model_s3_uri = f"s3://{bucket}/{model_key}"
   
    # Create or get model package group
    sm_client = boto3.client('sagemaker')
    model_package_group_name = f"EnergyForecastModels-{customer_profile}-{customer_segment}"        
   
    try:
        # Check if the model package group exists
        sm_client.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
        print(f"Using existing model package group: {model_package_group_name}")
    except sm_client.exceptions.ClientError as e:
        # If the error is that the group doesn't exist, create it
        if "ValidationException" in str(e) and "does not exist" in str(e):
            print(f"Creating new model package group: {model_package_group_name}")
            sm_client.create_model_package_group(
                ModelPackageGroupName=model_package_group_name,
                ModelPackageGroupDescription=f"Energy load forecasting models for {customer_profile}-{customer_segment}"
            )
        else:
            # If it's a different error, re-raise it
            raise
   
    # Use SageMaker SDK
    import sagemaker
    from sagemaker.xgboost import XGBoostModel
   
    # Create SageMaker session
    session = sagemaker.Session()
   
    # Create XGBoost model using environment-configured versions
    xgboost_model = XGBoostModel(
        model_data=model_s3_uri,
        role=role,
        framework_version=model_framework_version,
        py_version=model_python_version,
        name=f"energy-forecast-model-sdk-{run_id}",
        sagemaker_session=session
    )
   
    # Register the model with correct parameters
    model_package = xgboost_model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved"
    )
   
    # Extract the ARN from the ModelPackage object
    if hasattr(model_package, 'model_package_arn'):
        model_package_arn = model_package.model_package_arn
    elif hasattr(model_package, 'arn'):
        model_package_arn = model_package.arn
    else:
        # Get the ARN by describing the model package
        model_package_name = str(model_package).split('/')[-1]
        response = sm_client.describe_model_package(
            ModelPackageName=f"{model_package_group_name}/{model_package_name}"
        )
        model_package_arn = response['ModelPackageArn']
   
    print(f"Model registered successfully with ARN: {model_package_arn}")
   
    # Save model package ARN and run_id to environment for next steps
    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        env_file.write(f"MODEL_PACKAGE_ARN={model_package_arn}\n")
        env_file.write(f"RUN_ID={run_id}\n")
   
except Exception as e:
    print(f"Error registering model: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
