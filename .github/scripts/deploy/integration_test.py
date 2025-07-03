# Integration test script for validating pipeline outputs
import os
import sys
import tempfile
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd


def test_pipeline_definition():
    """Test that pipeline definition exists"""
    sm_client = boto3.client("sagemaker")

    try:
        response = sm_client.describe_pipeline(PipelineName=os.environ["PIPELINE_NAME"])
        print(f"✅ Pipeline definition exists: {response['PipelineName']}")
        return True
    except Exception as e:
        print(f"❌ Failed to find pipeline: {str(e)}")
        return False


def test_pipeline_execution():
    """Test that pipeline executed successfully"""
    sm_client = boto3.client("sagemaker")

    try:
        response = sm_client.describe_pipeline_execution(
            PipelineExecutionArn=os.environ["EXECUTION_ARN"]
        )

        status = response["PipelineExecutionStatus"]
        print(f"✅ Pipeline execution status: {status}")

        # Check steps
        for step in response.get("PipelineExecutionSteps", []):
            step_name = step["StepName"]
            step_status = step["StepStatus"]
            print(f"  - Step {step_name}: {step_status}")

        return status == "Succeeded"
    except Exception as e:
        print(f"❌ Failed to check execution: {str(e)}")
        return False


def test_output_data():
    """Test that output data exists and is valid"""
    s3 = boto3.client('s3')
    bucket = os.environ['S3_BUCKET']
    prefix = f"{os.environ['S3_PREFIX']}/processed"
   
    # Check training data
    try:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix}/training",
            MaxKeys=20
        )
       
        if 'Contents' in response and len(response['Contents']) > 0:
            print(f"✅ Training data exists: found {len(response['Contents'])} objects")
           
            # Look specifically for train.csv file
            train_csv_files = [item for item in response['Contents']
                              if item['Key'].endswith('train.csv')]
           
            if not train_csv_files:
                print("❌ No train.csv file found in training data")
                return False
               
            # Download the train.csv file to check its structure
            s3_resource = boto3.resource('s3')
            key = train_csv_files[0]['Key']
            local_file = '/tmp/sample_data.csv'
           
            print(f"Downloading {key} for validation")
            try:
                s3_resource.Bucket(bucket).download_file(key, local_file)
           
                # Check file structure
                df = pd.read_csv(local_file)
           
                if not df.empty:
                    print(f"✅ Data file has {len(df)} rows and {len(df.columns)} columns")
                    print(f"   Columns: {', '.join(df.columns[:5])}...")
               
                    # Check target column
                    if 'lossadjustedload' in df.columns:
                        target_stats = {
                            'mean': df['lossadjustedload'].mean(),
                            'min': df['lossadjustedload'].min(),
                            'max': df['lossadjustedload'].max(),
                            'null_count': df['lossadjustedload'].isna().sum()
                        }
                        print(f"✅ Target column stats: {target_stats}")
                        return True
                    else:
                        print("❌ Target column 'lossadjustedload' not found")
                        return False
                else:
                    print("❌ Data file is empty")
                    return False
            except Exception as download_err:
                print(f"❌ Error downloading or processing file: {str(download_err)}")
                return False
        else:
            print(f"❌ No training data found in s3://{bucket}/{prefix}/training")
            return False
    except Exception as e:
        print(f"❌ Error checking output data: {str(e)}")
        return False
   
    # This return should never be reached, but added for completeness
    return False


def validate_feature_engineering():
    """Validate feature engineering by checking for specific columns"""
    s3 = boto3.client('s3')
    bucket = os.environ['S3_BUCKET']
    prefix = f"{os.environ['S3_PREFIX']}/processed"
   
    try:
        # Look for any processed files (train/validation/test)
        for dataset_type in ['training', 'validation', 'test']:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=f"{prefix}/{dataset_type}",
                MaxKeys=5
            )
           
            if 'Contents' in response and len(response['Contents']) > 0:
                # Find a CSV file
                csv_files = [item for item in response['Contents']
                            if item['Key'].endswith('.csv')]
               
                if csv_files:
                    s3_resource = boto3.resource('s3')
                    key = csv_files[0]['Key']
                   
                    with tempfile.NamedTemporaryFile() as temp_file:
                        s3_resource.Bucket(bucket).download_file(key, temp_file.name)
                        df = pd.read_csv(temp_file.name)
                       
                        # Check for feature engineering columns
                        expected_features = [
                            'hour_sin', 'hour_cos', 'dayofweek',
                            'is_weekend', 'is_morning_peak', 'is_solar_period'
                        ]
                       
                        found_features = [col for col in expected_features if col in df.columns]
                       
                        if found_features:
                            print(f"✅ Found engineered features: {', '.join(found_features)}")
                            return True
                        else:
                            print("❌ No engineered features found in data")
                            return False
       
        print("❌ No CSV files found for feature validation")
        return False
   
    except Exception as e:
        print(f"❌ Error validating features: {str(e)}")
        return False


def check_data_splits():
    """Check if all three data splits (train/validation/test) exist"""
    s3 = boto3.client('s3')
    bucket = os.environ['S3_BUCKET']
    prefix = f"{os.environ['S3_PREFIX']}/processed"
   
    splits_found = 0
    splits_to_check = ['training', 'validation', 'test']
   
    try:
        for split in splits_to_check:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=f"{prefix}/{split}",
                MaxKeys=1
            )
           
            if 'Contents' in response and len(response['Contents']) > 0:
                print(f"✅ Found {split} data split")
                splits_found += 1
            else:
                print(f"❌ {split.capitalize()} data split missing")
       
        if splits_found == len(splits_to_check):
            print("✅ All data splits verified (train/validation/test)")
            return True
        else:
            print(f"❌ Only {splits_found}/{len(splits_to_check)} data splits found")
            return False
           
    except Exception as e:
        print(f"❌ Error checking data splits: {str(e)}")
        return False


# Run all tests
all_passed = True
all_passed = test_pipeline_definition() and all_passed
all_passed = test_pipeline_execution() and all_passed
all_passed = test_output_data() and all_passed

# Additional tests - only run if main tests pass
if all_passed:
    # Extra validation can be enabled by setting the environment variable
    if os.environ.get('RUN_EXTENDED_VALIDATION', 'false').lower() == 'true':
        print("\n🔍 Running extended validation checks...")
        all_passed = validate_feature_engineering() and all_passed
        all_passed = check_data_splits() and all_passed

# Exit with status
sys.exit(0 if all_passed else 1)
