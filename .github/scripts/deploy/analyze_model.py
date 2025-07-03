#!/usr/bin/env python3
"""
Model Analysis Script for Energy Load Forecasting Pipeline
"""
import boto3
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

print('=== MODEL ANALYSIS SCRIPT STARTED ===')

# Get S3 bucket and prefix from environment
bucket = os.environ['S3_BUCKET']
prefix = os.environ['S3_PREFIX']

print(f'Using bucket: {bucket}')
print(f'Using prefix: {prefix}')
print(f'Looking for models at: {prefix}/models/run_*')

# Find the latest run_id folder in S3
s3_client = boto3.client('s3')

try:
    # List all run_id folders in the models directory
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=f'{prefix}/models/run_'
    )
    
    print(f'S3 response keys found: {len(response.get("Contents", []))}')
    
    if 'Contents' not in response or not response['Contents']:
        print('No model runs found in S3 bucket.')
        print('Available objects in models directory:')
        try:
            models_response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=f'{prefix}/models/'
            )
            for obj in models_response.get('Contents', []):
                print(f'  - {obj["Key"]}')
        except Exception as e:
            print(f'Error listing models directory: {e}')
        sys.exit(1)
    
    # Extract all run_id directories
    run_dirs = set()
    for obj in response['Contents']:
        # Extract run_id from path like 'prefix/models/run_20250514_123456/...'
        key_parts = obj['Key'].split('/')
        print(f'Processing key: {obj["Key"]}')
        if len(key_parts) >= 3:
            run_dir = '/'.join(key_parts[:-1])  # Get the directory without filename
            if 'run_' in run_dir:
                run_dirs.add(run_dir)
                print(f'Added run directory: {run_dir}')
    
    if not run_dirs:
        print('No run directories found.')
        sys.exit(1)
    
    # Sort by name (which includes the timestamp) to get the latest run
    sorted_run_dirs = sorted(list(run_dirs), reverse=True)
    latest_run_dir = sorted_run_dirs[0]
    run_id = latest_run_dir.split('/')[-1]
    
    print(f'Found latest run: {run_id}')
    print(f'Run directory: {latest_run_dir}')
    
    # Check for evaluation results
    eval_key = f'{latest_run_dir}/evaluation.json'
    model_key = f'{latest_run_dir}/xgboost-model'
    
    print(f'Looking for evaluation file at: {eval_key}')
    print(f'Looking for model file at: {model_key}')
    
    try:
        # Load evaluation metrics
        s3_resource = boto3.resource('s3')
        obj = s3_resource.Object(bucket, eval_key)
        eval_results = json.loads(obj.get()['Body'].read().decode('utf-8'))
        
        print(' Successfully loaded evaluation results')
        
        # Print metrics
        print('\n=== MODEL EVALUATION SUMMARY ===')
        metrics = eval_results.get('metrics', {})
        
        # Print all available metrics
        print('Available metrics:')
        for key in metrics.keys():
            print(f'  - {key}: {metrics[key]}')
        
        # Export metrics to a file for later use
        print('\nExporting metrics to model_metrics.json')
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Check if model file exists
        try:
            s3_client.head_object(Bucket=bucket, Key=model_key)
            print(f' Model file exists at: s3://{bucket}/{model_key}')
            
            # Write results to files that can be read by the shell
            with open('run_id.txt', 'w') as f:
                f.write(run_id)
            with open('model_s3_uri.txt', 'w') as f:
                f.write(f's3://{bucket}/{model_key}')
            with open('pipeline_status.txt', 'w') as f:
                f.write('Succeeded')
            
            print(f' Run ID {run_id} saved for potential deployment')
            
        except Exception as e:
            print(f' Model file check failed: {str(e)}')
            with open('pipeline_status.txt', 'w') as f:
                f.write('Failed')
            sys.exit(1)
        
    except Exception as e:
        print(f' Error loading evaluation results: {str(e)}')
        print('Available files in run directory:')
        try:
            run_response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=f'{latest_run_dir}/'
            )
            for obj in run_response.get('Contents', []):
                print(f'  - {obj["Key"]}')
        except Exception as list_e:
            print(f'Error listing run directory: {list_e}')
        with open('pipeline_status.txt', 'w') as f:
            f.write('Failed')
        sys.exit(1)
    
except Exception as e:
    print(f' Error in model analysis: {str(e)}')
    import traceback
    traceback.print_exc()
    with open('pipeline_status.txt', 'w') as f:
        f.write('Failed')
    sys.exit(1)

print('=== MODEL ANALYSIS SCRIPT COMPLETED ===')
