#!/usr/bin/env python3
"""
SageMaker Pipeline Monitoring Script for Energy Load Forecasting
"""
import os
import boto3
import time
import sys
import json

def get_monitoring_parameters():
    """Get monitoring parameters from environment variables"""
    
    return {
        'region' : os.environ.get('AWS_REGION', 'us-west-2'),
        'pipeline_name': os.environ['PIPELINE_NAME'],
        'customer_profile': os.environ['CUSTOMER_PROFILE'],
        'customer_segment': os.environ['CUSTOMER_SEGMENT'],
        's3_bucket': os.environ['S3_BUCKET'],
        's3_prefix': os.environ['S3_PREFIX'],
        'max_wait_time': int(os.environ['PIPELINE_TIMEOUT']),  # 2 hours default os.environ['POLL_INTERVAL']
        'poll_interval': int(os.environ['POLL_INTERVAL']),  # 1 min default
    }

print('=== PIPELINE MONITORING SCRIPT STARTED ===')
print(f'Python version: {sys.version}')
print(f'Arguments received: {sys.argv}')

if len(sys.argv) < 2:
    print(' Error: No execution ARN provided as argument')
    print('Usage: python monitor_pipeline.py <execution_arn>')
    sys.exit(1)

execution_arn = sys.argv[1]
print(f'Monitoring execution ARN: {execution_arn}')

# Validate ARN format
if not execution_arn.startswith('arn:aws:sagemaker'):
    print(f' Invalid execution ARN format: {execution_arn}')
    sys.exit(1)

# Create SageMaker client with explicit region
try:
    params = get_monitoring_parameters()

    region = params['region']
    print(f'Using AWS region: {region}')
    sm_client = boto3.client('sagemaker', region_name=region)
    print(' SageMaker client created successfully')
except Exception as e:
    print(f' Error creating SageMaker client: {str(e)}')
    sys.exit(1)

# Test initial connection
try:
    print('Testing initial connection to pipeline...')
    initial_response = sm_client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    initial_status = initial_response['PipelineExecutionStatus']
    print(f' Successfully connected. Initial status: {initial_status}')
    print(f'Pipeline execution display name: {initial_response.get("PipelineExecutionDisplayName", "N/A")}')
    print(f'Creation time: {initial_response.get("CreationTime", "N/A")}')
except Exception as e:
    print(f' Error accessing pipeline execution: {str(e)}')
    print('This could mean:')
    print('- The execution ARN is invalid')
    print('- The pipeline execution does not exist')
    print('- AWS credentials lack permissions')
    print('- Wrong AWS region')
    sys.exit(1)

# Poll for execution status
max_wait_time = params['max_wait_time']
poll_interval = params['poll_interval']
waited_time = 0
last_status = None

print(f'Starting monitoring loop. Max wait time: {max_wait_time} seconds')
print(f'Poll interval: {poll_interval} seconds')

while waited_time < max_wait_time:
    try:
        print(f'--- Checking status at {waited_time}s ---')
       
        # Get execution details
        response = sm_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
       
        status = response['PipelineExecutionStatus']
       
        # Only print status change or every 5 minutes
        if status != last_status or waited_time % 300 == 0:
            print(f'Current status: {status} (waited {waited_time}s / {max_wait_time}s)')
            if status != last_status:
                print(f'Status changed from {last_status} to {status}')
            last_status = status
       
        # Get and print step details
        try:
            steps = response.get('PipelineExecutionSteps', [])
            if steps:
                print(f'Pipeline has {len(steps)} steps')
                for step in steps:
                    step_name = step.get('StepName', 'Unknown')
                    step_status = step.get('StepStatus', 'Unknown')
                    step_start = step.get('StartTime')
                    step_end = step.get('EndTime')
                   
                    # Calculate duration if step is complete
                    duration = ''
                    if step_start and step_end:
                        duration_sec = (step_end - step_start).total_seconds()
                        if duration_sec > 60:
                            duration = f' (Duration: {duration_sec/60:.1f} min)'
                        else:
                            duration = f' (Duration: {duration_sec:.1f} sec)'
                    elif step_start:
                        import datetime
                        now = datetime.datetime.now(step_start.tzinfo)
                        running_duration = (now - step_start).total_seconds()
                        duration = f' (Running: {running_duration/60:.1f} min)'
                   
                    print(f'  - Step {step_name}: {step_status}{duration}')
            else:
                print('No pipeline steps found in response')
        except Exception as e:
            print(f'Error getting pipeline steps: {str(e)}')
       
        # Check if terminal state reached
        if status in ['Succeeded', 'Failed', 'Stopped']:
            print(f'\n Pipeline reached terminal status: {status}')
            print(f'Total execution time: {waited_time} seconds ({waited_time/60:.1f} minutes)')
           
            # Print final step summary
            try:
                final_steps = response.get('PipelineExecutionSteps', [])
                print('\nFinal step summary:')
                for step in final_steps:
                    step_name = step.get('StepName', 'Unknown')
                    step_status = step.get('StepStatus', 'Unknown')
                    print(f'  {step_name}: {step_status}')
            except Exception as e:
                print(f'Error getting final step summary: {str(e)}')
           
            if status == 'Succeeded':
                print(' Pipeline completed successfully!')
                sys.exit(0)
            else:
                print(f' Pipeline failed with status: {status}')
                # Try to get failure reason
                failure_reason = response.get('FailureReason', 'No failure reason provided')
                print(f'Failure reason: {failure_reason}')
                sys.exit(1)
    except Exception as e:
        print(f' Error checking pipeline status: {str(e)}')
        print('Retrying in 5 seconds...')
        time.sleep(5)  # Brief pause before retry
        continue
   
    # Wait before checking again
    print(f'Waiting {poll_interval} seconds before next check...')
    time.sleep(poll_interval)
    waited_time += poll_interval

print(f'\n Monitoring timed out after {max_wait_time} seconds ({max_wait_time/3600:.1f} hours)')
print('Pipeline is still running but monitoring has reached the maximum wait time.')
print('You can check the pipeline status in the SageMaker console.')
sys.exit(1)
