#!/usr/bin/env python3
"""
CloudWatch Events Schedule Setup Script for Energy Load Forecasting Pipeline
Creates scheduled triggers for the forecasting Lambda function
"""
import boto3
import json
import os
import sys

print('=== SCHEDULE SETUP SCRIPT DEBUG ===')
function_name = os.environ.get('FORECAST_LAMBDA_NAME', '')
function_arn = os.environ.get('FORECAST_LAMBDA_ARN', '')
lambda_schedule = os.environ.get('LAMBDA_SCHEDULE', '')
env_name = os.environ.get('ENV_NAME', '')
customer_profile = os.environ.get('CUSTOMER_PROFILE', '')
customer_segment = os.environ.get('CUSTOMER_SEGMENT', '')
endpoint_name = os.environ.get('ENDPOINT_NAME', '')
run_id = os.environ.get('RUN_ID', '')

print(f'function_name: {function_name}')
print(f'function_arn: {function_arn}')
print(f'lambda_schedule: {lambda_schedule}')
print(f'env_name: {env_name}')
print(f'customer_profile: {customer_profile}')
print(f'customer_segment: {customer_segment}')
print(f'endpoint_name: {endpoint_name}')
print(f'run_id: {run_id}')
print('=== END DEBUG ===')

# Validate required parameters
if not lambda_schedule:
    print(' Error: lambda_schedule is empty')
    sys.exit(1)
if not function_name:
    print(' Error: function_name is empty')
    sys.exit(1)
if not function_arn:
    print(' Error: function_arn is empty')
    sys.exit(1)

print(f'Setting up schedule for: {function_name}')
print(f'Schedule expression: {lambda_schedule}')

try:
    events_client = boto3.client('events')
    lambda_client = boto3.client('lambda')
    
    # Create rule
    rule_name = f'EnergyForecastSchedule-{customer_profile}-{customer_segment}-{env_name}'
    print(f'Creating rule: {rule_name}')
    
    events_client.put_rule(
        Name=rule_name,
        ScheduleExpression=lambda_schedule,
        State='ENABLED',
        Description=f'Triggers energy load forecast Lambda for {customer_profile}-{customer_segment} in {env_name} environment'
    )
    print(' Rule created successfully')
    
    # Set Lambda as target
    target_input = {
        'endpoint_name': endpoint_name,
        'load_profile': customer_profile,
        'customer_segment': customer_segment,
        'model_version': 'latest',
        'run_id': run_id
    }
    print(f'Target input: {target_input}')
    
    events_client.put_targets(
        Rule=rule_name,
        Targets=[
            {
                'Id': '1',
                'Arn': function_arn,
                'Input': json.dumps(target_input)
            }
        ]
    )
    print(' Target added successfully')
    
    # Add permission for EventBridge to invoke Lambda
    try:
        rule_arn = events_client.describe_rule(Name=rule_name)['Arn']
        print(f'Rule ARN: {rule_arn}')
        
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId='AllowExecutionFromCloudWatch',
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=rule_arn
        )
        print(' Added permission for CloudWatch Events to invoke Lambda')
    except lambda_client.exceptions.ResourceConflictException:
        print(' Permission already exists for CloudWatch Events')
    
    print(f' Created schedule for Lambda function: {rule_name}')
    
except Exception as e:
    print(f' Error setting up schedule: {str(e)}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
