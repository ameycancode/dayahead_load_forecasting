# Script to wait for SageMaker pipeline execution to complete
import sys
import time

import boto3


def monitor_execution(execution_arn, max_wait_time=1800, poll_interval=30):
    """
    Monitor a SageMaker pipeline execution until completion or timeout.
   
    Args:
        execution_arn: The ARN of the pipeline execution to monitor
        max_wait_time: Maximum time to wait in seconds (default: 30 minutes)
        poll_interval: How frequently to check status in seconds (default: 30 seconds)
       
    Returns:
        bool: True if execution succeeded, False otherwise
    """
    print(f"Monitoring execution: {execution_arn}")
   
    # Create SageMaker client
    sm_client = boto3.client("sagemaker")
   
    # Poll for execution status
    waited_time = 0
   
    while waited_time < max_wait_time:
        # Get execution details
        response = sm_client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
   
        status = response["PipelineExecutionStatus"]
        print(f"Current status: {status} (waited {waited_time}s)")
   
        # Check if terminal state reached
        if status in ["Succeeded", "Failed", "Stopped"]:
            break
   
        # Wait before checking again
        time.sleep(poll_interval)
        waited_time += poll_interval
        sys.stdout.flush()  # Ensure logs are flushed
   
    # Final status check
    response = sm_client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    final_status = response["PipelineExecutionStatus"]
   
    print(f"Final execution status: {final_status}")
   
    return final_status == "Succeeded"

def main():
    """Main entry point for the script."""
    # Get execution ARN from command line
    if len(sys.argv) < 2:
        print("Error: Missing execution ARN argument")
        sys.exit(1)

    try:
        execution_arn = sys.argv[1]
   
        # Monitor the execution
        success = monitor_execution(execution_arn)
   
        # Exit with status code
        sys.exit(0 if success else 1)
    except IndexError:
        print("Error: Failed to access execution ARN")
        sys.exit(1)

if __name__ == "__main__":
    main()
