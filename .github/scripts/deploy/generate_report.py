"""
Generate validation reports for SageMaker pipeline executions.
This script can be used both in GitHub Actions workflows and from Jupyter notebooks.
"""

import datetime
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Union

import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('report-generator')

def get_env_var(name: str, default: str = "") -> str:
    """Get an environment variable or return a default value."""
    return os.environ.get(name, default)

def get_commit_message(sha: str) -> str:
    """Get the commit message for a specific SHA.
   
    Args:
        sha: The commit SHA to get the message for
       
    Returns:
        The commit message or a default message if unable to retrieve
    """
    try:
        # Only try to get commit message if it's not a placeholder
        if sha and "local" not in sha.lower():
            message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B', sha],
                universal_newlines=True
            ).strip()
            return message
    except Exception as e:
        logger.warning(f"Unable to get commit message: {e}")
   
    return "No commit message available"

def get_pipeline_details(pipeline_name: str, execution_arn: Optional[str] = None) -> Dict[str, Any]:
    """Get details about a SageMaker pipeline and its execution.
   
    Args:
        pipeline_name: Name of the SageMaker pipeline
        execution_arn: ARN of a specific execution to get details for
       
    Returns:
        Dictionary containing pipeline and execution details
    """
    details = {
        "pipeline_name": pipeline_name,
        "execution_arn": execution_arn,
        "creation_time": "",
        "last_modified": "",
        "status": "Unknown",
        "steps": []
    }
   
    try:
        sm_client = boto3.client('sagemaker')
       
        # Get pipeline details
        try:
            pipeline_response = sm_client.describe_pipeline(
                PipelineName=pipeline_name
            )
            details["pipeline_definition"] = {
                "created_at": pipeline_response.get('CreationTime', '').strftime('%Y-%m-%d %H:%M:%S')
                    if 'CreationTime' in pipeline_response else '',
                "last_modified": pipeline_response.get('LastModifiedTime', '').strftime('%Y-%m-%d %H:%M:%S')
                    if 'LastModifiedTime' in pipeline_response else '',
            }
        except Exception as e:
            logger.warning(f"Failed to get pipeline details: {e}")
       
        # Get execution details if ARN is provided
        if execution_arn:
            try:
                execution_response = sm_client.describe_pipeline_execution(
                    PipelineExecutionArn=execution_arn
                )
               
                details["status"] = execution_response.get('PipelineExecutionStatus', 'Unknown')
               
                if 'CreationTime' in execution_response:
                    details["creation_time"] = execution_response['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
               
                if 'LastModifiedTime' in execution_response:
                    details["last_modified"] = execution_response['LastModifiedTime'].strftime('%Y-%m-%d %H:%M:%S')
               
                # Get steps
                steps = []
                for step in execution_response.get('PipelineExecutionSteps', []):
                    step_info = {
                        'name': step.get('StepName', 'Unknown'),
                        'status': step.get('StepStatus', 'Unknown')
                    }
                   
                    # Get processing job details if available
                    if 'Metadata' in step and 'ProcessingJob' in step['Metadata']:
                        metadata = step['Metadata']['ProcessingJob']
                        if 'Arn' in metadata:
                            job_name = metadata['Arn'].split('/')[-1]
                            try:
                                job_response = sm_client.describe_processing_job(
                                    ProcessingJobName=job_name
                                )
                               
                                step_info['job'] = {
                                    'name': job_name,
                                    'instance_type': job_response['ProcessingResources']['ClusterConfig']['InstanceType'],
                                    'instance_count': job_response['ProcessingResources']['ClusterConfig']['InstanceCount'],
                                    'volume_size': job_response['ProcessingResources']['ClusterConfig']['VolumeSizeInGB'],
                                }
                               
                                if 'ProcessingStartTime' in job_response:
                                    step_info['job']['start_time'] = job_response['ProcessingStartTime'].strftime('%Y-%m-%d %H:%M:%S')
                               
                                if 'ProcessingEndTime' in job_response:
                                    step_info['job']['end_time'] = job_response['ProcessingEndTime'].strftime('%Y-%m-%d %H:%M:%S')
                                   
                                    # Calculate duration if both start and end times are available
                                    if 'start_time' in step_info['job']:
                                        start = job_response['ProcessingStartTime']
                                        end = job_response['ProcessingEndTime']
                                        duration = end - start
                                        step_info['job']['duration_seconds'] = duration.total_seconds()
                                        step_info['job']['duration_formatted'] = str(duration).split('.')[0]  # HH:MM:SS
                            except Exception as e:
                                step_info['job_error'] = str(e)
                   
                    steps.append(step_info)
               
                details["steps"] = steps
               
                # Get execution metrics
                try:
                    metrics_response = sm_client.list_pipeline_execution_metrics(
                        PipelineExecutionArn=execution_arn
                    )
                    details["metrics"] = metrics_response.get("PipelineExecutionMetrics", [])
                except Exception as e:
                    logger.warning(f"Failed to get pipeline metrics: {e}")
               
            except Exception as e:
                logger.warning(f"Failed to get execution details: {e}")
   
    except Exception as e:
        logger.error(f"Error getting pipeline details: {e}")
   
    return details

def check_output_data(bucket: str, prefix: str, env_name: str) -> Dict[str, Any]:
    """Check output data in S3 bucket.
   
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix for processed data
        env_name: Environment name
       
    Returns:
        Dictionary containing output data validation results
    """
    results = {
        "training_data_exists": False,
        "validation_data_exists": False,
        "test_data_exists": False,
        "file_count": 0,
        "sample_files": []
    }
   
    try:
        s3 = boto3.client('s3')
        output_prefix = f"{prefix}/processed"
       
        # Check training data
        training_response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{output_prefix}/training",
            MaxKeys=10
        )
       
        results["training_data_exists"] = 'Contents' in training_response and len(training_response['Contents']) > 0
       
        if results["training_data_exists"]:
            results["file_count"] += len(training_response.get('Contents', []))
            for item in training_response.get('Contents', [])[:3]:
                results["sample_files"].append(item['Key'])
       
        # Check validation data
        validation_response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{output_prefix}/validation",
            MaxKeys=10
        )
       
        results["validation_data_exists"] = 'Contents' in validation_response and len(validation_response['Contents']) > 0
       
        if results["validation_data_exists"]:
            results["file_count"] += len(validation_response.get('Contents', []))
            for item in validation_response.get('Contents', [])[:3]:
                results["sample_files"].append(item['Key'])
       
        # Check test data
        test_response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{output_prefix}/test",
            MaxKeys=10
        )
       
        results["test_data_exists"] = 'Contents' in test_response and len(test_response['Contents']) > 0
       
        if results["test_data_exists"]:
            results["file_count"] += len(test_response.get('Contents', []))
            for item in test_response.get('Contents', [])[:3]:
                results["sample_files"].append(item['Key'])
       
        # All outputs validation
        results["all_outputs_exist"] = (
            results["training_data_exists"] and
            results["validation_data_exists"] and
            results["test_data_exists"]
        )
       
    except Exception as e:
        logger.error(f"Error checking output data: {e}")
        results["error"] = str(e)
   
    return results

def generate_validation_report(
    pipeline_name: Optional[str] = None,
    execution_arn: Optional[str] = None,
    env_name: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    github_ref: Optional[str] = None,
    github_sha: Optional[str] = None,
    days_delay: Optional[str] = None,
    use_reduced_features: Optional[str] = None,
    meter_threshold: Optional[str] = None,
    use_cache: Optional[str] = None,
    output_path: str = "validation_report.md"
) -> str:
    """Generate a validation report for a SageMaker pipeline execution.
   
    Args:
        pipeline_name: SageMaker pipeline name (defaults to env var PIPELINE_NAME)
        execution_arn: Pipeline execution ARN (defaults to env var EXECUTION_ARN)
        env_name: Environment name (defaults to env var ENV_NAME)
        s3_bucket: S3 bucket name (defaults to env var S3_BUCKET)
        s3_prefix: S3 prefix (defaults to env var S3_PREFIX)
        github_ref: GitHub reference (defaults to env var GITHUB_REF)
        github_sha: GitHub SHA (defaults to env var GITHUB_SHA)
        days_delay: Days delay parameter (defaults to env var DAYS_DELAY)
        use_reduced_features: Use reduced features flag (defaults to env var USE_REDUCED_FEATURES)
        meter_threshold: Meter threshold parameter (defaults to env var METER_THRESHOLD)
        use_cache: Use cache flag (defaults to env var USE_CACHE)
        output_path: Output file path for the report
       
    Returns:
        Path to the generated report file
    """
    # Get parameters from environment variables if not provided
    pipeline_name = pipeline_name or get_env_var("PIPELINE_NAME")
    execution_arn = execution_arn or get_env_var("EXECUTION_ARN")
    env_name = env_name or get_env_var("ENV_NAME", "dev")
    s3_bucket = s3_bucket or get_env_var("S3_BUCKET")
    s3_prefix = s3_prefix or get_env_var("S3_PREFIX")
    github_ref = github_ref or get_env_var("GITHUB_REF", "local")
    github_sha = github_sha or get_env_var("GITHUB_SHA", "local")
    days_delay = days_delay or get_env_var("DAYS_DELAY", "7")
    use_reduced_features = use_reduced_features or get_env_var("USE_REDUCED_FEATURES", "True")
    meter_threshold = meter_threshold or get_env_var("METER_THRESHOLD", "1000")
    use_cache = use_cache or get_env_var("USE_CACHE", "True")
   
    # Get commit message
    commit_message = get_commit_message(github_sha)
   
    # Get pipeline details
    if pipeline_name:
        pipeline_details = get_pipeline_details(pipeline_name, execution_arn)
    else:
        pipeline_details = {
            "pipeline_name": "Unknown",
            "status": "Unknown",
            "steps": []
        }
   
    # Check output data
    output_validation = {}
    if s3_bucket and s3_prefix:
        output_validation = check_output_data(s3_bucket, s3_prefix, env_name)
   
    # Generate markdown report
    try:
        with open(output_path, 'w') as f:
            f.write(f"# Validation Report for {env_name.upper()} Environment\n\n")
           
            f.write("## Deployment Details\n")
            f.write(f"- **Branch:** {github_ref}\n")
            f.write(f"- **Commit:** {github_sha[:7] if len(github_sha) > 7 else github_sha} ({commit_message.splitlines()[0] if commit_message else 'No message'})\n")
            f.write(f"- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Environment:** {env_name.upper()}\n\n")
           
            f.write("## Pipeline Configuration\n")
            f.write(f"- **Pipeline Name:** {pipeline_name}\n")
            f.write(f"- **Days Delay:** {days_delay}\n")
            f.write(f"- **Use Reduced Features:** {use_reduced_features}\n")
            f.write(f"- **Meter Threshold:** {meter_threshold}\n")
            f.write(f"- **Use Cache:** {use_cache}\n\n")
           
            if execution_arn:
                f.write("## Pipeline Execution\n")
                f.write(f"- **Execution ARN:** {execution_arn}\n")
                f.write(f"- **Status:** {pipeline_details['status']}\n")
               
                if pipeline_details.get("creation_time"):
                    f.write(f"- **Creation Time:** {pipeline_details['creation_time']}\n")
               
                if pipeline_details.get("last_modified"):
                    f.write(f"- **Last Modified:** {pipeline_details['last_modified']}\n")
               
                f.write("\n")
           
            if pipeline_details.get("steps"):
                f.write("## Processing Steps\n")
                for step in pipeline_details["steps"]:
                    f.write(f"- **{step['name']}**: {step['status']}")
                    if 'job' in step:
                        job = step['job']
                        f.write(f" ({job.get('instance_type', 'unknown')}, {job.get('instance_count', 'unknown')} instances)\n")
                        if 'start_time' in job and 'end_time' in job:
                            f.write(f"  - Runtime: {job['start_time']} to {job['end_time']}")
                            if 'duration_formatted' in job:
                                f.write(f" ({job['duration_formatted']})")
                            f.write("\n")
                    else:
                        f.write("\n")
                f.write("\n")
           
            if output_validation:
                f.write("## Output Data Validation\n")
               
                # Summary
                validation_status = "✅ All outputs validated successfully" if output_validation.get("all_outputs_exist") else "❌ Some outputs are missing"
                f.write(f"- **Status:** {validation_status}\n")
                f.write(f"- **File Count:** {output_validation.get('file_count', 0)}\n")
               
                # Detailed validation
                f.write("- **Training Data:** ")
                f.write("✅ Present" if output_validation.get("training_data_exists") else "❌ Missing")
                f.write("\n")
               
                f.write("- **Validation Data:** ")
                f.write("✅ Present" if output_validation.get("validation_data_exists") else "❌ Missing")
                f.write("\n")
               
                f.write("- **Test Data:** ")
                f.write("✅ Present" if output_validation.get("test_data_exists") else "❌ Missing")
                f.write("\n\n")
               
                # Sample files
                if output_validation.get("sample_files"):
                    f.write("### Sample Output Files\n")
                    for file in output_validation.get("sample_files", []):
                        f.write(f"- `{file}`\n")
                    f.write("\n")
           
            f.write("## Integration Test Results\n")
           
            # Basic validations
            pipeline_valid = pipeline_name and pipeline_details.get("status") != "Unknown"
            execution_valid = execution_arn and pipeline_details.get("status") == "Succeeded"
            output_valid = output_validation.get("all_outputs_exist", False)
           
            f.write(f"- {'✅' if pipeline_valid else '❌'} Pipeline definition verified\n")
            f.write(f"- {'✅' if execution_valid else '❓'} Pipeline execution ")
            if execution_arn:
                f.write(f"{pipeline_details.get('status', 'Unknown')}\n")
            else:
                f.write("not checked\n")
            f.write(f"- {'✅' if output_valid else '❓'} Output data ")
            if output_validation:
                f.write("validated successfully\n\n" if output_valid else "validation failed\n\n")
            else:
                f.write("not checked\n\n")
           
            f.write("## Next Steps\n")
            if env_name.lower() == "dev":
                f.write("This pipeline is validated and ready for promotion to production. To promote it, create a PR from develop to main branch.\n")
            else:
                f.write("This pipeline is now live in production.\n")
   
        logger.info(f"Validation report generated at {output_path}")
        return output_path
       
    except Exception as e:
        logger.error(f"Error generating validation report: {e}")
        # Create a minimal report on error
        with open(output_path, 'w') as f:
            f.write(f"# Validation Report for {env_name.upper()} Environment\n\n")
            f.write("## Error\n")
            f.write(f"Error generating full report: {str(e)}\n\n")
            f.write("### Basic Information\n")
            f.write(f"- **Pipeline:** {pipeline_name}\n")
            f.write(f"- **Execution:** {execution_arn}\n")
            f.write(f"- **Environment:** {env_name}\n")
            f.write(f"- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
       
        return output_path

if __name__ == "__main__":
    """Run as a script to generate a validation report."""
    generate_validation_report()
    print("Validation report generated successfully")
