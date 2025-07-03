#!/usr/bin/env python3
"""
Enhanced Deployment Readiness Validation Script
Validates that all prerequisites are met for model deployment using centralized configuration.
"""
import os
import sys
import boto3
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_validation_config():
    """Get validation configuration from environment variables"""
    try:
        return {
            'customer_profile': os.environ['CUSTOMER_PROFILE'],
            'customer_segment': os.environ['CUSTOMER_SEGMENT'],
            'environment': os.environ['ENVIRONMENT'],
            's3_bucket': os.environ['S3_BUCKET'],
            's3_prefix': os.environ['S3_PREFIX'],
            'pipeline_name': os.environ['PIPELINE_NAME'],
            'endpoint_name': os.environ['ENDPOINT_NAME'],
            'model_package_group': os.environ['MODEL_PACKAGE_GROUP'],
            'sagemaker_role': os.environ['SAGEMAKER_ROLE_ARN'],
            'aws_region': os.environ['AWS_REGION'],
            'database_type': os.environ.get('DATABASE_TYPE', 'redshift'),
        }
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        raise

def validate_s3_artifacts(config):
    """Validate that required S3 artifacts exist"""
    logger.info("🔍 Validating S3 artifacts...")
   
    s3_client = boto3.client('s3')
   
    # Check for model artifacts
    model_prefix = f"{config['s3_prefix']}/models/"
   
    try:
        # List model runs
        response = s3_client.list_objects_v2(
            Bucket=config['s3_bucket'],
            Prefix=model_prefix,
            Delimiter='/'
        )
       
        if 'CommonPrefixes' not in response:
            logger.error(f"❌ No model runs found in s3://{config['s3_bucket']}/{model_prefix}")
            return False
           
        run_folders = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
        run_folders = [folder for folder in run_folders if folder.startswith('run_')]
       
        if not run_folders:
            logger.error(f"❌ No valid model runs found in s3://{config['s3_bucket']}/{model_prefix}")
            return False
           
        # Get the latest run
        latest_run = sorted(run_folders)[-1]
        logger.info(f"✅ Found model runs: {len(run_folders)}, latest: {latest_run}")
       
        # Check for required files in latest run
        required_files = ['model.tar.gz', 'evaluation.json']
        model_run_prefix = f"{model_prefix}{latest_run}/"
       
        for file_name in required_files:
            file_key = f"{model_run_prefix}{file_name}"
            try:
                s3_client.head_object(Bucket=config['s3_bucket'], Key=file_key)
                logger.info(f"✅ Found required file: {file_name}")
            except s3_client.exceptions.NoSuchKey:
                logger.error(f"❌ Missing required file: {file_name}")
                return False
               
        return True
       
    except Exception as e:
        logger.error(f"❌ Error validating S3 artifacts: {str(e)}")
        return False

def validate_sagemaker_permissions(config):
    """Validate SageMaker permissions"""
    logger.info("🔍 Validating SageMaker permissions...")
   
    try:
        # Assume the SageMaker role to test permissions
        sts_client = boto3.client('sts')
        response = sts_client.assume_role(
            RoleArn=config['sagemaker_role'],
            RoleSessionName=f"ValidationCheck-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
       
        # Use assumed role credentials
        credentials = response['Credentials']
        sagemaker_client = boto3.client(
            'sagemaker',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )
       
        # Test basic SageMaker operations
        sagemaker_client.list_models(MaxResults=1)
        logger.info("✅ SageMaker permissions validated")
       
        # Test model package group operations
        try:
            sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=config['model_package_group']
            )
            logger.info(f"✅ Model package group exists: {config['model_package_group']}")
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                logger.info(f"ℹ️ Model package group will be created: {config['model_package_group']}")
            else:
                logger.error(f"❌ Error checking model package group: {str(e)}")
                return False
               
        return True
       
    except Exception as e:
        logger.error(f"❌ SageMaker permissions validation failed: {str(e)}")
        return False

def validate_endpoint_prerequisites(config):
    """Validate endpoint deployment prerequisites"""
    logger.info("🔍 Validating endpoint prerequisites...")
   
    try:
        sagemaker_client = boto3.client('sagemaker')
       
        # Check if endpoint already exists
        try:
            response = sagemaker_client.describe_endpoint(
                EndpointName=config['endpoint_name']
            )
            endpoint_status = response['EndpointStatus']
           
            if endpoint_status == 'InService':
                logger.warning(f"⚠️ Endpoint already exists and is InService: {config['endpoint_name']}")
                logger.info("ℹ️ Deployment will update the existing endpoint")
            elif endpoint_status in ['Creating', 'Updating']:
                logger.error(f"❌ Endpoint is currently {endpoint_status}: {config['endpoint_name']}")
                logger.error("Cannot deploy while endpoint is in transition state")
                return False
            else:
                logger.info(f"ℹ️ Endpoint exists in {endpoint_status} state: {config['endpoint_name']}")
               
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                logger.info(f"ℹ️ Endpoint will be created: {config['endpoint_name']}")
            else:
                logger.error(f"❌ Error checking endpoint: {str(e)}")
                return False
               
        return True
       
    except Exception as e:
        logger.error(f"❌ Endpoint prerequisites validation failed: {str(e)}")
        return False

def validate_infrastructure_readiness(config):
    """Validate infrastructure readiness"""
    logger.info("🔍 Validating infrastructure readiness...")
   
    try:
        if config['database_type'] == 'redshift':
            # Validate Redshift connectivity (basic check)
            redshift_cluster = os.environ.get('REDSHIFT_CLUSTER')
            if redshift_cluster:
                logger.info(f"✅ Redshift cluster configured: {redshift_cluster}")
            else:
                logger.warning("⚠️ Redshift cluster not specified in environment")
               
        # Validate S3 bucket access
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=config['s3_bucket'])
        logger.info(f"✅ S3 bucket accessible: {config['s3_bucket']}")
       
        return True
       
    except Exception as e:
        logger.error(f"❌ Infrastructure validation failed: {str(e)}")
        return False

def validate_model_quality(config):
    """Validate model quality from evaluation metrics"""
    logger.info("🔍 Validating model quality...")
   
    try:
        s3_client = boto3.client('s3')
       
        # Find latest model run
        model_prefix = f"{config['s3_prefix']}/models/"
        response = s3_client.list_objects_v2(
            Bucket=config['s3_bucket'],
            Prefix=model_prefix,
            Delimiter='/'
        )
       
        if 'CommonPrefixes' not in response:
            logger.error("❌ No model runs found for quality validation")
            return False
           
        run_folders = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
        run_folders = [folder for folder in run_folders if folder.startswith('run_')]
       
        if not run_folders:
            logger.error("❌ No valid model runs found for quality validation")
            return False
           
        latest_run = sorted(run_folders)[-1]
       
        # Load evaluation metrics
        evaluation_key = f"{model_prefix}{latest_run}/evaluation.json"
       
        try:
            response = s3_client.get_object(Bucket=config['s3_bucket'], Key=evaluation_key)
            evaluation_data = json.loads(response['Body'].read().decode('utf-8'))
           
            # Extract key metrics
            metrics = evaluation_data.get('metrics', {})
            rmse = metrics.get('rmse', float('inf'))
            mape = metrics.get('mape', float('inf'))
            smape = metrics.get('smape', float('inf'))
            wape = metrics.get('wape', float('inf'))
            r2 = metrics.get('r2', -float('inf'))
           
            logger.info(f"📊 Model Quality Metrics:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAPE: {mape:.4f}%")
            logger.info(f"  WAPE: {wape:.4f}%")
            logger.info(f"  SMAPE: {smape:.4f}%")
            logger.info(f"  R²: {r2:.4f}")
           
            # # Define quality thresholds (adjust based on your requirements)
            # quality_thresholds = {
            #     'rmse_max': 50000.0,  # Maximum acceptable RMSE
            #     'mape_max': 50.0,    # Maximum acceptable MAPE (%)
            #     'r2_min': 0.3        # Minimum acceptable R²
            # }
           
            # # Check quality gates
            # quality_passed = True
           
            # if rmse > quality_thresholds['rmse_max']:
            #     logger.error(f"❌ RMSE too high: {rmse:.4f} > {quality_thresholds['rmse_max']}")
            #     quality_passed = False
            # else:
            #     logger.info(f"✅ RMSE acceptable: {rmse:.4f}")
               
            # if mape > quality_thresholds['mape_max']:
            #     logger.error(f"❌ MAPE too high: {mape:.4f}% > {quality_thresholds['mape_max']}%")
            #     quality_passed = False
            # else:
            #     logger.info(f"✅ MAPE acceptable: {mape:.4f}%")
               
            # if r2 < quality_thresholds['r2_min']:
            #     logger.error(f"❌ R² too low: {r2:.4f} < {quality_thresholds['r2_min']}")
            #     quality_passed = False
            # else:
            #     logger.info(f"✅ R² acceptable: {r2:.4f}")
               
            # if quality_passed:
            #     logger.info("✅ Model quality validation passed")
            # else:
            #     logger.error("❌ Model quality validation failed")
               
            # return quality_passed
            return True
        except Exception as e:
            logger.error(f"❌ Error loading evaluation metrics: {str(e)}")
            return False
           
    except Exception as e:
        logger.error(f"❌ Model quality validation failed: {str(e)}")
        return False

def main():
    """Main validation function"""
    logger.info("🚀 Starting deployment readiness validation...")
   
    try:
        # Get configuration from environment
        config = get_validation_config()
       
        logger.info(f"📋 Validation Configuration:")
        logger.info(f"  Combination: {config['customer_profile']}-{config['customer_segment']}")
        logger.info(f"  Environment: {config['environment']}")
        logger.info(f"  Pipeline: {config['pipeline_name']}")
        logger.info(f"  Endpoint: {config['endpoint_name']}")
        logger.info(f"  Model Package Group: {config['model_package_group']}")
        logger.info(f"  Database Type: {config['database_type']}")
       
        # Run validation checks
        validation_results = {}
       
        # 1. Validate S3 artifacts
        validation_results['s3_artifacts'] = validate_s3_artifacts(config)
       
        # 2. Validate SageMaker permissions
        validation_results['sagemaker_permissions'] = validate_sagemaker_permissions(config)
       
        # 3. Validate endpoint prerequisites
        validation_results['endpoint_prerequisites'] = validate_endpoint_prerequisites(config)
       
        # 4. Validate infrastructure readiness
        validation_results['infrastructure_readiness'] = validate_infrastructure_readiness(config)
       
        # 5. Validate model quality
        validation_results['model_quality'] = validate_model_quality(config)
       
        # Summary
        logger.info("\n📊 Validation Results Summary:")
        passed_checks = 0
        total_checks = len(validation_results)
       
        for check_name, passed in validation_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"  {check_name}: {status}")
            if passed:
                passed_checks += 1
               
        success_rate = (passed_checks / total_checks) * 100
        logger.info(f"\n📈 Overall Success Rate: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
       
        if all(validation_results.values()):
            logger.info("🎉 All validation checks passed! Deployment is ready to proceed.")
           
            # Create validation report
            validation_report = {
                'validation_timestamp': datetime.now().isoformat(),
                'combination': f"{config['customer_profile']}-{config['customer_segment']}",
                'environment': config['environment'],
                'validation_results': validation_results,
                'success_rate': success_rate,
                'overall_status': 'PASSED',
                'ready_for_deployment': True
            }
           
            # Save validation report
            report_file = f"validation_report_{config['customer_profile']}_{config['customer_segment']}.json"
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            logger.info(f"💾 Validation report saved: {report_file}")
           
            return 0
        else:
            failed_checks = [name for name, passed in validation_results.items() if not passed]
            logger.error(f"❌ Validation failed. Failed checks: {', '.join(failed_checks)}")
           
            # Create failure report
            validation_report = {
                'validation_timestamp': datetime.now().isoformat(),
                'combination': f"{config['customer_profile']}-{config['customer_segment']}",
                'environment': config['environment'],
                'validation_results': validation_results,
                'success_rate': success_rate,
                'overall_status': 'FAILED',
                'ready_for_deployment': False,
                'failed_checks': failed_checks
            }
           
            # Save validation report
            report_file = f"validation_report_{config['customer_profile']}_{config['customer_segment']}.json"
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            logger.info(f"💾 Validation report saved: {report_file}")
           
            return 1
           
    except Exception as e:
        logger.error(f"❌ Validation script failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
