#!/usr/bin/env python3
"""
Enhanced Endpoint Health Validation Script
Validates SageMaker endpoint health and readiness for production use.
"""
import os
import sys
import boto3
import json
import logging
import time
from datetime import datetime, timedelta

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
            'endpoint_name': os.environ['ENDPOINT_NAME'],
            'aws_region': os.environ['AWS_REGION'],
            'run_id': os.environ.get('RUN_ID', ''),
        }
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        raise

def validate_endpoint_status(config):
    """Validate that endpoint is in correct status"""
    logger.info(" Validating endpoint status...")
    
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=config['endpoint_name'])
        
        endpoint_status = response['EndpointStatus']
        creation_time = response['CreationTime']
        last_modified = response['LastModifiedTime']
        
        logger.info(f" Endpoint Status Information:")
        logger.info(f"  Name: {config['endpoint_name']}")
        logger.info(f"  Status: {endpoint_status}")
        logger.info(f"  Created: {creation_time}")
        logger.info(f"  Last Modified: {last_modified}")
        
        if endpoint_status != 'InService':
            logger.error(f" Endpoint is not InService: {endpoint_status}")
            if 'FailureReason' in response:
                logger.error(f"Failure reason: {response['FailureReason']}")
            return False
            
        # Check production variants
        if 'ProductionVariants' in response:
            logger.info(" Production Variants:")
            all_variants_healthy = True
            
            for variant in response['ProductionVariants']:
                variant_name = variant['VariantName']
                current_instances = variant['CurrentInstanceCount']
                desired_instances = variant['DesiredInstanceCount']
                current_weight = variant['CurrentWeight']
                
                logger.info(f"  Variant '{variant_name}':")
                logger.info(f"    Current Instances: {current_instances}")
                logger.info(f"    Desired Instances: {desired_instances}")
                logger.info(f"    Weight: {current_weight}")
                
                if current_instances != desired_instances:
                    logger.warning(f" Instance count mismatch for variant {variant_name}")
                    all_variants_healthy = False
                    
            if not all_variants_healthy:
                logger.error(" Not all production variants are healthy")
                return False
                
        logger.info(" Endpoint status validation passed")
        return True
        
    except Exception as e:
        logger.error(f" Error validating endpoint status: {str(e)}")
        return False

def create_test_payload(config):
    """Create a test payload for endpoint inference"""
    logger.info(" Creating test payload...")
    
    # Create a minimal test payload that mimics real inference data
    # This should be representative of your actual model input
    test_payload = {
        "instances": [
            {
                # Sample features - adjust based on your model's expected input
                "hour": 12,
                "dayofweek": 1,
                "month": 6,
                "is_weekend": 0,
                "temp_c": 25.0,
                "humidity": 65.0,
                "wind_speed": 10.0,
                "solar_radiation": 800.0,
                "lag_14_days": 1500.0,
                "lag_21_days": 1450.0,
                "ma_7_days": 1475.0,
                "customer_profile": config['customer_profile'],
                "customer_segment": config['customer_segment']
            }
        ]
    }
    
    logger.info(f" Test payload created for {config['customer_profile']}-{config['customer_segment']}")
    return test_payload

def test_endpoint_inference(config):
    """Test endpoint inference capability"""
    logger.info(" Testing endpoint inference...")
    
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    try:
        # Create test payload
        test_payload = create_test_payload(config)
        payload_json = json.dumps(test_payload)
        
        logger.info(f" Sending test request to endpoint: {config['endpoint_name']}")
        
        # Invoke endpoint
        start_time = time.time()
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=config['endpoint_name'],
            ContentType='application/json',
            Body=payload_json
        )
        end_time = time.time()
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f" Inference successful!")
        logger.info(f" Response time: {inference_time:.2f}ms")
        logger.info(f" Response format: {type(result)}")
        
        # Validate response structure
        if isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']
            logger.info(f" Predictions: {predictions}")
            
            # Basic sanity check on predictions
            if isinstance(predictions, list) and len(predictions) > 0:
                prediction_value = predictions[0]
                
                # Check if prediction is a reasonable number for energy load
                if isinstance(prediction_value, (int, float)) and 0 <= prediction_value <= 100000:
                    logger.info(f" Prediction value looks reasonable: {prediction_value}")
                else:
                    logger.warning(f" Prediction value may be unusual: {prediction_value}")
                    
            else:
                logger.warning(" Predictions list is empty or invalid format")
                
        elif isinstance(result, list):
            logger.info(f" Direct prediction result: {result}")
        else:
            logger.warning(f" Unexpected response format: {result}")
            
        # Performance check
        if inference_time > 5000:  # 5 seconds
            logger.warning(f" Slow inference time: {inference_time:.2f}ms")
        elif inference_time > 1000:  # 1 second
            logger.info(f" Moderate inference time: {inference_time:.2f}ms")
        else:
            logger.info(f" Fast inference time: {inference_time:.2f}ms")
            
        return True, inference_time
        
    except Exception as e:
        logger.error(f" Endpoint inference test failed: {str(e)}")
        return False, 0

def validate_endpoint_metrics(config):
    """Validate endpoint CloudWatch metrics"""
    logger.info(" Validating endpoint metrics...")
    
    cloudwatch = boto3.client('cloudwatch')
    
    try:
        # Define metric queries for the endpoint
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=10)  # Last 10 minutes
        
        metrics_to_check = [
            {
                'name': 'Invocations',
                'namespace': 'AWS/SageMaker',
                'stat': 'Sum'
            },
            {
                'name': 'InvocationLatency',
                'namespace': 'AWS/SageMaker', 
                'stat': 'Average'
            },
            {
                'name': 'ModelLatency',
                'namespace': 'AWS/SageMaker',
                'stat': 'Average'
            }
        ]
        
        metrics_available = False
        
        for metric in metrics_to_check:
            try:
                response = cloudwatch.get_metric_statistics(
                    Namespace=metric['namespace'],
                    MetricName=metric['name'],
                    Dimensions=[
                        {
                            'Name': 'EndpointName',
                            'Value': config['endpoint_name']
                        },
                        {
                            'Name': 'VariantName', 
                            'Value': 'AllTraffic'
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,  # 5 minutes
                    Statistics=[metric['stat']]
                )
                
                datapoints = response['Datapoints']
                if datapoints:
                    latest_value = datapoints[-1][metric['stat']]
                    logger.info(f" {metric['name']}: {latest_value}")
                    metrics_available = True
                else:
                    logger.info(f" No recent data for {metric['name']}")
                    
            except Exception as metric_error:
                logger.warning(f" Could not retrieve {metric['name']}: {str(metric_error)}")
                
        if metrics_available:
            logger.info(" Endpoint metrics are being collected")
        else:
            logger.info(" No metrics available yet (endpoint may be newly created)")
            
        return True
        
    except Exception as e:
        logger.warning(f" Metrics validation failed: {str(e)}")
        return True  # Don't fail validation just because of metrics issues

def run_load_test(config, num_requests=5):
    """Run a basic load test on the endpoint"""
    logger.info(f" Running basic load test ({num_requests} requests)...")
    
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    test_payload = create_test_payload(config)
    payload_json = json.dumps(test_payload)
    
    results = []
    
    try:
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=config['endpoint_name'],
                    ContentType='application/json',
                    Body=payload_json
                )
                
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000
                
                # Parse response to ensure it's valid
                result = json.loads(response['Body'].read().decode())
                
                results.append({
                    'success': True,
                    'time_ms': inference_time,
                    'request_num': i + 1
                })
                
                logger.info(f"  Request {i+1}: {inference_time:.2f}ms ")
                
            except Exception as request_error:
                results.append({
                    'success': False,
                    'error': str(request_error),
                    'request_num': i + 1
                })
                logger.error(f"  Request {i+1}: Failed - {str(request_error)}")
                
            # Small delay between requests
            if i < num_requests - 1:
                time.sleep(0.5)
                
        # Analyze results
        successful_requests = [r for r in results if r.get('success', False)]
        failed_requests = [r for r in results if not r.get('success', False)]
        
        success_rate = len(successful_requests) / len(results) * 100
        
        if successful_requests:
            response_times = [r['time_ms'] for r in successful_requests]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            logger.info(f" Load Test Results:")
            logger.info(f"  Success Rate: {success_rate:.1f}% ({len(successful_requests)}/{len(results)})")
            logger.info(f"  Avg Response Time: {avg_response_time:.2f}ms")
            logger.info(f"  Min Response Time: {min_response_time:.2f}ms")
            logger.info(f"  Max Response Time: {max_response_time:.2f}ms")
            
            if success_rate >= 100:
                logger.info(" Load test passed - all requests successful")
                return True
            elif success_rate >= 80:
                logger.warning(f" Load test partially passed - {success_rate:.1f}% success rate")
                return True
            else:
                logger.error(f" Load test failed - only {success_rate:.1f}% success rate")
                return False
        else:
            logger.error(" Load test failed - no successful requests")
            return False
            
    except Exception as e:
        logger.error(f" Load test failed: {str(e)}")
        return False

def main():
    """Main health validation function"""
    logger.info(" Starting endpoint health validation...")
    
    try:
        # Get configuration from environment
        config = get_validation_config()
        
        logger.info(f" Health Validation Configuration:")
        logger.info(f"  Combination: {config['customer_profile']}-{config['customer_segment']}")
        logger.info(f"  Environment: {config['environment']}")
        logger.info(f"  Endpoint: {config['endpoint_name']}")
        
        # Run validation tests
        validation_results = {}
        
        # 1. Validate endpoint status
        validation_results['endpoint_status'] = validate_endpoint_status(config)
        
        # 2. Test endpoint inference
        inference_success, inference_time = test_endpoint_inference(config)
        validation_results['inference_test'] = inference_success
        
        # 3. Validate endpoint metrics
        validation_results['metrics_validation'] = validate_endpoint_metrics(config)
        
        # 4. Run basic load test
        validation_results['load_test'] = run_load_test(config)
        
        # Summary
        logger.info("\n Health Validation Summary:")
        passed_tests = 0
        total_tests = len(validation_results)
        
        for test_name, passed in validation_results.items():
            status = " PASSED" if passed else " FAILED"
            logger.info(f"  {test_name}: {status}")
            if passed:
                passed_tests += 1
                
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"\n Overall Health Score: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if all(validation_results.values()):
            logger.info(" All health checks passed! Endpoint is ready for production use.")
            
            # Create health report
            health_report = {
                'validation_timestamp': datetime.now().isoformat(),
                'combination': f"{config['customer_profile']}-{config['customer_segment']}",
                'environment': config['environment'],
                'endpoint_name': config['endpoint_name'],
                'health_results': validation_results,
                'health_score': success_rate,
                'overall_status': 'HEALTHY',
                'inference_time_ms': inference_time,
                'ready_for_production': True
            }
            
            # Save health report
            report_file = f"endpoint_health_{config['customer_profile']}_{config['customer_segment']}.json"
            with open(report_file, 'w') as f:
                json.dump(health_report, f, indent=2)
            logger.info(f" Health report saved: {report_file}")
            
            return 0
        else:
            failed_tests = [name for name, passed in validation_results.items() if not passed]
            logger.error(f" Health validation failed. Failed tests: {', '.join(failed_tests)}")
            
            # Create failure report
            health_report = {
                'validation_timestamp': datetime.now().isoformat(),
                'combination': f"{config['customer_profile']}-{config['customer_segment']}",
                'environment': config['environment'],
                'endpoint_name': config['endpoint_name'],
                'health_results': validation_results,
                'health_score': success_rate,
                'overall_status': 'UNHEALTHY',
                'failed_tests': failed_tests,
                'ready_for_production': False
            }
            
            # Save health report
            report_file = f"endpoint_health_{config['customer_profile']}_{config['customer_segment']}.json"
            with open(report_file, 'w') as f:
                json.dump(health_report, f, indent=2)
            logger.info(f" Health report saved: {report_file}")
            
            return 1
            
    except Exception as e:
        logger.error(f" Health validation script failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
