#!/usr/bin/env python3
"""
Historical Prediction Generation Script

This script generates historical predictions by invoking Lambda functions
with optimized endpoint management for existing endpoints created by setup phase.
Maintains all existing functionality while adding test_invocation support.
"""

import boto3
import json
import time
import sys
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalForecastGenerator:
    def __init__(self):
        """Initialize the forecast generator with environment variables"""
        self.lambda_client = boto3.client('lambda')
        self.lambda_function_name = os.environ['LAMBDA_FUNCTION_NAME']
        self.profile = os.environ['PROFILE']
        self.segment = os.environ['SEGMENT']
        self.environment = os.environ['ENVIRONMENT']
        self.max_parallel = int(os.environ.get('MAX_PARALLEL_REQUESTS', 3))
        self.request_delay = int(os.environ.get('REQUEST_DELAY_SECONDS', 2))
        self.database_type = os.environ.get('DATABASE_TYPE', 'redshift')
        self.aws_region = os.environ.get('AWS_REGION')
       
        # Historical mode configuration
        self.historical_mode = os.environ.get('HISTORICAL_MODE', 'true').lower() == 'true'
        self.use_existing_endpoint = os.environ.get('USE_EXISTING_ENDPOINT', 'true').lower() == 'true'
        self.test_invocation = os.environ.get('TEST_INVOCATION', 'true').lower() == 'true'
       
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.prediction_results = []
        self.total_duration = 0
        self.start_time = time.time()
       
        logger.info(f"Generator initialized for {self.profile}-{self.segment}")
        logger.info(f"Lambda function: {self.lambda_function_name}")
        logger.info(f"Max parallel: {self.max_parallel}, Request delay: {self.request_delay}s")
        logger.info(f"Database type: {self.database_type}")
       
        # Log optimization settings
        logger.info(f" Historical mode: {self.historical_mode}")
        logger.info(f" Use existing endpoint: {self.use_existing_endpoint}")
        logger.info(f" Test invocation mode: {self.test_invocation}")

    def create_prediction_payload(self, forecast_date):
        """Create the JSON payload for Lambda function invocation"""
        payload = {
            "load_profile": self.profile,
            "customer_segment": self.segment,
            "forecast_date": forecast_date,
            # Use test_invocation=True to skip endpoint recreation
            "test_invocation": self.test_invocation,  # KEY CHANGE: Use existing endpoint
            # Additional parameters for historical mode
            "historical_mode": self.historical_mode,
            "use_existing_endpoint": self.use_existing_endpoint,
            "run_user": "historical_forecasting_system"
        }
       
        # Log payload details for first few calls
        if len(self.prediction_results) < 3:
            logger.info(f" Payload for {forecast_date}: test_invocation={payload['test_invocation']}")
       
        return payload

    def invoke_lambda_function(self, forecast_date):
        """Lambda function invocation with existing endpoint optimization"""
        request_start_time = time.time()
       
        try:
            logger.info(f" Generating prediction for {forecast_date} (using existing endpoint)")
           
            payload = self.create_prediction_payload(forecast_date)
           
            # Invoke the Lambda function
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_function_name,
                InvocationType='RequestResponse',  # Synchronous invocation
                Payload=json.dumps(payload)
            )
           
            # Check for Lambda function errors
            if response.get('FunctionError'):
                error_msg = f"Lambda function error for {forecast_date}: {response.get('FunctionError')}"
                logger.error(f" {error_msg}")
                return {
                    'success': False,
                    'date': forecast_date,
                    'error': error_msg,
                    'duration': time.time() - request_start_time
                }
           
            # Parse response
            result_payload = response['Payload'].read().decode()
           
            try:
                result = json.loads(result_payload)
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse Lambda response for {forecast_date}: {str(e)}"
                logger.error(f" {error_msg}")
                return {
                    'success': False,
                    'date': forecast_date,
                    'error': error_msg,
                    'duration': time.time() - request_start_time
                }
           
            # Check if Lambda execution was successful
            status_code = response.get('StatusCode', 0)
            lambda_status_code = result.get('statusCode', 0) if isinstance(result, dict) else 0
           
            if status_code != 200 or lambda_status_code != 200:
                # Extract error details
                if isinstance(result, dict):
                    if 'body' in result and isinstance(result['body'], str):
                        try:
                            body = json.loads(result['body'])
                            error_details = body.get('error', 'Unknown error')
                        except:
                            error_details = result.get('errorMessage', 'Unknown error')
                    else:
                        error_details = result.get('errorMessage', result.get('body', 'Unknown error'))
                else:
                    error_details = str(result)
               
                error_msg = f"Lambda execution failed for {forecast_date}: HTTP {status_code}, Lambda {lambda_status_code} - {error_details}"
                logger.error(f" {error_msg}")
                return {
                    'success': False,
                    'date': forecast_date,
                    'error': error_msg,
                    'duration': time.time() - request_start_time
                }
           
            # Extract prediction results
            duration = time.time() - request_start_time
           
            # Parse successful response
            if isinstance(result, dict) and 'body' in result:
                if isinstance(result['body'], str):
                    try:
                        body = json.loads(result['body'])
                    except:
                        body = result['body']
                else:
                    body = result['body']
               
                predictions_count = body.get('predictions_count', 0)
                records_inserted = body.get('records_inserted', 0)
               
                logger.info(f" SUCCESS for {forecast_date}: {predictions_count} predictions, {records_inserted} records ({duration:.1f}s)")
            else:
                logger.info(f" SUCCESS for {forecast_date} ({duration:.1f}s)")
                predictions_count = 0
                records_inserted = 0
           
            # Add metadata to result
            result_with_metadata = {
                'forecast_date': forecast_date,
                'profile': self.profile,
                'segment': self.segment,
                'lambda_function_name': self.lambda_function_name,
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'duration': duration,
                'predictions_count': predictions_count,
                'records_inserted': records_inserted,
                'used_existing_endpoint': self.use_existing_endpoint,
                'test_invocation_mode': self.test_invocation,
                'historical_mode': self.historical_mode,
                'response': result
            }
           
            return {
                'success': True,
                'date': forecast_date,
                'result': result_with_metadata,
                'duration': duration,
                'predictions_count': predictions_count,
                'records_inserted': records_inserted
            }
           
        except Exception as e:
            duration = time.time() - request_start_time
            error_msg = f"Failed to invoke Lambda function for {forecast_date}: {str(e)}"
            logger.error(f" {error_msg} ({duration:.1f}s)")
           
            return {
                'success': False,
                'date': forecast_date,
                'error': error_msg,
                'duration': duration
            }

    def generate_predictions_batch(self, dates):
        """Batch prediction with existing endpoint optimization"""
        logger.info(f" Starting batch prediction for {len(dates)} dates")
        logger.info(f"Using existing endpoint optimization: {self.use_existing_endpoint}")
       
        results = []
        batch_start_time = time.time()
       
        # Estimate time savings
        if self.use_existing_endpoint:
            estimated_time_without_optimization = len(dates) * 360  # 6 minutes per date
            logger.info(f" Estimated time without optimization: {estimated_time_without_optimization:.0f}s")
            logger.info(f" Expected time with optimization: ~{len(dates) * 30:.0f}s")
       
        # Use ThreadPoolExecutor for parallel requests (same as current)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(self.invoke_lambda_function, date): date
                for date in dates
            }
           
            # Process completed tasks
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    results.append(result)
                   
                    if result['success']:
                        self.successful_predictions += 1
                        logger.info(f" Completed {date}: SUCCESS ({result.get('duration', 0):.1f}s)")
                    else:
                        self.failed_predictions += 1
                        logger.error(f" Completed {date}: FAILED - {result.get('error', 'Unknown error')} ({result.get('duration', 0):.1f}s)")
                   
                    # Add delay between requests to avoid overwhelming the Lambda function
                    if len(results) < len(dates):  # Don't delay after the last request
                        time.sleep(self.request_delay)
                       
                except Exception as exc:
                    self.failed_predictions += 1
                    error_msg = f"Exception occurred for {date}: {str(exc)}"
                    logger.error(f" {error_msg}")
                    results.append({
                        'success': False,
                        'date': date,
                        'error': error_msg,
                        'duration': 0
                    })
       
        self.total_duration = time.time() - batch_start_time
        self.prediction_results = results
       
        # Logging with time savings
        logger.info(f" Batch prediction completed in {self.total_duration:.1f}s")
        if self.use_existing_endpoint:
            estimated_without_optimization = len(dates) * 360
            time_saved = estimated_without_optimization - self.total_duration
            savings_percentage = (time_saved / estimated_without_optimization) * 100 if estimated_without_optimization > 0 else 0
            logger.info(f" Time saved: {time_saved:.0f}s ({savings_percentage:.0f}% reduction)")
       
        return results

    def save_results_summary(self):
        """Results summary with optimization metrics"""
        # Calculate metrics
        total_dates = len(self.prediction_results)
        success_rate = (self.successful_predictions / total_dates * 100) if total_dates > 0 else 0
        average_duration_per_date = self.total_duration / total_dates if total_dates > 0 else 0
       
        # Calculate total predictions and records
        total_predictions_count = sum(r.get('predictions_count', 0) for r in self.prediction_results if r.get('success'))
        total_records_inserted = sum(r.get('records_inserted', 0) for r in self.prediction_results if r.get('success'))
       
        # Summary with optimization metrics
        summary = {
            'combination': f"{self.profile}-{self.segment}",
            'lambda_function_name': self.lambda_function_name,
            'total_dates': total_dates,
            'successful_dates': self.successful_predictions,
            'failed_dates': self.failed_predictions,
            'success_rate': success_rate,
            'total_predictions': total_predictions_count,
            'total_records_inserted': total_records_inserted,
            'total_duration': self.total_duration,
            'average_duration_per_date': average_duration_per_date,
           
            # Optimization metrics
            'historical_mode': self.historical_mode,
            'used_existing_endpoint': self.use_existing_endpoint,
            'test_invocation_mode': self.test_invocation,
            'optimization_enabled': self.use_existing_endpoint,
           
            # Processing metadata
            'processing_timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment,
            'database_type': self.database_type,
            'results': self.prediction_results
        }
       
        # Save detailed summary
        summary_file = f"prediction_summary_{self.profile}_{self.segment}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
       
        logger.info(f" Summary saved to {summary_file}")
       
        # Save successful dates list (same as current)
        successful_dates = [r['date'] for r in self.prediction_results if r['success']]
        if successful_dates:
            success_file = f"successful_dates_{self.profile}_{self.segment}.txt"
            with open(success_file, 'w') as f:
                for date in successful_dates:
                    f.write(f"{date}\n")
            logger.info(f" Successful dates saved to {success_file}")
       
        # Save failed dates list (same as current)
        failed_dates = [{'date': r['date'], 'error': r.get('error', 'Unknown error')}
                       for r in self.prediction_results if not r['success']]
        if failed_dates:
            failed_file = f"failed_dates_{self.profile}_{self.segment}.txt"
            with open(failed_file, 'w') as f:
                for item in failed_dates:
                    f.write(f"{item['date']}: {item['error']}\n")
            logger.info(f" Failed dates saved to {failed_file}")
       
        return summary

    def validate_database_connection(self):
        """Satabase connection validation (same as current)"""
        try:
            logger.info("Validating database connection parameters...")
           
            if self.database_type == 'athena':
                athena_database = os.environ.get('ATHENA_DATABASE')
                athena_table = os.environ.get('ATHENA_TABLE')
               
                if not athena_database or not athena_table:
                    logger.warning("Athena database/table information not available")
                else:
                    logger.info(f"Target Athena table: {athena_database}.{athena_table}")
                   
            elif self.database_type == 'redshift':
                redshift_cluster = os.environ.get('REDSHIFT_CLUSTER')
                redshift_schema = os.environ.get('REDSHIFT_SCHEMA')
                redshift_table = os.environ.get('REDSHIFT_TABLE')
               
                if not redshift_cluster or not redshift_schema or not redshift_table:
                    logger.warning("Redshift cluster/schema/table information not available")
                else:
                    logger.info(f"Target Redshift table: {redshift_schema}.{redshift_table} on {redshift_cluster}")
           
            logger.info(" Database configuration validated")
           
        except Exception as e:
            logger.warning(f"Could not validate database connection: {str(e)}")


def main():
    """Main prediction generation function"""
    try:
        logger.info(" Starting historical prediction generation...")
       
        # Get prediction dates from environment
        prediction_dates_json = os.environ.get('PREDICTION_DATES', '[]')
        prediction_dates = json.loads(prediction_dates_json)
       
        if not prediction_dates:
            logger.error("No prediction dates provided")
            sys.exit(1)
       
        logger.info(f"Processing {len(prediction_dates)} dates: {prediction_dates[:5]}{'...' if len(prediction_dates) > 5 else ''}")
       
        # Initialize generator
        generator = HistoricalForecastGenerator()
       
        # Validate database connection
        generator.validate_database_connection()
       
        # Generate predictions with existing endpoint optimization
        logger.info(" Starting prediction generation...")
        results = generator.generate_predictions_batch(prediction_dates)
       
        # Save results
        summary = generator.save_results_summary()
       
        # Final summary
        logger.info("=== PREDICTION GENERATION SUMMARY ===")
        logger.info(f"Combination: {generator.profile}-{generator.segment}")
        logger.info(f"Lambda function: {generator.lambda_function_name}")
        logger.info(f"Total dates processed: {len(prediction_dates)}")
        logger.info(f"Successful predictions: {generator.successful_predictions}")
        logger.info(f"Failed predictions: {generator.failed_predictions}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total predictions generated: {summary['total_predictions']}")
        logger.info(f"Total records inserted: {summary['total_records_inserted']}")
        logger.info(f"Total duration: {summary['total_duration']:.1f}s")
        logger.info(f"Average per date: {summary['average_duration_per_date']:.1f}s")
       
        # Optimization reporting
        logger.info(f" Used existing endpoint: {summary['used_existing_endpoint']}")
        logger.info(f" Test invocation mode: {summary['test_invocation_mode']}")
        if summary['used_existing_endpoint']:
            estimated_without_optimization = len(prediction_dates) * 360
            time_saved = estimated_without_optimization - summary['total_duration']
            savings_percentage = (time_saved / estimated_without_optimization) * 100 if estimated_without_optimization > 0 else 0
            logger.info(f" Estimated time saved: {time_saved:.0f}s ({savings_percentage:.0f}% reduction)")
       
        if generator.failed_predictions > 0:
            logger.warning(f" {generator.failed_predictions} predictions failed")
            failed_dates = [r['date'] for r in results if not r['success']]
            logger.warning(f"Failed dates: {failed_dates[:10]}{'...' if len(failed_dates) > 10 else ''}")
       
        # Set GitHub outputs (same as current)
        summary_file = f"prediction_summary_{generator.profile}_{generator.segment}.json"
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"total_predictions={len(prediction_dates)}\n")
            f.write(f"successful_predictions={generator.successful_predictions}\n")
            f.write(f"failed_predictions={generator.failed_predictions}\n")
            f.write(f"success_rate={summary['success_rate']:.1f}\n")
            f.write(f"summary_file={summary_file}\n")
            f.write(f"used_existing_endpoint={summary['used_existing_endpoint']}\n")
            f.write(f"total_duration={summary['total_duration']:.1f}\n")
       
        if generator.failed_predictions == 0:
            logger.info(" All predictions generated successfully with existing endpoint optimization!")
            return True
        elif generator.successful_predictions > 0:
            logger.info(" Partial success - some predictions completed with existing endpoint optimization")
            return True
        else:
            logger.error(" All predictions failed")
            return False
           
    except Exception as e:
        logger.error(f"Fatal error in prediction generation: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
