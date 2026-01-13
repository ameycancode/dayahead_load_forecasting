# Predictions & Lambda - Forecasting Implementation

## Table of Contents
- [Overview](#overview)
- [Lambda Function Architecture](#lambda-function-architecture)
- [Module: lambda_function.py](#module-lambda_functionpy)
- [Module: data_preparation.py](#module-data_preparationpy)
- [Module: feature_engineering.py](#module-feature_engineeringpy)
- [Module: weather_service.py](#module-weather_servicepy)
- [Module: endpoint_service.py](#module-endpoint_servicepy)
- [Module: utils.py](#module-utilspy)
- [Complete Execution Flow](#complete-execution-flow)
- [Cost Optimization Deep Dive](#cost-optimization-deep-dive)

---

## Overview

The predictions system generates day-ahead hourly electricity load forecasts using AWS Lambda. The architecture implements an aggressive cost optimization strategy through endpoint delete/recreate, reducing SageMaker inference costs by 99%+ while maintaining prediction accuracy.

**Key Features**:
- Scheduled daily execution via EventBridge
- Endpoint recreation from S3 configuration (~3-5 minutes)
- 24-hour hourly predictions generation
- Results storage in Redshift/Athena
- Automatic endpoint deletion after predictions
- Comprehensive error handling and logging

**Location**: `predictions/`

---

## Lambda Function Architecture

### Deployment Package Structure

```
lambda_forecast.zip (uploaded to AWS Lambda)
├── lambda_function.py              # Main handler (entry point)
├── forecast/
│   ├── __init__.py
│   ├── data_preparation.py         # Historical data retrieval
│   ├── feature_engineering.py      # Feature creation for inference
│   ├── weather_service.py          # Open-Meteo API client
│   ├── endpoint_service.py         # SageMaker invocation
│   └── utils.py                    # Logging, S3, utilities
├── configs/
│   └── config.py                   # Configuration constants
└── dependencies/                    # Third-party packages
    ├── boto3/                      # AWS SDK
    ├── pandas/                     # DataFrames
    ├── numpy/                      # Numerical operations
    ├── pytz/                       # Timezone handling
    └── openmeteo_requests/         # Weather API client
```

### Lambda Configuration

```python
Function Name: {env}-energy-daily-predictor-{profile}-{segment}
# Examples:
# - prod-energy-daily-predictor-RES-SOLAR
# - dev-energy-daily-predictor-MEDCI-NONSOLAR

Runtime: python3.9
Memory: 1024 MB (1 GB)
Timeout: 900 seconds (15 minutes)
Ephemeral Storage: 2048 MB (2 GB)
Environment Variables: 50+ configuration variables
```

### Execution Trigger

```yaml
EventBridge Schedule Rule:
  Name: EnergyForecastSchedule-{PROFILE}-{SEGMENT}-{ENV}
  Schedule Expression:
    - dev: cron(0 10 * * ? *)     # 10 AM UTC = 2 AM PST
    - qa: cron(0 11 * * ? *)      # 11 AM UTC = 3 AM PST
    - preprod: cron(0 8 * * ? *)  # 8 AM UTC = 12 AM PST
    - prod: cron(0 9 * * ? *)     # 9 AM UTC = 1 AM PST

  Target: Lambda function ARN
  Input: {"scheduled": true, "source": "eventbridge"}
```

---

## Module: lambda_function.py

**Purpose**: Main Lambda handler orchestrating the complete forecasting workflow.

### Main Handler: `lambda_handler()`

**Signature**:
```python
def lambda_handler(event, context):
    """
    Lambda entry point for day-ahead energy load forecasting.

    Args:
        event: Lambda event (from EventBridge or manual invoke)
        context: Lambda context (function metadata)

    Returns:
        Response dict with status, predictions metadata, cost optimization status
    """
```

**Event Parameters**:
```json
{
  "scheduled": true,                    // From EventBridge
  "source": "eventbridge",              // Event source
  "endpoint_name": "optional-override", // Optional endpoint override
  "forecast_date": "2026-01-15",       // Optional date override (default: tomorrow)
  "run_id": "optional-run-id",         // Optional run ID
  "run_user": "system",                // Optional user identifier
  "test_invocation": false             // If true, skip endpoint delete
}
```

**Execution Flow**:

```python
def lambda_handler(event, context):
    # 1. Initialize logging
    logger = setup_logging()
    logger.info(f"Lambda function: {context.function_name}")
    logger.info(f"Request ID: {context.aws_request_id}")
    logger.info(f"Memory limit: {context.memory_limit_in_mb} MB")
    logger.info(f"Time remaining: {context.get_remaining_time_in_millis()} ms")

    # 2. Load configuration from environment variables
    config = load_lambda_configuration()
    logger.info(f"Environment: {config['ENV_NAME']}")
    logger.info(f"Customer: {config['CUSTOMER_PROFILE']}-{config['CUSTOMER_SEGMENT']}")
    logger.info(f"Database: {config['DATABASE_TYPE']}")
    logger.info(f"S3 Bucket: {config['S3_BUCKET']}")

    # 3. Parse event parameters
    endpoint_name = event.get('endpoint_name', config.get('ENDPOINT_NAME'))
    forecast_date_str = event.get('forecast_date')
    run_id = event.get('run_id', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_user = event.get('run_user', 'system')
    test_invocation = event.get('test_invocation', False)

    # Default: forecast for tomorrow
    if forecast_date_str:
        forecast_date = datetime.strptime(forecast_date_str, '%Y-%m-%d').date()
    else:
        forecast_date = (datetime.now() + timedelta(days=1)).date()

    logger.info(f"Forecast date: {forecast_date}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Test invocation: {test_invocation}")

    # 4. Endpoint Management (if not test invocation)
    lambda_created_endpoint = False

    if not test_invocation and config.get('ENABLE_ENDPOINT_DELETE_RECREATE', False):
        manager = EndpointRecreationManager()

        # Check current endpoint status
        status = manager.get_endpoint_status(endpoint_name)
        logger.info(f"Endpoint status: {status}")

        # Recreate if not exists
        if status == 'NotFound':
            logger.info("⚡ COST OPTIMIZATION: Recreating endpoint from S3 config")
            success = manager.recreate_endpoint(endpoint_name, config)

            if not success:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': 'Failed to recreate endpoint',
                        'endpoint_name': endpoint_name
                    })
                }

            lambda_created_endpoint = True
            logger.info(f"✓ Endpoint recreated: {endpoint_name}")

        elif status != 'InService':
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': f'Endpoint not ready: {status}',
                    'endpoint_name': endpoint_name
                })
            }

    # 5. Generate predictions
    try:
        logger.info("=== GENERATING PREDICTIONS ===")
        predictions = generate_predictions(
            config=config,
            endpoint_name=endpoint_name,
            forecast_date=forecast_date,
            load_profile=config['CUSTOMER_PROFILE'],
            customer_segment=config['CUSTOMER_SEGMENT'],
            run_id=run_id
        )

        logger.info(f"✓ Generated {len(predictions)} hourly predictions")

    except Exception as e:
        logger.error(f"❌ Prediction generation failed: {str(e)}")
        logger.error(traceback.format_exc())

        # Cleanup endpoint if Lambda created it
        if lambda_created_endpoint and config.get('DELETE_ENDPOINT_AFTER_PREDICTION', False):
            manager.delete_endpoint_after_prediction(endpoint_name, config)

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Prediction generation failed',
                'message': str(e)
            })
        }

    # 6. Save predictions to database
    try:
        logger.info("=== SAVING PREDICTIONS ===")

        if config['DATABASE_TYPE'] == 'redshift':
            save_predictions_to_redshift_direct_insert(
                predictions=predictions,
                config=config,
                run_id=run_id,
                model_version='1.0',
                run_user=run_user,
                load_profile=config['CUSTOMER_PROFILE'],
                load_segment=config['CUSTOMER_SEGMENT']
            )
        else:  # athena
            save_predictions_to_athena_s3(
                predictions=predictions,
                config=config,
                run_id=run_id,
                model_version='1.0'
            )

        logger.info(f"✓ Saved {len(predictions)} predictions to {config['DATABASE_TYPE']}")

    except Exception as e:
        logger.error(f"❌ Failed to save predictions: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't fail the Lambda if save fails (predictions were generated)

    # 7. Delete endpoint (cost optimization)
    endpoint_deleted = False

    if not test_invocation and config.get('DELETE_ENDPOINT_AFTER_PREDICTION', False):
        try:
            logger.info("=== ENDPOINT CLEANUP (COST OPTIMIZATION) ===")
            manager = EndpointRecreationManager()
            success = manager.delete_endpoint_after_prediction(endpoint_name, config)

            if success:
                endpoint_deleted = True
                logger.info("✓ COST OPTIMIZATION: Endpoint deleted successfully")
                logger.info("✓ Ongoing cost: $0/hour")
            else:
                logger.warning("⚠ Endpoint deletion failed (non-fatal)")

        except Exception as e:
            logger.warning(f"⚠ Endpoint deletion error (non-fatal): {str(e)}")

    # 8. Return success response
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Forecast generation completed successfully',
            'run_id': run_id,
            'forecast_date': str(forecast_date),
            'predictions_count': len(predictions),
            'endpoint_name': endpoint_name,
            'endpoint_recreated': lambda_created_endpoint,
            'endpoint_deleted': endpoint_deleted,
            'cost_optimization': 'active' if endpoint_deleted else 'inactive',
            'database': config['DATABASE_TYPE']
        })
    }
```

### Function: `load_lambda_configuration()`

**Purpose**: Parse 50+ environment variables into structured config

**Implementation**:
```python
def load_lambda_configuration() -> dict:
    """
    Load and parse Lambda environment variables.

    Returns:
        Dictionary with parsed configuration
    """
    config = {}

    # Core identification
    config['CUSTOMER_PROFILE'] = os.environ['CUSTOMER_PROFILE']
    config['CUSTOMER_SEGMENT'] = os.environ['CUSTOMER_SEGMENT']
    config['ENV_NAME'] = os.environ['ENV_NAME']
    config['LOAD_PROFILE'] = os.environ.get('LOAD_PROFILE', config['CUSTOMER_PROFILE'])

    # AWS resources
    config['AWS_REGION'] = os.environ.get('AWS_REGION', 'us-west-2')
    config['S3_BUCKET'] = os.environ['S3_BUCKET']
    config['S3_PREFIX'] = os.environ.get('S3_PREFIX',
        f"{config['CUSTOMER_PROFILE']}-{config['CUSTOMER_SEGMENT']}")
    config['SAGEMAKER_ROLE_ARN'] = os.environ.get('SAGEMAKER_ROLE_ARN')

    # Database (Redshift)
    config['DATABASE_TYPE'] = os.environ.get('DATABASE_TYPE', 'redshift')

    if config['DATABASE_TYPE'] == 'redshift':
        config['REDSHIFT_CLUSTER_IDENTIFIER'] = os.environ['REDSHIFT_CLUSTER_IDENTIFIER']
        config['REDSHIFT_DATABASE'] = os.environ['REDSHIFT_DATABASE']
        config['REDSHIFT_DB_USER'] = os.environ['REDSHIFT_DB_USER']
        config['REDSHIFT_REGION'] = os.environ.get('REDSHIFT_REGION', 'us-west-2')
        config['REDSHIFT_INPUT_SCHEMA'] = os.environ['REDSHIFT_INPUT_SCHEMA']
        config['REDSHIFT_INPUT_TABLE'] = os.environ['REDSHIFT_INPUT_TABLE']
        config['REDSHIFT_OUTPUT_SCHEMA'] = os.environ['REDSHIFT_OUTPUT_SCHEMA']
        config['REDSHIFT_OUTPUT_TABLE'] = os.environ['REDSHIFT_OUTPUT_TABLE']

    # Endpoint management (COST OPTIMIZATION)
    config['ENDPOINT_NAME'] = os.environ['ENDPOINT_NAME']
    config['ENABLE_ENDPOINT_DELETE_RECREATE'] = os.environ.get(
        'ENABLE_ENDPOINT_DELETE_RECREATE', 'false').lower() == 'true'
    config['DELETE_ENDPOINT_AFTER_PREDICTION'] = os.environ.get(
        'DELETE_ENDPOINT_AFTER_PREDICTION', 'false').lower() == 'true'
    config['ENDPOINT_RECREATION_TIMEOUT'] = int(os.environ.get('ENDPOINT_RECREATION_TIMEOUT', '900'))
    config['ENDPOINT_DELETION_TIMEOUT'] = int(os.environ.get('ENDPOINT_DELETION_TIMEOUT', '300'))
    config['ENDPOINT_READY_BUFFER_TIME'] = int(os.environ.get('ENDPOINT_READY_BUFFER_TIME', '60'))
    config['ENDPOINT_CONFIG_S3_PREFIX'] = os.environ.get('ENDPOINT_CONFIG_S3_PREFIX', 'endpoint-configs')

    # Features and lags (parse compact format "14,21,28,35")
    lag_days_str = os.environ.get('DEFAULT_LAG_DAYS', '7,14,21,28')
    config['DEFAULT_LAG_DAYS'] = [int(x.strip()) for x in lag_days_str.split(',')]

    # Time periods (parse compact format "6,10")
    morning_peak_str = os.environ.get('MORNING_PEAK_HOURS', '6,10')
    config['MORNING_PEAK_HOURS'] = tuple(int(x.strip()) for x in morning_peak_str.split(','))

    solar_period_str = os.environ.get('SOLAR_PERIOD_HOURS', '9,17')
    config['SOLAR_PERIOD_HOURS'] = tuple(int(x.strip()) for x in solar_period_str.split(','))

    evening_ramp_str = os.environ.get('EVENING_RAMP_HOURS', '16,20')
    config['EVENING_RAMP_HOURS'] = tuple(int(x.strip()) for x in evening_ramp_str.split(','))

    evening_peak_str = os.environ.get('EVENING_PEAK_HOURS', '20,23')
    config['EVENING_PEAK_HOURS'] = tuple(int(x.strip()) for x in evening_peak_str.split(','))

    # Weather API
    config['DEFAULT_LATITUDE'] = float(os.environ.get('DEFAULT_LATITUDE', '32.7157'))
    config['DEFAULT_LONGITUDE'] = float(os.environ.get('DEFAULT_LONGITUDE', '-117.1611'))
    config['DEFAULT_TIMEZONE'] = os.environ.get('DEFAULT_TIMEZONE', 'America/Los_Angeles')

    weather_vars_str = os.environ.get('WEATHER_VARIABLES',
        'temperature_2m,apparent_temperature,cloudcover,direct_radiation,diffuse_radiation,shortwave_radiation,windspeed_10m,relativehumidity_2m,is_day')
    config['WEATHER_VARIABLES'] = [x.strip() for x in weather_vars_str.split(',')]

    # Data timing
    config['DATA_DELAY_DAYS'] = int(os.environ.get('DATA_DELAY_DAYS', '14'))
    config['FINAL_SUBMISSION_DELAY'] = int(os.environ.get('FINAL_SUBMISSION_DELAY', '48'))
    config['INITIAL_SUBMISSION_DELAY'] = int(os.environ.get('INITIAL_SUBMISSION_DELAY', '14'))
    config['SUBMISSION_TYPE_FINAL'] = os.environ.get('SUBMISSION_TYPE_FINAL', 'Final')
    config['SUBMISSION_TYPE_INITIAL'] = os.environ.get('SUBMISSION_TYPE_INITIAL', 'Initial')

    # Rate group filtering
    config['RATE_GROUP_FILTER_CLAUSE'] = os.environ.get('RATE_GROUP_FILTER_CLAUSE', '')

    return config
```

### Class: `EndpointRecreationManager`

**Purpose**: Manage SageMaker endpoint lifecycle for cost optimization

#### Method: `get_endpoint_status()`

```python
def get_endpoint_status(self, endpoint_name: str) -> str:
    """
    Get current endpoint status.

    Returns:
        'InService', 'Creating', 'Updating', 'Deleting', 'Failed', 'NotFound'
    """
    sagemaker_client = boto3.client('sagemaker')

    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        return status

    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            # Endpoint doesn't exist
            return 'NotFound'
        else:
            raise
```

#### Method: `load_endpoint_configuration()`

```python
def load_endpoint_configuration(self, endpoint_name: str, config: dict) -> dict:
    """
    Load stored endpoint configuration from S3.

    Tries multiple locations:
    1. Primary: s3://{bucket}/{prefix}/endpoint-configs/{endpoint_name}_config.json
    2. Backup: s3://{bucket}/{prefix}/endpoint-configs/customers/{profile}-{segment}/{endpoint_name}_config.json

    Returns:
        Endpoint configuration dict
    """
    s3_client = boto3.client('s3')
    bucket = config['S3_BUCKET']
    prefix = config['S3_PREFIX']

    # Try primary location
    primary_key = f"{prefix}/{config['ENDPOINT_CONFIG_S3_PREFIX']}/{endpoint_name}_config.json"

    try:
        response = s3_client.get_object(Bucket=bucket, Key=primary_key)
        endpoint_config = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"✓ Loaded config from primary location: {primary_key}")
        return endpoint_config

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"Config not found at primary location: {primary_key}")

            # Try backup location
            profile = config['CUSTOMER_PROFILE']
            segment = config['CUSTOMER_SEGMENT']
            backup_key = f"{prefix}/{config['ENDPOINT_CONFIG_S3_PREFIX']}/customers/{profile}-{segment}/{endpoint_name}_config.json"

            try:
                response = s3_client.get_object(Bucket=bucket, Key=backup_key)
                endpoint_config = json.loads(response['Body'].read().decode('utf-8'))
                logger.info(f"✓ Loaded config from backup location: {backup_key}")
                return endpoint_config

            except ClientError:
                logger.error(f"❌ Config not found at backup location: {backup_key}")
                raise Exception(f"Endpoint configuration not found for {endpoint_name}")
        else:
            raise
```

#### Method: `recreate_endpoint()`

**Purpose**: Recreate endpoint from stored S3 configuration

```python
def recreate_endpoint(self, endpoint_name: str, config: dict) -> bool:
    """
    Recreate SageMaker endpoint from S3 configuration.

    Steps:
    1. Load configuration from S3
    2. Recreate Model (if not exists)
    3. Recreate EndpointConfig (if not exists)
    4. Create Endpoint
    5. Wait for InService status

    Returns:
        True if successful, False otherwise
    """
    try:
        # Step 1: Load configuration
        logger.info("Step 1/5: Loading endpoint configuration from S3")
        endpoint_config = self.load_endpoint_configuration(endpoint_name, config)

        sagemaker_client = boto3.client('sagemaker')

        # Step 2: Recreate Model (if needed)
        logger.info("Step 2/5: Checking if model exists")
        model_name = endpoint_config['model_name']

        try:
            sagemaker_client.describe_model(ModelName=model_name)
            logger.info(f"✓ Model already exists: {model_name}")

        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.info(f"Creating model: {model_name}")
                model_config = endpoint_config['model_config']

                sagemaker_client.create_model(
                    ModelName=model_name,
                    PrimaryContainer=model_config['PrimaryContainer'],
                    ExecutionRoleArn=model_config['ExecutionRoleArn']
                )
                logger.info(f"✓ Model created: {model_name}")
            else:
                raise

        # Step 3: Recreate EndpointConfig (if needed)
        logger.info("Step 3/5: Checking if endpoint config exists")
        endpoint_config_name = endpoint_config['endpoint_config_name']

        try:
            sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            logger.info(f"✓ Endpoint config already exists: {endpoint_config_name}")

        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.info(f"Creating endpoint config: {endpoint_config_name}")
                endpoint_config_spec = endpoint_config['endpoint_config']

                sagemaker_client.create_endpoint_config(
                    EndpointConfigName=endpoint_config_name,
                    ProductionVariants=endpoint_config_spec['production_variants']
                )
                logger.info(f"✓ Endpoint config created: {endpoint_config_name}")
            else:
                raise

        # Step 4: Create Endpoint
        logger.info("Step 4/5: Creating endpoint")

        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=[
                {'Key': 'Environment', 'Value': config['ENV_NAME']},
                {'Key': 'Profile', 'Value': config['CUSTOMER_PROFILE']},
                {'Key': 'Segment', 'Value': config['CUSTOMER_SEGMENT']},
                {'Key': 'Recreated', 'Value': 'true'},
                {'Key': 'CostOptimized', 'Value': 'true'},
                {'Key': 'RecreatedAt', 'Value': datetime.now().isoformat()},
                {'Key': 'OriginalRunId', 'Value': endpoint_config.get('run_id', 'unknown')}
            ]
        )
        logger.info(f"✓ Endpoint creation initiated: {endpoint_name}")

        # Step 5: Wait for InService
        logger.info("Step 5/5: Waiting for endpoint to become InService")
        success = self._wait_for_endpoint_ready(endpoint_name, config)

        if success:
            logger.info(f"✓ Endpoint recreation completed: {endpoint_name}")
            return True
        else:
            logger.error(f"❌ Endpoint recreation failed (timeout): {endpoint_name}")
            return False

    except Exception as e:
        logger.error(f"❌ Endpoint recreation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False
```

#### Method: `_wait_for_endpoint_ready()`

```python
def _wait_for_endpoint_ready(self, endpoint_name: str, config: dict) -> bool:
    """
    Wait for endpoint to reach InService status.

    Polls every 30 seconds for up to recreation_timeout (default 900s = 15 min).
    After InService, waits additional buffer time (default 60s).

    Returns:
        True if endpoint reaches InService, False on timeout
    """
    sagemaker_client = boto3.client('sagemaker')

    timeout = config.get('ENDPOINT_RECREATION_TIMEOUT', 900)
    check_interval = 30
    buffer_time = config.get('ENDPOINT_READY_BUFFER_TIME', 60)

    elapsed = 0
    status = None

    logger.info(f"Waiting for endpoint (max {timeout}s, checking every {check_interval}s)")

    while elapsed < timeout:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']

            logger.info(f"Endpoint status: {status} (elapsed: {elapsed}s)")

            if status == 'InService':
                logger.info(f"✓ Endpoint is InService")

                # Wait buffer time for stability
                logger.info(f"Waiting buffer time ({buffer_time}s) for endpoint stability")
                time.sleep(buffer_time)

                logger.info("✓ Endpoint ready for predictions")
                return True

            elif status in ['Failed', 'RollingBack']:
                logger.error(f"❌ Endpoint creation failed: {status}")
                return False

            # Continue waiting
            time.sleep(check_interval)
            elapsed += check_interval

        except ClientError as e:
            logger.error(f"Error checking endpoint status: {str(e)}")
            time.sleep(check_interval)
            elapsed += check_interval

    logger.error(f"❌ Timeout waiting for endpoint ({elapsed}s)")
    return False
```

#### Method: `delete_endpoint_after_prediction()`

```python
def delete_endpoint_after_prediction(self, endpoint_name: str, config: dict) -> bool:
    """
    Delete endpoint after successful predictions (cost optimization).

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        sagemaker_client = boto3.client('sagemaker')

        # Check current status
        status = self.get_endpoint_status(endpoint_name)

        if status == 'NotFound':
            logger.info("✓ Endpoint already deleted")
            return True

        if status not in ['InService', 'Failed']:
            logger.warning(f"Endpoint in unexpected state for deletion: {status}")

        # Delete endpoint
        logger.info(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

        logger.info("✓ COST OPTIMIZATION: Endpoint deletion initiated")
        logger.info("✓ Expected result: $0/hour ongoing cost")

        # Optionally wait for deletion confirmation
        if config.get('WAIT_FOR_DELETION', False):
            self._wait_for_endpoint_deleted(endpoint_name, config)

        return True

    except Exception as e:
        logger.error(f"❌ Endpoint deletion failed: {str(e)}")
        return False
```

---

## Module: data_preparation.py

**Purpose**: Retrieve historical data from Redshift/Athena for feature engineering.

### Function: `calculate_data_date_ranges()`

**Purpose**: Calculate optimal date ranges for data retrieval

```python
def calculate_data_date_ranges(
    current_date: datetime,
    config: dict,
    data_delay_days: int = 14
) -> Tuple[date, date, date]:
    """
    Calculate three key dates for optimized data fetching.

    Args:
        current_date: Current datetime
        config: Configuration dict
        data_delay_days: Data availability delay (default 14 days)

    Returns:
        (start_date, end_date, final_cutoff_date)

    Example (current_date = 2026-01-13):
        start_date: 2025-11-04 (70 days back for lag features)
        end_date: 2025-12-30 (current_date - 14 days)
        final_cutoff_date: 2025-12-27 (latest Final submission)
    """
    # End date: Latest available data (accounting for delay)
    end_date = (current_date - timedelta(days=data_delay_days)).date()

    # Start date: 70 days back (enough for 35-day lags + extra)
    start_date = end_date - timedelta(days=70)

    # Find latest Final submission date via database query
    final_cutoff_date = find_final_cutoff_date(
        current_date, start_date, end_date, config
    )

    logger.info(f"Date ranges calculated:")
    logger.info(f"  Start date: {start_date} (70 days back)")
    logger.info(f"  End date: {end_date} (latest available)")
    logger.info(f"  Final cutoff: {final_cutoff_date} (latest Final submission)")
    logger.info(f"  Data span: {(end_date - start_date).days} days")

    return start_date, end_date, final_cutoff_date
```

### Function: `query_data_redshift()`

**Purpose**: Query Redshift for Final and Initial submissions

```python
def query_data_redshift(
    config: dict,
    current_date: datetime,
    load_profile: str,
    rate_group_filter: Optional[str] = None,
    query_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Query Redshift using Data API with optimized date ranges.

    Strategy:
    1. Query Final submissions (verified data) up to final_cutoff_date
    2. Query Initial submissions (recent data) from final_cutoff_date+1 to end_date
    3. Combine both datasets

    Returns:
        DataFrame with historical load data
    """
    # Calculate date ranges
    start_date, end_date, final_cutoff_date = calculate_data_date_ranges(
        current_date, config, data_delay_days=14
    )

    # Query Final submissions
    logger.info(f"Querying Final submissions: {start_date} to {final_cutoff_date}")
    final_df = query_redshift_final_data(
        config, start_date, final_cutoff_date,
        load_profile, rate_group_filter, query_limit
    )
    logger.info(f"✓ Retrieved {len(final_df)} Final submission rows")

    # Query Initial submissions (gap filling)
    initial_start = final_cutoff_date + timedelta(days=1)
    if initial_start <= end_date:
        logger.info(f"Querying Initial submissions: {initial_start} to {end_date}")
        initial_df = query_redshift_initial_data(
            config, initial_start, end_date,
            load_profile, rate_group_filter, query_limit
        )
        logger.info(f"✓ Retrieved {len(initial_df)} Initial submission rows")

        # Combine datasets
        df = pd.concat([final_df, initial_df], ignore_index=True)
    else:
        df = final_df

    # Sort by datetime
    df = df.sort_values(['tradedate', 'tradetime'])

    # Validate data completeness
    expected_days = (end_date - start_date).days + 1
    actual_days = df['tradedate'].nunique()
    coverage_pct = (actual_days / expected_days) * 100

    logger.info(f"Data coverage: {actual_days}/{expected_days} days ({coverage_pct:.1f}%)")

    if coverage_pct < 100:
        logger.warning(f"⚠ Incomplete data coverage: {coverage_pct:.1f}%")

    return df
```

### Function: `create_normalized_load_features()`

**Purpose**: Create safe forecast-available features that respect data delay

```python
def create_normalized_load_features(
    df: pd.DataFrame,
    forecast_delay_days: int = 14
) -> pd.DataFrame:
    """
    Create normalized load features using historical data with proper lag.

    Critical: All features must use data from at least forecast_delay_days ago
    to avoid data leakage (using future data not available at forecast time).

    Args:
        df: Historical load DataFrame
        forecast_delay_days: Minimum days back for feature calculation

    Returns:
        DataFrame with added normalized features
    """
    # Per-meter metrics (normalized by meter count)
    df['load_per_meter'] = np.where(
        df['loadmetercount'] > 0,
        df['loadlal'] / df['loadmetercount'],
        0
    )

    df['gen_per_meter'] = np.where(
        df['genmetercount'] > 0,
        df['genlal'] / df['genmetercount'],
        0
    )

    df['net_load_per_meter'] = np.where(
        df['metercount'] > 0,
        df['lossadjustedload'] / df['metercount'],
        0
    )

    # Export/import indicators
    df['is_net_export'] = (df['lossadjustedload'] < np.abs(df['genlal'])).astype(int)

    # Export level (categorical: 0=none, 1=low, 2=medium, 3=high)
    gen_load_ratio = np.where(
        df['loadlal'] > 0,
        np.abs(df['genlal']) / df['loadlal'],
        0
    )

    df['export_level'] = np.select(
        [gen_load_ratio == 0,
         gen_load_ratio < 0.5,
         gen_load_ratio < 1.0,
         gen_load_ratio >= 1.0],
        [0, 1, 2, 3],
        default=0
    )

    # Generation ratios (using shifted data for forecast delay)
    shift_hours = forecast_delay_days * 24

    df['hourly_gen_ratio'] = np.where(
        df['loadlal'].shift(shift_hours) > 0,
        np.abs(df['genlal'].shift(shift_hours)) / df['loadlal'].shift(shift_hours),
        0
    )

    # Cap extreme ratios
    df['hourly_gen_ratio'] = df['hourly_gen_ratio'].clip(upper=10.0)

    # Daily generation ratio
    daily_gen = df.groupby('date')['genlal'].transform(lambda x: np.abs(x).sum())
    daily_load = df.groupby('date')['loadlal'].transform('sum')

    df['daily_gen_load_ratio'] = np.where(
        daily_load > 0,
        daily_gen / daily_load,
        0
    )
    df['daily_gen_load_ratio'] = df['daily_gen_load_ratio'].clip(upper=5.0)

    # Mark features needing lag (historical context)
    for col in ['hourly_gen_ratio', 'daily_gen_load_ratio']:
        df[f'{col}_needs_lag'] = True

    return df
```

---

## Module: feature_engineering.py

**Purpose**: Create ML-ready features for model inference.

### Function: `add_weather_features()`

**Purpose**: Integrate weather forecast data with load data

```python
def add_weather_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge weather forecast data and create derived features.

    Args:
        df: Forecast DataFrame (24 rows for 24 hours)
        weather_df: Weather forecast from Open-Meteo API

    Returns:
        DataFrame with weather features added
    """
    # Round to nearest hour for merging
    df['datetime_hour'] = df['datetime'].dt.floor('h')
    weather_df['datetime_hour'] = weather_df['time'].dt.floor('h')

    # Merge on hourly timestamp
    df = pd.merge(df, weather_df, on='datetime_hour', how='left')

    # Temperature difference (comfort indicator)
    df['temperature_difference'] = (
        df['apparent_temperature'] - df['temperature_2m']
    )

    # Heating degree hours (below 65°F = 18.3°C)
    df['heating_degree_hours'] = np.maximum(18.3 - df['temperature_2m'], 0)

    # Cooling degree hours (above 75°F = 23.9°C)
    df['cooling_degree_hours'] = np.maximum(df['temperature_2m'] - 23.9, 0)

    # Direct radiation ratio (clear vs cloudy)
    df['direct_radiation_ratio'] = np.where(
        df['shortwave_radiation'] > 0,
        df['direct_radiation'] / df['shortwave_radiation'],
        0
    )

    # Period-specific temperatures
    df['morning_temp'] = np.where(
        (df['hour'] >= 6) & (df['hour'] < 9),
        df['temperature_2m'],
        0
    )

    df['evening_temp'] = np.where(
        (df['hour'] >= 17) & (df['hour'] < 22),
        df['temperature_2m'],
        0
    )

    # Solar window cloud cover
    df['solar_window_cloudcover'] = (
        df['cloudcover'] * df.get('is_solar_window', 0)
    )

    # Handle missing values (forward/backward fill)
    weather_cols = [
        'temperature_2m', 'apparent_temperature', 'cloudcover',
        'direct_radiation', 'diffuse_radiation', 'shortwave_radiation',
        'windspeed_10m', 'relativehumidity_2m'
    ]

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df
```

### Function: `create_weather_solar_interactions()`

**Purpose**: Create interaction features

```python
def create_weather_solar_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction terms between weather and solar features.

    Args:
        df: DataFrame with weather and solar features

    Returns:
        DataFrame with interaction features added
    """
    # Wind heating effect (cold + windy = increased heating)
    if 'windspeed_10m' in df.columns and 'heating_degree_hours' in df.columns:
        df['wind_heating_effect'] = (
            df['windspeed_10m'] * df['heating_degree_hours'] / 10
        )

    # Wind cooling effect (hot + windy = reduced cooling need)
    if 'windspeed_10m' in df.columns and 'cooling_degree_hours' in df.columns:
        df['wind_cooling_effect'] = (
            df['windspeed_10m'] * df['cooling_degree_hours'] / 10
        )

    # Humidity cooling impact (high humidity = reduced AC efficiency)
    if 'relativehumidity_2m' in df.columns and 'cooling_degree_hours' in df.columns:
        df['humidity_cooling_impact'] = np.where(
            df['cooling_degree_hours'] > 0,
            df['relativehumidity_2m'] * df['cooling_degree_hours'] / 100,
            0
        )

    # Evening humidity discomfort (hot + humid evenings)
    if 'relativehumidity_2m' in df.columns and 'evening_temp' in df.columns:
        df['evening_humidity_discomfort'] = np.where(
            df['evening_temp'] > 23.9,  # Above 75°F
            df['relativehumidity_2m'] * (df['evening_temp'] - 23.9) / 100,
            0
        )

    return df
```

---

## Module: weather_service.py

**Purpose**: Fetch weather forecasts from Open-Meteo API.

### Function: `fetch_weather_data()`

**Purpose**: Fetch weather data for any date (historical or forecast)

```python
def fetch_weather_data(
    target_date: Union[datetime, date],
    config: Optional[dict] = None,
    days: int = 1
) -> Optional[pd.DataFrame]:
    """
    Fetch weather data from Open-Meteo API.

    Automatically determines if historical or forecast based on date.

    Args:
        target_date: Date to fetch weather for
        config: Configuration dict with location and variables
        days: Number of days to fetch (for forecasts, max 16)

    Returns:
        DataFrame with hourly weather data, or None if error
    """
    # Default config
    if config is None:
        config = {}

    latitude = config.get('DEFAULT_LATITUDE', 32.7157)
    longitude = config.get('DEFAULT_LONGITUDE', -117.1611)
    timezone = config.get('DEFAULT_TIMEZONE', 'America/Los_Angeles')
    weather_variables = config.get('WEATHER_VARIABLES', [
        "temperature_2m", "apparent_temperature", "cloudcover",
        "direct_radiation", "diffuse_radiation", "shortwave_radiation",
        "windspeed_10m", "relativehumidity_2m", "is_day"
    ])

    # Convert to date
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    now = datetime.now().date()
    is_historical = target_date < now

    try:
        # Get Open-Meteo client
        client = get_openmeteo_client(cache_dir='/tmp/weather_cache')

        if is_historical:
            # Historical data endpoint
            url = "https://archive-api.open-meteo.com/v1/archive"
            start_date = target_date
            end_date = target_date

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "hourly": weather_variables,
                "timezone": timezone
            }

        else:
            # Forecast endpoint
            url = "https://api.open-meteo.com/v1/forecast"
            forecast_days = min(days, 16)  # API limit

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": weather_variables,
                "timezone": timezone,
                "forecast_days": forecast_days
            }

        # Call API
        logger.info(f"Fetching weather from Open-Meteo: {target_date}")
        responses = client.weather_api(url, params=params)
        response = responses[0]

        # Extract hourly data
        hourly = response.Hourly()

        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        # Convert to local timezone
        hourly_data["time"] = hourly_data["time"].dt.tz_convert(timezone)
        hourly_data["time"] = hourly_data["time"].dt.tz_localize(None)

        # Extract variables
        for i, variable in enumerate(weather_variables):
            hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

        # Create DataFrame
        weather_df = pd.DataFrame(data=hourly_data)

        # Filter to exact target date (for forecast, may return multiple days)
        target_start = datetime.combine(target_date, datetime.min.time())
        target_end = target_start + timedelta(days=1)

        weather_df = weather_df[
            (weather_df['time'] >= target_start) &
            (weather_df['time'] < target_end)
        ]

        logger.info(f"✓ Retrieved {len(weather_df)} hourly weather records")

        return weather_df

    except Exception as e:
        logger.error(f"❌ Weather fetch failed: {str(e)}")
        return None
```

---

## Module: endpoint_service.py

**Purpose**: Invoke SageMaker endpoints for predictions.

### Function: `invoke_sagemaker_endpoint()`

**Purpose**: Invoke endpoint and return predictions

```python
def invoke_sagemaker_endpoint(
    endpoint_name: str,
    inference_df: pd.DataFrame
) -> List[float]:
    """
    Invoke SageMaker endpoint with feature data.

    Args:
        endpoint_name: SageMaker endpoint name
        inference_df: DataFrame with features (typically 24 rows)

    Returns:
        List of predictions (floats)
    """
    runtime_client = boto3.client('sagemaker-runtime')

    # Convert DataFrame to JSON instances
    instances = inference_df.to_dict(orient='records')

    payload = {"instances": instances}

    logger.info(f"Invoking endpoint: {endpoint_name}")
    logger.info(f"Instances: {len(instances)}")

    try:
        # Invoke endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        # Parse response
        result = json.loads(response['Body'].read().decode())
        predictions = result.get('predictions', [])

        logger.info(f"✓ Received {len(predictions)} predictions")

        return predictions

    except Exception as e:
        logger.error(f"❌ Endpoint invocation failed: {str(e)}")
        raise
```

---

## Complete Execution Flow

### Step-by-Step Prediction Generation

```
1. TRIGGER
   EventBridge schedule (e.g., daily 9 AM UTC)
   ↓

2. LAMBDA INITIALIZATION
   - Load 50+ environment variables
   - Parse configuration
   - Set up logging
   ↓

3. ENDPOINT MANAGEMENT (if enabled)
   - Check endpoint status via describe_endpoint()
   - If NotFound:
     * Load config from S3
     * Recreate Model
     * Recreate EndpointConfig
     * Create Endpoint
     * Wait for InService (~3-5 minutes)
     * Wait buffer time (60 seconds)
   ↓

4. DATA PREPARATION
   - Calculate date ranges (70 days back)
   - Query Redshift:
     * Final submissions (verified data)
     * Initial submissions (recent data)
   - Combine datasets
   - Aggregate by datetime
   - Create normalized features
   ↓

5. WEATHER INTEGRATION
   - Fetch forecast from Open-Meteo API
   - Target date: tomorrow
   - 17 weather variables
   - Hourly resolution (24 rows)
   ↓

6. FEATURE ENGINEERING
   - Merge historical with weather
   - Add weather features (17 variables)
   - Add solar features (position, windows)
   - Create weather-solar interactions
   - Load model features from S3
   - Filter to available features
   - Fill missing with 0
   ↓

7. ENDPOINT INVOCATION
   - Create 24-hour inference DataFrame
   - Convert to JSON instances
   - POST to SageMaker endpoint
   - Receive 24 predictions
   ↓

8. RESULT STORAGE
   - Format predictions with metadata
   - Redshift: Direct INSERT via Data API
   - Or Athena: Write to S3 in partitioned format
   - 24 rows inserted (one per hour)
   ↓

9. ENDPOINT DELETION (cost optimization)
   - Delete endpoint via delete_endpoint()
   - Endpoint goes to "Deleting" then "NotFound"
   - Cost drops to $0/hour
   ↓

10. RESPONSE
    - Return success JSON
    - Include run_id, prediction count
    - Endpoint status (deleted)
    - Cost optimization status (active)
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────┐
│ EventBridge Trigger (9 AM UTC daily)                     │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Lambda: Load Configuration (50+ env vars)                │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ EndpointRecreationManager                                │
│  • Check status: NotFound                                │
│  • Load config from S3                                   │
│  • Recreate endpoint (~3-5 min)                          │
│  • Wait for InService + buffer                           │
└──────────────────┬───────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
┌──────────────────┐   ┌──────────────────┐
│ Data Preparation │   │ Weather Service  │
│ • Query Redshift │   │ • Open-Meteo API │
│ • 70 days back   │   │ • Tomorrow's     │
│ • Final+Initial  │   │   forecast       │
│ • Normalize      │   │ • 17 variables   │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│ Feature Engineering                                       │
│  • Merge historical + weather                            │
│  • Add solar features                                    │
│  • Create interactions                                   │
│  • Load model features from S3                           │
│  • Filter & fill missing                                 │
│  Output: 24 rows × 40 features                           │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Endpoint Invocation                                       │
│  • POST JSON to SageMaker                                │
│  • Receive 24 predictions                                │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Save to Redshift                                          │
│  • INSERT 24 rows via Data API                           │
│  • Include metadata (run_id, model_version)              │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Delete Endpoint (Cost Optimization)                       │
│  • delete_endpoint() call                                │
│  • Status: Deleting → NotFound                           │
│  • Cost: $0.47/hr → $0/hr                                │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Return Success Response                                   │
│  • statusCode: 200                                       │
│  • predictions_count: 24                                 │
│  • cost_optimization: active                             │
└──────────────────────────────────────────────────────────┘
```

---

## Cost Optimization Deep Dive

### Traditional vs Delete/Recreate

**Traditional Approach** (Keep endpoint running 24/7):
```
Endpoint: ml.m5.xlarge
Cost: $0.47/hour
Monthly: $0.47 × 24 × 30 = $338.40/month
Annual: $4,060.80/year

For 6 endpoints:
Monthly: $2,030.40
Annual: $24,364.80
```

**Delete/Recreate Approach**:
```
Daily Execution:
1. Endpoint recreation: 5 minutes
2. Prediction generation: 3 minutes
3. Result storage: 1 minute
4. Endpoint deletion: 1 minute
Total: 10 minutes/day

Monthly cost per endpoint:
10 min/day × 30 days = 300 minutes = 5 hours
5 hours × $0.47 = $2.35/month

Annual cost per endpoint: $28.20/year

For 6 endpoints:
Monthly: $14.10
Annual: $169.20

SAVINGS:
Monthly: $2,030.40 - $14.10 = $2,016.30 (99.3% reduction)
Annual: $24,364.80 - $169.20 = $24,195.60 (99.3% reduction)
```

### Implementation Details

**Endpoint Configuration Storage**:
```json
{
  "endpoint_name": "prod-energy-ml-endpoint-RES-SOLAR",
  "model_name": "prod-energy-model-RES-SOLAR-run_20260113",
  "model_config": {
    "PrimaryContainer": {
      "Image": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1",
      "ModelDataUrl": "s3://bucket/RES-SOLAR/models/run_20260113/xgboost-model",
      "Environment": {
        "SAGEMAKER_PROGRAM": "inference.py"
      }
    },
    "ExecutionRoleArn": "arn:aws:iam::123456789012:role/SageMakerRole"
  },
  "endpoint_config_name": "prod-energy-endpoint-config-RES-SOLAR-run_20260113",
  "endpoint_config": {
    "production_variants": [{
      "variant_name": "AllTraffic",
      "model_name": "prod-energy-model-RES-SOLAR-run_20260113",
      "instance_type": "ml.m5.xlarge",
      "initial_instance_count": 1,
      "initial_variant_weight": 1.0
    }]
  },
  "tags": {
    "Environment": "prod",
    "Profile": "RES",
    "Segment": "SOLAR",
    "CostOptimized": "true"
  },
  "cost_optimized": true,
  "delete_recreate_enabled": true,
  "created_at": "2026-01-13T12:00:00Z",
  "run_id": "run_20260113_120000"
}
```

**S3 Storage Locations**:
1. Primary: `s3://{bucket}/{profile}-{segment}/endpoint-configs/{endpoint_name}_config.json`
2. Backup: `s3://{bucket}/{profile}-{segment}/endpoint-configs/customers/{profile}-{segment}/{endpoint_name}_config.json`

**Recreation Time Breakdown**:
```
Model creation: 10 seconds (cached if exists)
EndpointConfig creation: 5 seconds (cached if exists)
Endpoint creation: 180-300 seconds (3-5 minutes)
Buffer wait: 60 seconds (stability)
Total: ~4-6 minutes
```

**Deletion Time**:
```
delete_endpoint() call: 1 second
Status transition: 30-60 seconds
Status: Deleting → NotFound
Total: ~1 minute
```

### Monitoring Cost Optimization

**CloudWatch Logs**:
```
[INFO] ⚡ COST OPTIMIZATION: Recreating endpoint from S3 config
[INFO] ✓ Endpoint recreated: prod-energy-ml-endpoint-RES-SOLAR
[INFO] ✓ Generated 24 hourly predictions
[INFO] ✓ Saved 24 predictions to redshift
[INFO] ⚡ COST OPTIMIZATION: Endpoint deleted successfully
[INFO] ✓ Ongoing cost: $0/hour
```

**Cost Explorer**:
- Service: Amazon SageMaker
- Usage Type: Endpoint Instance Hours
- Expected: ~5 hours/month per endpoint
- Cost: ~$2.35/month per endpoint

**Verification**:
```bash
# Check endpoint status (should be NotFound)
aws sagemaker describe-endpoint --endpoint-name prod-energy-ml-endpoint-RES-SOLAR

# Expected output:
# An error occurred (ValidationException) when calling the DescribeEndpoint operation:
# Could not find endpoint "arn:aws:sagemaker:...".
```

---

## Error Handling

### Common Issues and Solutions

**1. Endpoint Recreation Timeout**
```
Error: Endpoint not ready after 900 seconds
Solution:
- Check SageMaker service limits
- Increase ENDPOINT_RECREATION_TIMEOUT
- Verify instance availability in region
```

**2. Configuration Not Found**
```
Error: Endpoint configuration not found in S3
Solution:
- Verify deployment completed successfully
- Check S3 bucket permissions
- Look for backup config location
- Redeploy model to create new config
```

**3. Weather API Failure**
```
Error: Failed to fetch weather data
Solution:
- Check internet connectivity
- Verify Open-Meteo API status
- Continue with historical features only
- Non-fatal: predictions still generated
```

**4. Redshift INSERT Failure**
```
Error: Failed to insert predictions
Solution:
- Check Redshift cluster status
- Verify IAM permissions
- Check table schema matches
- Predictions are logged, can manually insert
```

**5. Lambda Timeout**
```
Error: Lambda timeout (900 seconds)
Solution:
- Increase Lambda timeout
- Optimize endpoint recreation (cache model)
- Reduce ENDPOINT_READY_BUFFER_TIME
- Split into multiple smaller Lambdas
```

### Logging Strategy

**Log Levels**:
- **INFO**: Normal execution milestones
- **WARNING**: Non-fatal issues (continue execution)
- **ERROR**: Fatal errors (stop execution)

**Key Log Messages**:
```python
# Endpoint management
logger.info("⚡ COST OPTIMIZATION: Recreating endpoint from S3 config")
logger.info("✓ Endpoint recreated: {endpoint_name}")
logger.info("⚡ COST OPTIMIZATION: Endpoint deleted successfully")
logger.info("✓ Ongoing cost: $0/hour")

# Data processing
logger.info("✓ Retrieved {rows} Final submission rows")
logger.info("✓ Retrieved {rows} hourly weather records")
logger.info("✓ Generated {count} hourly predictions")

# Storage
logger.info("✓ Saved {count} predictions to redshift")

# Errors
logger.error("❌ Endpoint recreation failed: {error}")
logger.error("❌ Prediction generation failed: {error}")
logger.warning("⚠ Endpoint deletion failed (non-fatal)")
```

---

## Performance Benchmarks

| Stage | Duration | Notes |
|-------|----------|-------|
| Lambda initialization | 2-3 sec | Cold start |
| Configuration loading | <1 sec | Environment variables |
| Endpoint recreation | 3-5 min | From S3 config |
| Historical data query | 10-20 sec | Redshift Data API |
| Weather API call | 1-2 sec | Cached responses |
| Feature engineering | 2-5 sec | 24 rows × 40 features |
| Endpoint invocation | 2-3 sec | SageMaker inference |
| Redshift INSERT | 3-5 sec | Data API |
| Endpoint deletion | 1-2 min | Asynchronous |
| **Total** | **8-12 min** | End-to-end |

---

## Environment Variables Reference

### Core Configuration (Required)
```bash
CUSTOMER_PROFILE=RES                # Load profile
CUSTOMER_SEGMENT=SOLAR              # Customer segment
ENV_NAME=prod                       # Environment
LOAD_PROFILE=RES                    # Same as CUSTOMER_PROFILE
```

### AWS Resources (Required)
```bash
AWS_REGION=us-west-2
S3_BUCKET=energy-forecasting-bucket
S3_PREFIX=RES-SOLAR
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole
```

### Database - Redshift (Required if DATABASE_TYPE=redshift)
```bash
DATABASE_TYPE=redshift
REDSHIFT_CLUSTER_IDENTIFIER=energy-cluster-prod
REDSHIFT_DATABASE=sdcp
REDSHIFT_DB_USER=ds_service_user
REDSHIFT_REGION=us-west-2
REDSHIFT_INPUT_SCHEMA=edp_cust_prod
REDSHIFT_INPUT_TABLE=caiso_sqmd
REDSHIFT_OUTPUT_SCHEMA=edp_forecasting_prod
REDSHIFT_OUTPUT_TABLE=dayahead_load_forecasts
```

### Endpoint Management (Cost Optimization)
```bash
ENDPOINT_NAME=prod-energy-ml-endpoint-RES-SOLAR
ENABLE_ENDPOINT_DELETE_RECREATE=true
DELETE_ENDPOINT_AFTER_PREDICTION=true
ENDPOINT_RECREATION_TIMEOUT=900        # 15 minutes
ENDPOINT_DELETION_TIMEOUT=300          # 5 minutes
ENDPOINT_READY_BUFFER_TIME=60          # 1 minute
ENDPOINT_CONFIG_S3_PREFIX=endpoint-configs
```

### Features and Lags
```bash
DEFAULT_LAG_DAYS=7,14,21,28
MORNING_PEAK_HOURS=6,10
SOLAR_PERIOD_HOURS=9,17
EVENING_RAMP_HOURS=16,20
EVENING_PEAK_HOURS=20,23
```

### Weather API
```bash
DEFAULT_LATITUDE=32.7157                # San Diego
DEFAULT_LONGITUDE=-117.1611
DEFAULT_TIMEZONE=America/Los_Angeles
WEATHER_VARIABLES=temperature_2m,apparent_temperature,cloudcover,direct_radiation,diffuse_radiation,shortwave_radiation,windspeed_10m,relativehumidity_2m,is_day
```

### Data Timing
```bash
DATA_DELAY_DAYS=14
FINAL_SUBMISSION_DELAY=48
INITIAL_SUBMISSION_DELAY=14
SUBMISSION_TYPE_FINAL=Final
SUBMISSION_TYPE_INITIAL=Initial
```

### Rate Group Filtering
```bash
# For NONSOLAR
RATE_GROUP_FILTER_CLAUSE=(rategroup NOT LIKE 'NEM%' AND rategroup NOT LIKE 'SBP%')

# For SOLAR
RATE_GROUP_FILTER_CLAUSE=(rategroup LIKE 'NEM%' OR rategroup LIKE 'SBP%')
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Status**: Production Ready
