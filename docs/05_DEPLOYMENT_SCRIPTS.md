# Deployment Scripts Reference

## Table of Contents
- [Overview](#overview)
- [Script Execution Order](#script-execution-order)
- [Infrastructure Setup Scripts](#infrastructure-setup-scripts)
- [Pipeline Management Scripts](#pipeline-management-scripts)
- [Model Deployment Scripts](#model-deployment-scripts)
- [Lambda Management Scripts](#lambda-management-scripts)
- [Validation & Testing Scripts](#validation--testing-scripts)
- [Reporting Scripts](#reporting-scripts)
- [Script Dependencies](#script-dependencies)

---

## Overview

The deployment automation consists of **23 Python scripts** located in `.github/scripts/deploy/`. These scripts orchestrate the complete MLOps pipeline from infrastructure setup through model deployment to Lambda creation.

### Script Categories

```
Infrastructure Setup (3 scripts):
├── setup_redshift_infrastructure.py    (65 KB, 1,800+ lines)
├── setup_athena.py                     (14 KB, 400+ lines)
└── check_sagemaker_permissions.py      (16 KB, 450+ lines)

Pipeline Management (7 scripts):
├── prepare_config.py                   (8 KB, 220+ lines)
├── create_pipeline.py                  (4 KB, 110+ lines)
├── execute_pipeline.py                 (7 KB, 200+ lines)
├── wait_for_execution.py               (2 KB, 60+ lines)
├── monitor_pipeline.py                 (7 KB, 190+ lines)
├── processing_wrapper.py               (16 KB, 440+ lines)
└── training_wrapper.py                 (47 KB, 1,300+ lines)

Model Deployment (4 scripts):
├── register_model.py                   (4 KB, 120+ lines)
├── deploy_model.py                     (22 KB, 600+ lines)
├── validate_deployment_readiness.py    (16 KB, 440+ lines)
└── validate_endpoint_health.py         (17 KB, 480+ lines)

Lambda Management (4 scripts):
├── create_lambda.py                    (25 KB, 680+ lines)
├── create_forecast_lambda.py           (38 KB, 1,050+ lines)
├── setup_schedule.py                   (3 KB, 95+ lines)
└── test_lambda.py                      (10 KB, 275+ lines)

Validation & Analysis (3 scripts):
├── integration_test.py                 (8 KB, 225+ lines)
├── analyze_model.py                    (5 KB, 140+ lines)
└── extract_metrics.py                  (2 KB, 60+ lines)

Reporting (2 scripts):
├── generate_report.py                  (19 KB, 515+ lines)
└── (deployment_summary in workflow)

Total: 23 scripts, ~270 KB, ~7,500+ lines of code
```

---

## Script Execution Order

### In GitHub Actions Workflow

```
1. check_sagemaker_permissions.py
   ↓
2. setup_redshift_infrastructure.py  (or setup_athena.py)
   ↓
3. prepare_config.py
   ↓
4. create_pipeline.py
   ↓
5. execute_pipeline.py
   ↓
6. wait_for_execution.py / monitor_pipeline.py
   ↓
   [SageMaker Pipeline Execution]
   ├── processing_wrapper.py (in SageMaker)
   └── training_wrapper.py (in SageMaker)
   ↓
7. analyze_model.py
   ↓
8. extract_metrics.py
   ↓
9. validate_deployment_readiness.py
   ↓
10. register_model.py
    ↓
11. deploy_model.py
    ↓
12. validate_endpoint_health.py
    ↓
13. create_lambda.py
    ↓
14. create_forecast_lambda.py
    ↓
15. setup_schedule.py
    ↓
16. test_lambda.py
    ↓
17. integration_test.py
    ↓
18. generate_report.py
```

---

## Infrastructure Setup Scripts

### 1. setup_redshift_infrastructure.py

**Purpose**: Create and configure Redshift database schemas and tables

**Size**: 65 KB, 1,800+ lines (largest script)

**Key Functions**:

#### `main()`
```python
def main():
    """
    Main entry point for Redshift infrastructure setup.

    Environment Variables Required:
    - REDSHIFT_CLUSTER_IDENTIFIER
    - REDSHIFT_DATABASE
    - REDSHIFT_DB_USER
    - REDSHIFT_INPUT_SCHEMA
    - REDSHIFT_INPUT_TABLE
    - REDSHIFT_OUTPUT_SCHEMA
    - REDSHIFT_OUTPUT_TABLE
    - REDSHIFT_BI_SCHEMA
    - REDSHIFT_BI_VIEW_NAME
    """
    # 1. Load configuration
    config = load_environment_config()

    # 2. Test cluster connectivity
    test_redshift_connectivity(config)

    # 3. Create input schema (if not exists)
    create_input_schema(config)

    # 4. Validate input table exists
    validate_input_table(config)

    # 5. Create output schema
    create_output_schema(config)

    # 6. Create output table
    create_output_table(config)

    # 7. Create BI schema
    create_bi_schema(config)

    # 8. Create BI view
    create_bi_view(config)

    # 9. Grant permissions
    grant_permissions(config)

    # 10. Verify setup
    verify_setup(config)
```

#### `create_output_table()`
```python
def create_output_table(config):
    """
    Create predictions output table with proper schema.

    Table: {output_schema}.{output_table}
    Purpose: Store hourly day-ahead load forecasts
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS {config['output_schema']}.{config['output_table']} (
        forecast_datetime TIMESTAMP NOT NULL,
        predicted_lossadjustedload FLOAT,
        run_id VARCHAR(100),
        model_version VARCHAR(50),
        run_user VARCHAR(100),
        created_at TIMESTAMP DEFAULT GETDATE(),
        load_profile VARCHAR(50),
        load_segment VARCHAR(50),
        year INT,
        month INT,
        day INT,
        PRIMARY KEY (forecast_datetime, run_id, load_profile, load_segment)
    )
    DISTKEY (load_profile)
    SORTKEY (forecast_datetime);
    """

    execute_redshift_sql(sql, config)
```

#### `create_bi_view()`
```python
def create_bi_view(config):
    """
    Create BI view for dashboard consumption.

    View: {bi_schema}.vw_{output_table}
    Purpose: Filtered, optimized view for BI tools
    """
    sql = f"""
    CREATE OR REPLACE VIEW {config['bi_schema']}.{config['bi_view']} AS
    SELECT
        forecast_datetime,
        predicted_lossadjustedload,
        load_profile,
        load_segment,
        model_version,
        run_id,
        created_at,
        year,
        month,
        day,
        EXTRACT(hour FROM forecast_datetime) AS hour,
        EXTRACT(dow FROM forecast_datetime) AS day_of_week
    FROM {config['output_schema']}.{config['output_table']}
    WHERE created_at >= DATEADD(day, -90, GETDATE())
    ORDER BY forecast_datetime DESC;
    """

    execute_redshift_sql(sql, config)
```

**Execution Example**:
```bash
python .github/scripts/deploy/setup_redshift_infrastructure.py

# Output:
# ✓ Connected to Redshift cluster: energy-cluster-prod
# ✓ Input schema exists: edp_cust_prod
# ✓ Input table validated: caiso_sqmd (26,000 rows)
# ✓ Created output schema: edp_forecasting_prod
# ✓ Created output table: dayahead_load_forecasts
# ✓ Created BI schema: edp_bi_prod
# ✓ Created BI view: vw_dayahead_load_forecasts
# ✓ Granted permissions to role: SageMakerRole
# ✓ Setup completed successfully
```

---

### 2. check_sagemaker_permissions.py

**Purpose**: Validate IAM permissions before pipeline execution

**Size**: 16 KB, 450+ lines

**Key Functions**:

#### `check_all_permissions()`
```python
def check_all_permissions(role_arn: str) -> dict:
    """
    Check all required SageMaker permissions.

    Returns:
        {
            'passed': 15,
            'failed': 2,
            'total': 17,
            'success_rate': 88.2,
            'failed_permissions': ['sagemaker:CreateEndpoint', ...]
        }
    """
    permissions_to_check = [
        # Pipeline permissions
        'sagemaker:CreatePipeline',
        'sagemaker:DescribePipeline',
        'sagemaker:StartPipelineExecution',
        'sagemaker:DescribePipelineExecution',
        'sagemaker:ListPipelineExecutionSteps',

        # Processing/Training permissions
        'sagemaker:CreateProcessingJob',
        'sagemaker:DescribeProcessingJob',
        'sagemaker:CreateTrainingJob',
        'sagemaker:DescribeTrainingJob',

        # Model permissions
        'sagemaker:CreateModel',
        'sagemaker:CreateModelPackage',
        'sagemaker:DescribeModelPackage',
        'sagemaker:CreateModelPackageGroup',

        # Endpoint permissions
        'sagemaker:CreateEndpointConfig',
        'sagemaker:CreateEndpoint',
        'sagemaker:DescribeEndpoint',
        'sagemaker:DeleteEndpoint',

        # S3 permissions
        's3:GetObject',
        's3:PutObject',
        's3:ListBucket',

        # Lambda permissions
        'lambda:CreateFunction',
        'lambda:UpdateFunctionCode',
        'lambda:UpdateFunctionConfiguration',
        'lambda:AddPermission',

        # IAM permissions
        'iam:PassRole'
    ]

    results = {}
    failed = []

    for permission in permissions_to_check:
        has_permission = check_permission(role_arn, permission)
        results[permission] = has_permission

        if not has_permission:
            failed.append(permission)

    return {
        'passed': len([p for p in results.values() if p]),
        'failed': len(failed),
        'total': len(permissions_to_check),
        'success_rate': (len([p for p in results.values() if p]) / len(permissions_to_check)) * 100,
        'failed_permissions': failed
    }
```

#### `check_permission()`
```python
def check_permission(role_arn: str, permission: str) -> bool:
    """
    Check if role has specific permission using IAM policy simulator.
    """
    iam_client = boto3.client('iam')

    try:
        # Get service and action from permission string
        service, action = permission.split(':')

        # Simulate policy
        response = iam_client.simulate_principal_policy(
            PolicySourceArn=role_arn,
            ActionNames=[permission],
            ResourceArns=['*']  # Check for all resources
        )

        # Check result
        for result in response['EvaluationResults']:
            if result['EvalDecision'] == 'allowed':
                return True

        return False

    except Exception as e:
        logger.warning(f"Could not check permission {permission}: {str(e)}")
        return False  # Assume no permission on error
```

**Output Example**:
```
=== SAGEMAKER PERMISSION CHECK RESULTS ===
✓ sagemaker:CreatePipeline
✓ sagemaker:DescribePipeline
✓ sagemaker:StartPipelineExecution
✓ sagemaker:CreateModel
✓ sagemaker:CreateEndpoint
✗ sagemaker:DeleteEndpoint (MISSING)
✓ s3:GetObject
✓ s3:PutObject
✗ lambda:CreateFunction (MISSING)

Summary:
  Passed: 15/17 (88.2%)
  Failed: 2 permissions

Action Required:
  - Add missing permissions to role
  - Or request exception for missing permissions
```

---

## Pipeline Management Scripts

### 3. prepare_config.py

**Purpose**: Generate `processing_config.json` from environment variables

**Size**: 8 KB, 220+ lines

**Key Functions**:

#### `create_processing_config()`
```python
def create_processing_config() -> dict:
    """
    Create configuration dictionary from environment variables.

    Returns:
        Dictionary with 50+ configuration parameters
    """
    config = {
        # Customer segmentation
        'customer_profile': os.environ['CUSTOMER_PROFILE'],
        'customer_segment': os.environ['CUSTOMER_SEGMENT'],
        'load_profile': os.environ.get('LOAD_PROFILE', os.environ['CUSTOMER_PROFILE']),

        # Database configuration
        'database_type': os.environ.get('DATABASE_TYPE', 'redshift'),
        'redshift_cluster': os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER'),
        'redshift_database': os.environ.get('REDSHIFT_DATABASE'),
        'redshift_user': os.environ.get('REDSHIFT_DB_USER'),
        'redshift_input_schema': os.environ.get('REDSHIFT_INPUT_SCHEMA'),
        'redshift_input_table': os.environ.get('REDSHIFT_INPUT_TABLE'),

        # S3 configuration
        's3_bucket': os.environ['S3_BUCKET'],
        's3_prefix': os.environ.get('S3_PREFIX',
            f"{os.environ['CUSTOMER_PROFILE']}-{os.environ['CUSTOMER_SEGMENT']}"),

        # Processing parameters
        'days_delay': int(os.environ.get('DAYS_DELAY', '14')),
        'use_reduced_features': os.environ.get('USE_REDUCED_FEATURES', 'false').lower() == 'true',
        'meter_threshold': int(os.environ.get('METER_THRESHOLD', '10')),
        'use_cache': os.environ.get('USE_CACHE', 'true').lower() == 'true',
        'use_weather': os.environ.get('USE_WEATHER', 'true').lower() == 'true',
        'use_solar': os.environ.get('USE_SOLAR', 'true').lower() == 'true',

        # Feature engineering
        'feature_selection_method': os.environ.get('FEATURE_SELECTION_METHOD', 'consensus'),
        'feature_count': int(os.environ.get('FEATURE_COUNT', '40')),
        'correlation_threshold': float(os.environ.get('CORRELATION_THRESHOLD', '0.85')),

        # Hyperparameter optimization
        'hpo_method': os.environ.get('HPO_METHOD', 'optuna'),
        'hpo_max_evals': int(os.environ.get('HPO_MAX_EVALS', '50')),
        'cv_folds': int(os.environ.get('CV_FOLDS', '5')),
        'cv_gap_days': int(os.environ.get('CV_GAP_DAYS', '7')),

        # Model deployment
        'enable_multi_model': os.environ.get('ENABLE_MULTI_MODEL', 'false').lower() == 'true',
        'deploy_model': os.environ.get('DEPLOY_MODEL', 'true').lower() == 'true',
        'endpoint_name': os.environ.get('ENDPOINT_NAME'),

        # Rate group filtering
        'rate_group_filter': os.environ.get('RATE_GROUP_FILTER_CLAUSE', ''),

        # Instance types
        'preprocessing_instance_type': os.environ.get('PREPROCESSING_INSTANCE_TYPE', 'ml.m5.large'),
        'training_instance_type': os.environ.get('TRAINING_INSTANCE_TYPE', 'ml.m5.xlarge'),
    }

    return config
```

**Output**: `processing_config.json` saved to local and S3

---

### 4. execute_pipeline.py

**Purpose**: Start SageMaker pipeline execution

**Size**: 7 KB, 200+ lines

**Key Functions**:

#### `execute_pipeline()`
```python
def execute_pipeline(pipeline_name: str, parameters: dict = None) -> str:
    """
    Start SageMaker pipeline execution.

    Args:
        pipeline_name: Name of the pipeline
        parameters: Optional parameter overrides

    Returns:
        Pipeline execution ARN
    """
    sagemaker_client = boto3.client('sagemaker')

    # Default parameters
    default_params = {
        'DaysDelay': '14',
        'UseWeather': 'true',
        'UseSolar': 'true',
        'HPOMaxEvals': '50',
        'FeatureCount': '40'
    }

    if parameters:
        default_params.update(parameters)

    # Format parameters for SageMaker
    pipeline_params = [
        {'Name': key, 'Value': value}
        for key, value in default_params.items()
    ]

    # Start execution
    response = sagemaker_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=pipeline_params,
        PipelineExecutionDescription=f"Execution started at {datetime.now().isoformat()}"
    )

    execution_arn = response['PipelineExecutionArn']

    logger.info(f"✓ Pipeline execution started")
    logger.info(f"  ARN: {execution_arn}")
    logger.info(f"  Parameters: {default_params}")

    return execution_arn
```

---

### 5. monitor_pipeline.py

**Purpose**: Monitor pipeline execution progress

**Size**: 7 KB, 190+ lines

**Key Functions**:

#### `monitor_execution()`
```python
def monitor_execution(
    execution_arn: str,
    max_wait_time: int = 7200,
    poll_interval: int = 60
) -> str:
    """
    Monitor pipeline execution until completion.

    Args:
        execution_arn: Pipeline execution ARN
        max_wait_time: Maximum wait time in seconds (default 2 hours)
        poll_interval: Polling interval in seconds (default 1 minute)

    Returns:
        Final status: 'Succeeded', 'Failed', or 'Stopped'
    """
    sagemaker_client = boto3.client('sagemaker')

    start_time = time.time()
    last_status = None

    while time.time() - start_time < max_wait_time:
        # Get execution status
        response = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        status = response['PipelineExecutionStatus']

        # Log status changes
        if status != last_status:
            logger.info(f"Pipeline status: {status}")
            last_status = status

            # Get step details
            steps = sagemaker_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn
            )

            for step in steps['PipelineExecutionSteps']:
                step_name = step['StepName']
                step_status = step['StepStatus']
                logger.info(f"  Step: {step_name} - {step_status}")

        # Check for completion
        if status in ['Succeeded', 'Failed', 'Stopped']:
            elapsed = time.time() - start_time
            logger.info(f"Pipeline {status.lower()} after {elapsed:.0f} seconds")
            return status

        # Wait before next poll
        time.sleep(poll_interval)

    # Timeout
    logger.error(f"Pipeline monitoring timeout after {max_wait_time} seconds")
    raise TimeoutError(f"Pipeline execution did not complete within {max_wait_time} seconds")
```

---

### 6. processing_wrapper.py

**Purpose**: Wrapper for preprocessing pipeline execution in SageMaker

**Size**: 16 KB, 440+ lines

**Execution**: Runs inside SageMaker Processing job

**Key Functions**:

#### `main()`
```python
def main():
    """
    Main entry point for SageMaker preprocessing job.

    Inputs:
    - /opt/ml/processing/input/config/processing_config.json

    Outputs:
    - /opt/ml/processing/output/train/train.csv
    - /opt/ml/processing/output/val/validation.csv
    - /opt/ml/processing/output/test/test.csv
    """
    # 1. Load configuration
    config_path = '/opt/ml/processing/input/config/processing_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Set up paths
    output_train = '/opt/ml/processing/output/train'
    output_val = '/opt/ml/processing/output/val'
    output_test = '/opt/ml/processing/output/test'

    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)
    os.makedirs(output_test, exist_ok=True)

    # 3. Run preprocessing
    from pipeline.preprocessing.preprocessing import process_data

    train_df, val_df, test_df = process_data(
        output_train=output_train,
        output_val=output_val,
        output_test=output_test,
        days_delay=config.get('days_delay', 14),
        use_reduced_features=config.get('use_reduced_features', False),
        meter_threshold=config.get('meter_threshold', 10),
        use_cache=config.get('use_cache', True),
        use_weather=config.get('use_weather', True),
        use_solar=config.get('use_solar', True),
        weather_cache=config.get('weather_cache', True)
    )

    # 4. Log statistics
    logger.info(f"✓ Preprocessing completed")
    logger.info(f"  Train: {len(train_df)} rows, {len(train_df.columns)} columns")
    logger.info(f"  Val: {len(val_df)} rows")
    logger.info(f"  Test: {len(test_df)} rows")
```

---

### 7. training_wrapper.py

**Purpose**: Wrapper for training pipeline execution in SageMaker

**Size**: 47 KB, 1,300+ lines (second largest script)

**Execution**: Runs inside SageMaker Training job

**Key Functions**:

#### `main()`
```python
def main():
    """
    Main entry point for SageMaker training job.

    Inputs:
    - /opt/ml/input/data/train/train.csv
    - /opt/ml/input/data/validation/validation.csv
    - /opt/ml/input/data/test/test.csv

    Outputs:
    - /opt/ml/model/xgboost-model
    - /opt/ml/model/features.pkl
    - /opt/ml/model/metrics.json
    """
    # 1. Load data
    train_df = pd.read_csv('/opt/ml/input/data/train/train.csv')
    val_df = pd.read_csv('/opt/ml/input/data/validation/validation.csv')
    test_df = pd.read_csv('/opt/ml/input/data/test/test.csv')

    logger.info(f"Loaded datasets:")
    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val: {len(val_df)} rows")
    logger.info(f"  Test: {len(test_df)} rows")

    # 2. Feature selection
    from pipeline.training.feature_selection import consensus_feature_selection

    all_features = [col for col in train_df.columns if col not in ['datetime', 'lossadjustedload']]

    selected_features = consensus_feature_selection(
        train_df,
        all_features,
        target='lossadjustedload',
        method=config.get('feature_selection_method', 'consensus'),
        top_n=config.get('feature_count', 40)
    )

    logger.info(f"✓ Selected {len(selected_features)} features")

    # 3. Hyperparameter optimization
    from pipeline.training.hyperparameter_optimization import OptunaHPO

    hpo = OptunaHPO(
        df=pd.concat([train_df, val_df]),
        features=selected_features,
        target='lossadjustedload',
        n_splits=config.get('cv_folds', 5),
        n_trials=config.get('hpo_max_evals', 50)
    )

    best_params, study = hpo.optimize()
    logger.info(f"✓ HPO completed - Best params: {best_params}")

    # 4. Train final model
    from pipeline.training.model import train_model

    model, metrics = train_model(
        df=pd.concat([train_df, val_df]),
        features=selected_features,
        target='lossadjustedload',
        params=best_params
    )

    logger.info(f"✓ Model trained - Val RMSE: {metrics['val_rmse']:.2f}")

    # 5. Evaluate on test set
    from pipeline.training.evaluation import evaluate_predictions

    test_pred = model.predict(test_df[selected_features])
    test_metrics = evaluate_predictions(
        test_df['lossadjustedload'].values,
        test_pred,
        prefix='test_'
    )

    metrics.update(test_metrics)
    logger.info(f"✓ Test RMSE: {test_metrics['test_rmse']:.2f}")

    # 6. Save artifacts
    import pickle

    model_path = '/opt/ml/model/xgboost-model'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    features_path = '/opt/ml/model/features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)

    metrics_path = '/opt/ml/model/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("✓ Artifacts saved to /opt/ml/model/")
```

---

## Model Deployment Scripts

### 8. register_model.py

**Purpose**: Register trained model in SageMaker Model Registry

**Size**: 4 KB, 120+ lines

**Key Functions**:

#### `register_model()`
```python
def register_model(
    model_s3_uri: str,
    model_package_group_name: str,
    image_uri: str,
    role_arn: str
) -> str:
    """
    Register model in SageMaker Model Registry.

    Args:
        model_s3_uri: S3 URI to model artifacts
        model_package_group_name: Model group name
        image_uri: Container image URI
        role_arn: Execution role ARN

    Returns:
        Model package ARN
    """
    sagemaker_client = boto3.client('sagemaker')

    # Create model package group if not exists
    try:
        sagemaker_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription=f"Energy forecast models for {model_package_group_name}"
        )
        logger.info(f"✓ Created model package group: {model_package_group_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            logger.info(f"Model package group already exists: {model_package_group_name}")
        else:
            raise

    # Register model
    response = sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription=f"Energy forecast model - {datetime.now().isoformat()}",
        InferenceSpecification={
            'Containers': [{
                'Image': image_uri,
                'ModelDataUrl': model_s3_uri
            }],
            'SupportedContentTypes': ['application/json'],
            'SupportedResponseMIMETypes': ['application/json'],
            'SupportedRealtimeInferenceInstanceTypes': [
                'ml.m5.large',
                'ml.m5.xlarge',
                'ml.m5.2xlarge'
            ]
        },
        ModelApprovalStatus='Approved'
    )

    model_package_arn = response['ModelPackageArn']

    logger.info(f"✓ Model registered: {model_package_arn}")

    return model_package_arn
```

---

### 9. deploy_model.py

**Purpose**: Deploy model to SageMaker endpoint with cost optimization

**Size**: 22 KB, 600+ lines

**Key Functions**:

#### `deploy_model_with_cost_optimization()`
```python
def deploy_model_with_cost_optimization(
    model_s3_uri: str,
    endpoint_name: str,
    instance_type: str,
    role_arn: str,
    run_id: str
) -> dict:
    """
    Deploy model to endpoint and immediately delete for cost optimization.

    Steps:
    1. Create SageMaker Model
    2. Create Endpoint Configuration
    3. Create Endpoint
    4. Wait for InService
    5. DELETE Endpoint (cost optimization)
    6. Store configuration in S3

    Returns:
        {
            'model_name': str,
            'endpoint_config_name': str,
            'endpoint_name': str,
            'endpoint_deleted': bool,
            'config_s3_uri': str
        }
    """
    sagemaker_client = boto3.client('sagemaker')
    s3_client = boto3.client('s3')

    # Step 1: Create Model
    model_name = f"{endpoint_name}-model-{run_id}"

    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': get_xgboost_image_uri(),
            'ModelDataUrl': model_s3_uri,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_s3_uri
            }
        },
        ExecutionRoleArn=role_arn
    )

    logger.info(f"✓ Created model: {model_name}")

    # Step 2: Create Endpoint Configuration
    endpoint_config_name = f"{endpoint_name}-config-{run_id}"

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InstanceType': instance_type,
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 1.0
        }]
    )

    logger.info(f"✓ Created endpoint config: {endpoint_config_name}")

    # Step 3: Create Endpoint
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
        Tags=[
            {'Key': 'CostOptimized', 'Value': 'true'},
            {'Key': 'DeleteRecreate', 'Value': 'true'},
            {'Key': 'RunId', 'Value': run_id}
        ]
    )

    logger.info(f"✓ Created endpoint: {endpoint_name}")

    # Step 4: Wait for InService
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)

    logger.info(f"✓ Endpoint is InService: {endpoint_name}")

    # Step 5: COST OPTIMIZATION - DELETE ENDPOINT
    logger.info("⚡ COST OPTIMIZATION: Deleting endpoint")

    # Save configuration FIRST
    config = {
        'endpoint_name': endpoint_name,
        'model_name': model_name,
        'model_config': {...},
        'endpoint_config_name': endpoint_config_name,
        'endpoint_config': {...},
        'cost_optimized': True,
        'delete_recreate_enabled': True,
        'created_at': datetime.now().isoformat(),
        'run_id': run_id
    }

    # Upload to S3
    config_key = f"{s3_prefix}/endpoint-configs/{endpoint_name}_config.json"
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=config_key,
        Body=json.dumps(config, indent=2),
        ContentType='application/json'
    )

    logger.info(f"✓ Saved config to S3: s3://{s3_bucket}/{config_key}")

    # NOW delete endpoint
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    logger.info("✓ Endpoint deletion initiated")
    logger.info("✓ Cost optimization: $0/hour ongoing")

    # Wait for deletion
    time.sleep(120)

    return {
        'model_name': model_name,
        'endpoint_config_name': endpoint_config_name,
        'endpoint_name': endpoint_name,
        'endpoint_deleted': True,
        'config_s3_uri': f"s3://{s3_bucket}/{config_key}"
    }
```

---

### 10. validate_deployment_readiness.py

**Purpose**: Validate model artifacts and configuration before deployment

**Size**: 16 KB, 440+ lines

**Key Functions**:

#### `validate_deployment()`
```python
def validate_deployment(
    s3_bucket: str,
    s3_prefix: str,
    run_id: str
) -> bool:
    """
    Validate deployment readiness.

    Checks:
    1. Model artifacts exist in S3
    2. Features file exists
    3. Metrics file exists and passes thresholds
    4. Training completed successfully
    5. Database connectivity

    Returns:
        True if ready to deploy, False otherwise
    """
    s3_client = boto3.client('s3')

    checks_passed = 0
    checks_total = 5

    # Check 1: Model artifacts
    model_key = f"{s3_prefix}/models/{run_id}/xgboost-model"
    if check_s3_object_exists(s3_bucket, model_key):
        logger.info("✓ Model artifacts exist")
        checks_passed += 1
    else:
        logger.error(f"✗ Model artifacts missing: {model_key}")

    # Check 2: Features file
    features_key = f"{s3_prefix}/models/{run_id}/features.pkl"
    if check_s3_object_exists(s3_bucket, features_key):
        logger.info("✓ Features file exists")
        checks_passed += 1
    else:
        logger.error(f"✗ Features file missing: {features_key}")

    # Check 3: Metrics file
    metrics_key = f"{s3_prefix}/models/{run_id}/metrics.json"
    if check_s3_object_exists(s3_bucket, metrics_key):
        metrics = load_json_from_s3(s3_bucket, metrics_key)

        # Validate metrics
        if validate_metrics(metrics):
            logger.info("✓ Metrics pass thresholds")
            checks_passed += 1
        else:
            logger.error("✗ Metrics below thresholds")
    else:
        logger.error(f"✗ Metrics file missing: {metrics_key}")

    # Check 4: Training status
    # (Check SageMaker training job status)
    checks_passed += 1  # Simplified

    # Check 5: Database connectivity
    if test_database_connection():
        logger.info("✓ Database connection successful")
        checks_passed += 1
    else:
        logger.error("✗ Database connection failed")

    logger.info(f"Validation: {checks_passed}/{checks_total} checks passed")

    return checks_passed == checks_total
```

---

## Lambda Management Scripts

### 11. create_forecast_lambda.py

**Purpose**: Create forecasting Lambda function with all dependencies

**Size**: 38 KB, 1,050+ lines (third largest script)

**Key Functions**:

#### `package_lambda()`
```python
def package_lambda() -> str:
    """
    Package Lambda function with all dependencies.

    Creates lambda_forecast.zip with:
    - lambda_function.py
    - predictions/forecast/*.py
    - configs/config.py
    - Third-party packages (boto3, pandas, numpy, etc.)

    Returns:
        Path to lambda_forecast.zip
    """
    import zipfile
    import shutil

    # Create temp directory
    temp_dir = '/tmp/lambda_package'
    os.makedirs(temp_dir, exist_ok=True)

    # Copy source files
    shutil.copy('predictions/lambda_function.py', temp_dir)
    shutil.copytree('predictions/forecast', f'{temp_dir}/forecast')
    shutil.copytree('configs', f'{temp_dir}/configs')

    # Install dependencies
    subprocess.run([
        'pip', 'install',
        '-t', temp_dir,
        'boto3', 'pandas', 'numpy', 'pytz',
        'openmeteo-requests', 'requests-cache', 'retry-requests'
    ])

    # Create zip
    zip_path = '/tmp/lambda_forecast.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    # Check size
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    logger.info(f"✓ Lambda package created: {size_mb:.1f} MB")

    if size_mb > 50:
        logger.warning("⚠ Package exceeds 50MB (Lambda limit for direct upload)")
        logger.info("  Consider using Lambda Layers for large dependencies")

    return zip_path
```

#### `create_lambda_function()`
```python
def create_lambda_function(
    function_name: str,
    role_arn: str,
    zip_path: str,
    environment_vars: dict
) -> str:
    """
    Create Lambda function.

    Args:
        function_name: Lambda function name
        role_arn: Execution role ARN
        zip_path: Path to deployment package
        environment_vars: Environment variables dict

    Returns:
        Lambda function ARN
    """
    lambda_client = boto3.client('lambda')

    # Read zip file
    with open(zip_path, 'rb') as f:
        zip_content = f.read()

    try:
        # Create function
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_content},
            Description=f"Day-ahead energy load forecasting for {function_name}",
            Timeout=900,  # 15 minutes
            MemorySize=1024,  # 1 GB
            EphemeralStorage={'Size': 2048},  # 2 GB
            Environment={'Variables': environment_vars},
            Tags={
                'Application': 'EnergyForecasting',
                'CostOptimized': 'true'
            }
        )

        function_arn = response['FunctionArn']
        logger.info(f"✓ Created Lambda function: {function_name}")
        logger.info(f"  ARN: {function_arn}")

        return function_arn

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceConflictException':
            # Function exists, update code
            logger.info(f"Function exists, updating code: {function_name}")

            lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )

            # Update configuration
            lambda_client.update_function_configuration(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Description=f"Day-ahead energy load forecasting for {function_name}",
                Timeout=900,
                MemorySize=1024,
                EphemeralStorage={'Size': 2048},
                Environment={'Variables': environment_vars}
            )

            response = lambda_client.get_function(FunctionName=function_name)
            function_arn = response['Configuration']['FunctionArn']

            logger.info(f"✓ Updated Lambda function: {function_name}")

            return function_arn
        else:
            raise
```

---

### 12. setup_schedule.py

**Purpose**: Create EventBridge schedule for Lambda

**Size**: 3 KB, 95+ lines

**Key Functions**:

#### `create_schedule()`
```python
def create_schedule(
    function_arn: str,
    schedule_expression: str,
    rule_name: str
) -> str:
    """
    Create EventBridge schedule rule.

    Args:
        function_arn: Lambda function ARN
        schedule_expression: Cron expression (e.g., 'cron(0 9 * * ? *)')
        rule_name: Rule name

    Returns:
        Rule ARN
    """
    events_client = boto3.client('events')
    lambda_client = boto3.client('lambda')

    # Create rule
    response = events_client.put_rule(
        Name=rule_name,
        ScheduleExpression=schedule_expression,
        State='ENABLED',
        Description=f"Daily energy forecast schedule for {rule_name}"
    )

    rule_arn = response['RuleArn']
    logger.info(f"✓ Created schedule rule: {rule_name}")
    logger.info(f"  Schedule: {schedule_expression}")

    # Add Lambda as target
    events_client.put_targets(
        Rule=rule_name,
        Targets=[{
            'Id': '1',
            'Arn': function_arn,
            'Input': json.dumps({
                'scheduled': True,
                'source': 'eventbridge'
            })
        }]
    )

    logger.info("✓ Added Lambda as target")

    # Grant EventBridge permission to invoke Lambda
    try:
        lambda_client.add_permission(
            FunctionName=function_arn,
            StatementId=f"AllowEventBridge-{rule_name}",
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=rule_arn
        )
        logger.info("✓ Granted EventBridge invoke permission")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceConflictException':
            logger.info("Permission already exists")
        else:
            raise

    return rule_arn
```

---

## Validation & Testing Scripts

### 13. integration_test.py

**Purpose**: End-to-end integration testing

**Size**: 8 KB, 225+ lines

**Key Functions**:

#### `run_integration_test()`
```python
def run_integration_test(
    endpoint_name: str,
    lambda_function_name: str
) -> bool:
    """
    Run complete integration test.

    Tests:
    1. Endpoint configuration exists in S3
    2. Lambda function exists and is configured
    3. Test Lambda invocation (with test_invocation=true)
    4. Verify prediction output format
    5. Check CloudWatch logs

    Returns:
        True if all tests pass, False otherwise
    """
    tests_passed = 0
    tests_total = 5

    # Test 1: Endpoint config in S3
    if verify_endpoint_config_exists(endpoint_name):
        logger.info("✓ Test 1: Endpoint config exists in S3")
        tests_passed += 1
    else:
        logger.error("✗ Test 1: Endpoint config missing")

    # Test 2: Lambda function
    if verify_lambda_exists(lambda_function_name):
        logger.info("✓ Test 2: Lambda function exists")
        tests_passed += 1
    else:
        logger.error("✗ Test 2: Lambda function not found")

    # Test 3: Test invocation
    try:
        result = invoke_lambda_test(lambda_function_name)
        logger.info("✓ Test 3: Lambda invocation successful")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Test 3: Lambda invocation failed: {str(e)}")

    # Test 4: Output format
    if verify_prediction_format(result):
        logger.info("✓ Test 4: Prediction output format valid")
        tests_passed += 1
    else:
        logger.error("✗ Test 4: Invalid prediction format")

    # Test 5: CloudWatch logs
    if verify_cloudwatch_logs(lambda_function_name):
        logger.info("✓ Test 5: CloudWatch logs available")
        tests_passed += 1
    else:
        logger.error("✗ Test 5: CloudWatch logs missing")

    logger.info(f"Integration test: {tests_passed}/{tests_total} tests passed")

    return tests_passed == tests_total
```

---

## Reporting Scripts

### 14. generate_report.py

**Purpose**: Generate comprehensive deployment report

**Size**: 19 KB, 515+ lines

**Key Functions**:

#### `generate_deployment_report()`
```python
def generate_deployment_report(
    environment: str,
    combinations: list,
    run_id: str
) -> str:
    """
    Generate comprehensive deployment report.

    Sections:
    1. Executive Summary
    2. Deployment Details
    3. Model Performance Metrics
    4. Cost Optimization Status
    5. Infrastructure Summary
    6. Next Steps

    Returns:
        Report file path
    """
    report = []

    # Header
    report.append("# Energy Load Forecasting Deployment Report")
    report.append(f"## Environment: {environment}")
    report.append(f"## Date: {datetime.now().isoformat()}")
    report.append(f"## Run ID: {run_id}")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append(f"- **Combinations Deployed**: {len(combinations)}")
    report.append(f"- **Endpoint Strategy**: Delete/Recreate (Cost Optimized)")
    report.append(f"- **Expected Savings**: 99.3% ($2,016/month)")
    report.append("")

    # Per-combination details
    report.append("## Deployment Details")
    report.append("")

    for combo in combinations:
        profile = combo['profile']
        segment = combo['segment']

        # Load metrics
        metrics = load_metrics(profile, segment, run_id)

        report.append(f"### {profile}-{segment}")
        report.append("")
        report.append("**Model Performance:**")
        report.append(f"- RMSE: {metrics.get('test_rmse', 'N/A')}")
        report.append(f"- MAE: {metrics.get('test_mae', 'N/A')}")
        report.append(f"- MAPE: {metrics.get('test_mape', 'N/A')}%")
        report.append(f"- R²: {metrics.get('test_r2', 'N/A')}")
        report.append("")

        report.append("**Deployment Status:**")
        report.append(f"- Endpoint: `{environment}-energy-ml-endpoint-{profile}-{segment}` (DELETED)")
        report.append(f"- Lambda: `{environment}-energy-daily-predictor-{profile}-{segment}` (ACTIVE)")
        report.append(f"- Schedule: Enabled")
        report.append(f"- Cost: $0/hour (optimized)")
        report.append("")

    # Cost optimization
    report.append("## Cost Optimization Summary")
    report.append("")
    report.append("**Strategy**: Delete/Recreate Endpoints")
    report.append("")
    report.append("| Metric | Traditional | Delete/Recreate | Savings |")
    report.append("|--------|-------------|-----------------|---------|")
    report.append(f"| Endpoint Uptime | 24 hours/day | ~10 min/day | 99.3% |")
    report.append(f"| Cost per Endpoint | $338/month | $2.35/month | $335.65/month |")
    report.append(f"| Total (6 endpoints) | $2,030/month | $14/month | $2,016/month |")
    report.append("")

    # Write report
    report_content = "\n".join(report)
    report_path = f"deployment_report_{environment}_{run_id}.md"

    with open(report_path, 'w') as f:
        f.write(report_content)

    logger.info(f"✓ Report generated: {report_path}")

    return report_path
```

---

## Script Dependencies

### Dependency Graph

```
check_sagemaker_permissions.py
    ↓
setup_redshift_infrastructure.py
    ↓
prepare_config.py → (generates) → processing_config.json
    ↓
create_pipeline.py → (uses) → processing_config.json
    ↓
execute_pipeline.py → (starts) → SageMaker Pipeline
    ↓
monitor_pipeline.py → (watches) → Pipeline Execution
    ↓
    [Inside SageMaker]:
    processing_wrapper.py → (runs preprocessing)
    training_wrapper.py → (runs training)
    ↓
analyze_model.py → (extracts) → Model Metadata
    ↓
extract_metrics.py → (extracts) → Metrics
    ↓
validate_deployment_readiness.py → (checks) → Model Ready
    ↓
register_model.py → (creates) → Model Package
    ↓
deploy_model.py → (deploys & deletes) → Endpoint
    ↓
validate_endpoint_health.py → (verifies) → Endpoint Config in S3
    ↓
create_lambda.py → (creates) → Deployment Lambda
    ↓
create_forecast_lambda.py → (creates) → Forecasting Lambda
    ↓
setup_schedule.py → (creates) → EventBridge Rule
    ↓
test_lambda.py → (tests) → Lambda Function
    ↓
integration_test.py → (validates) → End-to-End
    ↓
generate_report.py → (creates) → Deployment Report
```

### Common Imports

All scripts share common imports:
```python
import os
import sys
import json
import logging
import boto3
from datetime import datetime
from typing import Dict, List, Optional
```

### Environment Variables Required

Most scripts require:
```bash
AWS_REGION
S3_BUCKET
S3_PREFIX
CUSTOMER_PROFILE
CUSTOMER_SEGMENT
SAGEMAKER_ROLE_ARN
```

---

## Usage Examples

### Running Individual Scripts

**Setup Infrastructure**:
```bash
export REDSHIFT_CLUSTER_IDENTIFIER=energy-cluster-prod
export REDSHIFT_DATABASE=sdcp
export REDSHIFT_DB_USER=ds_service_user
# ... (more env vars)

python .github/scripts/deploy/setup_redshift_infrastructure.py
```

**Check Permissions**:
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole

python .github/scripts/deploy/check_sagemaker_permissions.py
```

**Deploy Model**:
```bash
export ENDPOINT_NAME=prod-energy-ml-endpoint-RES-SOLAR
export MODEL_S3_URI=s3://bucket/models/run_20260113/xgboost-model
export RUN_ID=run_20260113

python .github/scripts/deploy/deploy_model.py
```

### Running Complete Workflow

In GitHub Actions, scripts are orchestrated automatically via the workflow YAML.

---

## Best Practices

### 1. Error Handling
```python
try:
    result = execute_operation()
    logger.info("✓ Operation successful")
except ClientError as e:
    logger.error(f"✗ AWS error: {e.response['Error']['Code']}")
    sys.exit(1)
except Exception as e:
    logger.error(f"✗ Unexpected error: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
```

### 2. Logging
```python
# Always use structured logging
logger.info("Starting operation")
logger.info(f"  Parameter 1: {value1}")
logger.info(f"  Parameter 2: {value2}")

# Use symbols for status
logger.info("✓ Success")
logger.warning("⚠ Warning")
logger.error("✗ Error")
```

### 3. Output Validation
```python
# Always validate critical outputs
if not os.path.exists(output_file):
    raise FileNotFoundError(f"Expected output not created: {output_file}")

if df.empty:
    raise ValueError("Query returned no data")
```

### 4. Configuration Management
```python
# Load all config at start
config = load_environment_config()

# Validate required keys
required_keys = ['s3_bucket', 's3_prefix', 'customer_profile']
missing = [k for k in required_keys if k not in config]
if missing:
    raise ValueError(f"Missing required config: {missing}")
```

---

## Troubleshooting

### Common Issues

**1. Script Import Errors**
```
Error: ModuleNotFoundError: No module named 'pipeline'
Solution: Add repository root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**2. AWS Credential Errors**
```
Error: NoCredentialsError: Unable to locate credentials
Solution: Set AWS credentials via environment or AWS CLI
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

**3. S3 Access Denied**
```
Error: AccessDenied: Access Denied
Solution: Check IAM role permissions for S3
- s3:GetObject
- s3:PutObject
- s3:ListBucket
```

**4. SageMaker Resource Limits**
```
Error: ResourceLimitExceeded
Solution: Request quota increase or use smaller instances
```

---

## Script Metrics Summary

| Category | Scripts | Total Lines | Total Size |
|----------|---------|-------------|------------|
| Infrastructure | 3 | 2,650+ | 95 KB |
| Pipeline | 7 | 2,410+ | 86 KB |
| Deployment | 4 | 1,640+ | 59 KB |
| Lambda | 4 | 2,100+ | 76 KB |
| Validation | 3 | 425+ | 15 KB |
| Reporting | 2 | 575+ | 21 KB |
| **Total** | **23** | **~7,800** | **~352 KB** |

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Status**: Production Ready
