# GitHub Actions Workflows - Complete Guide

## Table of Contents
- [Overview](#overview)
- [Deploy Workflow (deploy.yml)](#deploy-workflow-deployyml)
- [Historical Forecasting Workflow](#historical-forecasting-workflow)
- [Workflow Comparison](#workflow-comparison)
- [Environment Configuration](#environment-configuration)
- [Troubleshooting Workflows](#troubleshooting-workflows)

---

## Overview

This project uses two main GitHub Actions workflows:

1. **deploy.yml** (4,554 lines): Main deployment pipeline for training and deploying ML models
2. **historical_forecasting.yml** (1,109 lines): Generates historical predictions for backfilling data

Both workflows support multiple environments (dev, qa, preprod, prod) and implement sophisticated cost optimization strategies.

---

## Deploy Workflow (deploy.yml)

### Purpose
Automated end-to-end MLOps pipeline that:
- Sets up cloud infrastructure (Redshift/Athena)
- Executes SageMaker preprocessing and training pipelines
- Deploys models with cost-optimized endpoints
- Creates scheduled Lambda functions for daily forecasting

### Trigger Conditions

```yaml
on:
  push:
    branches: [main, develop]
    paths:
      - 'pipeline/**'
      - 'predictions/**'
      - 'configs/**'
      - '.github/workflows/deploy.yml'
      - '.github/scripts/deploy/**'

  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        type: choice
        options: [dev, qa, preprod, prod]

      deploy_combinations:
        description: 'Which combinations to deploy'
        type: choice
        options:
          - all
          - res_only
          - medci_only
          - smlcom_only
          - solar_only
          - nonsolar_only
          - single_combination

      single_customer_profile:
        description: 'Profile (if single_combination)'
        type: choice
        options: [RES, MEDCI, SMLCOM]

      single_customer_segment:
        description: 'Segment (if single_combination)'
        type: choice
        options: [SOLAR, NONSOLAR]

      skip_tests:
        description: 'Skip unit tests'
        type: boolean
        default: false
```

### Complete Job Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. determine_environment                                         │
│    • Detect trigger type (push vs manual)                       │
│    • Set environment (dev/qa/preprod/prod)                      │
│    • Generate combinations matrix (up to 6 segments)            │
│    • Configure schedules and timeouts                           │
│    Output: environment, combinations_matrix, config parameters  │
└────────────────┬────────────────────────────────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
     ▼                       ▼
┌─────────────┐    ┌──────────────────────┐
│ 2. test     │    │ 3. check_sagemaker_  │
│    (skip if │    │    permissions       │
│    push)    │    │  • Validate IAM      │
│             │    │  • Test S3 access    │
└──────┬──────┘    └──────────┬───────────┘
       │                      │
       └──────────┬───────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. setup_redshift_infrastructure (conditional on database_type)  │
│    • Create schemas (input, operational, BI)                    │
│    • Create tables (historical, predictions)                    │
│    • Set up BI views for dashboards                             │
│    • Test connectivity and permissions                          │
│    Output: cluster, database, schemas, tables                   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. consolidate_infrastructure                                    │
│    • Consolidate Redshift/Athena outputs                        │
│    • Create unified configuration                               │
│    • Validate setup success                                     │
│    Output: Unified database configuration                       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. approve_pipeline (environment-specific approval gate)         │
│    • prod/preprod: Requires manual GitHub approval             │
│    • dev/qa: Automatic approval                                 │
│    • Logs approval details                                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. deploy_combination [MATRIX JOB - 6 PARALLEL]                 │
│    Matrix: RES-SOLAR, RES-NONSOLAR, MEDCI-SOLAR, etc.          │
│    Max Parallel: 6 (all combinations simultaneously)            │
│                                                                  │
│    For each combination:                                         │
│    ├─ a. Prepare configuration (processing_config.json)         │
│    ├─ b. Upload scripts to S3                                   │
│    ├─ c. Create SageMaker Pipeline definition                   │
│    ├─ d. Execute preprocessing step                             │
│    │     • Query 3 years of data from Redshift                  │
│    │     • Feature engineering (weather, solar, lags)           │
│    │     • Train/validation/test splits                         │
│    │     • Output: CSV files to S3                              │
│    ├─ e. Execute training step                                  │
│    │     • Feature selection (consensus method)                 │
│    │     • Hyperparameter optimization (Optuna, 50 trials)      │
│    │     • XGBoost model training                               │
│    │     • Segment-specific evaluation                          │
│    │     • Output: Model artifacts, metrics, plots              │
│    ├─ f. Monitor pipeline execution (poll every 60s)            │
│    ├─ g. Extract metrics and generate reports                   │
│    └─ h. Upload artifacts to S3                                 │
│                                                                  │
│    Output: Pipeline execution ARN, model S3 URI, metrics        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. approve_deployment [MATRIX JOB - 6 PARALLEL]                 │
│    • Validate model artifacts exist in S3                       │
│    • Check training completion status                           │
│    • Verify metrics meet thresholds                             │
│    • Confirm database connectivity                              │
│    Output: Deployment approval per combination                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. deploy_model [MATRIX JOB - 6 PARALLEL]                       │
│    Max Parallel: 6                                              │
│                                                                  │
│    For each combination:                                         │
│    ├─ a. Find latest model artifacts (run_id)                   │
│    ├─ b. Register model in SageMaker Model Registry             │
│    ├─ c. Create deployment Lambda (model-deployer)              │
│    ├─ d. Create SageMaker Model                                 │
│    ├─ e. Create EndpointConfig                                  │
│    ├─ f. Deploy to SageMaker Endpoint                           │
│    ├─ g. Wait for InService status                              │
│    ├─ h. ⚡ COST OPTIMIZATION: Delete endpoint immediately      │
│    ├─ i. Store endpoint configuration in S3                     │
│    │     Location: s3://{bucket}/{profile}-{segment}/           │
│    │               endpoint-configs/{endpoint_name}_config.json │
│    ├─ j. Generate endpoint info documentation                   │
│    └─ k. Upload deployment artifacts                            │
│                                                                  │
│    Output: Endpoint ARN (deleted), config in S3, run_id        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. approve_lambda [MATRIX JOB - 6 PARALLEL]                    │
│     • Validate endpoint configuration stored in S3              │
│     • Verify endpoint deletion status (NotFound or Deleting)    │
│     • Confirm cost optimization achieved ($0/hour)              │
│     • Check configuration content validity                      │
│     Output: Approval for forecasting Lambda creation            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 11. create_forecasting_lambda [MATRIX JOB - 6 PARALLEL]         │
│     Max Parallel: 6                                             │
│                                                                  │
│     For each combination:                                        │
│     ├─ a. Package Lambda code                                   │
│     │     • predictions/lambda_function.py                      │
│     │     • predictions/forecast/*.py modules                   │
│     │     • Dependencies (boto3, pandas, etc.)                  │
│     │     • Output: lambda_forecast.zip                         │
│     ├─ b. Create Lambda function                                │
│     │     Name: {env}-energy-daily-predictor-{profile}-{segment}│
│     │     Runtime: python3.9                                    │
│     │     Memory: 1024 MB                                       │
│     │     Timeout: 900 seconds (15 minutes)                     │
│     ├─ c. Configure environment variables (50+ vars)            │
│     │     • Database: Redshift cluster, schemas, tables         │
│     │     • Endpoint: Name, recreation settings                 │
│     │     • Features: Lag days, evaluation periods              │
│     │     • Weather: API config, location, variables            │
│     │     • Cost optimization: Delete/recreate flags            │
│     ├─ d. Create EventBridge schedule rule                      │
│     │     • dev: cron(0 10 * * ? *) - 10 AM UTC                │
│     │     • qa: cron(0 11 * * ? *)  - 11 AM UTC                │
│     │     • preprod: cron(0 8 * * ? *) - 8 AM UTC              │
│     │     • prod: cron(0 9 * * ? *) - 9 AM UTC (2 AM PST)      │
│     ├─ e. Attach rule to Lambda function                        │
│     ├─ f. Test Lambda invocation (optional)                     │
│     └─ g. Generate forecasting documentation                    │
│                                                                  │
│     Output: Lambda ARN, schedule rule, test results             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 12. deployment_summary (always runs)                            │
│     • Aggregate results from all matrix jobs                    │
│     • Generate cost optimization report                         │
│     • List endpoint statuses (all should be NotFound)           │
│     • Document Lambda functions and schedules                   │
│     • Create operational guidance                               │
│     • Upload comprehensive summary to S3                        │
│     Output: enhanced_deployment_summary_{env}_{run_id}.md       │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Job Descriptions

#### 1. determine_environment

**Purpose**: Central configuration hub that determines execution parameters

**Key Logic**:
```bash
# Push trigger defaults
if [ "$GITHUB_EVENT_NAME" == "push" ]; then
  ENVIRONMENT="dev"
  DEPLOY_COMBINATIONS="single_combination"
  SINGLE_PROFILE="RES"
  SINGLE_SEGMENT="NONSOLAR"

# Manual trigger uses inputs
else
  ENVIRONMENT="${{ github.event.inputs.environment }}"
  DEPLOY_COMBINATIONS="${{ github.event.inputs.deploy_combinations }}"
fi

# Generate combinations matrix based on selection
case "$DEPLOY_COMBINATIONS" in
  "all")
    # All 6 combinations
    COMBINATIONS='[
      {"profile":"RES","segment":"SOLAR"},
      {"profile":"RES","segment":"NONSOLAR"},
      {"profile":"MEDCI","segment":"SOLAR"},
      {"profile":"MEDCI","segment":"NONSOLAR"},
      {"profile":"SMLCOM","segment":"SOLAR"},
      {"profile":"SMLCOM","segment":"NONSOLAR"}
    ]'
    ;;
  "res_only")
    # RES-SOLAR + RES-NONSOLAR
    COMBINATIONS='[
      {"profile":"RES","segment":"SOLAR"},
      {"profile":"RES","segment":"NONSOLAR"}
    ]'
    ;;
  "single_combination")
    # Single specified combination
    COMBINATIONS='[
      {"profile":"'"$SINGLE_PROFILE"'","segment":"'"$SINGLE_SEGMENT"'"}
    ]'
    ;;
esac

# Environment-specific schedules
case "$ENVIRONMENT" in
  "dev")    LAMBDA_SCHEDULE="cron(0 10 * * ? *)" ;;
  "qa")     LAMBDA_SCHEDULE="cron(0 11 * * ? *)" ;;
  "preprod") LAMBDA_SCHEDULE="cron(0 8 * * ? *)" ;;
  "prod")   LAMBDA_SCHEDULE="cron(0 9 * * ? *)" ;;
esac
```

**Outputs**:
- `environment`: dev/qa/preprod/prod
- `database_type`: redshift (or athena)
- `pipeline_type`: complete
- `deploy_model`: true
- `create_lambda`: true
- `combinations_matrix`: JSON array
- `lambda_schedule`: Cron expression
- `max_wait_time`: 7200 (2 hours)
- `poll_interval`: 60 (seconds)

#### 2. test

**Purpose**: Run Python unit tests with coverage

**Condition**: Runs only if `skip_tests != 'true'` AND `event_name != 'push'`

**Steps**:
```bash
# Install dependencies
pip install pytest pytest-cov coverage

# Run tests with coverage
pytest tests/ --cov=pipeline --cov=predictions --cov-report=xml --cov-report=term

# Check coverage threshold
coverage report --fail-under=80
```

**Artifacts**: `coverage.xml`, test logs

#### 3. check_sagemaker_permissions

**Purpose**: Validate IAM permissions before expensive operations

**Script**: `.github/scripts/deploy/check_sagemaker_permissions.py`

**Permissions Checked**:
```python
REQUIRED_PERMISSIONS = [
    # SageMaker Pipeline
    'sagemaker:CreatePipeline',
    'sagemaker:DescribePipeline',
    'sagemaker:StartPipelineExecution',
    'sagemaker:DescribePipelineExecution',

    # SageMaker Processing/Training
    'sagemaker:CreateProcessingJob',
    'sagemaker:CreateTrainingJob',
    'sagemaker:DescribeProcessingJob',
    'sagemaker:DescribeTrainingJob',

    # Model Registry
    'sagemaker:CreateModel',
    'sagemaker:CreateModelPackage',
    'sagemaker:DescribeModelPackage',

    # Endpoints
    'sagemaker:CreateEndpointConfig',
    'sagemaker:CreateEndpoint',
    'sagemaker:DescribeEndpoint',
    'sagemaker:DeleteEndpoint',

    # S3
    's3:GetObject',
    's3:PutObject',
    's3:ListBucket',

    # Lambda
    'lambda:CreateFunction',
    'lambda:UpdateFunctionCode',
    'lambda:AddPermission',

    # IAM
    'iam:PassRole'
]
```

**Output**: `permission_check_passed` (true/false), failed permissions list

#### 4. setup_redshift_infrastructure

**Purpose**: Create database schemas and tables

**Script**: `.github/scripts/deploy/setup_redshift_infrastructure.py`

**Environment-Aware Naming**:
```python
# Production: No suffix
# Non-production: Add environment suffix

# Example for 'dev' environment:
REDSHIFT_INPUT_SCHEMA = "edp_cust_dev"
REDSHIFT_OUTPUT_SCHEMA = "edp_forecasting_dev"
REDSHIFT_BI_SCHEMA = "edp_bi_dev"

# Example for 'prod' environment:
REDSHIFT_INPUT_SCHEMA = "edp_cust"
REDSHIFT_OUTPUT_SCHEMA = "edp_forecasting"
REDSHIFT_BI_SCHEMA = "edp_bi"
```

**Tables Created**:

1. **Input Table** (already exists, validated):
   ```sql
   {REDSHIFT_INPUT_SCHEMA}.caiso_sqmd
   -- Contains: historical CAISO SQMD submissions
   -- Columns: tradedatelocal, tradehourstartlocal, loadprofile,
   --          rategroup, loadlal, genlal, metercount, submission, etc.
   ```

2. **Operational Table** (created if not exists):
   ```sql
   CREATE TABLE IF NOT EXISTS {REDSHIFT_OUTPUT_SCHEMA}.dayahead_load_forecasts (
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
   SORTKEY (forecast_datetime)
   DISTKEY (load_profile);
   ```

3. **BI View** (for dashboard):
   ```sql
   CREATE OR REPLACE VIEW {REDSHIFT_BI_SCHEMA}.vw_dayahead_load_forecasts AS
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
       day
   FROM {REDSHIFT_OUTPUT_SCHEMA}.dayahead_load_forecasts
   WHERE created_at >= DATEADD(day, -90, GETDATE())  -- Last 90 days
   ORDER BY forecast_datetime DESC;
   ```

**Validation Steps**:
- Test Redshift cluster connectivity
- Verify S3 IAM role for COPY/UNLOAD
- Test schema creation permissions
- Validate table structure

#### 7. deploy_combination (Matrix Job)

**Purpose**: Deploy complete ML pipeline for each customer combination

**Matrix Configuration**:
```yaml
strategy:
  matrix:
    combination: ${{ fromJson(needs.determine_environment.outputs.combinations_matrix) }}
  fail-fast: false
  max-parallel: 6
```

**Per-Combination Environment Variables**:
```bash
# Matrix-specific
CUSTOMER_PROFILE=${{ matrix.combination.profile }}     # RES, MEDCI, SMLCOM
CUSTOMER_SEGMENT=${{ matrix.combination.segment }}     # SOLAR, NONSOLAR
S3_PREFIX=${CUSTOMER_PROFILE}-${CUSTOMER_SEGMENT}      # RES-SOLAR

# Pipeline naming
PIPELINE_NAME="${ENV_NAME}-energy-forecasting-${CUSTOMER_PROFILE}-${CUSTOMER_SEGMENT}-complete-${RUN_ID}"

# SageMaker instances (environment-dependent)
PREPROCESSING_INSTANCE_TYPE=ml.m5.large   # ml.t3.medium for dev
TRAINING_INSTANCE_TYPE=ml.m5.xlarge       # ml.m5.large for dev

# HPO settings
HPO_MAX_EVALS=50                          # 10 for dev
CV_FOLDS=5
FEATURE_COUNT=40

# Data settings
DAYS_DELAY=14
METER_THRESHOLD=10
USE_WEATHER=true
USE_SOLAR=true
USE_CACHE=true
```

**Execution Steps**:

**a. Prepare Configuration**:
```python
# Script: prepare_config.py
config = {
    "customer_profile": CUSTOMER_PROFILE,
    "customer_segment": CUSTOMER_SEGMENT,
    "database_type": "redshift",
    "redshift_cluster": REDSHIFT_CLUSTER,
    "redshift_database": REDSHIFT_DATABASE,
    "redshift_input_schema": REDSHIFT_INPUT_SCHEMA,
    "redshift_input_table": REDSHIFT_INPUT_TABLE,
    "s3_bucket": S3_BUCKET,
    "s3_prefix": S3_PREFIX,
    "days_delay": DAYS_DELAY,
    "use_weather": True,
    "use_solar": True,
    "hpo_max_evals": HPO_MAX_EVALS,
    # ... 50+ more parameters
}

# Save to processing_config.json
# Upload to S3: s3://{bucket}/{prefix}/scripts/processing_config.json
```

**b. Create SageMaker Pipeline**:
```python
# Script: create_pipeline.py
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator

# Define preprocessing step
preprocessing_step = ProcessingStep(
    name="Preprocessing",
    processor=SKLearnProcessor(
        framework_version='1.0-1',
        instance_type=PREPROCESSING_INSTANCE_TYPE,
        instance_count=1,
        role=SAGEMAKER_ROLE_ARN
    ),
    code='s3://{bucket}/{prefix}/scripts/processing_wrapper.py',
    inputs=[...],
    outputs=[...]
)

# Define training step
training_step = TrainingStep(
    name="Training",
    estimator=Estimator(
        image_uri='<xgboost-image>',
        instance_type=TRAINING_INSTANCE_TYPE,
        instance_count=1,
        role=SAGEMAKER_ROLE_ARN
    ),
    inputs={...}
)

# Create pipeline
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[...],
    steps=[preprocessing_step, training_step]
)

# Upsert (create or update)
pipeline.upsert(role_arn=SAGEMAKER_ROLE_ARN)
```

**c. Execute Pipeline**:
```python
# Script: execute_pipeline.py
execution = pipeline.start(
    parameters={
        'DaysDelay': 14,
        'UseWeather': 'true',
        'HPOMaxEvals': 50
    }
)

print(f"Pipeline execution ARN: {execution.arn}")
```

**d. Monitor Pipeline**:
```python
# Script: monitor_pipeline.py
import time

max_wait = 7200  # 2 hours
poll_interval = 60  # 1 minute
elapsed = 0

while elapsed < max_wait:
    status = execution.describe()['PipelineExecutionStatus']

    if status == 'Succeeded':
        print("✓ Pipeline completed successfully")
        break
    elif status in ['Failed', 'Stopped']:
        raise Exception(f"Pipeline {status}")

    time.sleep(poll_interval)
    elapsed += poll_interval

if elapsed >= max_wait:
    raise TimeoutError("Pipeline execution timeout")
```

**Pipeline Steps Executed**:

1. **Preprocessing Step** (~15-20 min):
   - Query 3 years of data from Redshift
   - Aggregate by datetime
   - Add weather features (Open-Meteo API)
   - Add solar features (position, radiation)
   - Create lag features (14, 21, 28, 35 days)
   - Split into train/val/test
   - Output CSVs to S3

2. **Training Step** (~2-4 hours):
   - Load train/val/test CSVs
   - Feature selection (consensus across 3 methods)
   - Hyperparameter optimization (Optuna, 50 trials)
   - Train final XGBoost model
   - Evaluate on test set (segment-specific metrics)
   - Generate visualizations
   - Save model, features, metrics to S3

**Output Artifacts**:
```
s3://{bucket}/{profile}-{segment}/
├── models/run_{run_id}/
│   ├── xgboost-model (pickled model)
│   ├── features.pkl (feature list)
│   ├── model_metadata.json
│   └── metrics.json
├── evaluation/run_{run_id}/
│   ├── training_results.png
│   ├── hpo_results.png
│   ├── feature_importance.png
│   └── segment_metrics.json
└── reports/
    ├── pipeline_report_{timestamp}.json
    └── deployment_summary_{timestamp}.md
```

#### 9. deploy_model (Matrix Job) - COST OPTIMIZATION CORE

**Purpose**: Deploy model to SageMaker endpoint and IMMEDIATELY DELETE for cost savings

**Key Steps**:

**a. Find Latest Model**:
```bash
# List all run directories
aws s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/models/ | grep run_

# Sort by timestamp and get latest
LATEST_RUN=$(aws s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/models/ \
  | grep run_ | sort | tail -1 | awk '{print $2}' | tr -d '/')

MODEL_S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}/models/${LATEST_RUN}/xgboost-model"
```

**b. Register Model**:
```python
# Script: register_model.py
model_package = sagemaker_client.create_model_package(
    ModelPackageGroupName=f"EnergyForecastModels-{PROFILE}-{SEGMENT}",
    ModelPackageDescription=f"Energy forecast model for {PROFILE}-{SEGMENT}",
    InferenceSpecification={
        'Containers': [{
            'Image': xgboost_image_uri,
            'ModelDataUrl': MODEL_S3_URI
        }],
        'SupportedContentTypes': ['application/json'],
        'SupportedResponseMIMETypes': ['application/json']
    },
    ModelApprovalStatus='Approved'
)
```

**c. Deploy to Endpoint**:
```python
# Script: deploy_model.py

# Step 1: Create SageMaker Model
model_name = f"{ENV}-energy-model-{PROFILE}-{SEGMENT}-{RUN_ID}"
sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': xgboost_image_uri,
        'ModelDataUrl': MODEL_S3_URI,
        'Environment': {
            'SAGEMAKER_PROGRAM': 'inference.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': model_s3_uri
        }
    },
    ExecutionRoleArn=SAGEMAKER_ROLE_ARN
)

# Step 2: Create Endpoint Configuration
endpoint_config_name = f"{ENV}-energy-endpoint-config-{PROFILE}-{SEGMENT}-{RUN_ID}"
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InstanceType': 'ml.m5.xlarge',
        'InitialInstanceCount': 1,
        'InitialVariantWeight': 1.0
    }]
)

# Step 3: Create Endpoint
endpoint_name = f"{ENV}-energy-ml-endpoint-{PROFILE}-{SEGMENT}"
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name,
    Tags=[
        {'Key': 'Environment', 'Value': ENV},
        {'Key': 'Profile', 'Value': PROFILE},
        {'Key': 'Segment', 'Value': SEGMENT},
        {'Key': 'RunId', 'Value': RUN_ID},
        {'Key': 'CostOptimized', 'Value': 'true'}
    ]
)

# Step 4: Wait for InService
waiter = sagemaker_client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)
print(f"✓ Endpoint {endpoint_name} is InService")

# Step 5: ⚡ COST OPTIMIZATION - IMMEDIATE DELETE ⚡
print("⚡ COST OPTIMIZATION: Deleting endpoint to save costs")

# Save endpoint configuration to S3 FIRST
endpoint_config = {
    'endpoint_name': endpoint_name,
    'model_name': model_name,
    'model_config': {...},
    'endpoint_config_name': endpoint_config_name,
    'endpoint_config': {...},
    'cost_optimized': True,
    'delete_recreate_enabled': True,
    'created_at': datetime.now().isoformat(),
    'run_id': RUN_ID
}

s3_key = f"{S3_PREFIX}/endpoint-configs/{endpoint_name}_config.json"
s3_client.put_object(
    Bucket=S3_BUCKET,
    Key=s3_key,
    Body=json.dumps(endpoint_config, indent=2),
    ContentType='application/json'
)

# Also save in customer-specific location (backup)
backup_key = f"{S3_PREFIX}/endpoint-configs/customers/{PROFILE}-{SEGMENT}/{endpoint_name}_config.json"
s3_client.put_object(
    Bucket=S3_BUCKET,
    Key=backup_key,
    Body=json.dumps(endpoint_config, indent=2)
)

# Now DELETE the endpoint
sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
print(f"✓ Endpoint {endpoint_name} deleted")
print("✓ Configuration saved to S3 for recreation")
print("✓ Cost optimization: $0/hour ongoing (100% savings)")

# Wait a bit for deletion to start
time.sleep(120)

# Verify deletion
try:
    status = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    print(f"  Endpoint status: {status['EndpointStatus']}")  # Should be "Deleting"
except ClientError as e:
    if e.response['Error']['Code'] == 'ValidationException':
        print("✓ Endpoint fully deleted (NotFound)")
```

**Cost Calculation**:
```
Traditional Approach:
- Endpoint instance: ml.m5.xlarge @ $0.47/hour
- Runtime: 24 hours/day × 30 days = 720 hours/month
- Cost per endpoint: $338.40/month
- 6 endpoints: $2,030.40/month

Delete/Recreate Approach:
- Endpoint creation: ~5 minutes/day
- Prediction runtime: ~3 minutes/day
- Total runtime: ~8 minutes/day × 30 days = 240 minutes/month = 4 hours/month
- Cost per endpoint: 4 hours × $0.47 = $1.88/month
- 6 endpoints: $11.28/month

Monthly Savings: $2,030.40 - $11.28 = $2,019.12 (99.4% reduction)
Annual Savings: $24,229.44
```

#### 11. create_forecasting_lambda (Matrix Job)

**Purpose**: Deploy Lambda function that will recreate endpoints and generate predictions

**Lambda Package Structure**:
```
lambda_forecast.zip
├── lambda_function.py                    # Main handler
├── forecast/
│   ├── __init__.py
│   ├── data_preparation.py               # Historical data retrieval
│   ├── feature_engineering.py            # Feature creation
│   ├── weather_service.py                # Open-Meteo API
│   ├── endpoint_service.py               # SageMaker invocation
│   └── utils.py                          # Utilities
├── configs/
│   └── config.py                         # Configuration
└── dependencies/
    ├── boto3/
    ├── pandas/
    ├── numpy/
    └── ... (other packages)
```

**Lambda Configuration**:
```python
# Script: create_forecast_lambda.py

lambda_client.create_function(
    FunctionName=f"{ENV}-energy-daily-predictor-{PROFILE}-{SEGMENT}",
    Runtime='python3.9',
    Role=LAMBDA_EXECUTION_ROLE_ARN,
    Handler='lambda_function.lambda_handler',
    Code={'ZipFile': open('lambda_forecast.zip', 'rb').read()},

    # Resource configuration
    MemorySize=1024,           # 1 GB
    Timeout=900,               # 15 minutes
    EphemeralStorage={'Size': 2048},  # 2 GB temp storage

    # Environment variables (50+ variables)
    Environment={'Variables': {
        # Core identification
        'CUSTOMER_PROFILE': PROFILE,
        'CUSTOMER_SEGMENT': SEGMENT,
        'ENV_NAME': ENV,
        'LOAD_PROFILE': PROFILE,

        # AWS resources
        'AWS_REGION': 'us-west-2',
        'S3_BUCKET': S3_BUCKET,
        'S3_PREFIX': S3_PREFIX,
        'SAGEMAKER_ROLE_ARN': SAGEMAKER_ROLE_ARN,

        # Redshift configuration
        'REDSHIFT_CLUSTER_IDENTIFIER': REDSHIFT_CLUSTER,
        'REDSHIFT_DATABASE': REDSHIFT_DATABASE,
        'REDSHIFT_DB_USER': REDSHIFT_DB_USER,
        'REDSHIFT_REGION': 'us-west-2',
        'REDSHIFT_INPUT_SCHEMA': REDSHIFT_INPUT_SCHEMA,
        'REDSHIFT_INPUT_TABLE': REDSHIFT_INPUT_TABLE,
        'REDSHIFT_OUTPUT_SCHEMA': REDSHIFT_OUTPUT_SCHEMA,
        'REDSHIFT_OUTPUT_TABLE': REDSHIFT_OUTPUT_TABLE,

        # Endpoint management (COST OPTIMIZATION)
        'ENDPOINT_NAME': f"{ENV}-energy-ml-endpoint-{PROFILE}-{SEGMENT}",
        'ENABLE_ENDPOINT_DELETE_RECREATE': 'true',
        'DELETE_ENDPOINT_AFTER_PREDICTION': 'true',
        'ENDPOINT_RECREATION_TIMEOUT': '900',
        'ENDPOINT_DELETION_TIMEOUT': '300',
        'ENDPOINT_READY_BUFFER_TIME': '60',
        'ENDPOINT_CONFIG_S3_PREFIX': 'endpoint-configs',

        # Features and lags
        'DEFAULT_LAG_DAYS': '7,14,21,28',
        'MORNING_PEAK_HOURS': '6,10',
        'SOLAR_PERIOD_HOURS': '9,17',
        'EVENING_RAMP_HOURS': '16,20',
        'EVENING_PEAK_HOURS': '20,23',

        # Weather API
        'DEFAULT_LATITUDE': '32.7157',
        'DEFAULT_LONGITUDE': '-117.1611',
        'DEFAULT_TIMEZONE': 'America/Los_Angeles',
        'WEATHER_VARIABLES': 'temperature_2m,apparent_temperature,cloudcover,direct_radiation,diffuse_radiation,shortwave_radiation,windspeed_10m,relativehumidity_2m,is_day',

        # Data timing
        'DATA_DELAY_DAYS': '14',
        'FINAL_SUBMISSION_DELAY': '48',
        'INITIAL_SUBMISSION_DELAY': '14',
        'SUBMISSION_TYPE_FINAL': 'Final',
        'SUBMISSION_TYPE_INITIAL': 'Initial',

        # Rate group filtering
        'RATE_GROUP_FILTER_CLAUSE': '(rategroup NOT LIKE "NEM%" AND rategroup NOT LIKE "SBP%")' if SEGMENT == 'NONSOLAR' else '(rategroup LIKE "NEM%" OR rategroup LIKE "SBP%")'
    }},

    # Tags
    Tags={
        'Environment': ENV,
        'Profile': PROFILE,
        'Segment': SEGMENT,
        'CostOptimized': 'true',
        'DeleteRecreate': 'true'
    }
)
```

**EventBridge Schedule Creation**:
```python
# Script: setup_schedule.py

# Create rule
events_client.put_rule(
    Name=f"EnergyForecastSchedule-{PROFILE}-{SEGMENT}-{ENV}",
    ScheduleExpression=LAMBDA_SCHEDULE,  # e.g., cron(0 9 * * ? *)
    State='ENABLED',
    Description=f"Daily energy forecast for {PROFILE}-{SEGMENT} in {ENV}"
)

# Add Lambda as target
events_client.put_targets(
    Rule=f"EnergyForecastSchedule-{PROFILE}-{SEGMENT}-{ENV}",
    Targets=[{
        'Id': '1',
        'Arn': lambda_arn,
        'Input': json.dumps({
            'scheduled': True,
            'source': 'eventbridge'
        })
    }]
)

# Grant EventBridge permission to invoke Lambda
lambda_client.add_permission(
    FunctionName=lambda_name,
    StatementId=f"AllowEventBridge-{ENV}-{PROFILE}-{SEGMENT}",
    Action='lambda:InvokeFunction',
    Principal='events.amazonaws.com',
    SourceArn=rule_arn
)
```

**Schedule by Environment**:
```
dev:     cron(0 10 * * ? *)  →  10:00 AM UTC (2:00 AM PST)
qa:      cron(0 11 * * ? *)  →  11:00 AM UTC (3:00 AM PST)
preprod: cron(0 8 * * ? *)   →   8:00 AM UTC (12:00 AM PST)
prod:    cron(0 9 * * ? *)   →   9:00 AM UTC (1:00 AM PST)
```

#### 12. deployment_summary

**Purpose**: Generate comprehensive deployment report

**Report Contents**:
```markdown
# Energy Load Forecasting Deployment Summary

## Deployment Details
- **Environment**: prod
- **Date**: 2026-01-13 12:34:56 UTC
- **GitHub Run ID**: 1234567890
- **Triggered By**: user@company.com
- **Branch**: main
- **Commit**: abc123def456

## Cost Optimization Status
✓ **Strategy**: Delete/Recreate Endpoints
✓ **Expected Savings**: 99.4% ($2,019/month)
✓ **Current Status**: All endpoints deleted
✓ **Ongoing Costs**: $0.00/hour

## Deployed Combinations

| Profile | Segment | Status | Pipeline | Model | Endpoint | Lambda | Schedule |
|---------|---------|--------|----------|-------|----------|--------|----------|
| RES | SOLAR | ✓ | Succeeded | Deployed | Deleted | Created | Enabled |
| RES | NONSOLAR | ✓ | Succeeded | Deployed | Deleted | Created | Enabled |
| MEDCI | SOLAR | ✓ | Succeeded | Deployed | Deleted | Created | Enabled |
| MEDCI | NONSOLAR | ✓ | Succeeded | Deployed | Deleted | Created | Enabled |
| SMLCOM | SOLAR | ✓ | Succeeded | Deployed | Deleted | Created | Enabled |
| SMLCOM | NONSOLAR | ✓ | Succeeded | Deployed | Deleted | Created | Enabled |

## SageMaker Endpoints Status

| Endpoint Name | Status | Cost | Configuration |
|---------------|--------|------|---------------|
| prod-energy-ml-endpoint-RES-SOLAR | NotFound | $0/hour | ✓ Stored in S3 |
| prod-energy-ml-endpoint-RES-NONSOLAR | NotFound | $0/hour | ✓ Stored in S3 |
| prod-energy-ml-endpoint-MEDCI-SOLAR | NotFound | $0/hour | ✓ Stored in S3 |
| prod-energy-ml-endpoint-MEDCI-NONSOLAR | NotFound | $0/hour | ✓ Stored in S3 |
| prod-energy-ml-endpoint-SMLCOM-SOLAR | NotFound | $0/hour | ✓ Stored in S3 |
| prod-energy-ml-endpoint-SMLCOM-NONSOLAR | NotFound | $0/hour | ✓ Stored in S3 |

## Lambda Functions

| Function Name | Status | Schedule | Next Run | Capabilities |
|---------------|--------|----------|----------|--------------|
| prod-energy-daily-predictor-RES-SOLAR | Active | 9 AM UTC | 2026-01-14 09:00 | Delete/Recreate |
| prod-energy-daily-predictor-RES-NONSOLAR | Active | 9 AM UTC | 2026-01-14 09:00 | Delete/Recreate |
| prod-energy-daily-predictor-MEDCI-SOLAR | Active | 9 AM UTC | 2026-01-14 09:00 | Delete/Recreate |
| prod-energy-daily-predictor-MEDCI-NONSOLAR | Active | 9 AM UTC | 2026-01-14 09:00 | Delete/Recreate |
| prod-energy-daily-predictor-SMLCOM-SOLAR | Active | 9 AM UTC | 2026-01-14 09:00 | Delete/Recreate |
| prod-energy-daily-predictor-SMLCOM-NONSOLAR | Active | 9 AM UTC | 2026-01-14 09:00 | Delete/Recreate |

## Model Performance Metrics

| Combination | RMSE | MAE | MAPE | R² | Duck Curve RMSE |
|-------------|------|-----|------|----|----|
| RES-SOLAR | 1,234 | 987 | 8.5% | 0.92 | 1,456 |
| RES-NONSOLAR | 1,567 | 1,234 | 9.2% | 0.89 | N/A |
| MEDCI-SOLAR | 2,345 | 1,876 | 7.8% | 0.94 | 2,123 |
| MEDCI-NONSOLAR | 2,678 | 2,134 | 8.1% | 0.93 | N/A |
| SMLCOM-SOLAR | 876 | 654 | 6.5% | 0.96 | 923 |
| SMLCOM-NONSOLAR | 1,012 | 789 | 7.2% | 0.95 | N/A |

## Operational Guidance

### Monitoring
- **CloudWatch Logs**: `/aws/lambda/prod-energy-daily-predictor-*`
- **Key Messages**: "endpoint deleted", "prediction successful", "cost optimization achieved"

### Configuration Storage
- **Primary Location**: `s3://{bucket}/{profile}-{segment}/endpoint-configs/`
- **Backup Location**: `s3://{bucket}/{profile}-{segment}/endpoint-configs/customers/`

### Testing Recommendations
1. Monitor first scheduled execution (tomorrow at 9 AM UTC)
2. Verify endpoint recreation in CloudWatch Logs
3. Check predictions in Redshift output table
4. Confirm endpoint deletion after predictions
5. Review Lambda execution time (should be 8-12 minutes)

### Troubleshooting
- **Endpoint Recreation Issues**: Check S3 configuration exists
- **Prediction Failures**: Review CloudWatch Logs for Lambda
- **Data Issues**: Verify Redshift connectivity
- **Weather API Failures**: Check Open-Meteo API status

## Next Steps
1. ✓ Deployment complete - all systems operational
2. ⏳ First scheduled forecast: 2026-01-14 09:00 UTC
3. ⏳ Monitor initial execution
4. ⏳ Validate predictions in dashboard
5. ⏳ Review cost reports after 1 week

---
**Generated by**: GitHub Actions Deploy Workflow
**Artifact Retention**: 90 days
```

---

## Historical Forecasting Workflow (historical_forecasting.yml)

### Purpose
Generate historical predictions for backfilling data or testing models on past dates.

### Key Differences from Deploy Workflow
1. **No Training**: Uses existing deployed models
2. **Endpoint Lifecycle**: Creates endpoints → Generates predictions → Deletes endpoints
3. **Date Ranges**: Supports multiple date input methods
4. **Parallel Predictions**: Generates predictions for multiple dates simultaneously

### Trigger Options

```yaml
workflow_dispatch:
  inputs:
    environment:
      description: 'Environment'
      type: choice
      options: [dev, qa, preprod, prod]

    combinations:
      description: 'Which combinations'
      type: choice
      options:
        - all
        - res_only
        - medci_only
        - smlcom_only
        - solar_only
        - nonsolar_only
        - single_combination

    prediction_type:
      description: 'Type of historical prediction'
      type: choice
      options:
        - date_range       # Between start_date and end_date
        - days_past        # Number of days back from reference_date
        - multiple_dates   # Specific comma-separated dates

    # Date range inputs
    start_date: '2025-05-01'
    end_date: '2025-05-31'

    # Days past inputs
    reference_date: ''     # Defaults to today
    days_back: '7'

    # Multiple dates input
    multiple_dates_list: '2025-06-01,2025-06-04,2025-06-07'
```

### Job Flow

```
1. determine_trigger_and_inputs
   • Parse workflow inputs or use defaults

2. validate_inputs
   • Validate date formats
   • Generate prediction dates list
   • Calculate total predictions
   • Create combinations matrix

3. setup_infrastructure_info
   • Load database configuration
   • Verify connectivity

4. setup_historical_endpoints [MATRIX: 6 parallel]
   • Load endpoint config from S3
   • Recreate SageMaker endpoint
   • Wait for InService

5. generate_historical_predictions [MATRIX: 6 parallel]
   • For each date in prediction_dates:
     - Fetch historical data
     - Generate features
     - Invoke endpoint
     - Save predictions

6. cleanup_historical_endpoints [MATRIX: 6 parallel]
   • Delete all endpoints
   • Verify deletion
   • Log cost optimization

7. generate_final_summary
   • Aggregate results
   • Create comprehensive report
   • Upload artifacts
```

### Prediction Type Examples

**1. Date Range**:
```yaml
prediction_type: date_range
start_date: '2025-05-01'
end_date: '2025-05-31'

# Generates predictions for May 1-31, 2025 (31 days)
# Total predictions: 31 dates × 6 combinations = 186 prediction sets
```

**2. Days Past**:
```yaml
prediction_type: days_past
reference_date: '2025-06-15'  # or empty for today
days_back: '7'

# Generates predictions for past 7 days from June 15
# Dates: June 8, 9, 10, 11, 12, 13, 14, 15
# Total: 8 dates × 6 combinations = 48 prediction sets
```

**3. Multiple Dates**:
```yaml
prediction_type: multiple_dates
multiple_dates_list: '2025-06-01,2025-06-15,2025-06-30'

# Generates predictions for exactly these 3 dates
# Total: 3 dates × 6 combinations = 18 prediction sets
```

### Cost Optimization in Historical Workflow

```
Traditional Approach (keep endpoints running):
- 6 endpoints × $0.47/hour
- Prediction time: 2 hours for 30 days
- Cost: 6 × $0.47 × 2 = $5.64

Delete/Recreate Approach:
- Endpoint creation: 6 × 5 minutes = 30 minutes
- Prediction time: ~1 hour
- Endpoint deletion: 6 × 2 minutes = 12 minutes
- Total time: ~1.7 hours
- Cost: 6 × $0.47 × 1.7 = $4.80

Savings: $0.84 per historical generation run (15% savings)

BUT: After predictions complete, all endpoints deleted
→ Ongoing cost: $0/hour vs $2.82/hour
→ Monthly savings: $2,058 if kept running
```

---

## Workflow Comparison

| Feature | deploy.yml | historical_forecasting.yml |
|---------|------------|----------------------------|
| **Purpose** | Train & deploy models | Generate historical predictions |
| **Trigger** | Push or manual | Manual only |
| **Training** | Yes (full pipeline) | No (uses existing models) |
| **Endpoints** | Create → Delete immediately | Create → Use → Delete after |
| **Duration** | 3-5 hours | 1-2 hours |
| **Schedules** | Creates Lambda schedules | No schedules |
| **Outputs** | Models, endpoints configs, Lambdas | Historical predictions in Redshift |
| **Cost** | ~$5-10 per deployment | ~$5 per historical generation |

---

## Environment Configuration

### Environment Variables by Job

**All Jobs**:
```bash
ENVIRONMENT=prod                    # dev, qa, preprod, prod
DATABASE_TYPE=redshift             # or athena
AWS_REGION=us-west-2
S3_BUCKET=sdcp-prod-sagemaker-energy-forecasting-data
```

**Matrix Jobs** (per combination):
```bash
CUSTOMER_PROFILE=RES               # RES, MEDCI, SMLCOM
CUSTOMER_SEGMENT=SOLAR             # SOLAR, NONSOLAR
S3_PREFIX=RES-SOLAR

ENDPOINT_NAME=prod-energy-ml-endpoint-RES-SOLAR
LAMBDA_FUNCTION_NAME=prod-energy-daily-predictor-RES-SOLAR

REDSHIFT_CLUSTER_IDENTIFIER=sdcp-cluster-prod
REDSHIFT_DATABASE=sdcp
REDSHIFT_INPUT_SCHEMA=edp_cust_prod
REDSHIFT_OUTPUT_SCHEMA=edp_forecasting_prod
```

### Secrets Required

```yaml
secrets:
  AWS_ACCESS_KEY_ID: <IAM user access key>
  AWS_SECRET_ACCESS_KEY: <IAM user secret key>
  AWS_REGION: us-west-2
  S3_BUCKET: <bucket name>
  SAGEMAKER_ROLE_ARN: <SageMaker execution role ARN>
```

### Variables Required

```yaml
vars:
  REDSHIFT_CLUSTER_IDENTIFIER_PREFIX: sdcp-cluster
  REDSHIFT_DATABASE: sdcp
  REDSHIFT_DB_USER: ds_service_user
  REDSHIFT_INPUT_SCHEMA_PREFIX: edp_cust
  REDSHIFT_INPUT_TABLE: caiso_sqmd
  REDSHIFT_OPERATIONAL_SCHEMA_PREFIX: edp_forecasting
  REDSHIFT_OPERATIONAL_TABLE: dayahead_load_forecasts
  REDSHIFT_BI_SCHEMA_PREFIX: edp_bi
```

---

## Troubleshooting Workflows

### Common Issues

**1. Pipeline Timeout**
```
Error: Pipeline execution exceeded max_wait_time (7200s)
Solution:
- Check SageMaker pipeline execution logs
- Increase max_wait_time for large datasets
- Reduce HPO_MAX_EVALS for faster training
```

**2. Endpoint Creation Failure**
```
Error: ResourceLimitExceeded: Endpoint quota exceeded
Solution:
- Request quota increase via AWS Service Quotas
- Delete unused endpoints
- Use smaller instance types for dev/qa
```

**3. Lambda Packaging Failure**
```
Error: Lambda package exceeds 50MB limit
Solution:
- Remove unnecessary dependencies
- Use Lambda layers for large packages
- Optimize numpy/pandas installations
```

**4. Redshift Connection Timeout**
```
Error: Unable to connect to Redshift cluster
Solution:
- Verify VPC/security group settings
- Check Redshift cluster status
- Validate IAM role permissions
```

**5. Matrix Job Failure**
```
Error: One combination failed but others succeeded
Solution:
- fail-fast: false allows other jobs to continue
- Check individual job logs
- Re-run failed combinations manually
```

### Debugging Tips

**View Logs**:
```bash
# GitHub Actions logs
# Navigate to: https://github.com/{org}/{repo}/actions

# SageMaker pipeline logs
aws sagemaker describe-pipeline-execution \
  --pipeline-execution-arn <arn>

# Lambda logs
aws logs tail /aws/lambda/{function-name} --follow

# Redshift query logs
aws redshift-data describe-statement --id <query-id>
```

**Manual Rerun**:
```bash
# Rerun specific job
# GitHub Actions UI → Failed job → Re-run jobs

# Rerun specific combination
# Workflow dispatch with single_combination
```

**Check Endpoint Status**:
```bash
aws sagemaker describe-endpoint \
  --endpoint-name prod-energy-ml-endpoint-RES-SOLAR

# Should return: ValidationException (endpoint not found)
# Indicates successful deletion
```

---

## Best Practices

### 1. Environment Progression
- Deploy to **dev** first (automatic)
- Test in **qa** with manual trigger
- Validate in **preprod** with production-like data
- Deploy to **prod** with approval gate

### 2. Testing Before Production
- Run integration tests in qa/preprod
- Validate historical predictions
- Monitor Lambda executions
- Review cost reports

### 3. Rollback Strategy
- Keep previous model artifacts in S3
- Maintain endpoint configs for all versions
- Document model version in tags
- Can redeploy previous version via workflow dispatch

### 4. Monitoring
- Set up CloudWatch Alarms for Lambda failures
- Monitor SageMaker pipeline success rates
- Track cost trends in Cost Explorer
- Review prediction accuracy in dashboard

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Status**: Production Ready
