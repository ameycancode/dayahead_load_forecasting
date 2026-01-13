# Configuration and Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS Account Setup](#aws-account-setup)
3. [GitHub Repository Setup](#github-repository-setup)
4. [Local Development Setup](#local-development-setup)
5. [Configuration Files](#configuration-files)
6. [Environment Variables](#environment-variables)
7. [Infrastructure Setup](#infrastructure-setup)
8. [Verification and Testing](#verification-and-testing)

---

## 1. Prerequisites

### Required Tools
- **Python 3.9+**: Primary development language
- **AWS CLI v2**: For interacting with AWS services
- **Git**: Version control
- **Docker** (optional): For local testing of Lambda functions
- **SAM CLI** (optional): For local Lambda development

### Required Knowledge
- Basic understanding of AWS services (SageMaker, Lambda, Redshift, S3, Athena)
- Python programming
- Machine learning concepts (XGBoost, time series forecasting)
- SQL and database operations
- GitHub Actions CI/CD

### AWS Account Requirements
- Active AWS account with administrative access
- Credit card for AWS billing
- Understanding of AWS pricing for:
  - SageMaker training and inference
  - Lambda function execution
  - S3 storage
  - Redshift cluster
  - Data transfer

---

## 2. AWS Account Setup

### 2.1 Create IAM User for GitHub Actions

Create a dedicated IAM user with programmatic access for GitHub Actions:

```bash
# User name suggestion
aws iam create-user --user-name github-actions-sagemaker
```

### 2.2 Required IAM Policies

Attach the following managed policies to the IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "s3:*",
        "lambda:*",
        "events:*",
        "iam:PassRole",
        "iam:GetRole",
        "redshift:*",
        "athena:*",
        "glue:*",
        "logs:*",
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

**Key Permissions Breakdown:**
- `sagemaker:*`: Full access to create pipelines, training jobs, models, endpoints
- `s3:*`: Read/write access to S3 buckets for data and model artifacts
- `lambda:*`: Create and manage Lambda functions for predictions
- `events:*`: Set up EventBridge schedules for daily forecasts
- `iam:PassRole`: Pass execution roles to SageMaker and Lambda
- `redshift:*`: Query data and write predictions to Redshift
- `athena:*`: Alternative data source using Athena

### 2.3 Create SageMaker Execution Role

Create an IAM role that SageMaker and Lambda can assume:

```bash
# Create trust policy file
cat > sagemaker-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "sagemaker.amazonaws.com",
          "lambda.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
  --role-name SageMakerEnergyForecastingRole \
  --assume-role-policy-document file://sagemaker-trust-policy.json

# Attach required policies
aws iam attach-role-policy \
  --role-name SageMakerEnergyForecastingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerEnergyForecastingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name SageMakerEnergyForecastingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonRedshiftFullAccess

aws iam attach-role-policy \
  --role-name SageMakerEnergyForecastingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonAthenaFullAccess

aws iam attach-role-policy \
  --role-name SageMakerEnergyForecastingRole \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
```

**Get the Role ARN** (save this for GitHub Secrets):
```bash
aws iam get-role --role-name SageMakerEnergyForecastingRole --query 'Role.Arn' --output text
```

### 2.4 Create S3 Bucket

```bash
# Set your environment and region
ENVIRONMENT="dev"  # or "stage", "prod"
AWS_REGION="us-west-2"

# Create bucket
BUCKET_NAME="sdcp-${ENVIRONMENT}-sagemaker-energy-forecasting-data"
aws s3 mb s3://${BUCKET_NAME} --region ${AWS_REGION}

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
  --bucket ${BUCKET_NAME} \
  --versioning-configuration Status=Enabled

# Block public access (security best practice)
aws s3api put-public-access-block \
  --bucket ${BUCKET_NAME} \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

### 2.5 Setup Redshift Cluster (or Athena)

#### Option A: Redshift Setup

```bash
# Create Redshift cluster
aws redshift create-cluster \
  --cluster-identifier sdcp-${ENVIRONMENT}-energy-forecasting \
  --node-type dc2.large \
  --master-username admin \
  --master-user-password <YourSecurePassword> \
  --cluster-type single-node \
  --publicly-accessible false \
  --region ${AWS_REGION}

# Wait for cluster to be available
aws redshift wait cluster-available \
  --cluster-identifier sdcp-${ENVIRONMENT}-energy-forecasting
```

**Create Required Schemas and Tables:**
```sql
-- Connect to Redshift and run:

-- Input schema for historical load data
CREATE SCHEMA IF NOT EXISTS edp_ods;

-- Output schema for forecasts
CREATE SCHEMA IF NOT EXISTS edp_forecasting;

-- Create output table for forecasts
CREATE TABLE IF NOT EXISTS edp_forecasting.dayahead_load_forecasts (
    forecast_date DATE NOT NULL,
    forecast_hour INTEGER NOT NULL,
    forecast_timestamp TIMESTAMP NOT NULL,
    customer_profile VARCHAR(50) NOT NULL,
    customer_segment VARCHAR(50) NOT NULL,
    predicted_load DECIMAL(18, 6) NOT NULL,
    model_version VARCHAR(100),
    created_at TIMESTAMP DEFAULT GETDATE(),
    PRIMARY KEY (forecast_date, forecast_hour, customer_profile, customer_segment)
);

-- Create BI view for reporting
CREATE OR REPLACE VIEW edp_forecasting.vw_dayahead_forecasts AS
SELECT
    forecast_date,
    forecast_hour,
    forecast_timestamp,
    customer_profile,
    customer_segment,
    predicted_load,
    model_version,
    created_at
FROM edp_forecasting.dayahead_load_forecasts
ORDER BY forecast_date, forecast_hour, customer_profile, customer_segment;

-- Create service user for predictions Lambda
CREATE USER ds_service_user PASSWORD '<SecurePassword>';
GRANT USAGE ON SCHEMA edp_ods TO ds_service_user;
GRANT SELECT ON ALL TABLES IN SCHEMA edp_ods TO ds_service_user;
GRANT USAGE ON SCHEMA edp_forecasting TO ds_service_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA edp_forecasting TO ds_service_user;
```

#### Option B: Athena Setup

```bash
# Create Athena database
aws athena start-query-execution \
  --query-string "CREATE DATABASE IF NOT EXISTS sdcp_energy_forecasting" \
  --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
  --region ${AWS_REGION}

# Create Athena query results bucket
aws s3 mb s3://aws-athena-query-results-${AWS_REGION}-<YOUR_ACCOUNT_ID>/ --region ${AWS_REGION}
```

---

## 3. GitHub Repository Setup

### 3.1 Configure GitHub Secrets

Navigate to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

Add the following secrets:

| Secret Name | Description | Example Value |
|------------|-------------|---------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key ID | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret access key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_REGION` | AWS region for deployments | `us-west-2` |
| `AWS_ACCOUNT_ID` | Your AWS account ID (12 digits) | `123456789012` |
| `S3_BUCKET` | S3 bucket name for model artifacts | `sdcp-dev-sagemaker-energy-forecasting-data` |
| `SAGEMAKER_ROLE_ARN` | ARN of SageMaker execution role | `arn:aws:iam::123456789012:role/SageMakerEnergyForecastingRole` |

### 3.2 Configure GitHub Variables

Add repository variables (Settings → Secrets and variables → Actions → Variables tab):

| Variable Name | Description | Example Value |
|--------------|-------------|---------------|
| `ENVIRONMENT` | Deployment environment | `dev`, `stage`, or `prod` |
| `REDSHIFT_CLUSTER_IDENTIFIER` | Redshift cluster name | `sdcp-dev-energy-forecasting` |
| `REDSHIFT_DATABASE` | Redshift database name | `sdcp` |
| `REDSHIFT_INPUT_SCHEMA` | Schema for input data | `edp_ods` |
| `REDSHIFT_OUTPUT_SCHEMA` | Schema for forecasts | `edp_forecasting` |
| `DATABASE_TYPE` | Database type (`redshift` or `athena`) | `redshift` |

### 3.3 Verify Secrets Configuration

Create a test workflow to verify secrets are accessible:

```yaml
# .github/workflows/verify-secrets.yml
name: Verify Secrets
on:
  workflow_dispatch:

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - name: Check AWS Credentials
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ${{ secrets.AWS_REGION }}
        run: |
          if [ -z "$AWS_ACCESS_KEY_ID" ]; then
            echo "❌ AWS_ACCESS_KEY_ID is not set"
            exit 1
          fi
          echo "✅ AWS_ACCESS_KEY_ID is set"

          if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
            echo "❌ AWS_SECRET_ACCESS_KEY is not set"
            exit 1
          fi
          echo "✅ AWS_SECRET_ACCESS_KEY is set"

          if [ -z "$AWS_REGION" ]; then
            echo "❌ AWS_REGION is not set"
            exit 1
          fi
          echo "✅ AWS_REGION is set"

          echo "All required secrets are configured!"
```

---

## 4. Local Development Setup

### 4.1 Clone Repository

```bash
git clone https://github.com/yourusername/dayahead_load_forecasting.git
cd dayahead_load_forecasting
```

### 4.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4.3 Install Dependencies

The project has multiple requirements files for different components:

#### Preprocessing Pipeline Dependencies
```bash
pip install -r pipeline/preprocessing/requirements.txt
```

**Key packages:**
- `pandas==2.0.3`: Data manipulation
- `numpy==1.24.4`: Numerical operations
- `scikit-learn==1.3.0`: ML utilities
- `boto3==1.28.38`: AWS SDK
- `pyathena==2.25.2`: Athena connectivity
- `openmeteo-requests==1.4.0`: Weather API client

#### Lambda/Predictions Dependencies
```bash
pip install -r predictions/requirements.txt
```

**Key packages:**
- `pandas==2.0.3`: Data processing
- `boto3==1.28.38`: AWS services
- `psycopg2-binary>=2.8.0`: Redshift connectivity
- `redshift_connector>=2.0.0`: Redshift driver
- `openmeteo-requests==1.1.0`: Weather forecasts

#### Test Automation Dependencies
```bash
pip install -r test-automation/requirements-test.txt
```

**Key packages:**
- `pytest>=7.0.0`: Testing framework
- `pytest-cov>=4.0.0`: Code coverage
- `moto>=4.0.0`: AWS service mocking
- `black>=22.0.0`: Code formatting

### 4.4 Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# Verify configuration
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDAEXAMPLEUSERID",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/github-actions-sagemaker"
}
```

### 4.5 Set Environment Variables

Create a `.env` file in the project root (DO NOT commit this file):

```bash
# .env
export ENVIRONMENT=dev
export AWS_REGION=us-west-2
export S3_BUCKET=sdcp-dev-sagemaker-energy-forecasting-data
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerEnergyForecastingRole

# Database Configuration
export DATABASE_TYPE=redshift
export REDSHIFT_CLUSTER_IDENTIFIER=sdcp-dev-energy-forecasting
export REDSHIFT_DATABASE=sdcp
export REDSHIFT_DB_USER=ds_service_user
export REDSHIFT_REGION=us-west-2
export REDSHIFT_INPUT_SCHEMA=edp_ods
export REDSHIFT_INPUT_TABLE=caiso_sqmd
export REDSHIFT_OUTPUT_SCHEMA=edp_forecasting
export REDSHIFT_OUTPUT_TABLE=dayahead_load_forecasts
export REDSHIFT_BI_SCHEMA=edp_forecasting
export REDSHIFT_BI_VIEW=vw_dayahead_forecasts

# SageMaker Instance Configuration
export PREPROCESSING_INSTANCE_TYPE=ml.m5.large
export PREPROCESSING_INSTANCE_COUNT=1
export TRAINING_INSTANCE_TYPE=ml.m5.large
export TRAINING_INSTANCE_COUNT=1
export INFERENCE_INSTANCE_TYPE=ml.m5.large
export INFERENCE_INSTANCE_COUNT=1

# Data Processing Configuration
export INITIAL_SUBMISSION_DELAY=14
export FINAL_SUBMISSION_DELAY=48
export METER_THRESHOLD=1000
export USE_CACHE=true
```

Load the environment variables:
```bash
source .env
```

---

## 5. Configuration Files

### 5.1 Main Configuration File: `configs/config.py`

This is the central configuration file containing business domain knowledge and operational parameters.

#### Key Configuration Sections

**A. Customer Profiles and Segments**
```python
PROFILE_CONFIGS = {
    "RES": {
        "SOLAR": {
            "METER_THRESHOLD": 100,
            "USE_SOLAR_FEATURES": True,
            "LOAD_PROFILE": "RES",
            "MODEL_BASE_NAME": "res-solar",
            "BASE_JOB_NAME": "res-solar-load-forecasting"
        },
        "NONSOLAR": {
            "METER_THRESHOLD": 100,
            "USE_SOLAR_FEATURES": False,
            # ...
        }
    },
    # MEDCI and SMLCOM profiles...
}
```

**B. Rate Group Filters**
```python
RATE_GROUP_FILTERS = {
    "RES": {
        "SOLAR": {
            "include": ["NEM", "SBP"],  # Net Energy Metering, Solar Billing Plan
            "exclude": [],
            "operator": "LIKE",
            "logic": "OR"
        },
        # ...
    }
}
```

**C. Time Period Definitions**
```python
MORNING_PEAK_HOURS = (6, 9)      # 6 AM - 9 AM
SOLAR_PERIOD_HOURS = (9, 16)     # 9 AM - 4 PM
EVENING_RAMP_HOURS = (14, 18)    # 2 PM - 6 PM
EVENING_PEAK_HOURS = (17, 21)    # 5 PM - 9 PM
```

**D. Feature Engineering Parameters**
```python
DEFAULT_LAG_DAYS = [14, 21]           # Default lag days for features
EXTENDED_LAG_DAYS = [14, 21, 28, 35]  # Extended lag configuration
OUTLIER_IQR_FACTOR = 1.5              # Outlier detection threshold
EXTREME_IQR_FACTOR = 3.0              # Extreme outlier capping
TEST_DAYS = 30                         # Test set size
VALIDATION_DAYS = 60                   # Validation set size
```

**E. Segment-Specific Evaluation Periods**
```python
SEGMENT_EVALUATION_PERIODS = {
    "RES_SOLAR": {
        "morning_high": (6, 9),
        "solar_ramp_down": (9, 12),
        "midday_low": (10, 14),
        "duck_curve_critical": (14, 18),  # Most critical for solar
        "evening_peak": (17, 21),
        "night_baseload": (21, 6)
    },
    # Other segments...
}
```

**F. Metric Weights for Model Evaluation**
```python
SEGMENT_METRIC_WEIGHTS = {
    "RES_SOLAR": {
        "duck_curve_critical": 0.35,    # Highest weight
        "evening_peak": 0.25,
        "midday_low": 0.15,
        "solar_ramp_down": 0.1,
        "morning_high": 0.1,
        "night_baseload": 0.05
    },
    # Other segments...
}
```

#### Using the Configuration

```python
from configs.config import get_config_for_profile_segment

# Get configuration for specific profile/segment
config = get_config_for_profile_segment(profile="RES", segment="SOLAR")

# Access configuration values
s3_bucket = config["S3_BUCKET"]
meter_threshold = config["METER_THRESHOLD"]
use_solar_features = config["USE_SOLAR_FEATURES"]
rate_filter = config["RATE_GROUP_FILTER_CLAUSE"]
```

### 5.2 Pipeline Configuration Generation

Pipeline configurations are dynamically generated during workflow execution:

**Location:** `.github/scripts/deploy/prepare_config.py`

**Generated Configuration Example:**
```json
{
  "customer_profile": "RES",
  "customer_segment": "SOLAR",
  "s3_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
  "s3_prefix": "RES-SOLAR",
  "model_base_name": "res-solar",
  "endpoint_name": "res-solar-forecasting-endpoint",
  "meter_threshold": 100,
  "use_solar_features": true,
  "rate_group_filter": "(rategroup LIKE 'NEM%' OR rategroup LIKE 'SBP%')",
  "database_type": "redshift",
  "redshift_cluster_identifier": "sdcp-dev-energy-forecasting",
  "redshift_database": "sdcp",
  "redshift_input_schema": "edp_ods",
  "redshift_input_table": "caiso_sqmd",
  "redshift_output_schema": "edp_forecasting",
  "redshift_output_table": "dayahead_load_forecasts"
}
```

---

## 6. Environment Variables Reference

### 6.1 AWS Configuration

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `ENVIRONMENT` | Deployment environment | Yes | `dev`, `stage`, `prod` |
| `AWS_REGION` | AWS region | Yes | `us-west-2` |
| `AWS_ACCOUNT_ID` | AWS account ID | Yes | `123456789012` |
| `S3_BUCKET` | S3 bucket for artifacts | Yes | `sdcp-dev-sagemaker-energy-forecasting-data` |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN | Yes | `arn:aws:iam::123456789012:role/SageMakerRole` |

### 6.2 Database Configuration

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `DATABASE_TYPE` | Database type | Yes | `redshift` or `athena` |
| `REDSHIFT_CLUSTER_IDENTIFIER` | Redshift cluster ID | If using Redshift | `sdcp-dev-energy-forecasting` |
| `REDSHIFT_DATABASE` | Database name | If using Redshift | `sdcp` |
| `REDSHIFT_DB_USER` | Database user | If using Redshift | `ds_service_user` |
| `REDSHIFT_REGION` | Redshift region | If using Redshift | `us-west-2` |
| `REDSHIFT_INPUT_SCHEMA` | Input schema | If using Redshift | `edp_ods` |
| `REDSHIFT_INPUT_TABLE` | Input table | If using Redshift | `caiso_sqmd` |
| `REDSHIFT_OUTPUT_SCHEMA` | Output schema | If using Redshift | `edp_forecasting` |
| `REDSHIFT_OUTPUT_TABLE` | Output table | If using Redshift | `dayahead_load_forecasts` |
| `REDSHIFT_BI_SCHEMA` | BI view schema | If using Redshift | `edp_forecasting` |
| `REDSHIFT_BI_VIEW` | BI view name | If using Redshift | `vw_dayahead_forecasts` |
| `ATHENA_DATABASE` | Athena database | If using Athena | `sdcp_energy_forecasting` |
| `ATHENA_TABLE` | Athena table | If using Athena | `raw_agg_caiso_sqmd` |

### 6.3 SageMaker Instance Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PREPROCESSING_INSTANCE_TYPE` | Preprocessing instance type | No | `ml.m5.large` |
| `PREPROCESSING_INSTANCE_COUNT` | Preprocessing instance count | No | `1` |
| `TRAINING_INSTANCE_TYPE` | Training instance type | No | `ml.m5.large` |
| `TRAINING_INSTANCE_COUNT` | Training instance count | No | `1` |
| `INFERENCE_INSTANCE_TYPE` | Inference instance type | No | `ml.m5.large` |
| `INFERENCE_INSTANCE_COUNT` | Inference instance count | No | `1` |

### 6.4 Data Processing Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `INITIAL_SUBMISSION_DELAY` | Hours to wait for initial data | No | `14` |
| `FINAL_SUBMISSION_DELAY` | Hours to wait for final data | No | `48` |
| `METER_THRESHOLD` | Minimum meter count | No | `1000` |
| `USE_CACHE` | Enable CSV caching | No | `true` |

### 6.5 Customer Profile/Segment Variables

These are set by the GitHub Actions matrix strategy:

| Variable | Description | Values |
|----------|-------------|--------|
| `CUSTOMER_PROFILE` | Customer profile type | `RES`, `MEDCI`, `SMLCOM` |
| `CUSTOMER_SEGMENT` | Customer segment | `SOLAR`, `NONSOLAR` |

---

## 7. Infrastructure Setup

### 7.1 Initial Infrastructure Setup

Run the infrastructure setup scripts in order:

#### Step 1: Setup Redshift Infrastructure
```bash
python .github/scripts/deploy/setup_redshift_infrastructure.py \
  --cluster-id sdcp-dev-energy-forecasting \
  --database sdcp \
  --input-schema edp_ods \
  --output-schema edp_forecasting \
  --region us-west-2
```

This script:
- Creates required schemas
- Creates output tables for forecasts
- Creates BI views
- Sets up necessary permissions

#### Step 2: Setup Athena (if using Athena)
```bash
python .github/scripts/deploy/setup_athena.py \
  --database sdcp_energy_forecasting \
  --s3-bucket sdcp-dev-sagemaker-energy-forecasting-data \
  --region us-west-2
```

This script:
- Creates Athena database
- Creates external tables
- Sets up partitions
- Configures query result location

#### Step 3: Check SageMaker Permissions
```bash
python .github/scripts/deploy/check_sagemaker_permissions.py \
  --role-arn arn:aws:iam::123456789012:role/SageMakerEnergyForecastingRole \
  --s3-bucket sdcp-dev-sagemaker-energy-forecasting-data
```

This script verifies:
- SageMaker role exists
- S3 bucket access
- Redshift/Athena permissions
- Lambda execution permissions

### 7.2 Manual GitHub Actions Deployment

Trigger the main deployment workflow:

```bash
# Navigate to GitHub Actions
# Select "Deploy Energy Forecasting Models" workflow
# Click "Run workflow"
# Select branch: main (or your development branch)
# Click "Run workflow" button
```

The workflow will:
1. Setup infrastructure
2. Create SageMaker pipelines for all 6 customer segments
3. Execute preprocessing and training
4. Deploy models to endpoints
5. Create Lambda functions for daily predictions
6. Setup EventBridge schedules

---

## 8. Verification and Testing

### 8.1 Verify S3 Bucket Structure

After first deployment, verify S3 structure:

```bash
aws s3 ls s3://${S3_BUCKET}/ --recursive | head -20
```

Expected structure:
```
RES-SOLAR/
├── processed/
│   ├── training/
│   ├── validation/
│   └── test/
├── models/
├── evaluation/
├── deployment/
├── scripts/
└── forecasts/

RES-NONSOLAR/
├── [same structure]

MEDCI-SOLAR/
├── [same structure]

# ... (6 customer segments total)
```

### 8.2 Verify SageMaker Pipelines

List created pipelines:
```bash
aws sagemaker list-pipelines --region ${AWS_REGION}
```

Expected pipelines:
- `res-solar-forecasting-pipeline`
- `res-nonsolar-forecasting-pipeline`
- `medci-solar-forecasting-pipeline`
- `medci-nonsolar-forecasting-pipeline`
- `smlcom-solar-forecasting-pipeline`
- `smlcom-nonsolar-forecasting-pipeline`

### 8.3 Verify Endpoints

List SageMaker endpoints:
```bash
aws sagemaker list-endpoints --region ${AWS_REGION}
```

**Note:** Endpoints are deleted after deployment to save costs. They are recreated dynamically by Lambda functions during prediction time.

### 8.4 Verify Lambda Functions

List Lambda functions:
```bash
aws lambda list-functions --region ${AWS_REGION} | grep energy-forecasting
```

Expected Lambda functions:
- `res-solar-dayahead-forecasting`
- `res-nonsolar-dayahead-forecasting`
- `medci-solar-dayahead-forecasting`
- `medci-nonsolar-dayahead-forecasting`
- `smlcom-solar-dayahead-forecasting`
- `smlcom-nonsolar-dayahead-forecasting`

### 8.5 Verify EventBridge Schedules

List EventBridge rules:
```bash
aws events list-rules --region ${AWS_REGION} | grep dayahead-forecast
```

Expected schedules (daily at 7 AM PT):
- `res-solar-dayahead-forecast-schedule`
- `res-nonsolar-dayahead-forecast-schedule`
- (etc. for all 6 segments)

### 8.6 Test Lambda Function

Manually invoke a Lambda function to test:

```bash
# Create test event
cat > test-event.json << EOF
{
  "forecast_date": "2024-01-15",
  "run_type": "test"
}
EOF

# Invoke Lambda
aws lambda invoke \
  --function-name res-solar-dayahead-forecasting \
  --payload file://test-event.json \
  --region ${AWS_REGION} \
  response.json

# Check response
cat response.json
```

Expected response:
```json
{
  "statusCode": 200,
  "body": {
    "message": "Forecast completed successfully",
    "forecast_date": "2024-01-15",
    "records_written": 24,
    "model_version": "res-solar-20240110-v1"
  }
}
```

### 8.7 Verify Redshift Forecasts

Query Redshift to verify predictions were written:

```sql
-- Connect to Redshift
-- Check recent forecasts
SELECT
    forecast_date,
    forecast_hour,
    customer_profile,
    customer_segment,
    predicted_load,
    model_version,
    created_at
FROM edp_forecasting.dayahead_load_forecasts
WHERE forecast_date = CURRENT_DATE + 1
ORDER BY customer_profile, customer_segment, forecast_hour
LIMIT 50;
```

### 8.8 Run Integration Tests

Execute the full test suite:

```bash
# Install test dependencies
pip install -r test-automation/requirements-test.txt

# Run all tests
pytest test-automation/ -v --cov=pipeline --cov=predictions

# Run specific test categories
pytest test-automation/test_preprocessing.py -v
pytest test-automation/test_training.py -v
pytest test-automation/test_predictions.py -v
```

### 8.9 Check CloudWatch Logs

Monitor Lambda execution logs:

```bash
# Get latest log stream
LOG_GROUP="/aws/lambda/res-solar-dayahead-forecasting"
LOG_STREAM=$(aws logs describe-log-streams \
  --log-group-name ${LOG_GROUP} \
  --order-by LastEventTime \
  --descending \
  --max-items 1 \
  --query 'logStreams[0].logStreamName' \
  --output text)

# Tail logs
aws logs tail ${LOG_GROUP} --follow
```

---

## Common Issues and Solutions

### Issue 1: "Access Denied" to S3 Bucket

**Cause:** IAM role lacks S3 permissions

**Solution:**
```bash
# Add S3 policy to SageMaker role
aws iam attach-role-policy \
  --role-name SageMakerEnergyForecastingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Issue 2: Pipeline Creation Fails

**Cause:** SageMaker role cannot be assumed

**Solution:**
```bash
# Update trust policy to allow SageMaker
aws iam update-assume-role-policy \
  --role-name SageMakerEnergyForecastingRole \
  --policy-document file://sagemaker-trust-policy.json
```

### Issue 3: Lambda Cannot Connect to Redshift

**Cause:** Lambda not in VPC or security group misconfigured

**Solution:**
- Ensure Lambda is in same VPC as Redshift
- Update Redshift security group to allow inbound from Lambda security group
- Verify NAT Gateway for Lambda internet access (for Open-Meteo API)

### Issue 4: GitHub Actions Workflow Fails

**Cause:** Secrets not configured

**Solution:**
- Verify all required secrets are set in GitHub repository
- Check secret names match exactly (case-sensitive)
- Re-run workflow after adding missing secrets

---

## Best Practices

### Security
1. **Never commit AWS credentials** to repository
2. **Use IAM roles** with least privilege principle
3. **Enable MFA** on AWS accounts
4. **Rotate access keys** regularly (every 90 days)
5. **Use AWS Secrets Manager** for database passwords
6. **Enable CloudTrail** for audit logging

### Cost Optimization
1. **Delete endpoints after deployment** (handled automatically)
2. **Use Spot instances** for training (optional)
3. **Enable S3 lifecycle policies** to archive old data
4. **Use Redshift pause/resume** for non-production clusters
5. **Monitor costs** with AWS Cost Explorer and set up billing alerts

### Development Workflow
1. **Use feature branches** for development
2. **Test locally** before pushing to GitHub
3. **Use pull requests** for code review
4. **Run tests** before deployment
5. **Tag releases** for production deployments

### Monitoring
1. **Set up CloudWatch alarms** for Lambda errors
2. **Monitor SageMaker training jobs** for failures
3. **Track model performance** metrics over time
4. **Set up SNS notifications** for critical alerts
5. **Review logs regularly** for anomalies

---

## Next Steps

After completing this setup:

1. **Read the Troubleshooting Guide** (07_TROUBLESHOOTING_OPERATIONS.md)
2. **Review the KT Plan** (08_KNOWLEDGE_TRANSFER_PLAN.md)
3. **Run a full deployment** using GitHub Actions
4. **Monitor the first few forecast cycles** to ensure accuracy
5. **Set up dashboards** for ongoing monitoring
6. **Document any customizations** specific to your environment

---

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)

---

**Document Version:** 1.0
**Last Updated:** 2024-01-15
**Maintained By:** MLOps Team
