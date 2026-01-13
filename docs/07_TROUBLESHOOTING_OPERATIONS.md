# Troubleshooting and Operations Guide

## Table of Contents
1. [Operational Overview](#operational-overview)
2. [Daily Operations](#daily-operations)
3. [Monitoring and Alerts](#monitoring-and-alerts)
4. [Common Issues and Solutions](#common-issues-and-solutions)
5. [Debugging Techniques](#debugging-techniques)
6. [Performance Troubleshooting](#performance-troubleshooting)
7. [Data Quality Issues](#data-quality-issues)
8. [Model Performance Issues](#model-performance-issues)
9. [Emergency Procedures](#emergency-procedures)
10. [Maintenance Tasks](#maintenance-tasks)

---

## 1. Operational Overview

### System Health Checklist

Daily checks to ensure system is operating correctly:

- [ ] **Lambda Functions:** All 6 forecast Lambda functions executed successfully
- [ ] **Redshift:** Forecast data written to output table
- [ ] **Models:** Predictions are reasonable and within expected ranges
- [ ] **Costs:** Daily AWS costs are within budget
- [ ] **Logs:** No critical errors in CloudWatch logs
- [ ] **EventBridge:** Schedules are enabled and triggering correctly

### Key Operational Metrics

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| Daily Lambda Success Rate | 100% | < 95% | < 90% |
| Prediction Latency | < 5 minutes | > 10 minutes | > 15 minutes |
| RMSE (RES Solar) | < 1.5 | > 2.0 | > 3.0 |
| RMSE (RES NonSolar) | < 2.0 | > 2.5 | > 4.0 |
| Daily AWS Cost | $7-10 | > $15 | > $25 |
| Data Freshness | < 24 hours | > 48 hours | > 72 hours |

---

## 2. Daily Operations

### 2.1 Morning Routine (8:00 AM PT)

**Check Daily Forecasts Completion**

```bash
# 1. Check Lambda execution status
aws lambda list-functions --region us-west-2 | grep dayahead-forecasting

# 2. Check CloudWatch Logs for errors
LOG_GROUP="/aws/lambda/res-solar-dayahead-forecasting"
aws logs tail ${LOG_GROUP} --since 1h --filter-pattern "ERROR"

# 3. Query Redshift for today's forecasts
psql -h <redshift-endpoint> -U ds_service_user -d sdcp -c "
SELECT
    customer_profile,
    customer_segment,
    COUNT(*) as forecast_count,
    MIN(predicted_load) as min_load,
    MAX(predicted_load) as max_load,
    AVG(predicted_load) as avg_load
FROM edp_forecasting.dayahead_load_forecasts
WHERE forecast_date = CURRENT_DATE + 1
GROUP BY customer_profile, customer_segment
ORDER BY customer_profile, customer_segment;
"
```

**Expected Output:**
```
 customer_profile | customer_segment | forecast_count | min_load | max_load | avg_load
------------------+------------------+----------------+----------+----------+----------
 MEDCI           | NONSOLAR         |             24 |     2.15 |    12.34 |     6.78
 MEDCI           | SOLAR            |             24 |     1.85 |     9.87 |     5.23
 RES             | NONSOLAR         |             24 |     3.45 |    18.92 |     9.12
 RES             | SOLAR            |             24 |     2.21 |    11.45 |     5.89
 SMLCOM          | NONSOLAR         |             24 |     1.87 |     8.34 |     4.56
 SMLCOM          | SOLAR            |             24 |     1.45 |     6.78 |     3.89
(6 rows)
```

If any segment is missing, investigate immediately.

### 2.2 Weekly Routine (Monday 9:00 AM PT)

**Review Model Performance**

```sql
-- Check forecast accuracy for the past week
SELECT
    customer_profile,
    customer_segment,
    forecast_date,
    AVG(ABS(predicted_load - actual_load)) as mae,
    SQRT(AVG(POWER(predicted_load - actual_load, 2))) as rmse
FROM (
    SELECT
        f.customer_profile,
        f.customer_segment,
        f.forecast_date,
        f.forecast_hour,
        f.predicted_load,
        a.actual_load
    FROM edp_forecasting.dayahead_load_forecasts f
    LEFT JOIN edp_ods.caiso_sqmd a
        ON f.forecast_date = a.usage_date
        AND f.forecast_hour = EXTRACT(HOUR FROM a.usage_timestamp)
        AND f.customer_profile = a.customer_profile
        AND f.customer_segment = a.customer_segment
    WHERE f.forecast_date >= CURRENT_DATE - 7
        AND a.actual_load IS NOT NULL
) AS comparison
GROUP BY customer_profile, customer_segment, forecast_date
ORDER BY customer_profile, customer_segment, forecast_date;
```

**Review Cost Report**

```bash
# Get AWS cost for the past week
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --group-by Type=SERVICE,Key=SERVICE \
  --filter file://filter.json

# filter.json
{
  "Tags": {
    "Key": "Project",
    "Values": ["EnergyForecasting"]
  }
}
```

### 2.3 Monthly Routine (1st of Month)

**Model Retraining Review**

- Review if models need retraining based on performance drift
- Check for seasonal pattern changes
- Evaluate new data availability
- Plan retraining deployment if needed

**Infrastructure Review**

- Review Redshift storage usage and growth
- Clean up old S3 artifacts (> 90 days)
- Review IAM policies for any needed updates
- Check for AWS service updates or deprecations

---

## 3. Monitoring and Alerts

### 3.1 CloudWatch Alarms Setup

**Lambda Error Alarm**

```bash
# Create alarm for Lambda errors
aws cloudwatch put-metric-alarm \
  --alarm-name energy-forecasting-lambda-errors \
  --alarm-description "Alert when Lambda forecast function errors" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=res-solar-dayahead-forecasting \
  --alarm-actions arn:aws:sns:us-west-2:123456789012:mlops-alerts
```

**Lambda Duration Alarm**

```bash
# Create alarm for long-running Lambda
aws cloudwatch put-metric-alarm \
  --alarm-name energy-forecasting-lambda-duration \
  --alarm-description "Alert when Lambda takes too long" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 600000 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=res-solar-dayahead-forecasting \
  --alarm-actions arn:aws:sns:us-west-2:123456789012:mlops-alerts
```

**Cost Alarm**

```bash
# Create billing alarm
aws cloudwatch put-metric-alarm \
  --alarm-name energy-forecasting-daily-cost \
  --alarm-description "Alert when daily cost exceeds budget" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 15 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:us-west-2:123456789012:billing-alerts
```

### 3.2 SNS Topic Setup

```bash
# Create SNS topic for alerts
aws sns create-topic --name mlops-alerts

# Subscribe email
aws sns subscribe \
  --topic-arn arn:aws:sns:us-west-2:123456789012:mlops-alerts \
  --protocol email \
  --notification-endpoint mlops-team@company.com
```

### 3.3 CloudWatch Dashboard

Create a custom dashboard for monitoring:

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Invocations", {"stat": "Sum"}],
          [".", "Errors", {"stat": "Sum"}],
          [".", "Duration", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-west-2",
        "title": "Lambda Performance",
        "yAxis": {
          "left": {"min": 0}
        }
      }
    },
    {
      "type": "log",
      "properties": {
        "query": "SOURCE '/aws/lambda/res-solar-dayahead-forecasting' | fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20",
        "region": "us-west-2",
        "title": "Recent Lambda Errors"
      }
    }
  ]
}
```

---

## 4. Common Issues and Solutions

### Issue 1: Lambda Function Timeout

**Symptoms:**
- Lambda execution time exceeds 15 minutes
- CloudWatch logs show "Task timed out after 900.00 seconds"
- No forecasts written to Redshift

**Diagnosis:**
```bash
# Check Lambda execution duration
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=res-solar-dayahead-forecasting \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum
```

**Root Causes:**
1. **Endpoint creation taking too long** (cold start)
2. **Large amount of historical data** to process
3. **Slow Redshift queries**
4. **Weather API rate limiting**

**Solutions:**

```python
# Solution 1: Increase Lambda timeout (config in create_forecast_lambda.py)
lambda_config = {
    'Timeout': 900,  # Increase from 600 to 900 seconds
    'MemorySize': 512  # Increase memory for faster processing
}

# Solution 2: Optimize Redshift queries (add indexes)
CREATE INDEX idx_caiso_sqmd_date ON edp_ods.caiso_sqmd(usage_date);
CREATE INDEX idx_caiso_sqmd_profile ON edp_ods.caiso_sqmd(customer_profile, customer_segment);

# Solution 3: Cache weather data
# Already implemented in predictions/weather_service.py with requests-cache

# Solution 4: Pre-warm endpoints (keep them running during peak hours)
# Update delete_endpoint logic to skip deletion during business hours
```

### Issue 2: Redshift Connection Failure

**Symptoms:**
- Lambda logs show "could not connect to server"
- Error: "timeout expired"
- Forecasts not written to database

**Diagnosis:**
```bash
# Check Redshift cluster status
aws redshift describe-clusters \
  --cluster-identifier sdcp-dev-energy-forecasting \
  --query 'Clusters[0].ClusterStatus'

# Check VPC connectivity
aws ec2 describe-security-groups \
  --group-ids <lambda-security-group-id>

# Test connection from Lambda
aws lambda invoke \
  --function-name res-solar-dayahead-forecasting \
  --payload '{"test_connection": true}' \
  response.json
```

**Root Causes:**
1. **Redshift cluster paused or unavailable**
2. **Security group not allowing Lambda access**
3. **Lambda not in correct VPC/subnet**
4. **Network ACLs blocking traffic**

**Solutions:**

```bash
# Solution 1: Resume Redshift cluster
aws redshift resume-cluster \
  --cluster-identifier sdcp-dev-energy-forecasting

# Solution 2: Update security group
aws ec2 authorize-security-group-ingress \
  --group-id <redshift-security-group-id> \
  --protocol tcp \
  --port 5439 \
  --source-group <lambda-security-group-id>

# Solution 3: Move Lambda to correct VPC
aws lambda update-function-configuration \
  --function-name res-solar-dayahead-forecasting \
  --vpc-config SubnetIds=<subnet-id>,SecurityGroupIds=<sg-id>

# Solution 4: Check NAT Gateway
# Ensure Lambda subnet has route to NAT Gateway for internet access
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=<vpc-id>"
```

### Issue 3: Endpoint Creation Failure

**Symptoms:**
- Lambda logs show "Failed to create endpoint"
- Error: "ResourceLimitExceeded"
- Model predictions unavailable

**Diagnosis:**
```bash
# Check SageMaker endpoint limits
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-93D95A38  # Endpoint instances

# List current endpoints
aws sagemaker list-endpoints --region us-west-2

# Check endpoint status
aws sagemaker describe-endpoint \
  --endpoint-name res-solar-forecasting-endpoint
```

**Root Causes:**
1. **AWS account limits reached**
2. **Insufficient capacity in availability zone**
3. **Model artifact not found in S3**
4. **IAM role lacks permissions**

**Solutions:**

```bash
# Solution 1: Request limit increase
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-93D95A38 \
  --desired-value 10

# Solution 2: Delete old/unused endpoints
aws sagemaker list-endpoints --query 'Endpoints[?EndpointStatus==`Failed`].EndpointName' --output text | \
  xargs -I {} aws sagemaker delete-endpoint --endpoint-name {}

# Solution 3: Verify model in S3
aws s3 ls s3://${S3_BUCKET}/RES-SOLAR/models/ --recursive

# Solution 4: Check IAM role permissions
aws iam simulate-principal-policy \
  --policy-source-arn ${SAGEMAKER_ROLE_ARN} \
  --action-names sagemaker:CreateEndpoint \
  --resource-arns "*"
```

### Issue 4: Weather API Errors

**Symptoms:**
- Lambda logs show "Weather API request failed"
- Missing weather features in predictions
- Forecasts have NaN values

**Diagnosis:**
```bash
# Check Lambda logs for weather API errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/res-solar-dayahead-forecasting \
  --filter-pattern "WeatherService" \
  --start-time $(date -d '1 hour ago' +%s)000

# Test weather API directly
curl "https://api.open-meteo.com/v1/forecast?latitude=32.7157&longitude=-117.1611&hourly=temperature_2m&timezone=America/Los_Angeles"
```

**Root Causes:**
1. **Open-Meteo API downtime**
2. **Rate limiting** (unlikely for free tier)
3. **Network connectivity issues**
4. **Invalid coordinates or parameters**

**Solutions:**

```python
# Solution 1: Add retry logic (already in retry-requests library)
from retry_requests import retry
import openmeteo_requests

retry_session = retry(retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

# Solution 2: Use cached data as fallback
def get_weather_with_fallback(date, latitude, longitude):
    try:
        return fetch_weather_from_api(date, latitude, longitude)
    except Exception as e:
        logger.warning(f"Weather API failed: {e}, using historical average")
        return get_historical_weather_average(date, latitude, longitude)

# Solution 3: Pre-fetch weather data for next day
# Add scheduled Lambda to pre-fetch and cache weather at midnight

# Solution 4: Monitor API status
# Subscribe to Open-Meteo status page: https://status.open-meteo.com/
```

### Issue 5: Data Quality Problems

**Symptoms:**
- Forecasts have unrealistic values (negative, extremely high)
- RMSE suddenly increases
- Missing data for certain hours

**Diagnosis:**
```sql
-- Check for missing data
SELECT
    usage_date,
    customer_profile,
    customer_segment,
    COUNT(*) as hour_count
FROM edp_ods.caiso_sqmd
WHERE usage_date >= CURRENT_DATE - 7
GROUP BY usage_date, customer_profile, customer_segment
HAVING COUNT(*) < 24
ORDER BY usage_date DESC;

-- Check for outliers
SELECT
    usage_date,
    customer_profile,
    customer_segment,
    MIN(actual_load) as min_load,
    MAX(actual_load) as max_load,
    AVG(actual_load) as avg_load,
    STDDEV(actual_load) as stddev_load
FROM edp_ods.caiso_sqmd
WHERE usage_date >= CURRENT_DATE - 30
GROUP BY usage_date, customer_profile, customer_segment
HAVING MAX(actual_load) > AVG(actual_load) + 5 * STDDEV(actual_load)
ORDER BY usage_date DESC;
```

**Root Causes:**
1. **Missing data from source system**
2. **Data pipeline failures upstream**
3. **Schema changes in source tables**
4. **Data corruption or invalid values**

**Solutions:**

```python
# Solution 1: Add data validation in preprocessing
def validate_input_data(df):
    """Validate data quality before training"""
    # Check for required columns
    required_cols = ['usage_date', 'usage_timestamp', 'actual_load']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found: {null_counts}")

    # Check for unrealistic values
    if (df['actual_load'] < 0).any():
        raise ValueError("Negative load values detected")

    if (df['actual_load'] > 1000).any():  # Threshold depends on customer type
        logger.warning("Extremely high load values detected")

    # Check for complete time series
    expected_hours = 24 * len(df['usage_date'].unique())
    if len(df) < expected_hours:
        logger.warning(f"Missing hours: expected {expected_hours}, got {len(df)}")

    return True

# Solution 2: Implement data imputation
def impute_missing_values(df):
    """Impute missing values using forward fill and interpolation"""
    df = df.sort_values(['usage_date', 'usage_timestamp'])
    df['actual_load'] = df['actual_load'].fillna(method='ffill', limit=2)
    df['actual_load'] = df['actual_load'].interpolate(method='linear')
    return df

# Solution 3: Add data quality monitoring
def monitor_data_quality(df):
    """Calculate and log data quality metrics"""
    metrics = {
        'completeness': 1 - df['actual_load'].isnull().sum() / len(df),
        'uniqueness': len(df.drop_duplicates()) / len(df),
        'validity': ((df['actual_load'] >= 0) & (df['actual_load'] <= 1000)).sum() / len(df)
    }
    logger.info(f"Data quality metrics: {metrics}")
    return metrics
```

### Issue 6: Model Performance Degradation

**Symptoms:**
- Forecast accuracy (RMSE) increasing over time
- Business stakeholders reporting inaccurate forecasts
- Seasonal patterns not captured

**Diagnosis:**
```sql
-- Calculate rolling RMSE for past 30 days
WITH forecast_actual AS (
    SELECT
        f.forecast_date,
        f.customer_profile,
        f.customer_segment,
        SQRT(AVG(POWER(f.predicted_load - a.actual_load, 2))) as rmse
    FROM edp_forecasting.dayahead_load_forecasts f
    JOIN edp_ods.caiso_sqmd a
        ON f.forecast_date = a.usage_date
        AND f.forecast_hour = EXTRACT(HOUR FROM a.usage_timestamp)
    WHERE f.forecast_date >= CURRENT_DATE - 30
    GROUP BY f.forecast_date, f.customer_profile, f.customer_segment
)
SELECT
    customer_profile,
    customer_segment,
    AVG(rmse) as avg_rmse,
    STDDEV(rmse) as stddev_rmse,
    MIN(rmse) as min_rmse,
    MAX(rmse) as max_rmse
FROM forecast_actual
GROUP BY customer_profile, customer_segment
ORDER BY avg_rmse DESC;
```

**Root Causes:**
1. **Concept drift** - customer behavior patterns changing
2. **Data distribution shift** - new data patterns not in training set
3. **Seasonal patterns** - model trained on limited seasonal data
4. **Feature degradation** - weather or lag features losing predictive power

**Solutions:**

```bash
# Solution 1: Retrain model with recent data
# Trigger retraining workflow in GitHub Actions
gh workflow run deploy.yml \
  --ref main \
  --field customer_profile=RES \
  --field customer_segment=SOLAR

# Solution 2: Add more training data
# Extend training window from 3 years to 5 years
# Update configs/config.py
TRAINING_DATA_YEARS = 5  # Increased from 3

# Solution 3: Adjust hyperparameters for recent data
# Update hyperparameter_optimization.py to weight recent data more
sample_weights = np.exp(0.01 * np.arange(len(X_train)))  # Exponential weighting

# Solution 4: Implement ensemble models
# Combine multiple models trained on different time windows
def ensemble_predict(X, models, weights):
    predictions = [model.predict(X) for model in models]
    return np.average(predictions, axis=0, weights=weights)
```

### Issue 7: GitHub Actions Workflow Failure

**Symptoms:**
- Workflow run fails with errors
- Jobs stuck in pending state
- Deployment incomplete

**Diagnosis:**
```bash
# Check workflow run status
gh run list --workflow deploy.yml --limit 5

# View failed run logs
gh run view <run-id> --log-failed

# Check specific job logs
gh run view <run-id> --job <job-id> --log
```

**Root Causes:**
1. **GitHub Actions runner timeout**
2. **Concurrent job limits reached**
3. **Secrets expired or incorrect**
4. **AWS service quotas exceeded**
5. **Network connectivity issues**

**Solutions:**

```yaml
# Solution 1: Increase workflow timeout
jobs:
  preprocessing:
    timeout-minutes: 180  # Increase from 120

# Solution 2: Reduce parallelism
strategy:
  matrix:
    customer:
      - profile: RES
        segment: SOLAR
  max-parallel: 2  # Limit concurrent jobs

# Solution 3: Verify secrets
- name: Verify AWS Credentials
  run: |
    aws sts get-caller-identity
    if [ $? -ne 0 ]; then
      echo "AWS credentials invalid or expired"
      exit 1
    fi

# Solution 4: Add retry logic
- name: Create Pipeline
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 30
    max_attempts: 3
    command: python .github/scripts/deploy/create_pipeline.py

# Solution 5: Use GitHub Actions cache
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

---

## 5. Debugging Techniques

### 5.1 Lambda Function Debugging

**Enable Detailed Logging**

```python
# Add to Lambda function code
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Change from INFO to DEBUG

def lambda_handler(event, context):
    logger.debug(f"Event received: {json.dumps(event)}")
    logger.debug(f"Environment variables: {dict(os.environ)}")

    # Add timing information
    import time
    start_time = time.time()

    # Your code here

    elapsed_time = time.time() - start_time
    logger.info(f"Execution time: {elapsed_time:.2f} seconds")
```

**Test Lambda Locally**

```bash
# Install SAM CLI
pip install aws-sam-cli

# Create test event
cat > event.json << EOF
{
  "forecast_date": "2024-01-15",
  "customer_profile": "RES",
  "customer_segment": "SOLAR"
}
EOF

# Invoke locally
sam local invoke res-solar-dayahead-forecasting --event event.json
```

**CloudWatch Logs Insights Queries**

```sql
-- Find slow operations
fields @timestamp, @message, @duration
| filter @message like /Execution time/
| parse @message /Execution time: (?<execution_time>[\d.]+)/
| sort execution_time desc
| limit 20

-- Find memory usage
fields @timestamp, @message, @maxMemoryUsed, @memorySize
| filter @type = "REPORT"
| stats avg(@maxMemoryUsed), max(@maxMemoryUsed), avg(@memorySize)

-- Find specific errors
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 50
```

### 5.2 SageMaker Debugging

**View Processing Job Logs**

```bash
# List recent processing jobs
aws sagemaker list-processing-jobs \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 5

# Get job details
aws sagemaker describe-processing-job \
  --processing-job-name res-solar-load-preprocessing-20240115

# View CloudWatch logs
LOG_GROUP="/aws/sagemaker/ProcessingJobs"
aws logs tail ${LOG_GROUP} --follow --filter-pattern "res-solar"
```

**View Training Job Logs**

```bash
# List recent training jobs
aws sagemaker list-training-jobs \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 5

# Get job details
aws sagemaker describe-training-job \
  --training-job-name res-solar-training-20240115

# View CloudWatch logs
LOG_GROUP="/aws/sagemaker/TrainingJobs"
aws logs tail ${LOG_GROUP} --follow --filter-pattern "res-solar"
```

**Debug Script in SageMaker Studio**

```bash
# Create SageMaker Studio notebook
# Upload your preprocessing/training script
# Run interactively with sample data

# Example: Debug preprocessing script
import pandas as pd
import sys
sys.path.append('/opt/ml/processing/input')

from preprocessing import preprocess_data
from config import get_config_for_profile_segment

# Load sample data
df = pd.read_csv('/opt/ml/processing/input/sample_data.csv')

# Get config
config = get_config_for_profile_segment('RES', 'SOLAR')

# Run preprocessing with debug prints
processed_df = preprocess_data(df, config, debug=True)
```

### 5.3 Redshift Query Debugging

**Enable Query Logging**

```sql
-- Check query performance
SELECT
    query,
    TRIM(querytxt) as query_text,
    starttime,
    endtime,
    DATEDIFF(seconds, starttime, endtime) as duration_seconds,
    status
FROM stl_query
WHERE userid > 1
    AND starttime >= DATEADD(hour, -1, GETDATE())
ORDER BY duration_seconds DESC
LIMIT 20;

-- Check for locks
SELECT
    l.table_id,
    t.schemaname || '.' || t.tablename as table_name,
    l.transaction_id,
    l.pid,
    l.mode,
    l.granted
FROM pg_locks l
JOIN pg_class c ON l.relation = c.oid
JOIN pg_namespace n ON c.relnamespace = n.oid
JOIN stv_tbl_perm t ON l.table_id = t.id
WHERE t.schemaname = 'edp_forecasting'
ORDER BY l.granted, l.transaction_id;

-- Check table statistics
SELECT
    schemaname,
    tablename,
    size_in_mb,
    pct_used,
    unsorted
FROM svv_table_info
WHERE schemaname = 'edp_forecasting'
ORDER BY size_in_mb DESC;
```

**Optimize Slow Queries**

```sql
-- Add EXPLAIN to understand query plan
EXPLAIN
SELECT * FROM edp_forecasting.dayahead_load_forecasts
WHERE forecast_date >= CURRENT_DATE - 30;

-- Add indexes for frequently queried columns
CREATE INDEX idx_forecasts_date ON edp_forecasting.dayahead_load_forecasts(forecast_date);
CREATE INDEX idx_forecasts_profile ON edp_forecasting.dayahead_load_forecasts(customer_profile, customer_segment);

-- Analyze table to update statistics
ANALYZE edp_forecasting.dayahead_load_forecasts;

-- Vacuum table to reclaim space
VACUUM edp_forecasting.dayahead_load_forecasts;
```

---

## 6. Performance Troubleshooting

### 6.1 Lambda Performance Optimization

**Cold Start Optimization**

```python
# Move imports outside handler
import boto3
import pandas as pd
import numpy as np

# Initialize clients globally
sagemaker_client = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

# Cache configuration
CONFIG_CACHE = {}

def lambda_handler(event, context):
    # Handler code here
    pass
```

**Memory Optimization**

```bash
# Test different memory configurations
for memory in 256 512 1024 2048 3008; do
    aws lambda update-function-configuration \
      --function-name res-solar-dayahead-forecasting \
      --memory-size $memory

    # Run test
    aws lambda invoke \
      --function-name res-solar-dayahead-forecasting \
      --payload file://test-event.json \
      response.json

    # Check duration from CloudWatch
    aws logs filter-log-events \
      --log-group-name /aws/lambda/res-solar-dayahead-forecasting \
      --filter-pattern "REPORT" \
      --limit 1
done
```

### 6.2 SageMaker Training Performance

**Instance Type Selection**

```python
# Test different instance types
instance_types = [
    'ml.m5.large',    # 2 vCPU, 8 GB RAM - $0.115/hr
    'ml.m5.xlarge',   # 4 vCPU, 16 GB RAM - $0.23/hr
    'ml.c5.xlarge',   # 4 vCPU, 8 GB RAM - $0.204/hr (compute optimized)
]

for instance_type in instance_types:
    training_job = estimator.fit(
        inputs={'train': train_data, 'validation': val_data},
        instance_type=instance_type,
        instance_count=1
    )

    # Compare training time vs. cost
    print(f"Instance: {instance_type}")
    print(f"Training time: {training_job.training_time}")
    print(f"Cost: {calculate_cost(instance_type, training_job.training_time)}")
```

**Data Loading Optimization**

```python
# Use parquet instead of CSV for faster I/O
df.to_parquet('train.parquet', compression='snappy')

# Use S3 Transfer Acceleration
s3_client = boto3.client('s3', config=Config(
    s3={'use_accelerate_endpoint': True}
))

# Load data in chunks for large datasets
chunk_size = 100000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### 6.3 Redshift Query Performance

**Distribution and Sort Keys**

```sql
-- Analyze query patterns
SELECT
    query,
    TRIM(querytxt) as query_text,
    COUNT(*) as execution_count,
    AVG(DATEDIFF(seconds, starttime, endtime)) as avg_duration
FROM stl_query
WHERE userid > 1
    AND starttime >= DATEADD(day, -7, GETDATE())
    AND querytxt LIKE '%dayahead_load_forecasts%'
GROUP BY query, querytxt
ORDER BY execution_count DESC
LIMIT 10;

-- Optimize table with distribution and sort keys
CREATE TABLE edp_forecasting.dayahead_load_forecasts_optimized (
    forecast_date DATE NOT NULL,
    forecast_hour INTEGER NOT NULL,
    forecast_timestamp TIMESTAMP NOT NULL,
    customer_profile VARCHAR(50) NOT NULL,
    customer_segment VARCHAR(50) NOT NULL,
    predicted_load DECIMAL(18, 6) NOT NULL,
    model_version VARCHAR(100),
    created_at TIMESTAMP DEFAULT GETDATE()
)
DISTKEY(forecast_date)  -- Distribute by most frequently joined column
SORTKEY(forecast_date, customer_profile, customer_segment);  -- Sort by query patterns

-- Copy data to optimized table
INSERT INTO edp_forecasting.dayahead_load_forecasts_optimized
SELECT * FROM edp_forecasting.dayahead_load_forecasts;

-- Rename tables
ALTER TABLE edp_forecasting.dayahead_load_forecasts RENAME TO dayahead_load_forecasts_old;
ALTER TABLE edp_forecasting.dayahead_load_forecasts_optimized RENAME TO dayahead_load_forecasts;
```

---

## 7. Data Quality Issues

### 7.1 Data Validation Framework

```python
# data_validator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataValidator:
    """Comprehensive data validation for energy forecasting"""

    def __init__(self, config: Dict):
        self.config = config
        self.validation_results = []

    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Run all validation checks"""
        checks = [
            self.check_schema,
            self.check_completeness,
            self.check_duplicates,
            self.check_value_ranges,
            self.check_time_continuity,
            self.check_statistical_anomalies
        ]

        all_passed = True
        messages = []

        for check in checks:
            passed, message = check(df)
            if not passed:
                all_passed = False
            messages.append(message)

        return all_passed, messages

    def check_schema(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Verify required columns exist"""
        required_columns = [
            'usage_date', 'usage_timestamp', 'actual_load',
            'customer_profile', 'customer_segment'
        ]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            return False, f"Missing required columns: {missing}"
        return True, "Schema validation passed"

    def check_completeness(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for missing values"""
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df) * 100).round(2)

        critical_nulls = null_pct[null_pct > 5]  # More than 5% missing

        if len(critical_nulls) > 0:
            return False, f"High null percentage: {critical_nulls.to_dict()}"
        return True, f"Completeness check passed (max null: {null_pct.max()}%)"

    def check_duplicates(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for duplicate records"""
        key_columns = ['usage_date', 'usage_timestamp', 'customer_profile', 'customer_segment']
        duplicates = df[df.duplicated(subset=key_columns, keep=False)]

        if len(duplicates) > 0:
            return False, f"Found {len(duplicates)} duplicate records"
        return True, "No duplicates found"

    def check_value_ranges(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for values outside expected ranges"""
        issues = []

        # Check actual_load
        if (df['actual_load'] < 0).any():
            issues.append(f"{(df['actual_load'] < 0).sum()} negative load values")

        # Check for unrealistically high values (depends on customer type)
        max_threshold = self.config.get('MAX_LOAD_THRESHOLD', 1000)
        if (df['actual_load'] > max_threshold).any():
            issues.append(f"{(df['actual_load'] > max_threshold).sum()} values exceed {max_threshold}")

        if issues:
            return False, f"Value range issues: {', '.join(issues)}"
        return True, "All values within expected ranges"

    def check_time_continuity(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for gaps in time series"""
        df_sorted = df.sort_values(['customer_profile', 'customer_segment', 'usage_timestamp'])

        gaps = []
        for (profile, segment), group in df_sorted.groupby(['customer_profile', 'customer_segment']):
            time_diffs = group['usage_timestamp'].diff()
            expected_diff = pd.Timedelta(hours=1)

            # Find gaps larger than expected
            large_gaps = time_diffs[time_diffs > expected_diff * 1.5]
            if len(large_gaps) > 0:
                gaps.append(f"{profile}-{segment}: {len(large_gaps)} gaps")

        if gaps:
            return False, f"Time continuity issues: {', '.join(gaps)}"
        return True, "Time series is continuous"

    def check_statistical_anomalies(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for statistical outliers"""
        anomalies = []

        for (profile, segment), group in df.groupby(['customer_profile', 'customer_segment']):
            Q1 = group['actual_load'].quantile(0.25)
            Q3 = group['actual_load'].quantile(0.75)
            IQR = Q3 - Q1

            outliers = group[
                (group['actual_load'] < Q1 - 3 * IQR) |
                (group['actual_load'] > Q3 + 3 * IQR)
            ]

            if len(outliers) > len(group) * 0.05:  # More than 5% outliers
                anomalies.append(f"{profile}-{segment}: {len(outliers)} outliers")

        if anomalies:
            return False, f"Statistical anomalies detected: {', '.join(anomalies)}"
        return True, "No significant statistical anomalies"

# Usage in preprocessing pipeline
validator = DataValidator(config)
is_valid, messages = validator.validate_all(df)

for message in messages:
    logger.info(message)

if not is_valid:
    logger.error("Data validation failed!")
    # Decide: raise exception or continue with data cleaning
```

---

## 8. Model Performance Issues

### 8.1 Model Monitoring Dashboard

```sql
-- Create materialized view for model performance metrics
CREATE MATERIALIZED VIEW edp_forecasting.mv_model_performance AS
SELECT
    DATE_TRUNC('day', f.forecast_date) as metric_date,
    f.customer_profile,
    f.customer_segment,
    f.model_version,

    -- Overall metrics
    COUNT(*) as prediction_count,
    AVG(f.predicted_load) as avg_predicted,
    AVG(a.actual_load) as avg_actual,

    -- Error metrics
    AVG(ABS(f.predicted_load - a.actual_load)) as mae,
    SQRT(AVG(POWER(f.predicted_load - a.actual_load, 2))) as rmse,
    AVG(ABS(f.predicted_load - a.actual_load) / NULLIF(a.actual_load, 0)) * 100 as mape,

    -- Bias metrics
    AVG(f.predicted_load - a.actual_load) as bias,

    -- R-squared
    1 - (SUM(POWER(a.actual_load - f.predicted_load, 2)) /
         SUM(POWER(a.actual_load - AVG(a.actual_load), 2))) as r_squared

FROM edp_forecasting.dayahead_load_forecasts f
JOIN edp_ods.caiso_sqmd a
    ON f.forecast_date = a.usage_date
    AND f.forecast_hour = EXTRACT(HOUR FROM a.usage_timestamp)
    AND f.customer_profile = a.customer_profile
    AND f.customer_segment = a.customer_segment
WHERE f.forecast_date >= CURRENT_DATE - 90
GROUP BY 1, 2, 3, 4;

-- Refresh materialized view daily
-- Add to cron job or Lambda schedule
REFRESH MATERIALIZED VIEW edp_forecasting.mv_model_performance;
```

### 8.2 Performance Alerting

```python
# model_performance_monitor.py
import boto3
import pandas as pd
from datetime import datetime, timedelta

def check_model_performance():
    """Monitor model performance and send alerts if degraded"""

    # Query performance metrics
    query = """
    SELECT
        customer_profile,
        customer_segment,
        AVG(rmse) as avg_rmse,
        AVG(mape) as avg_mape
    FROM edp_forecasting.mv_model_performance
    WHERE metric_date >= CURRENT_DATE - 7
    GROUP BY customer_profile, customer_segment
    """

    df = execute_redshift_query(query)

    # Define thresholds
    thresholds = {
        'RES_SOLAR': {'rmse': 2.0, 'mape': 15},
        'RES_NONSOLAR': {'rmse': 2.5, 'mape': 18},
        'MEDCI_SOLAR': {'rmse': 1.5, 'mape': 12},
        'MEDCI_NONSOLAR': {'rmse': 1.8, 'mape': 15},
        'SMLCOM_SOLAR': {'rmse': 1.2, 'mape': 10},
        'SMLCOM_NONSOLAR': {'rmse': 1.5, 'mape': 12}
    }

    # Check for threshold violations
    alerts = []
    for _, row in df.iterrows():
        segment_key = f"{row['customer_profile']}_{row['customer_segment']}"
        threshold = thresholds.get(segment_key, {})

        if row['avg_rmse'] > threshold.get('rmse', float('inf')):
            alerts.append({
                'segment': segment_key,
                'metric': 'RMSE',
                'value': row['avg_rmse'],
                'threshold': threshold['rmse']
            })

        if row['avg_mape'] > threshold.get('mape', float('inf')):
            alerts.append({
                'segment': segment_key,
                'metric': 'MAPE',
                'value': row['avg_mape'],
                'threshold': threshold['mape']
            })

    # Send alerts if any violations
    if alerts:
        send_performance_alert(alerts)

    return alerts

def send_performance_alert(alerts):
    """Send SNS notification for performance degradation"""
    sns = boto3.client('sns')

    message = "Model Performance Alert\n\n"
    for alert in alerts:
        message += f"Segment: {alert['segment']}\n"
        message += f"Metric: {alert['metric']}\n"
        message += f"Current: {alert['value']:.2f}\n"
        message += f"Threshold: {alert['threshold']:.2f}\n\n"

    sns.publish(
        TopicArn='arn:aws:sns:us-west-2:123456789012:model-performance-alerts',
        Subject='Energy Forecasting Model Performance Degradation',
        Message=message
    )

# Schedule this to run daily
if __name__ == '__main__':
    alerts = check_model_performance()
    print(f"Found {len(alerts)} performance alerts")
```

---

## 9. Emergency Procedures

### 9.1 Emergency Contacts

| Role | Name | Phone | Email | Availability |
|------|------|-------|-------|--------------|
| MLOps Lead | [Name] | [Phone] | [Email] | 24/7 |
| Data Engineer | [Name] | [Phone] | [Email] | Business hours |
| AWS Support | AWS | N/A | Support Portal | 24/7 (Premium) |
| Business Owner | [Name] | [Phone] | [Email] | Business hours |

### 9.2 Incident Response Runbook

**Severity Levels**

- **P1 (Critical)**: All forecasts failing, production outage, data loss
- **P2 (High)**: Multiple segments failing, significant accuracy degradation
- **P3 (Medium)**: Single segment failing, moderate accuracy issues
- **P4 (Low)**: Minor issues, no business impact

**P1 Incident Response**

```bash
# 1. Assess the situation
aws lambda list-functions | grep dayahead-forecasting
aws sagemaker list-endpoints --status-equals Failed

# 2. Check recent deployments
git log --oneline -10
gh run list --workflow deploy.yml --limit 5

# 3. Rollback if recent deployment caused issue
git revert <commit-hash>
git push origin main

# 4. Disable failing Lambda functions
for func in $(aws lambda list-functions --query 'Functions[?contains(FunctionName, `dayahead-forecasting`)].FunctionName' --output text); do
    # Note: EventBridge schedules will continue to trigger but won't execute
    echo "Disabling $func"
    aws lambda put-function-concurrency --function-name $func --reserved-concurrent-executions 0
done

# 5. Use backup/historical forecasts
python scripts/use_historical_forecast.py --date tomorrow

# 6. Notify stakeholders
python scripts/send_incident_notification.py --severity P1 --message "Forecasting system down, using historical data"

# 7. Start incident call
# Follow incident management process
```

**P2 Incident Response**

```bash
# 1. Identify failing segments
python scripts/check_forecast_status.py --date today

# 2. Attempt restart of affected Lambda functions
aws lambda update-function-configuration \
  --function-name res-solar-dayahead-forecasting \
  --environment Variables={FORCE_REFRESH=true}

# 3. Check logs for root cause
aws logs tail /aws/lambda/res-solar-dayahead-forecasting --since 1h

# 4. If data issue, use backup
python scripts/use_backup_data.py --segment RES_SOLAR

# 5. Monitor closely for next 24 hours
```

### 9.3 Data Loss Recovery

```sql
-- Backup forecasts daily
CREATE TABLE edp_forecasting.dayahead_load_forecasts_backup AS
SELECT * FROM edp_forecasting.dayahead_load_forecasts
WHERE forecast_date >= CURRENT_DATE - 30;

-- Restore from backup
INSERT INTO edp_forecasting.dayahead_load_forecasts
SELECT * FROM edp_forecasting.dayahead_load_forecasts_backup
WHERE forecast_date = '2024-01-15';

-- Point-in-time recovery from S3
aws s3 cp s3://${S3_BUCKET}/forecasts/RES-SOLAR/2024-01-15/ . --recursive
python scripts/restore_forecasts_from_s3.py --date 2024-01-15
```

---

## 10. Maintenance Tasks

### 10.1 Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

# 1. Clean up old CloudWatch logs (> 30 days)
for log_group in $(aws logs describe-log-groups --query 'logGroups[*].logGroupName' --output text | grep energy-forecasting); do
    aws logs put-retention-policy \
        --log-group-name $log_group \
        --retention-in-days 30
done

# 2. Clean up old S3 artifacts (> 90 days)
aws s3 ls s3://${S3_BUCKET}/ --recursive | \
    awk '{if ($1 < "'$(date -d '90 days ago' +%Y-%m-%d)'") print $4}' | \
    xargs -I {} aws s3 rm s3://${S3_BUCKET}/{}

# 3. Vacuum Redshift tables
psql -h ${REDSHIFT_ENDPOINT} -U admin -d sdcp -c "
VACUUM DELETE ONLY edp_forecasting.dayahead_load_forecasts;
ANALYZE edp_forecasting.dayahead_load_forecasts;
"

# 4. Check for unused endpoints
aws sagemaker list-endpoints --query 'Endpoints[?EndpointStatus==`OutOfService`].EndpointName' --output text | \
    xargs -I {} aws sagemaker delete-endpoint --endpoint-name {}

# 5. Review and rotate IAM access keys (if > 60 days old)
python scripts/check_access_key_age.py --rotate-if-older-than 60
```

### 10.2 Monthly Maintenance

```bash
#!/bin/bash
# monthly_maintenance.sh

# 1. Review model performance trends
python scripts/generate_monthly_report.py --month $(date +%Y-%m)

# 2. Check for AWS service updates
aws health describe-events --filter eventTypeCategories=scheduledChange

# 3. Review and update dependencies
pip list --outdated
npm outdated  # if using Node.js for any tooling

# 4. Backup critical configuration
aws s3 sync configs/ s3://${S3_BUCKET}/backups/configs/$(date +%Y-%m-%d)/

# 5. Review costs and optimize
python scripts/cost_optimization_recommendations.py

# 6. Update documentation if needed
git diff --name-only HEAD~30 HEAD | grep -E '\.(py|yml|md)$'
```

### 10.3 Quarterly Maintenance

```bash
#!/bin/bash
# quarterly_maintenance.sh

# 1. Model retraining evaluation
python scripts/evaluate_retrain_necessity.py --lookback-months 3

# 2. Security audit
python scripts/security_audit.py
aws iam get-account-authorization-details > iam_audit_$(date +%Y%m%d).json

# 3. Disaster recovery drill
python scripts/test_disaster_recovery.py

# 4. Review and update alerting thresholds
python scripts/review_alert_thresholds.py --adjust-based-on-history

# 5. Conduct architecture review
# Schedule meeting with team to review:
# - System performance
# - Cost optimization opportunities
# - New AWS services to leverage
# - Technical debt to address
```

---

## Summary

This troubleshooting and operations guide provides comprehensive coverage of:

1. **Daily Operations**: Routine checks and monitoring procedures
2. **Monitoring**: CloudWatch alarms, SNS alerts, and dashboards
3. **Common Issues**: Detailed troubleshooting for frequent problems
4. **Debugging**: Techniques for Lambda, SageMaker, and Redshift
5. **Performance**: Optimization strategies for all components
6. **Data Quality**: Validation framework and quality checks
7. **Model Performance**: Monitoring and alerting for model drift
8. **Emergency Procedures**: Incident response and recovery
9. **Maintenance**: Regular tasks to keep system healthy

**Key Takeaways:**

- **Monitor proactively** with CloudWatch alarms and daily checks
- **Document all incidents** to build institutional knowledge
- **Automate maintenance** tasks to reduce operational burden
- **Test disaster recovery** procedures regularly
- **Keep stakeholders informed** with clear communication

For additional support, refer to:
- **Project Overview**: docs/01_PROJECT_OVERVIEW.md
- **Configuration Guide**: docs/06_CONFIGURATION_SETUP.md
- **KT Plan**: docs/08_KNOWLEDGE_TRANSFER_PLAN.md

---

**Document Version:** 1.0
**Last Updated:** 2024-01-15
**Maintained By:** MLOps Team
