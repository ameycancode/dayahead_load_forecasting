# Knowledge Transfer Plan - Day-Ahead Energy Load Forecasting

## Table of Contents
1. [Overview](#overview)
2. [KT Timeline and Structure](#kt-timeline-and-structure)
3. [Week 1: Foundation and Overview](#week-1-foundation-and-overview)
4. [Week 2: Deep Dive into Components](#week-2-deep-dive-into-components)
5. [Week 3: Operations and Hands-On](#week-3-operations-and-hands-on)
6. [Week 4: Advanced Topics and Handoff](#week-4-advanced-topics-and-handoff)
7. [Assessment Checkpoints](#assessment-checkpoints)
8. [Reference Materials](#reference-materials)
9. [Support and Resources](#support-and-resources)

---

## 1. Overview

### Purpose

This Knowledge Transfer (KT) plan is designed to onboard new team members to the Day-Ahead Energy Load Forecasting MLOps project within 4 weeks. The plan covers:

- **Architecture understanding**: How the entire system works
- **Technical implementation**: Code, workflows, and infrastructure
- **Operations**: Daily monitoring, troubleshooting, and maintenance
- **Business context**: Why we forecast, customer segments, and use cases

### Target Audience

- Data Scientists transitioning to MLOps
- MLOps Engineers new to the project
- DevOps Engineers supporting ML infrastructure
- Technical Team Leads overseeing the project

### Prerequisites

Before starting this KT plan, ensure you have:

- [ ] Active AWS account access with appropriate permissions
- [ ] GitHub repository access
- [ ] Python 3.9+ development environment
- [ ] Basic understanding of:
  - Python programming
  - AWS services (SageMaker, Lambda, S3, Redshift)
  - Machine learning concepts
  - CI/CD with GitHub Actions
  - Time series forecasting basics

### Success Criteria

By the end of this KT, you should be able to:

1. **Explain** the architecture and data flow to stakeholders
2. **Deploy** a new model using GitHub Actions
3. **Monitor** daily operations and identify issues
4. **Troubleshoot** common problems independently
5. **Make code changes** to preprocessing, training, or prediction components
6. **Retrain models** when performance degrades
7. **Respond to incidents** following the runbook

---

## 2. KT Timeline and Structure

### 4-Week Timeline

| Week | Focus Area | Time Commitment | Deliverables |
|------|-----------|-----------------|--------------|
| **Week 1** | Foundation & Overview | 30-40 hours | Complete setup, understand architecture |
| **Week 2** | Deep Dive into Components | 35-45 hours | Understand code, run local tests |
| **Week 3** | Operations & Hands-On | 30-40 hours | Deploy changes, monitor operations |
| **Week 4** | Advanced Topics & Handoff | 25-35 hours | Independent project, final assessment |

**Total:** 120-160 hours (3-4 weeks of full-time work)

### Daily Structure

Each day follows a structured approach:

1. **Morning (9:00-12:00)**: Reading documentation, code review
2. **Afternoon (1:00-3:00)**: Hands-on exercises and experimentation
3. **Late Afternoon (3:00-5:00)**: Q&A session with departing team member
4. **Evening**: Optional self-study and additional practice

### KT Methods

- **Documentation Reading** (40%): Self-paced review of all documentation
- **Code Walkthrough** (20%): Guided sessions through key components
- **Hands-On Exercises** (30%): Practical tasks to reinforce learning
- **Q&A Sessions** (10%): Direct knowledge transfer from current team

---

## 3. Week 1: Foundation and Overview

### Objective

Understand the big picture: Why does this system exist? What problem does it solve? How does it work at a high level?

### Day 1: Project Context and Setup

**Morning (9:00-12:00): Business Context**

Reading:
- [ ] Read: `docs/01_PROJECT_OVERVIEW.md` (complete)
- [ ] Read: Business requirements document (if available)

Key concepts to understand:
- What is day-ahead load forecasting?
- Why do we need 6 separate models (RES/MEDCI/SMLCOM × SOLAR/NONSOLAR)?
- What is the cost optimization strategy (delete/recreate endpoints)?
- Who are the stakeholders and how do they use forecasts?

**Afternoon (1:00-3:00): Environment Setup**

Tasks:
- [ ] Clone the GitHub repository
- [ ] Set up local Python environment
- [ ] Install all dependencies (preprocessing, predictions, test)
- [ ] Configure AWS CLI with credentials
- [ ] Verify access to AWS services (S3, SageMaker, Redshift, Lambda)

Follow: `docs/06_CONFIGURATION_SETUP.md` (Section 4: Local Development Setup)

**Late Afternoon (3:00-5:00): Q&A Session**

Questions to ask:
1. What are the most common operational issues?
2. What parts of the system are most critical?
3. What changes are planned or needed?
4. Who are the key stakeholders and how often do they interact with the team?

**Homework:**
- Review AWS console: Browse through S3 bucket structure, Lambda functions, SageMaker endpoints
- Write a 1-page summary of your understanding of the project

---

### Day 2: Architecture Deep Dive

**Morning (9:00-12:00): System Architecture**

Reading:
- [ ] Re-read: `docs/01_PROJECT_OVERVIEW.md` (Architecture section)
- [ ] Review: AWS architecture diagram (if available)

Exercise:
- [ ] Draw the data flow diagram from raw data to predictions in Redshift
- [ ] List all AWS services used and their purpose
- [ ] Create a timeline of when each component runs (preprocessing → training → deployment → daily predictions)

**Afternoon (1:00-3:00): GitHub Actions Workflows**

Reading:
- [ ] Read: `docs/02_WORKFLOWS_GUIDE.md` (Section 1-3)
- [ ] Review: `.github/workflows/deploy.yml` (browse through file)

Tasks:
- [ ] Navigate to GitHub Actions tab in repository
- [ ] Review recent workflow runs (successful and failed)
- [ ] Identify the 12 jobs in deploy.yml workflow
- [ ] Understand the matrix strategy for 6 customer segments

**Late Afternoon (3:00-5:00): Walkthrough Session**

Live demonstration:
- Current team member triggers a workflow
- Walk through each job as it executes
- Show CloudWatch logs, S3 artifacts being created
- Show final Lambda function being deployed

**Homework:**
- Sketch a flowchart of the deploy.yml workflow
- List 3 questions about the workflow to discuss tomorrow

---

### Day 3: Data and Configuration

**Morning (9:00-12:00): Data Understanding**

Reading:
- [ ] Read: `docs/03_PIPELINE_COMPONENTS.md` (Section 1: Preprocessing Pipeline)

SQL Exercise:
- [ ] Connect to Redshift using psql or SQL client
- [ ] Run queries to explore input data:
  ```sql
  -- Explore historical load data
  SELECT * FROM edp_ods.caiso_sqmd LIMIT 100;

  -- Check data coverage
  SELECT
      customer_profile,
      customer_segment,
      MIN(usage_date) as earliest_date,
      MAX(usage_date) as latest_date,
      COUNT(*) as record_count
  FROM edp_ods.caiso_sqmd
  GROUP BY customer_profile, customer_segment;

  -- Explore forecast output
  SELECT * FROM edp_forecasting.dayahead_load_forecasts
  WHERE forecast_date >= CURRENT_DATE
  ORDER BY forecast_date, forecast_hour
  LIMIT 100;
  ```

- [ ] Understand data schema and columns
- [ ] Identify data quality patterns (missing data, outliers)

**Afternoon (1:00-3:00): Configuration Files**

Reading:
- [ ] Read: `docs/06_CONFIGURATION_SETUP.md` (Section 5: Configuration Files)
- [ ] Review: `configs/config.py` file

Exercise:
- [ ] Identify all customer profile configurations
- [ ] Understand rate group filters (NEM, SBP for solar customers)
- [ ] Review time period definitions (morning peak, solar period, etc.)
- [ ] Understand how configuration is used in preprocessing and training

Task:
- [ ] Run Python script to generate configuration for RES-SOLAR:
  ```python
  from configs.config import get_config_for_profile_segment

  config = get_config_for_profile_segment("RES", "SOLAR")
  print(f"S3 Bucket: {config['S3_BUCKET']}")
  print(f"Meter Threshold: {config['METER_THRESHOLD']}")
  print(f"Use Solar Features: {config['USE_SOLAR_FEATURES']}")
  print(f"Rate Group Filter: {config['RATE_GROUP_FILTER_CLAUSE']}")
  ```

**Late Afternoon (3:00-5:00): Configuration Walkthrough**

Discussion topics:
- Why separate configurations for each customer segment?
- How are environment variables passed from GitHub Actions to scripts?
- What happens if configuration is wrong?

**Homework:**
- Create a table mapping each customer segment to its key configuration parameters
- Write down any unclear configuration parameters

---

### Day 4: ML Pipeline Components

**Morning (9:00-12:00): Preprocessing Pipeline**

Reading:
- [ ] Read: `docs/03_PIPELINE_COMPONENTS.md` (Section 1: Preprocessing Pipeline)

Code Review:
- [ ] Review: `pipeline/preprocessing/preprocessing.py`
- [ ] Review: `pipeline/preprocessing/data_processing.py`
- [ ] Review: `pipeline/preprocessing/solar_features.py`
- [ ] Review: `pipeline/preprocessing/weather_features.py`

Key concepts:
- How is data extracted from Redshift?
- What features are engineered (lag features, rolling averages, calendar features)?
- How are solar-specific features added?
- How is weather data fetched from Open-Meteo API?
- How is data split into train/validation/test sets?

**Afternoon (1:00-3:00): Training Pipeline**

Reading:
- [ ] Read: `docs/03_PIPELINE_COMPONENTS.md` (Section 2: Training Pipeline)

Code Review:
- [ ] Review: `pipeline/training/model.py`
- [ ] Review: `pipeline/training/hyperparameter_optimization.py`
- [ ] Review: `pipeline/training/feature_selection.py`
- [ ] Review: `pipeline/training/evaluation.py`

Key concepts:
- What algorithm is used (XGBoost)?
- How are hyperparameters optimized (Bayesian optimization)?
- What metrics are calculated (RMSE, MAE, MAPE, R²)?
- How are segment-specific evaluation periods used?

**Late Afternoon (3:00-5:00): Code Walkthrough**

Live code walkthrough:
- Run preprocessing script locally with sample data
- Show feature engineering in action
- Explain cross-validation strategy
- Show model training output

**Homework:**
- Write a summary of the preprocessing steps
- List all features created (lag features, calendar features, weather features, solar features)

---

### Day 5: Predictions and Operations

**Morning (9:00-12:00): Lambda Predictions**

Reading:
- [ ] Read: `docs/04_PREDICTIONS_LAMBDA.md` (complete)

Code Review:
- [ ] Review: `predictions/lambda_function.py`
- [ ] Review: `predictions/endpoint_manager.py`
- [ ] Review: `predictions/data_preparation.py`
- [ ] Review: `predictions/weather_service.py`

Key concepts:
- How does Lambda recreate endpoints dynamically?
- How is tomorrow's data prepared for prediction?
- How are weather forecasts fetched?
- How are predictions written to Redshift?
- What is the execution flow (10 steps)?

**Afternoon (1:00-3:00): Monitoring and Troubleshooting**

Reading:
- [ ] Read: `docs/07_TROUBLESHOOTING_OPERATIONS.md` (Sections 1-3)

Tasks:
- [ ] Access CloudWatch Logs for Lambda functions
- [ ] Review recent execution logs
- [ ] Identify any errors or warnings
- [ ] Check Redshift for recent forecasts
- [ ] Verify EventBridge schedules are enabled

**Late Afternoon (3:00-5:00): Daily Operations Walkthrough**

Live demonstration:
- Show morning operational routine
- Check Lambda execution status
- Query Redshift for forecasts
- Review CloudWatch dashboard
- Explain what to do if forecasts are missing

**Homework:**
- Write down the daily operations checklist in your own words
- Create a troubleshooting decision tree for common issues

---

### Week 1 Assessment

**Knowledge Check:**

1. Explain the cost optimization strategy and calculate monthly savings
2. Draw the complete architecture diagram from memory
3. List all 6 customer segments and their key differences
4. Describe the data flow from Redshift to predictions
5. Explain the purpose of each GitHub Actions workflow job

**Practical Assessment:**

- [ ] Successfully run a query to check today's forecasts
- [ ] Navigate to S3 and find model artifacts for RES-SOLAR
- [ ] View CloudWatch logs for Lambda function
- [ ] Identify a recent workflow run and explain what it did

**Sign-off:**
- [ ] Departing team member confirms understanding of foundation
- [ ] New team member feels comfortable with basic concepts

---

## 4. Week 2: Deep Dive into Components

### Objective

Understand the code in detail and be able to make changes to preprocessing, training, and prediction components.

### Day 6: Preprocessing Deep Dive

**Morning (9:00-12:00): Data Processing**

Code Study:
- [ ] Detailed review of `pipeline/preprocessing/data_processing.py`
- [ ] Understand SQL query construction for data extraction
- [ ] Study outlier detection and handling logic
- [ ] Review data transformation functions

Exercise:
- [ ] Run preprocessing script locally with test data
- [ ] Modify outlier threshold and observe changes
- [ ] Add debug logging to understand data flow

**Afternoon (1:00-3:00): Feature Engineering**

Code Study:
- [ ] Detailed review of `pipeline/preprocessing/preprocessing.py`
- [ ] Understand lag feature creation
- [ ] Study rolling window calculations
- [ ] Review calendar feature engineering

Exercise:
- [ ] Create a new lag feature (e.g., 7-day lag)
- [ ] Test feature engineering with sample data
- [ ] Verify feature correlations with target

**Late Afternoon (3:00-5:00): Hands-On Modification**

Task:
- [ ] Make a small change to preprocessing (e.g., add a new calendar feature)
- [ ] Test the change locally
- [ ] Document the change and its impact

**Homework:**
- Write a detailed explanation of how rolling window features are calculated
- Identify potential improvements to feature engineering

---

### Day 7: Weather and Solar Features

**Morning (9:00-12:00): Weather Integration**

Code Study:
- [ ] Detailed review of `pipeline/preprocessing/weather_features.py`
- [ ] Understand Open-Meteo API integration
- [ ] Study weather feature transformations
- [ ] Review error handling for API failures

Exercise:
- [ ] Test Open-Meteo API directly with curl
- [ ] Fetch weather data for different locations
- [ ] Understand weather variable meanings (irradiance, cloudcover, etc.)

**Afternoon (1:00-3:00): Solar Features**

Code Study:
- [ ] Detailed review of `pipeline/preprocessing/solar_features.py`
- [ ] Understand duck curve calculation
- [ ] Study solar-specific feature creation
- [ ] Review how solar features differ by customer segment

Exercise:
- [ ] Plot duck curve pattern for RES-SOLAR customers
- [ ] Compare solar vs. non-solar load patterns
- [ ] Analyze peak hours for each segment

**Late Afternoon (3:00-5:00): Feature Analysis Session**

Data analysis:
- Load training data and analyze feature importance
- Compare feature distributions across segments
- Identify most predictive features for each segment

**Homework:**
- Create visualizations of load patterns by customer segment
- Write a report on solar vs. non-solar feature differences

---

### Day 8: Training Pipeline Deep Dive

**Morning (9:00-12:00): Model Training**

Code Study:
- [ ] Detailed review of `pipeline/training/model.py`
- [ ] Understand XGBoost model configuration
- [ ] Study cross-validation implementation
- [ ] Review model persistence and versioning

Exercise:
- [ ] Train a simple XGBoost model with sample data
- [ ] Experiment with different hyperparameters
- [ ] Understand impact of tree depth, learning rate, etc.

**Afternoon (1:00-3:00): Hyperparameter Optimization**

Code Study:
- [ ] Detailed review of `pipeline/training/hyperparameter_optimization.py`
- [ ] Understand Bayesian optimization approach
- [ ] Study search space definition
- [ ] Review scoring function (weighted RMSE)

Exercise:
- [ ] Run hyperparameter optimization on sample data
- [ ] Modify search space and observe differences
- [ ] Compare Bayesian vs. grid search

**Late Afternoon (3:00-5:00): Optimization Walkthrough**

Discussion:
- Why Bayesian optimization over grid search?
- How are segment-specific metric weights used?
- When should hyperparameters be re-tuned?

**Homework:**
- Document current hyperparameter ranges and their rationale
- Propose alternative optimization strategies

---

### Day 9: Model Evaluation and Deployment

**Morning (9:00-12:00): Evaluation Metrics**

Code Study:
- [ ] Detailed review of `pipeline/training/evaluation.py`
- [ ] Understand segment-specific evaluation periods
- [ ] Study metric weight calculations
- [ ] Review performance reporting

Exercise:
- [ ] Calculate evaluation metrics manually
- [ ] Compare metrics across different segments
- [ ] Understand why certain periods are weighted higher

**Afternoon (1:00-3:00): Model Deployment**

Reading:
- [ ] Read: `docs/05_DEPLOYMENT_SCRIPTS.md` (Section 4: Model Deployment Scripts)

Code Study:
- [ ] Review: `.github/scripts/deploy/register_model.py`
- [ ] Review: `.github/scripts/deploy/deploy_model.py`
- [ ] Review: `.github/scripts/deploy/validate_endpoint_health.py`

Key concepts:
- How are models registered in SageMaker Model Registry?
- What metadata is stored with each model?
- How are endpoints validated before production use?

**Late Afternoon (3:00-5:00): Deployment Walkthrough**

Live demonstration:
- Register a model manually
- Create an endpoint configuration
- Deploy an endpoint
- Test the endpoint with sample data
- Delete the endpoint (cost optimization)

**Homework:**
- Document the deployment process step-by-step
- Write test cases for endpoint validation

---

### Day 10: Lambda Function Deep Dive

**Morning (9:00-12:00): Lambda Architecture**

Code Study:
- [ ] Detailed review of `predictions/lambda_function.py` (main handler)
- [ ] Understand event structure and error handling
- [ ] Study logging and monitoring integration
- [ ] Review environment variable usage

Exercise:
- [ ] Test Lambda function locally using SAM CLI
- [ ] Modify Lambda timeout and memory settings
- [ ] Add custom logging for debugging

**Afternoon (1:00-3:00): Endpoint Management**

Code Study:
- [ ] Detailed review of `predictions/endpoint_manager.py`
- [ ] Understand endpoint recreation logic
- [ ] Study endpoint cleanup after predictions
- [ ] Review error handling for endpoint failures

Exercise:
- [ ] Create a test script to simulate endpoint recreation
- [ ] Measure time taken for endpoint creation
- [ ] Test error handling when model not found

**Late Afternoon (3:00-5:00): Lambda Testing Session**

Task:
- [ ] Manually invoke Lambda function for tomorrow's date
- [ ] Monitor CloudWatch logs in real-time
- [ ] Verify predictions written to Redshift
- [ ] Check endpoint is deleted after completion

**Homework:**
- Document Lambda execution flow with timing
- Identify potential optimization opportunities

---

### Week 2 Assessment

**Knowledge Check:**

1. Explain the complete preprocessing pipeline
2. Describe how weather features are created
3. Explain the hyperparameter optimization process
4. Describe the Lambda endpoint recreation strategy
5. List all evaluation metrics and their purposes

**Practical Assessment:**

- [ ] Make a code change to add a new feature
- [ ] Run preprocessing pipeline locally with the change
- [ ] Train a model with the new feature
- [ ] Test the trained model with sample data

**Code Review Exercise:**

- [ ] Review a provided code snippet and identify issues
- [ ] Suggest improvements for performance or maintainability

**Sign-off:**
- [ ] Departing team member confirms code understanding
- [ ] New team member can explain code to others

---

## 5. Week 3: Operations and Hands-On

### Objective

Gain operational experience by monitoring, troubleshooting, and deploying changes through GitHub Actions.

### Day 11: GitHub Actions Deployment

**Morning (9:00-12:00): Workflow Deep Dive**

Reading:
- [ ] Read: `docs/02_WORKFLOWS_GUIDE.md` (complete, second pass)

Code Study:
- [ ] Detailed review of `.github/workflows/deploy.yml`
- [ ] Understand job dependencies
- [ ] Study matrix strategy implementation
- [ ] Review secret and environment variable usage

Exercise:
- [ ] Create a test workflow that runs on push
- [ ] Add custom logging to understand job execution
- [ ] Test workflow with different matrix combinations

**Afternoon (1:00-3:00): Manual Deployment**

Task:
- [ ] Trigger deploy.yml workflow manually
- [ ] Monitor all 12 jobs as they execute
- [ ] Identify any failures or warnings
- [ ] Verify successful deployment of all 6 segments

Follow each job:
1. Setup infrastructure
2. Prepare configuration
3. Create pipelines (matrix: 6 segments)
4. Execute pipelines (matrix: 6 segments)
5. Register models
6. Validate deployments
7. Deploy endpoints
8. Create Lambda functions
9. Setup schedules
10. Validate deployment
11. Cleanup
12. Generate report

**Late Afternoon (3:00-5:00): Deployment Review**

Review:
- What succeeded and what failed?
- How long did each job take?
- What artifacts were created?
- What could be optimized?

**Homework:**
- Write a post-deployment checklist
- Document any issues encountered and how they were resolved

---

### Day 12: Monitoring and Alerting

**Morning (9:00-12:00): CloudWatch Setup**

Reading:
- [ ] Read: `docs/07_TROUBLESHOOTING_OPERATIONS.md` (Section 3: Monitoring)

Tasks:
- [ ] Set up CloudWatch alarms for Lambda errors
- [ ] Create alarm for Lambda duration
- [ ] Set up SNS topic for alerts
- [ ] Subscribe to alert notifications
- [ ] Create a CloudWatch dashboard

Reference: Follow alarm setup examples in troubleshooting guide

**Afternoon (1:00-3:00): Daily Operations Practice**

Exercise:
- [ ] Perform morning operational routine as documented
- [ ] Check Lambda execution status for all 6 segments
- [ ] Query Redshift for forecast completeness
- [ ] Review CloudWatch logs for errors
- [ ] Verify EventBridge schedules

Create a checklist:
```markdown
## Daily Operations Checklist

### Morning Routine (8:00 AM PT)
- [ ] Check all 6 Lambda functions executed successfully
- [ ] Verify forecasts in Redshift (24 records × 6 segments = 144 total)
- [ ] Review CloudWatch logs for errors
- [ ] Check for CloudWatch alarms
- [ ] Verify AWS costs are within budget

### Issues Found:
- [List any issues and actions taken]

### Notes:
- [Any observations or concerns]
```

**Late Afternoon (3:00-5:00): Operational Discussion**

Discussion topics:
- What are the most common operational issues?
- How do you prioritize alerts?
- When should you escalate to senior team members?
- What are the on-call responsibilities?

**Homework:**
- Refine your daily operations checklist
- Set up mobile notifications for critical alarms

---

### Day 13: Troubleshooting Practice

**Morning (9:00-12:00): Common Issues**

Reading:
- [ ] Read: `docs/07_TROUBLESHOOTING_OPERATIONS.md` (Section 4: Common Issues)

Exercise:
Simulate and resolve common issues:

1. **Lambda Timeout**
   - Manually increase Lambda timeout
   - Test with longer execution
   - Understand when timeouts occur

2. **Redshift Connection Failure**
   - Review security group settings
   - Test connection from Lambda
   - Verify VPC configuration

3. **Endpoint Creation Failure**
   - Check SageMaker endpoint limits
   - Review IAM permissions
   - Test endpoint creation manually

**Afternoon (1:00-3:00): Debugging Session**

Exercise:
- [ ] Review a failed Lambda execution from logs
- [ ] Identify the root cause
- [ ] Propose and implement a fix
- [ ] Test the fix

Use CloudWatch Logs Insights:
```sql
fields @timestamp, @message, @duration
| filter @message like /ERROR/
| sort @timestamp desc
| limit 50
```

**Late Afternoon (3:00-5:00): Incident Response Practice**

Scenario-based exercise:
- **Scenario 1**: All forecasts failed overnight. What do you do?
- **Scenario 2**: RMSE for RES-SOLAR increased by 50%. What actions do you take?
- **Scenario 3**: Weather API is down. How do you handle it?

Follow: `docs/07_TROUBLESHOOTING_OPERATIONS.md` (Section 9: Emergency Procedures)

**Homework:**
- Create a personal incident response cheat sheet
- Document 3 troubleshooting scenarios and solutions

---

### Day 14: Making Code Changes

**Morning (9:00-12:00): Feature Development**

Task: Implement a new feature
- [ ] Create a feature branch in Git
- [ ] Add a new feature to preprocessing (e.g., hour-of-week interaction)
- [ ] Update tests to cover the new feature
- [ ] Run tests locally

Example:
```python
# Add to preprocessing.py
def add_hour_of_week_feature(df):
    """Create hour-of-week feature (0-167)"""
    df['hour_of_week'] = df['usage_timestamp'].dt.dayofweek * 24 + df['usage_timestamp'].dt.hour
    return df
```

**Afternoon (1:00-3:00): Testing Changes**

Tasks:
- [ ] Write unit tests for new feature
- [ ] Run preprocessing pipeline with new feature locally
- [ ] Verify feature is created correctly
- [ ] Check feature correlation with target

```bash
# Run tests
pytest test-automation/test_preprocessing.py -v

# Run preprocessing locally
python pipeline/preprocessing/preprocessing.py \
  --profile RES \
  --segment SOLAR \
  --test-mode
```

**Late Afternoon (3:00-5:00): Code Review**

Task:
- [ ] Create a pull request with your changes
- [ ] Request code review from departing team member
- [ ] Address review comments
- [ ] Understand code review standards

**Homework:**
- Complete the pull request
- Document the feature and its expected impact

---

### Day 15: Deployment Practice

**Morning (9:00-12:00): Deploy Your Changes**

Task:
- [ ] Merge your feature branch to main (after approval)
- [ ] Trigger the deploy.yml workflow
- [ ] Monitor the deployment closely
- [ ] Take notes on any issues

**Afternoon (1:00-3:00): Validation**

Task:
- [ ] Verify new feature is in S3 scripts
- [ ] Check pipeline execution created new feature
- [ ] Review training logs for feature usage
- [ ] Compare model performance before/after

**Late Afternoon (3:00-5:00): Retrospective**

Discussion:
- What went well with the deployment?
- What challenges did you face?
- What would you do differently next time?
- What improvements can be made to the deployment process?

**Homework:**
- Write a deployment report documenting your experience
- Identify any gaps in documentation that confused you

---

### Week 3 Assessment

**Knowledge Check:**

1. Explain the daily operations routine
2. Describe how to troubleshoot a Lambda timeout
3. Walk through the GitHub Actions deployment process
4. Explain how to set up monitoring and alerts
5. Describe the incident response process

**Practical Assessment:**

- [ ] Successfully monitor daily operations for 3 consecutive days
- [ ] Identify and resolve a simulated issue
- [ ] Deploy a code change through GitHub Actions
- [ ] Set up a new CloudWatch alarm

**Operational Readiness:**

- [ ] Can perform daily operations independently
- [ ] Knows when to escalate issues
- [ ] Can deploy changes with supervision

**Sign-off:**
- [ ] Departing team member confirms operational competence
- [ ] New team member feels confident in day-to-day operations

---

## 6. Week 4: Advanced Topics and Handoff

### Objective

Master advanced topics, handle complex scenarios, and complete the knowledge transfer.

### Day 16: Model Retraining

**Morning (9:00-12:00): When to Retrain**

Reading:
- [ ] Read: `docs/07_TROUBLESHOOTING_OPERATIONS.md` (Section 8: Model Performance)

Discussion:
- How do you identify model performance degradation?
- What triggers a retraining decision?
- How often should models be retrained?
- What data is used for retraining?

Exercise:
- [ ] Query model performance metrics for past 30 days
- [ ] Calculate rolling RMSE
- [ ] Identify performance trends
- [ ] Make a retraining recommendation

```sql
-- Calculate rolling RMSE
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
    GROUP BY 1, 2, 3
)
SELECT
    customer_profile,
    customer_segment,
    AVG(rmse) as avg_rmse,
    STDDEV(rmse) as stddev_rmse
FROM forecast_actual
GROUP BY 1, 2
ORDER BY avg_rmse DESC;
```

**Afternoon (1:00-3:00): Retraining Execution**

Task:
- [ ] Trigger a retraining workflow for one segment
- [ ] Monitor the entire pipeline execution
- [ ] Compare new model performance with old model
- [ ] Decide whether to deploy new model

**Late Afternoon (3:00-5:00): Model Version Management**

Discussion:
- How are model versions tracked?
- How do you rollback to a previous model?
- What metadata should be stored with each model?
- How do you compare model versions?

**Homework:**
- Create a model retraining checklist
- Document model version management process

---

### Day 17: Performance Optimization

**Morning (9:00-12:00): Cost Optimization**

Reading:
- [ ] Read: `docs/01_PROJECT_OVERVIEW.md` (Section 2.3: Cost Optimization)

Exercise:
- [ ] Calculate current monthly AWS costs
- [ ] Break down costs by service (Lambda, SageMaker, Redshift, S3)
- [ ] Identify optimization opportunities
- [ ] Propose cost reduction strategies

Task:
- [ ] Review CloudWatch metrics for Lambda execution times
- [ ] Analyze whether Lambda memory can be reduced
- [ ] Check S3 storage usage and apply lifecycle policies
- [ ] Review Redshift query performance

**Afternoon (1:00-3:00): Performance Tuning**

Exercise:
- [ ] Profile preprocessing script for bottlenecks
- [ ] Optimize slow SQL queries in Redshift
- [ ] Test different Lambda memory configurations
- [ ] Benchmark model training with different instance types

**Late Afternoon (3:00-5:00): Optimization Review**

Discussion:
- What are the biggest cost drivers?
- What performance improvements can be made?
- What trade-offs exist between cost and performance?

**Homework:**
- Write a cost optimization report with recommendations
- Create a performance benchmarking dashboard

---

### Day 18: Data Quality and Validation

**Morning (9:00-12:00): Data Quality Framework**

Reading:
- [ ] Read: `docs/07_TROUBLESHOOTING_OPERATIONS.md` (Section 7: Data Quality)

Code Study:
- [ ] Review data validation framework in troubleshooting guide
- [ ] Understand validation checks (schema, completeness, duplicates, ranges)

Exercise:
- [ ] Implement the DataValidator class
- [ ] Run validation on recent data
- [ ] Identify any data quality issues
- [ ] Create a data quality report

**Afternoon (1:00-3:00): Data Quality Monitoring**

Task:
- [ ] Set up automated data quality checks
- [ ] Create alerts for data quality issues
- [ ] Build a data quality dashboard
- [ ] Document data quality standards

**Late Afternoon (3:00-5:00): Data Issue Response**

Scenario-based exercise:
- **Scenario 1**: Missing 6 hours of data for RES-SOLAR
- **Scenario 2**: Outliers detected in MEDCI-NONSOLAR data
- **Scenario 3**: Schema change in source Redshift table

**Homework:**
- Create a data quality incident response plan
- Document acceptable data quality thresholds

---

### Day 19: Advanced Troubleshooting

**Morning (9:00-12:00): Complex Issues**

Case studies:
- **Case 1**: Performance degradation after AWS service update
- **Case 2**: Intermittent Redshift connection failures
- **Case 3**: Weather API rate limiting during forecast burst

For each case:
- [ ] Analyze the problem
- [ ] Identify root cause
- [ ] Propose multiple solutions
- [ ] Implement the best solution
- [ ] Document the resolution

**Afternoon (1:00-3:00): Debugging Deep Dive**

Exercise:
- [ ] Review complex CloudWatch Logs Insights queries
- [ ] Use AWS X-Ray for Lambda function tracing (if available)
- [ ] Analyze SageMaker training job metrics
- [ ] Debug a multi-component failure

**Late Afternoon (3:00-5:00): War Room Simulation**

Simulation:
- All 6 forecast Lambda functions failed overnight
- Business stakeholders need forecasts immediately
- Current team member plays incident commander
- New team member leads the resolution

Follow: Emergency Procedures runbook

**Homework:**
- Write a post-incident report
- Identify preventive measures

---

### Day 20: Knowledge Transfer Completion

**Morning (9:00-12:00): Final Assessment**

**Written Assessment:**

1. Architecture Questions (30 minutes)
   - Draw the complete system architecture
   - Explain data flow from source to predictions
   - Describe the cost optimization strategy

2. Operational Questions (30 minutes)
   - Describe the daily operations routine
   - Explain how to troubleshoot common issues
   - Walk through the deployment process

3. Technical Questions (30 minutes)
   - Explain the preprocessing pipeline
   - Describe hyperparameter optimization
   - Explain Lambda endpoint recreation logic

**Afternoon (1:00-3:00): Practical Assessment**

**Hands-On Tasks:**

1. **Deployment Task (30 minutes)**
   - Deploy a change to one customer segment
   - Monitor and verify successful deployment

2. **Troubleshooting Task (30 minutes)**
   - Given a simulated issue, diagnose and resolve it
   - Document your troubleshooting steps

3. **Code Task (30 minutes)**
   - Make a small code improvement
   - Write tests and verify the change

**Late Afternoon (3:00-5:00): Final Handoff**

**Handoff Meeting:**

- [ ] Review all documentation created during KT
- [ ] Discuss any remaining questions or concerns
- [ ] Transfer ownership of operational responsibilities
- [ ] Exchange contact information for future questions
- [ ] Schedule 1-month and 3-month follow-up check-ins

**Transition Plan:**

- **Week 5**: New team member operates with light oversight
- **Week 6-8**: New team member operates independently, current team member available for questions
- **Month 2**: Current team member available for urgent issues only
- **Month 3+**: Full knowledge transfer complete

**Sign-off:**

- [ ] New team member confirms readiness to take over
- [ ] Departing team member confirms successful knowledge transfer
- [ ] Manager approves the transition
- [ ] Documentation is complete and up-to-date

---

## 7. Assessment Checkpoints

### Week 1 Checkpoint: Foundation

**Must achieve:**
- [ ] Understand business context and project goals
- [ ] Complete environment setup
- [ ] Navigate AWS services and GitHub repository
- [ ] Understand high-level architecture
- [ ] Explain the 6 customer segments

**If not achieved:** Extend Week 1 by 2-3 days

---

### Week 2 Checkpoint: Technical Depth

**Must achieve:**
- [ ] Understand preprocessing pipeline code
- [ ] Understand training pipeline code
- [ ] Understand Lambda prediction code
- [ ] Can make small code changes locally
- [ ] Can run tests locally

**If not achieved:** Extend Week 2 by 2-3 days

---

### Week 3 Checkpoint: Operational Readiness

**Must achieve:**
- [ ] Can perform daily operations independently
- [ ] Can monitor and identify issues
- [ ] Can troubleshoot common problems
- [ ] Can deploy changes with minimal supervision
- [ ] Understands incident response procedures

**If not achieved:** Extend Week 3 by 3-5 days (critical)

---

### Week 4 Checkpoint: Full Ownership

**Must achieve:**
- [ ] Can operate the system independently
- [ ] Can make code changes and deploy them
- [ ] Can handle incidents following runbooks
- [ ] Knows when to escalate issues
- [ ] Can explain the system to stakeholders

**If not achieved:** Extend handoff period

---

## 8. Reference Materials

### Essential Documentation

1. **Project Overview** (`docs/01_PROJECT_OVERVIEW.md`)
   - Read first for context
   - Reference architecture diagrams
   - Understand cost optimization

2. **Workflows Guide** (`docs/02_WORKFLOWS_GUIDE.md`)
   - Detailed GitHub Actions breakdown
   - Job dependencies and execution order
   - Matrix strategy explanation

3. **Pipeline Components** (`docs/03_PIPELINE_COMPONENTS.md`)
   - Preprocessing pipeline details
   - Training pipeline details
   - Feature engineering explanations

4. **Predictions & Lambda** (`docs/04_PREDICTIONS_LAMBDA.md`)
   - Lambda function architecture
   - Endpoint recreation strategy
   - Daily prediction execution flow

5. **Deployment Scripts** (`docs/05_DEPLOYMENT_SCRIPTS.md`)
   - All 23 deployment scripts
   - Script execution order
   - Script usage examples

6. **Configuration & Setup** (`docs/06_CONFIGURATION_SETUP.md`)
   - AWS account setup
   - GitHub repository configuration
   - Local development setup
   - Environment variables

7. **Troubleshooting & Operations** (`docs/07_TROUBLESHOOTING_OPERATIONS.md`)
   - Daily operations routine
   - Common issues and solutions
   - Debugging techniques
   - Emergency procedures

8. **Knowledge Transfer Plan** (`docs/08_KNOWLEDGE_TRANSFER_PLAN.md`)
   - This document
   - Structured 4-week onboarding

### External Resources

**AWS Documentation:**
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [Amazon Redshift Documentation](https://docs.aws.amazon.com/redshift/)

**Machine Learning:**
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)

**MLOps:**
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLOps Best Practices](https://ml-ops.org/)

---

## 9. Support and Resources

### During Knowledge Transfer

**Daily Q&A Sessions (3:00-5:00 PM):**
- Departing team member available for questions
- Review progress and clarify concepts
- Pair programming on complex topics

**Slack/Teams Channel:**
- Create a dedicated channel for KT
- Share resources and links
- Ask quick questions asynchronously

**Documentation:**
- All documentation in `docs/` directory
- Keep a running list of questions and answers
- Update documentation with lessons learned

### After Knowledge Transfer

**Follow-up Schedule:**

**Week 5-8:**
- Twice-weekly check-ins (30 minutes each)
- Available for urgent questions via Slack

**Month 2:**
- Weekly check-ins (30 minutes)
- Available for urgent issues

**Month 3:**
- Monthly check-in (1 hour)
- Review operational metrics and challenges

**Escalation Path:**

1. **First Level**: Review documentation
2. **Second Level**: Check troubleshooting guide
3. **Third Level**: Contact departing team member (if within 3 months)
4. **Fourth Level**: Escalate to manager or AWS support

**Knowledge Base:**
- Maintain a shared document of Q&A
- Document new issues and resolutions
- Contribute back to documentation

---

## Final Checklist

Before considering knowledge transfer complete, ensure:

**Documentation:**
- [ ] All 8 documentation files reviewed
- [ ] Personal notes and cheat sheets created
- [ ] Any unclear areas documented and resolved

**Access and Permissions:**
- [ ] AWS account access verified
- [ ] GitHub repository access confirmed
- [ ] Redshift database access working
- [ ] All necessary permissions granted

**Operational Readiness:**
- [ ] Can perform daily operations independently
- [ ] Can monitor system health
- [ ] Can troubleshoot common issues
- [ ] Can deploy changes via GitHub Actions
- [ ] Knows incident response procedures

**Technical Competence:**
- [ ] Understands preprocessing pipeline
- [ ] Understands training pipeline
- [ ] Understands Lambda predictions
- [ ] Can make code changes
- [ ] Can write tests

**Stakeholder Engagement:**
- [ ] Met key business stakeholders
- [ ] Understand reporting requirements
- [ ] Know escalation procedures
- [ ] Understand SLAs and expectations

**Confidence Level:**
- [ ] Feel confident operating the system (self-assessment: 7+/10)
- [ ] Comfortable making decisions independently
- [ ] Know when to ask for help

---

## Success Metrics

**1-Month Post-KT:**
- [ ] Zero production incidents requiring external help
- [ ] All daily forecasts successful (>98% success rate)
- [ ] Made at least one code improvement
- [ ] Responded to at least one operational issue independently

**3-Month Post-KT:**
- [ ] Successfully retrained models based on performance monitoring
- [ ] Implemented at least one cost optimization
- [ ] Trained another team member on a component
- [ ] Contributed improvements to documentation

---

**Congratulations on completing the Knowledge Transfer!**

You are now the owner of the Day-Ahead Energy Load Forecasting system. Remember:
- The documentation is your friend
- Don't hesitate to ask questions
- Continuous learning and improvement are key
- You've got this!

**Document Version:** 1.0
**Last Updated:** 2024-01-15
**Maintained By:** MLOps Team
