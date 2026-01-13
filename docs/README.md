# Day-Ahead Energy Load Forecasting - Documentation

## Welcome

This directory contains comprehensive documentation for the Day-Ahead Energy Load Forecasting MLOps project. The documentation is designed to onboard new team members and serve as a reference for operational and development tasks.

---

## Documentation Structure

The documentation is organized into 8 comprehensive guides, designed to be read in sequence for onboarding:

### 1. [Project Overview](01_PROJECT_OVERVIEW.md)
**Start here for context and high-level understanding**

- System architecture and design principles
- Customer segmentation strategy (6 models: RES/MEDCI/SMLCOM × SOLAR/NONSOLAR)
- Cost optimization strategy ($2,016/month savings via endpoint deletion)
- Technology stack and AWS services
- Quick start guide

**Read this if you:**
- Are new to the project
- Need to explain the system to stakeholders
- Want to understand the business context

---

### 2. [Workflows Guide](02_WORKFLOWS_GUIDE.md)
**Deep dive into GitHub Actions CI/CD workflows**

- Complete breakdown of `deploy.yml` workflow (12 jobs)
- Historical forecasting workflow analysis
- Matrix strategy for parallel deployments
- Job dependencies and execution order
- Environment variables and secrets

**Read this if you:**
- Need to understand the deployment process
- Are troubleshooting workflow failures
- Want to modify the CI/CD pipeline

---

### 3. [Pipeline Components](03_PIPELINE_COMPONENTS.md)
**Detailed code documentation for ML pipelines**

- **Preprocessing Pipeline**: Data extraction, feature engineering, solar features, weather integration
- **Training Pipeline**: XGBoost model, hyperparameter optimization, cross-validation, evaluation
- **Code Examples**: Key functions and their usage

**Read this if you:**
- Need to understand the ML code
- Are making changes to preprocessing or training
- Want to add new features or improve model performance

---

### 4. [Predictions and Lambda](04_PREDICTIONS_LAMBDA.md)
**Lambda function architecture for daily forecasting**

- Lambda function implementation details
- Endpoint recreation manager (cost optimization)
- Data preparation and feature engineering for predictions
- Weather service integration
- 10-step execution flow
- Environment variables reference (50+ variables)

**Read this if you:**
- Are troubleshooting prediction failures
- Need to understand Lambda execution
- Want to modify the forecasting logic

---

### 5. [Deployment Scripts Reference](05_DEPLOYMENT_SCRIPTS.md)
**Complete reference for all 23 deployment scripts**

- **Infrastructure Setup** (3 scripts): Redshift, Athena, permissions
- **Pipeline Management** (7 scripts): Create, execute, update pipelines
- **Model Deployment** (4 scripts): Register, deploy, validate models
- **Lambda Management** (4 scripts): Create, test, schedule Lambda functions
- **Validation & Testing** (3 scripts): Integration tests, model analysis
- **Reporting** (2 scripts): Generate deployment reports

**Read this if you:**
- Need to understand what each script does
- Are debugging deployment issues
- Want to run scripts manually

---

### 6. [Configuration and Setup Guide](06_CONFIGURATION_SETUP.md)
**Complete setup instructions from scratch**

- Prerequisites and AWS account setup
- IAM roles and permissions
- GitHub repository configuration (secrets, variables)
- Local development environment setup
- Configuration files explained (`configs/config.py`)
- Environment variables reference
- Infrastructure setup procedures
- Verification and testing

**Read this if you:**
- Are setting up a new environment
- Need to configure AWS resources
- Are troubleshooting access/permission issues

---

### 7. [Troubleshooting and Operations Guide](07_TROUBLESHOOTING_OPERATIONS.md)
**Operational procedures and problem resolution**

- Daily, weekly, and monthly operations checklists
- Monitoring and alerting setup (CloudWatch, SNS)
- Common issues and solutions (Lambda timeouts, Redshift connections, endpoint failures)
- Debugging techniques (Lambda, SageMaker, Redshift)
- Performance troubleshooting and optimization
- Data quality validation framework
- Model performance monitoring
- Emergency procedures and incident response
- Maintenance tasks

**Read this if you:**
- Are on-call or handling operations
- Need to troubleshoot an issue
- Want to set up monitoring and alerts
- Are responding to an incident

---

### 8. [Knowledge Transfer Plan](08_KNOWLEDGE_TRANSFER_PLAN.md)
**Structured 4-week onboarding plan**

- **Week 1**: Foundation and overview
- **Week 2**: Deep dive into components
- **Week 3**: Operations and hands-on practice
- **Week 4**: Advanced topics and handoff
- Daily learning activities with exercises
- Assessment checkpoints
- Success criteria and metrics

**Read this if you:**
- Are onboarding to the project
- Are responsible for training new team members
- Want a structured learning path

---

## Quick Navigation

### By Role

**Data Scientist / ML Engineer:**
1. Start: [Project Overview](01_PROJECT_OVERVIEW.md)
2. Then: [Pipeline Components](03_PIPELINE_COMPONENTS.md)
3. Then: [Predictions and Lambda](04_PREDICTIONS_LAMBDA.md)
4. Reference: [Troubleshooting](07_TROUBLESHOOTING_OPERATIONS.md) as needed

**MLOps Engineer:**
1. Start: [Project Overview](01_PROJECT_OVERVIEW.md)
2. Then: [Workflows Guide](02_WORKFLOWS_GUIDE.md)
3. Then: [Deployment Scripts](05_DEPLOYMENT_SCRIPTS.md)
4. Then: [Troubleshooting](07_TROUBLESHOOTING_OPERATIONS.md)
5. Reference: [Configuration](06_CONFIGURATION_SETUP.md) as needed

**DevOps Engineer:**
1. Start: [Configuration and Setup](06_CONFIGURATION_SETUP.md)
2. Then: [Workflows Guide](02_WORKFLOWS_GUIDE.md)
3. Then: [Troubleshooting](07_TROUBLESHOOTING_OPERATIONS.md)
4. Reference: [Deployment Scripts](05_DEPLOYMENT_SCRIPTS.md) as needed

**New Team Member (Full Onboarding):**
- Follow: [Knowledge Transfer Plan](08_KNOWLEDGE_TRANSFER_PLAN.md)
- Read all documents in sequence (1-8)

### By Task

**Deploying a New Model:**
1. [Workflows Guide](02_WORKFLOWS_GUIDE.md) - Understand the workflow
2. [Deployment Scripts](05_DEPLOYMENT_SCRIPTS.md) - Reference scripts
3. [Troubleshooting](07_TROUBLESHOOTING_OPERATIONS.md) - If issues occur

**Adding a New Feature:**
1. [Pipeline Components](03_PIPELINE_COMPONENTS.md) - Understand current features
2. [Configuration](06_CONFIGURATION_SETUP.md) - Local development setup
3. [Workflows Guide](02_WORKFLOWS_GUIDE.md) - Deploy the change

**Troubleshooting an Issue:**
1. [Troubleshooting](07_TROUBLESHOOTING_OPERATIONS.md) - Start here
2. [Predictions and Lambda](04_PREDICTIONS_LAMBDA.md) - If Lambda issue
3. [Pipeline Components](03_PIPELINE_COMPONENTS.md) - If training issue

**Setting Up from Scratch:**
1. [Configuration and Setup](06_CONFIGURATION_SETUP.md) - Complete setup
2. [Project Overview](01_PROJECT_OVERVIEW.md) - Understand architecture
3. [Workflows Guide](02_WORKFLOWS_GUIDE.md) - Deploy first time

---

## Documentation Statistics

| Document | Lines | Topics Covered | Estimated Reading Time |
|----------|-------|----------------|----------------------|
| [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) | 704 | Architecture, Cost, Quick Start | 45 minutes |
| [02_WORKFLOWS_GUIDE.md](02_WORKFLOWS_GUIDE.md) | 1,396 | GitHub Actions, Deploy Process | 90 minutes |
| [03_PIPELINE_COMPONENTS.md](03_PIPELINE_COMPONENTS.md) | 2,243 | Preprocessing, Training Code | 2 hours |
| [04_PREDICTIONS_LAMBDA.md](04_PREDICTIONS_LAMBDA.md) | 1,654 | Lambda, Daily Forecasting | 90 minutes |
| [05_DEPLOYMENT_SCRIPTS.md](05_DEPLOYMENT_SCRIPTS.md) | 1,679 | All 23 Scripts | 90 minutes |
| [06_CONFIGURATION_SETUP.md](06_CONFIGURATION_SETUP.md) | 1,004 | Setup, Configuration | 60 minutes |
| [07_TROUBLESHOOTING_OPERATIONS.md](07_TROUBLESHOOTING_OPERATIONS.md) | 1,518 | Operations, Debugging | 90 minutes |
| [08_KNOWLEDGE_TRANSFER_PLAN.md](08_KNOWLEDGE_TRANSFER_PLAN.md) | 1,385 | 4-Week Onboarding | 60 minutes |
| **Total** | **11,583** | **Complete System** | **10+ hours** |

---

## Key Concepts Summary

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GitHub Actions                          │
│  (Triggered manually or on schedule - retraining quarterly)    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SageMaker Pipelines                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Preprocessing│→ │   Training   │→ │   Evaluate   │         │
│  │  (Redshift)  │  │  (XGBoost)   │  │  (Metrics)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Model Registry (S3)                           │
│  6 Models: RES/MEDCI/SMLCOM × SOLAR/NONSOLAR                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Lambda Functions (Daily 7 AM PT)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Recreate   │→ │   Predict    │→ │ Write to     │         │
│  │   Endpoint   │  │  (24 hours)  │  │  Redshift    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Customer Segments

| Profile | Segment | Meter Threshold | Solar Features | Key Characteristics |
|---------|---------|----------------|----------------|---------------------|
| **RES** | SOLAR | 100 | Yes | Residential with solar panels, duck curve pattern |
| **RES** | NONSOLAR | 100 | No | Residential without solar, evening peak dominant |
| **MEDCI** | SOLAR | 50 | Yes | Medium commercial with solar |
| **MEDCI** | NONSOLAR | 50 | No | Medium commercial, business hours pattern |
| **SMLCOM** | SOLAR | 30 | Yes | Small commercial with solar |
| **SMLCOM** | NONSOLAR | 30 | No | Small commercial, business hours pattern |

### Cost Optimization

**Problem:** Running 6 SageMaker endpoints 24/7 costs $14,599/month

**Solution:** Delete endpoints after deployment, recreate dynamically when needed

**Savings:** 99.3% reduction → $2,016/month (includes Lambda, S3, Redshift)

**Implementation:** Lambda endpoint manager recreates endpoint in ~5 minutes before prediction

---

## Technology Stack

### AWS Services
- **SageMaker**: ML training, pipelines, model registry, endpoints
- **Lambda**: Daily predictions (6 functions, scheduled via EventBridge)
- **Redshift**: Data warehouse (input data and forecast storage)
- **S3**: Model artifacts, scripts, processed data
- **Athena**: Alternative data source (optional)
- **CloudWatch**: Logging and monitoring
- **EventBridge**: Scheduling (daily 7 AM PT triggers)
- **IAM**: Roles and permissions

### ML/Data Stack
- **XGBoost 1.7.6**: Gradient boosting algorithm
- **Python 3.9+**: Primary language
- **Pandas 2.0.3**: Data manipulation
- **NumPy 1.24.4**: Numerical computing
- **Scikit-learn 1.3.0**: ML utilities
- **Bayesian Optimization**: Hyperparameter tuning
- **Open-Meteo API**: Free weather forecasts

### DevOps Stack
- **GitHub Actions**: CI/CD workflows
- **pytest**: Testing framework
- **moto**: AWS service mocking
- **boto3**: AWS SDK for Python

---

## Best Practices

### Development
1. **Always test locally** before deploying to AWS
2. **Use feature branches** and pull requests
3. **Write tests** for new features
4. **Update documentation** when making changes
5. **Follow code review** process

### Operations
1. **Check forecasts daily** (morning routine at 8 AM PT)
2. **Monitor CloudWatch** alarms and dashboards
3. **Review costs weekly** to catch anomalies
4. **Keep documentation updated** with lessons learned
5. **Follow runbooks** for incident response

### Security
1. **Never commit secrets** to Git
2. **Rotate IAM keys** every 90 days
3. **Use least privilege** IAM policies
4. **Enable MFA** on AWS accounts
5. **Review security groups** regularly

---

## Getting Help

### Documentation Issues
- If documentation is unclear, create an issue in GitHub
- Update documentation as you learn (contribute back!)
- Keep a running Q&A document for your team

### Technical Issues
1. **First**: Check [Troubleshooting Guide](07_TROUBLESHOOTING_OPERATIONS.md)
2. **Second**: Review CloudWatch logs
3. **Third**: Check AWS service health dashboard
4. **Fourth**: Escalate to team lead or AWS support

### Learning Resources
- AWS Documentation: https://docs.aws.amazon.com/
- XGBoost Docs: https://xgboost.readthedocs.io/
- GitHub Actions: https://docs.github.com/en/actions
- Time Series Forecasting: https://otexts.com/fpp3/

---

## Contributing to Documentation

Found an error or want to improve documentation?

1. Create a feature branch
2. Make your changes
3. Submit a pull request
4. Update the "Last Updated" date in the document

**Documentation Standards:**
- Use clear, concise language
- Include code examples where helpful
- Add screenshots for UI-heavy sections
- Keep table of contents updated
- Use consistent formatting

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2024-01-15 | Initial documentation release | MLOps Team |

---

## Contact

For questions or support:

- **MLOps Team**: mlops-team@company.com
- **On-Call**: [On-call rotation schedule]
- **Slack Channel**: #energy-forecasting
- **GitHub Issues**: [Create an issue](https://github.com/yourorg/dayahead_load_forecasting/issues)

---

**Last Updated:** 2024-01-15
**Maintained By:** MLOps Team
**Documentation Status:** Complete and Active
