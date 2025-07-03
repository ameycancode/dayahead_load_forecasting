#!/usr/bin/env python3
"""
Create GitHub Workflow Summary Script

This script creates the GitHub workflow summary for the historical forecasting execution.
"""

import json
import os
import sys
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_date_range_for_query(prediction_dates):
    """Extract start and end dates for SQL query"""
    if not prediction_dates:
        return "2025-01-01", "2025-12-31"
   
    if len(prediction_dates) == 1:
        return prediction_dates[0], prediction_dates[0]
   
    return prediction_dates[0], prediction_dates[-1]


def create_workflow_summary():
    """Create GitHub workflow summary"""
    try:
        # Get environment variables
        environment = os.environ.get('ENVIRONMENT', 'unknown')
        database_type = os.environ.get('DATABASE_TYPE', 'unknown')
        total_predictions = os.environ.get('TOTAL_PREDICTIONS', '0')
        github_run_id = os.environ.get('GITHUB_RUN_ID', 'unknown')
        aws_region = os.environ.get('AWS_REGION', 'us-west-2')
       
        # Get calculated statistics
        combo_count = os.environ.get('COMBO_COUNT', '0')
        dates_count = os.environ.get('DATES_COUNT', '0')
        date_range = os.environ.get('DATE_RANGE', 'N/A')
       
        # Get prediction dates for query
        prediction_dates_json = os.environ.get('PREDICTION_DATES', '[]')
        prediction_dates = json.loads(prediction_dates_json)
        start_date, end_date = get_date_range_for_query(prediction_dates)
       
        # Get current timestamp
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
       
        # Determine table name based on database type
        if database_type == 'athena':
            table_reference = "your_athena_database.dayahead_load_forecasts"
        else:
            table_reference = "your_schema.dayahead_load_forecasts"
       
        # Create the summary content
        summary_content = f"""## Historical Energy Load Forecasting Results

**Environment:** {environment}  
**Database:** {database_type}  
**Execution Date:** {current_time}

### Configuration
- **Total Combinations:** {combo_count}
- **Total Dates:** {dates_count}  
- **Date Range:** {date_range}
- **Expected Predictions:** {total_predictions}

### Quick Actions
- [Download Execution Summary](../../actions/runs/{github_run_id}/artifacts)
- [View CloudWatch Logs](https://console.aws.amazon.com/cloudwatch/home?region={aws_region}#logs:)
- [Check SageMaker Endpoints](https://console.aws.amazon.com/sagemaker/home?region={aws_region}#/endpoints)

### Database Query
Use this query to verify the generated forecasts in your database:

```sql
SELECT load_profile, load_segment, forecast_date, COUNT(*) as forecast_count
FROM {table_reference}
WHERE forecast_date BETWEEN '{start_date}' AND '{end_date}'
GROUP BY load_profile, load_segment, forecast_date
ORDER BY load_profile, load_segment, forecast_date;
```

### Additional Verification
```sql
-- Check total predictions by combination
SELECT
    load_profile,
    load_segment,
    COUNT(*) as total_forecasts,
    MIN(forecast_date) as first_date,
    MAX(forecast_date) as last_date
FROM {table_reference}
WHERE forecast_date BETWEEN '{start_date}' AND '{end_date}'
GROUP BY load_profile, load_segment
ORDER BY load_profile, load_segment;
```
"""
       
        # Write to GITHUB_STEP_SUMMARY
        github_step_summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
        if github_step_summary_path:
            with open(github_step_summary_path, 'a') as f:
                f.write(summary_content)
            logger.info("✅ GitHub workflow summary created successfully")
        else:
            # If not in GitHub Actions, just print the summary
            print("GITHUB_STEP_SUMMARY content:")
            print(summary_content)
            logger.info("✅ Summary content generated (GITHUB_STEP_SUMMARY not available)")
       
        logger.info(f"Summary details:")
        logger.info(f"  Environment: {environment}")
        logger.info(f"  Database: {database_type}")
        logger.info(f"  Combinations: {combo_count}")
        logger.info(f"  Dates: {dates_count}")
        logger.info(f"  Date range: {date_range}")
       
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON data: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating workflow summary: {str(e)}")
        sys.exit(1)


def main():
    """Main function"""
    try:
        logger.info("Creating GitHub workflow summary...")
        create_workflow_summary()
        logger.info("Workflow summary creation completed")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
