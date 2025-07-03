#!/usr/bin/env python3
"""
Historical Forecasting Input Validation Script

This script validates all input parameters for the historical forecasting workflow
and generates the prediction plan including combinations matrix and date lists.
"""

import sys
import json
import os
from datetime import datetime, timedelta
from dateutil.parser import parse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_date_format(date_str):
    """Validate date format YYYY-MM-DD"""
    try:
        parsed_date = parse(date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except:
        return None


def get_date_list(prediction_type, **kwargs):
    """Generate list of dates based on prediction type"""
    dates = []
   
    if prediction_type == 'date_range':
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
       
        if not start_date or not end_date:
            raise ValueError("start_date and end_date required for date_range")
           
        start_date = validate_date_format(start_date)
        end_date = validate_date_format(end_date)
       
        if not start_date or not end_date:
            raise ValueError("Invalid date format for start_date or end_date")
       
        start = parse(start_date)
        end = parse(end_date)
       
        if start > end:
            raise ValueError("start_date must be before end_date")
       
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
           
    elif prediction_type == 'single_date':
        single_date = kwargs.get('single_forecast_date')
        if not single_date:
            raise ValueError("single_forecast_date required for single_date")
       
        validated_date = validate_date_format(single_date)
        if not validated_date:
            raise ValueError("Invalid date format for single_forecast_date")
       
        dates.append(validated_date)
       
    elif prediction_type == 'days_past':
        reference_date = kwargs.get('reference_date')
        days_back = kwargs.get('days_back')
       
        if not days_back:
            raise ValueError("days_back required for days_past")
       
        try:
            days_back = int(days_back)
        except:
            raise ValueError("days_back must be a number")
       
        if days_back <= 0:
            raise ValueError("days_back must be positive")
       
        if reference_date:
            ref_date = validate_date_format(reference_date)
            if not ref_date:
                raise ValueError("Invalid date format for reference_date")
            ref_date = parse(ref_date)
        else:
            ref_date = datetime.now()
       
        for i in range(days_back):
            date_to_add = ref_date - timedelta(days=i+1)
            dates.append(date_to_add.strftime('%Y-%m-%d'))
       
        dates.reverse()  # Chronological order
       
    elif prediction_type == 'multiple_dates':
        multiple_dates = kwargs.get('multiple_dates_list', '')
        if not multiple_dates:
            raise ValueError("multiple_dates_list required for multiple_dates")
       
        date_strings = [d.strip() for d in multiple_dates.split(',')]
        for date_str in date_strings:
            validated_date = validate_date_format(date_str)
            if not validated_date:
                raise ValueError(f"Invalid date format: {date_str}")
            dates.append(validated_date)
       
        # Remove duplicates and sort
        dates = sorted(list(set(dates)))
   
    return dates


def get_combinations_matrix(combinations, single_profile=None, single_segment=None):
    """Generate combinations matrix similar to main workflow"""
    all_combinations = [
        {"profile": "RES", "segment": "SOLAR"},
        {"profile": "RES", "segment": "NONSOLAR"},
        {"profile": "MEDCI", "segment": "SOLAR"},
        {"profile": "MEDCI", "segment": "NONSOLAR"},
        {"profile": "SMLCOM", "segment": "SOLAR"},
        {"profile": "SMLCOM", "segment": "NONSOLAR"}
    ]
   
    if combinations == "all":
        return all_combinations
    elif combinations == "res_only":
        return [c for c in all_combinations if c["profile"] == "RES"]
    elif combinations == "medci_only":
        return [c for c in all_combinations if c["profile"] == "MEDCI"]
    elif combinations == "smlcom_only":
        return [c for c in all_combinations if c["profile"] == "SMLCOM"]
    elif combinations == "solar_only":
        return [c for c in all_combinations if c["segment"] == "SOLAR"]
    elif combinations == "nonsolar_only":
        return [c for c in all_combinations if c["segment"] == "NONSOLAR"]
    elif combinations == "single_combination":
        if not single_profile or not single_segment:
            raise ValueError("single_customer_profile and single_customer_segment required for single_combination")
        return [{"profile": single_profile, "segment": single_segment}]
    else:
        raise ValueError(f"Invalid combinations value: {combinations}")


def main():
    """Main validation function"""
    try:
        logger.info("Starting historical forecasting input validation...")
       
        # Get inputs from environment variables
        environment = os.environ.get('ENVIRONMENT')
        database_type = os.environ.get('DATABASE_TYPE')
        combinations = os.environ.get('COMBINATIONS')
        prediction_type = os.environ.get('PREDICTION_TYPE')
        single_profile = os.environ.get('SINGLE_PROFILE')
        single_segment = os.environ.get('SINGLE_SEGMENT')
        max_parallel = os.environ.get('MAX_PARALLEL_REQUESTS')
        request_delay = os.environ.get('REQUEST_DELAY_SECONDS')
       
        logger.info(f"Environment: {environment}")
        logger.info(f"Database type: {database_type}")
        logger.info(f"Combinations: {combinations}")
        logger.info(f"Prediction type: {prediction_type}")
       
        # Validate environment
        if environment not in ['dev', 'qa', 'preprod', 'prod']:
            raise ValueError(f"Invalid environment: {environment}")
       
        # Validate database type
        if database_type not in ['athena', 'redshift']:
            raise ValueError(f"Invalid database_type: {database_type}")
       
        # Validate parallel requests and delay
        try:
            max_parallel = int(max_parallel) if max_parallel else 3
            request_delay = int(request_delay) if request_delay else 2
        except:
            raise ValueError("max_parallel_requests and request_delay_seconds must be numbers")
       
        if max_parallel < 1 or max_parallel > 10:
            raise ValueError("max_parallel_requests must be between 1 and 10")
       
        if request_delay < 1 or request_delay > 30:
            raise ValueError("request_delay_seconds must be between 1 and 30")
       
        # Generate combinations matrix
        combinations_matrix = get_combinations_matrix(
            combinations,
            single_profile if single_profile else None,
            single_segment if single_segment else None
        )
       
        logger.info(f"Combinations matrix: {combinations_matrix}")
       
        # Generate prediction dates
        prediction_dates = get_date_list(
            prediction_type,
            start_date=os.environ.get('START_DATE'),
            end_date=os.environ.get('END_DATE'),
            single_forecast_date=os.environ.get('SINGLE_FORECAST_DATE'),
            reference_date=os.environ.get('REFERENCE_DATE'),
            days_back=os.environ.get('DAYS_BACK'),
            multiple_dates_list=os.environ.get('MULTIPLE_DATES_LIST')
        )
       
        logger.info(f"Prediction dates: {prediction_dates[:5]}{'...' if len(prediction_dates) > 5 else ''}")
       
        # Calculate total predictions
        total_predictions = len(combinations_matrix) * len(prediction_dates)
       
        # Validate reasonable limits
        if total_predictions > 1000:
            raise ValueError(f"Total predictions ({total_predictions}) exceeds limit of 1000. Reduce date range or combinations.")
       
        if len(prediction_dates) > 100:
            raise ValueError(f"Number of dates ({len(prediction_dates)}) exceeds limit of 100. Reduce date range.")
       
        # Validate dates are not too far in the future (with some tolerance)
        today = datetime.now().date()
        future_dates = [d for d in prediction_dates if parse(d).date() > today + timedelta(days=7)]
        if future_dates:
            logger.warning(f"Future dates detected: {future_dates[:5]}{'...' if len(future_dates) > 5 else ''}")
            logger.warning("Forecasting future dates is allowed but ensure models can handle it.")
       
        # Success outputs
        logger.info("✅ Input validation successful")
        logger.info(f"Total combinations: {len(combinations_matrix)}")
        logger.info(f"Total dates: {len(prediction_dates)}")
        logger.info(f"Total predictions to generate: {total_predictions}")
        logger.info(f"Max parallel requests: {max_parallel}")
        logger.info(f"Request delay: {request_delay} seconds")
       
        # Set GitHub outputs
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"environment={environment}\n")
            f.write(f"database_type={database_type}\n")
            f.write(f"combinations_matrix={json.dumps(combinations_matrix)}\n")
            f.write(f"prediction_dates={json.dumps(prediction_dates)}\n")
            f.write(f"total_predictions={total_predictions}\n")
            f.write(f"validation_status=success\n")
            f.write(f"max_parallel_requests={max_parallel}\n")
            f.write(f"request_delay_seconds={request_delay}\n")
       
        logger.info("Validation completed successfully")
       
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"validation_status=failed\n")
            f.write(f"error_message={str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
