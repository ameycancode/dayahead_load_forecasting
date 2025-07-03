"""
Data preparation module for energy load forecasting.
Handles data retrieval and basic preprocessing.
"""
import os
import logging
import traceback
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Optional, Union

import numpy as np
import pandas as pd
import boto3
from pyathena import connect
from pyathena.pandas.util import as_pandas

from forecast.utils import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


def calculate_data_date_ranges(current_date, config, data_delay_days=14):
    """
    Calculate the three required dates for optimized data fetching
   
    Args:
        current_date: Current date (datetime or date object)
        data_delay_days: Data availability delay (default 14 days)
   
    Returns:
        tuple: (start_date, end_date, final_cutoff_date)
    """
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, '%Y-%m-%d').date()
    elif isinstance(current_date, datetime):
        current_date = current_date.date()
   
    # Calculate the three dates as per requirements
    end_date = current_date - timedelta(days=data_delay_days)  # current date - 14 days
    start_date = current_date - timedelta(days=70)  # going back 10 weeks (70 days)
   
    # Final cutoff date: Find the last available Final submission date
    # This should be calculated by querying the actual data, not using SUBMISSION_DELAY
    final_cutoff_date = find_final_cutoff_date(current_date, start_date, end_date, config)
   
    logger.info("=== CALCULATED DATE RANGES ===")
    logger.info(f"Current date: {current_date}")
    logger.info(f"Start date (10 weeks back): {start_date}")
    logger.info(f"End date (current - {data_delay_days} days): {end_date}")
    logger.info(f"Total historical period: {(end_date - start_date).days} days")

    if final_cutoff_date is not None:
        logger.info(f"Final Submission cutoff date (actual): {final_cutoff_date}")
        logger.info(f"Final Submission data period: {start_date} to {final_cutoff_date}")
        logger.info(f"Initial Submission data period: {final_cutoff_date + timedelta(days=1)} to {end_date}")
    else:
        logger.info("No Final Submission data used for prediction")
        logger.info(f"Initial Submission data period: {start_date} to {end_date}")

    logger.info(f"=== END DATE RANGES ===")
   
    return start_date, end_date, final_cutoff_date

def find_final_cutoff_date(current_date, start_date, end_date, config):
    """
    Find the actual final cutoff date by querying the database
    This finds the latest date with Final submission data available
    """
    try:
        # Get database configuration
        database = config.get('REDSHIFT_DATABASE', 'sdcp')
        cluster_identifier = config.get('REDSHIFT_CLUSTER_IDENTIFIER')
        db_user = config.get('REDSHIFT_DB_USER', 'ds_service_user')
        region = config.get('REDSHIFT_REGION', 'us-west-2')
        schema_name = config.get('REDSHIFT_INPUT_SCHEMA', 'edp_cust_dev')
        table_name = config.get('REDSHIFT_INPUT_TABLE', 'caiso_sqmd')
        load_profile = config.get('LOAD_PROFILE', 'RES')
       
        # Query to find the latest Final submission date within our range
        query = f"""
        SELECT MAX(tradedatelocal) as latest_final_date
        FROM {schema_name}.{table_name}
        WHERE submission = 'Final'
        AND loadprofile = '{load_profile}'
        AND tradedatelocal >= '{start_date}'
        AND tradedatelocal <= '{end_date}'
        """
       
        logger.info("Finding actual final cutoff date from database...")
        logger.info(f"Query: {query}")
       
        # Execute query using existing method
        redshift_data_client = boto3.client('redshift-data')
       
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=query
        )
       
        query_id = response['Id']
       
        # Wait for completion
        while True:
            status_response = redshift_data_client.describe_statement(Id=query_id)
            status = status_response['Status']
           
            if status == 'FINISHED':
                break
            elif status in ['FAILED', 'ABORTED']:
                error_msg = status_response.get('Error', 'Query failed')
                logger.error(f"Query to find final cutoff failed: {error_msg}")
                # Fallback to estimated date
                return current_date - timedelta(days=config.get('FINAL_SUBMISSION_DELAY'))
           
            time.sleep(2)
       
        # Get result
        result_response = redshift_data_client.get_statement_result(Id=query_id)
        records = result_response.get('Records', [])
       
        if records and len(records) > 0 and len(records[0]) > 0:
            latest_final_str = records[0][0].get('stringValue')
            if latest_final_str:
                final_cutoff_date = datetime.strptime(latest_final_str, '%Y-%m-%d').date()
                logger.info(f"Found actual final cutoff date: {final_cutoff_date}")
                return final_cutoff_date
       
        # Fallback if no Final data found
        logger.warning("No Final data found in range, using estimated cutoff")
        # return current_date - timedelta(days=config.get('FINAL_SUBMISSION_DELAY'))  # Conservative estimate
        return None
       
    except Exception as e:
        logger.error(f"Error finding final cutoff date: {str(e)}")
        # Fallback to conservative estimate
        # return current_date - timedelta(days=config.get('FINAL_SUBMISSION_DELAY'))
        return None

def execute_redshift_query_via_data_api(query, database=None, cluster_identifier=None, db_user=None, region=None):  
    """
    Execute Redshift query using Data API (no direct connection needed)
    This approach avoids connection timeout issues
    """
    try:
        logger.info(f"Executing query via Data API on cluster: {cluster_identifier}")
        logger.info(f"Database: {database}, User: {db_user}, Region: {region}")
       
        # Create Redshift Data API client
        redshift_data_client = boto3.client('redshift-data', region_name=region)
       
        # Execute the query
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=query
        )
       
        query_id = response['Id']
        logger.info(f"Query submitted with ID: {query_id}")
       
        # Wait for query completion
        wait_for_query_completion(redshift_data_client, query_id)
       
        # Get ALL results with pagination - THIS IS THE KEY FIX
        df = get_paginated_results(redshift_data_client, query_id)
       
        logger.info(f"Query completed successfully. Retrieved {len(df)} rows")
        return df
       
    except Exception as e:
        logger.error(f"Error executing query via Data API: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_paginated_results(redshift_data_client, query_id):
    """
    Get all results from Data API with proper pagination
    This fixes the row count discrepancy issue
    """
    all_records = []
    next_token = None
    page_count = 0
   
    logger.info(f"Starting pagination for query {query_id}")
   
    while True:
        try:
            # Get results page
            params = {'Id': query_id}
            if next_token:
                params['NextToken'] = next_token
               
            result_response = redshift_data_client.get_statement_result(**params)
            page_count += 1
           
            # Extract records from this page
            page_records = result_response.get('Records', [])
            all_records.extend(page_records)
           
            logger.info(f"Page {page_count}: Retrieved {len(page_records)} records (Total: {len(all_records)})")
           
            # Check if there are more pages
            next_token = result_response.get('NextToken')
            if not next_token:
                logger.info(f"Pagination complete: {page_count} pages, {len(all_records)} total records")
                break
               
        except Exception as e:
            logger.error(f"Error getting paginated results: {str(e)}")
            raise
   
    # Convert to DataFrame
    if not all_records:
        # Get column metadata for empty DataFrame
        column_metadata = result_response.get('ColumnMetadata', [])
        column_names = [col['name'] for col in column_metadata]
        return pd.DataFrame(columns=column_names)
   
    return convert_data_api_result_to_dataframe_with_all_records(result_response, all_records)

def convert_data_api_result_to_dataframe_with_all_records(result_response, all_records):
    """Convert Data API result to pandas DataFrame with all paginated records"""
    try:
        # Get column metadata
        column_metadata = result_response.get('ColumnMetadata', [])
        column_names = [col['name'] for col in column_metadata]
       
        if not all_records:
            return pd.DataFrame(columns=column_names)
       
        # Convert records to list of lists
        data_rows = []
        for record in all_records:
            row = []
            for field in record:
                # Extract value based on type
                if 'stringValue' in field:
                    row.append(field['stringValue'])
                elif 'longValue' in field:
                    row.append(field['longValue'])
                elif 'doubleValue' in field:
                    row.append(field['doubleValue'])
                elif 'booleanValue' in field:
                    row.append(field['booleanValue'])
                elif 'isNull' in field and field['isNull']:
                    row.append(None)
                else:
                    row.append(str(field))
            data_rows.append(row)
       
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=column_names)
       
        logger.info(f"Successfully converted {len(all_records)} records to DataFrame with {len(df)} rows and {len(column_names)} columns")
        return df
       
    except Exception as e:
        logger.error(f"Error converting Data API result to DataFrame: {str(e)}")
        raise

def wait_for_query_completion(redshift_data_client, query_id, max_wait_seconds=1800):
    """
    Wait for Redshift Data API query to complete
    """
    logger.info(f"Waiting for query {query_id} to complete...")
   
    waited = 0
    while waited < max_wait_seconds:
        try:
            status_response = redshift_data_client.describe_statement(Id=query_id)
            status = status_response['Status']
           
            if status == 'FINISHED':
                logger.info(f"Query {query_id} completed successfully")
                return
            elif status == 'FAILED':
                error_msg = status_response.get('Error', 'Unknown error')
                logger.error(f"Query {query_id} failed: {error_msg}")
                raise Exception(f'Query failed: {error_msg}')
            elif status == 'ABORTED':
                logger.error(f"Query {query_id} was aborted")
                raise Exception(f'Query was aborted')
           
            # Still running
            if waited % 60 == 0 and waited > 0:  # Log every minute
                logger.info(f"Query still running... waited {waited}s (status: {status})")
           
            time.sleep(10)  # Check every 10 seconds
            waited += 10
           
        except Exception as e:
            if 'failed:' in str(e) or 'aborted' in str(e):
                raise  # Re-raise query failures
            else:
                logger.warning(f"Error checking query status: {str(e)}")
                time.sleep(10)
                waited += 10
                continue
   
    raise Exception(f'Query timed out after {max_wait_seconds} seconds')

def convert_data_api_result_to_dataframe(result_response):
    """
    Convert Redshift Data API result to pandas DataFrame
    """
    try:
        # Get column metadata
        column_metadata = result_response.get('ColumnMetadata', [])
        column_names = [col['name'] for col in column_metadata]
       
        # Get records
        records = result_response.get('Records', [])
       
        if not records:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=column_names)
       
        # Convert records to list of lists
        data_rows = []
        for record in records:
            row = []
            for field in record:
                # Extract value based on type
                if 'stringValue' in field:
                    row.append(field['stringValue'])
                elif 'longValue' in field:
                    row.append(field['longValue'])
                elif 'doubleValue' in field:
                    row.append(field['doubleValue'])
                elif 'booleanValue' in field:
                    row.append(field['booleanValue'])
                elif 'isNull' in field and field['isNull']:
                    row.append(None)
                else:
                    row.append(str(field))  # Fallback
            data_rows.append(row)
       
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=column_names)
       
        logger.info(f"Converted Data API result to DataFrame: {len(df)} rows, {len(column_names)} columns")
        return df
       
    except Exception as e:
        logger.error(f"Error converting Data API result to DataFrame: {str(e)}")
        raise


def query_data(
    config,
    current_date=None,
    load_profile=None,
    rate_group_filter=None,
    use_cache=None,
    query_limit=None,
):
    """
    Query data from Redshift or Athena combining Final and Initial submissions
    Updated for Redshift migration in prediction pipeline.
    """
    try:
        if current_date is None:
            current_date = datetime.now()

        # Check database type from config
        database_type = config.get('DATABASE_TYPE', 'redshift')
       
        if database_type == 'redshift':
            logger.info("Using Redshift data source for predictions")
            return query_data_redshift(config, current_date, load_profile, rate_group_filter, query_limit)
        else:
            logger.info("Using Athena data source (legacy) for predictions")
            return query_data_athena(config, current_date, load_profile, rate_group_filter, use_cache, query_limit)
           
    except Exception as e:
        import traceback
        logger.error(f"Error in query_data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def query_data_redshift(config, current_date=None, load_profile=None, rate_group_filter=None, query_limit=None):
    """
    Query data with optimized date ranges as per new requirements
    """
    try:
        if current_date is None:
            current_date = datetime.now()

        # Get data delay from environment or use default
        data_delay_days = int(config.get('DATA_DELAY_DAYS', '14'))
       
        # Calculate optimized date ranges
        start_date, end_date, final_cutoff_date = calculate_data_date_ranges(current_date, config, data_delay_days)
       
        # Use defaults from config if not provided
        load_profile = load_profile or config.get('CUSTOMER_PROFILE', 'RES')
        rate_group_filter = config.get('RATE_GROUP_FILTER_CLAUSE')

        logger.info(f"Querying optimized Redshift data for {load_profile} as of: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"Rate group filter: {rate_group_filter}")

        final_df = pd.DataFrame()
        initial_df = pd.DataFrame()

        # Query Final data from start_date to final_cutoff_date
        if final_cutoff_date is not None:
            final_df = query_redshift_final_data(
                config, start_date, final_cutoff_date, load_profile, rate_group_filter, query_limit
            )
            final_df = convert_column_types(final_df)

            # Query Initial data from final_cutoff_date + 1 to end_date
            initial_start_date = final_cutoff_date + timedelta(days=1)
        else:
            initial_start_date = start_date
            
        initial_df = query_redshift_initial_data(
            config, initial_start_date, end_date, load_profile, rate_group_filter, query_limit
        )
        initial_df = convert_column_types(initial_df)

        # Combine datasets
        if final_df.empty and initial_df.empty:
            logger.warning("No data retrieved from either Final or Initial submissions")
            return pd.DataFrame()
        elif final_df.empty:
            logger.warning("No Final submission data available, using only Initial data")
            combined_df = initial_df
        elif initial_df.empty:
            logger.warning("No Initial submission data available")
            combined_df = final_df
        else:
            combined_df = pd.concat([final_df, initial_df], ignore_index=True)
            logger.info(f"Combined historical data: {len(combined_df)} rows")

        # Validate the date range in final dataset
        if not combined_df.empty and 'datetime' in combined_df.columns:
            actual_start = combined_df['datetime'].min().date()
            actual_end = combined_df['datetime'].max().date()
            logger.info(f"Final historical data range: {actual_start} to {actual_end}")
           
            # Check data completeness
            expected_days = (end_date - start_date).days + 1
            actual_days = len(combined_df['datetime'].dt.date.unique())
            coverage_pct = (actual_days / expected_days) * 100
            logger.info(f"Data coverage: {actual_days}/{expected_days} days ({coverage_pct:.1f}%)")

        return combined_df

    except Exception as e:
        logger.error(f"Error querying Redshift data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def query_redshift_final_data(config, start_date, final_cutoff_date, load_profile, rate_group_filter, query_limit=None):
    """Query Final data from start_date to final_cutoff_date"""
    try:
        # Build WHERE clause for Final data
        where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.get('SUBMISSION_TYPE_FINAL')}'",
            f"tradedatelocal >= '{start_date}'",
            f"tradedatelocal <= '{final_cutoff_date}'"
        ]

        if rate_group_filter:
            where_clauses.append(rate_group_filter)

        where_clause = " AND ".join(where_clauses)

        # Build query
        schema_name = config.get('REDSHIFT_INPUT_SCHEMA', 'edp_cust_dev')
        table_name = config.get('REDSHIFT_INPUT_TABLE', 'caiso_sqmd')

        query = f"""
        SELECT
            tradedatelocal as tradedate,
            tradehourstartlocal as tradetime,
            loadprofile, rategroup, baseload, lossadjustedload, metercount,
            loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
            submission, createddate as created
        FROM {schema_name}.{table_name}
        WHERE {where_clause}
        ORDER BY tradedatelocal, tradehourstartlocal
        """

        if query_limit and query_limit > 0:
            query += f" LIMIT {query_limit}"

        logger.info(f"Final Submission Data Fetch Query: {query}")
        logger.info(f"Executing Final data query from {start_date} to {final_cutoff_date}")
       
        final_df = query_redshift_data(config, query)
        logger.info(f"Retrieved {len(final_df)} rows of Final data")

        return final_df
       
    except Exception as e:
        logger.error(f"Error querying Final data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def query_redshift_initial_data(config, initial_start_date, end_date, load_profile, rate_group_filter, query_limit=None):
    """Query Initial data from final_cutoff_date + 1 to end_date"""
    try:
        # Get database configuration
        if initial_start_date > end_date:
            logger.info("No Initial data needed - final cutoff date is after end date")
            return pd.DataFrame()

        # Build WHERE clause for Initial data
        where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.get('SUBMISSION_TYPE_INITIAL')}'",
            f"tradedatelocal >= '{initial_start_date}'",
            f"tradedatelocal <= '{end_date}'"
        ]

        if rate_group_filter:
            where_clauses.append(rate_group_filter)

        where_clause = " AND ".join(where_clauses)

        # Build query
        schema_name = config.get('REDSHIFT_INPUT_SCHEMA', 'edp_cust_dev')
        table_name = config.get('REDSHIFT_INPUT_TABLE', 'caiso_sqmd')
       
        query = f"""
        SELECT
            tradedatelocal as tradedate,
            tradehourstartlocal as tradetime,
            loadprofile, rategroup, baseload, lossadjustedload, metercount,
            loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
            submission, createddate as created
        FROM {schema_name}.{table_name}
        WHERE {where_clause}
        ORDER BY tradedatelocal, tradehourstartlocal
        """

        if query_limit and query_limit > 0:
            query += f" LIMIT {query_limit}"

        logger.info(f"Initial Submission Data Fetch Query: {query}")
        logger.info(f"Executing Initial data query from {initial_start_date} to {end_date}")
        initial_df = query_redshift_data(config, query)
        logger.info(f"Retrieved {len(initial_df)} rows of Initial data")

        return initial_df
       
    except Exception as e:
        logger.error(f"Error querying Initial data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def query_redshift_data(config, query):
    """
    Execute query on Redshift and return DataFrame
    """
    try:
        logger.info("Executing Redshift query via Data API...")
        # logger.debug(f"Query: {query[:200]}..." if len(query) > 200 else f"Query: {query}")

        database = config.get('REDSHIFT_DATABASE', 'sdcp')
        cluster_identifier = config.get('REDSHIFT_CLUSTER_IDENTIFIER', 'sdcp-edp-backend-dev')
        db_user = config.get('REDSHIFT_DB_USER', 'ds_service_user')
        region = config.get('REDSHIFT_REGION', 'us-west-2')
       
        df = execute_redshift_query_via_data_api(query, database, cluster_identifier, db_user, region)
       
        logger.info(f"Retrieved {len(df)} rows from Redshift via Data API")
        return df
       
    except Exception as e:
        logger.error(f"Error querying Redshift via Data API: {str(e)}")
        raise


def query_data_athena(
    config,
    current_date=None,
    load_profile=None,
    rate_group_filter=None,
    use_cache=None,
    query_limit=None,
):
    """
    Query data from Athena combining Final and Initial submissions

    Args:
        current_date: Current date (defaults to today)
        load_profile: Load profile to filter by (default from config)
        rate_group_filter: Rate group filter pattern (default from config)

    Returns:
        DataFrame with combined data
    """
    try:
        if current_date is None:
            current_date = datetime.now()

        # Load from cache if available and recent
        final_df = pd.DataFrame()
        initial_df = pd.DataFrame()

        # Use defaults from config if not provided
        load_profile = load_profile or config.get('LOAD_PROFILE')
       
        # UPDATED LOGIC: Use dynamic rate group filter clause like in data processing
        logger.info(f"RATE_GROUP_FILTER_CLAUSE: {config.get('RATE_GROUP_FILTER_CLAUSE')}")

        if rate_group_filter is None:
            # First try to get the dynamic filter clause
            if hasattr(config, "RATE_GROUP_FILTER_CLAUSE"):
                rate_group_filter = config.get('RATE_GROUP_FILTER_CLAUSE')
                logger.info(f"Using dynamic RATE_GROUP_FILTER_CLAUSE: {rate_group_filter}")
            # Fallback to old simple filter for backward compatibility
            elif hasattr(config, "RATE_GROUP_FILTER"):
                rate_group_filter = config.get('RATE_GROUP_FILTER')
                logger.warning(f"Fallback to legacy RATE_GROUP_FILTER: {rate_group_filter}")
            else:
                rate_group_filter = None
                logger.warning("No rate group filter configuration found")

        logger.info(f"Querying data as of: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"Load profile: {load_profile}")
        logger.info(f"Rate group filter: {rate_group_filter}")

        # Calculate cutoff dates based on submission type delays
        initial_cutoff_date = current_date - timedelta(
            days=config.get('INITIAL_SUBMISSION_DELAY')
        )
        final_cutoff_date = current_date - timedelta(days=config.get('FINAL_SUBMISSION_DELAY'))

        logger.info(f"Initial cutoff: {initial_cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"Final cutoff: {final_cutoff_date.strftime('%Y-%m-%d')}")

        final_cutoff_date = current_date - timedelta(
            days=config.get('FINAL_SUBMISSION_DELAY')
        )
        logger.info(
            f"Querying Athena for Final submission data (cutoff: {final_cutoff_date})"
        )

        # Connect to Athena and make query
        final_df = query_athena_final_data(
            config, current_date, load_profile, rate_group_filter, query_limit
        )

        # Convert types
        final_df = convert_column_types(final_df)

        initial_cutoff_date = current_date - timedelta(
            days=config.get('INITIAL_SUBMISSION_DELAY')
        )
        logger.info(
            f"Querying Athena for Initial submission data (cutoff: {initial_cutoff_date})"
        )

        # Get the latest Final date from data or Athena
        if final_df.empty:
            latest_final_date = final_cutoff_date
        else:
            if "datetime" in final_df.columns:
                latest_final_date = final_df["datetime"].max()
            elif "tradedate" in final_df.columns:
                latest_final_date = pd.to_datetime(final_df["tradedate"].max())
            else:
                latest_final_date = final_cutoff_date

        # Connect to Athena and make query for Initial data
        initial_df = query_athena_initial_data(
            config,
            current_date,
            latest_final_date,
            load_profile,
            rate_group_filter,
            query_limit,
        )

        # Convert types
        initial_df = convert_column_types(initial_df)

        # Combine datasets
        if final_df.empty and initial_df.empty:
            logger.warning("No data retrieved from either Final or Initial submissions")
            return pd.DataFrame()
        elif final_df.empty:
            logger.warning(
                "No Final submission data available, using only Initial data"
            )
            combined_df = initial_df
        elif initial_df.empty:
            logger.warning("No Initial submission data available")
            combined_df = final_df
        else:
            combined_df = pd.concat([final_df, initial_df], ignore_index=True)
            logger.info(f"Combined data: {len(combined_df)} rows")

        return combined_df

    except Exception as e:
        import traceback
        logger.error(f"Error querying data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def query_athena_final_data(
    config, current_date, load_profile, rate_group_filter, query_limit=None
):
    """Query Athena for Final submission data"""
    try:
        # Connect to Athena
        conn = connect(
            s3_staging_dir=config.get('ATHENA_STAGING_DIR'), region_name=config.get('REDSHIFT_REGION')
        )

        final_cutoff_date = current_date - timedelta(days=config.get('FINAL_SUBMISSION_DELAY'))

        # Build WHERE clause for filtering
        where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.get('SUBMISSION_TYPE_FINAL')}'",  # Using Final submission type as default
        ]

        # UPDATED LOGIC: Add rate group filter clause if provided
        if rate_group_filter:
            # The rate_group_filter now contains the complete clause
            # e.g., "(rategroup NOT LIKE 'NEM%' AND rategroup NOT LIKE 'SBP%')"
            where_clauses.append(rate_group_filter)
            logger.info(f"Added rate group filter clause: {rate_group_filter}")

        where_clause = " AND ".join(where_clauses)
        logger.info(f"Final WHERE clause: {where_clause}")

        # Get latest Final submission date
        cursor = conn.cursor()
        latest_final_query = f"""
        SELECT MAX(tradedate) as max_date
        FROM {config.get('ATHENA_DATABASE')}.{config.get('ATHENA_TABLE')}
        WHERE {where_clause}
        """

        if query_limit is not None and query_limit > 0:
            latest_final_query += f" LIMIT {query_limit}"

        logger.info(
            f"Executing Athena query for Final submission date: {latest_final_query}"
        )
        cursor.execute(latest_final_query)
        latest_final_result = as_pandas(cursor)

        if latest_final_result.empty or pd.isna(
            latest_final_result["max_date"].iloc[0]
        ):
            latest_final_date = final_cutoff_date
            logger.warning(
                f"No Final submission date found, using cutoff: {latest_final_date}"
            )
        else:
            latest_final_date = pd.to_datetime(latest_final_result["max_date"].iloc[0])
            logger.info(f"Latest Final submission date: {latest_final_date}")

        # Base query with all fields
        base_select = """
        SELECT id, tradedate, tradetime, loadprofile, rategroup,
               baseload, lossadjustedload, metercount, loadbl,
               loadlal, loadmetercount, genbl, genlal, genmetercount,
               submission, created
        FROM {}.{}
        WHERE {}
        """

        # Query Final submission data
        final_where = f"{where_clause} AND CAST(tradedate AS DATE) <= CAST('{latest_final_date.strftime('%Y-%m-%d')}' AS DATE)"
        final_query = base_select.format(
            config.get('ATHENA_DATABASE'), config.get('ATHENA_TABLE'), final_where
        )

        if query_limit is not None and query_limit > 0:
            final_query += f" LIMIT {query_limit}"

        logger.info(f"Executing Athena query for Final submission data: {final_query}")
        cursor.execute(final_query)
        final_df = as_pandas(cursor)
        logger.info(f"Retrieved {len(final_df)} rows of Final data")

        return final_df
    except Exception as e:
        import traceback
        logger.error(f"Error querying Athena for Final data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def query_athena_initial_data(
    config, current_date, latest_final_date, load_profile, rate_group_filter, query_limit=None
):
    """Query Athena for Initial submission data"""
    try:
        # Connect to Athena
        conn = connect(
            s3_staging_dir=config.get('ATHENA_STAGING_DIR'), region_name=config.get('REDSHIFT_REGION')
        )

        initial_cutoff_date = current_date - timedelta(
            days=config.get('INITIAL_SUBMISSION_DELAY')
        )

        # Format latest_final_date if needed
        if isinstance(latest_final_date, pd.Timestamp):
            latest_final_date = latest_final_date.strftime("%Y-%m-%d")
        elif isinstance(latest_final_date, datetime):
            latest_final_date = latest_final_date.strftime("%Y-%m-%d")

        latest_final_date = pd.to_datetime(latest_final_date)

        # Modify where clause for Initial data
        init_where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.get('SUBMISSION_TYPE_INITIAL')}'",  # Using Initial instead of config default
        ]

        # UPDATED LOGIC: Add rate group filter clause if provided
        if rate_group_filter:
            # The rate_group_filter now contains the complete clause
            init_where_clauses.append(rate_group_filter)
            logger.info(f"Added rate group filter clause for Initial data: {rate_group_filter}")

        # Add date constraints for Initial data
        init_where_clauses.append(
            f"CAST(tradedate AS DATE) <= CAST('{initial_cutoff_date.strftime('%Y-%m-%d')}' AS DATE)"
        )
        init_where_clauses.append(
            f"CAST(tradedate AS DATE) > CAST('{latest_final_date.strftime('%Y-%m-%d')}' AS DATE)"
        )

        init_where_clause = " AND ".join(init_where_clauses)
        logger.info(f"Initial WHERE clause: {init_where_clause}")

        # Base query with all fields
        base_select = """
        SELECT id, tradedate, tradetime, loadprofile, rategroup,
               baseload, lossadjustedload, metercount, loadbl,
               loadlal, loadmetercount, genbl, genlal, genmetercount,
               submission, created
        FROM {}.{}
        WHERE {}
        """

        # Query Initial submission data
        initial_query = base_select.format(
            config.get('ATHENA_DATABASE'), config.get('ATHENA_TABLE'), init_where_clause
        )

        logger.info(
            f"Executing Athena query for Initial submission data: {initial_query}"
        )

        if query_limit is not None and query_limit > 0:
            initial_query += f" LIMIT {query_limit}"

        cursor = conn.cursor()
        cursor.execute(initial_query)
        initial_df = as_pandas(cursor)
        logger.info(f"Retrieved {len(initial_df)} rows of Initial data")

        return initial_df
    except Exception as e:
        import traceback
        logger.error(f"Error querying Athena for Initial data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def convert_column_types(df):
    """
    Convert DataFrame columns to appropriate types.
    Updated to handle Redshift column names (tradedatelocal, tradehourstartlocal)
    """
    if df.empty:
        return df

    # Make a copy to avoid warnings
    df = df.copy()

    # Define column type mapping (same as before)
    type_mapping = {
        "id": "int64",
        "baseload": "float64",
        "lossadjustedload": "float64",
        "metercount": "int64",
        "loadbl": "float64",
        "loadlal": "float64",
        "loadmetercount": "int64",
        "genbl": "float64",
        "genlal": "float64",
        "genmetercount": "int64",
    }

    # Convert each column if it exists
    for col, dtype in type_mapping.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Error converting column {col} to {dtype}: {str(e)}")

    # Convert date columns - handle both Athena and Redshift column names
    date_col = "tradedate" if "tradedate" in df.columns else "tradedatelocal"
    time_col = "tradetime" if "tradetime" in df.columns else "tradehourstartlocal"
   
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            logger.warning(f"Error converting {date_col} to datetime")

    # Create datetime column if not present
    if "datetime" not in df.columns and date_col in df.columns and time_col in df.columns:
        try:
            # Handle both time formats (HH:MM:SS from Athena vs hour number from Redshift)
            if df[time_col].dtype == 'object':
                # Athena format: "14:00:00"
                df["datetime"] = pd.to_datetime(
                    df[date_col].astype(str) + " " + df[time_col].astype(str)
                )
            else:
                # Redshift format: hour number (0-23)
                df["datetime"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[time_col], unit='h')
        except:
            logger.warning("Error creating datetime column")

    return df


def aggregate_timeseries(df):
    """
    Aggregate data by timestamp.
    From your data_processing.py.
   
    Args:
        df: DataFrame with datetime column
   
    Returns:
        Aggregated DataFrame
    """
    try:
        if df.empty:
            return df
       
        logger.info("Aggregating by timestamp")
       
        # Numeric columns to sum
        sum_cols = [
            "lossadjustedload",
            "baseload",
            "metercount",
            "loadbl",
            "loadlal",
            "loadmetercount",
            "genbl",
            "genlal",
            "genmetercount",
        ]
       
        # Create aggregation dict
        agg_dict = {col: 'sum' for col in sum_cols if col in df.columns}
       
        # Categorical columns - take first
        cat_cols = ['loadprofile', 'submission']
        for col in cat_cols:
            if col in df.columns:
                agg_dict[col] = 'first'
       
        # Aggregate
        before_count = len(df)
        agg_df = df.groupby('datetime').agg(agg_dict).reset_index()
        logger.info(f"Aggregated rows: {before_count} → {len(agg_df)}")
       
        return agg_df
   
    except Exception as e:
        logger.error(f"Error in aggregation: {e}")
        return df


def handle_missing(df):
    """
    Handle missing values

    Args:
        df: DataFrame with potential missing values

    Returns:
        DataFrame with handled missing values
    """
    # Original implementation - unchanged
    try:
        if df.empty:
            return df

        logger.info("Handling missing values")

        # Key numeric columns
        numeric_cols = [
            "lossadjustedload",
            "baseload",
            "metercount",
            "loadbl",
            "loadlal",
            "loadmetercount",
            "genbl",
            "genlal",
            "genmetercount",
        ]

        # Only process columns that exist
        cols_to_fix = [col for col in numeric_cols if col in df.columns]

        # Log missing counts
        missing_before = {col: df[col].isna().sum() for col in cols_to_fix}
        total_missing = sum(missing_before.values())

        if total_missing > 0:
            logger.info(f"Missing values before: {missing_before}")

            # Use time-based interpolation
            temp_df = df.set_index("datetime")
            for col in cols_to_fix:
                if df[col].isna().sum() > 0:
                    temp_df[col] = temp_df[col].interpolate(method="time")

            df = temp_df.reset_index()

            # Fill any remaining NaNs with ffill/bfill
            for col in cols_to_fix:
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

            # Check remaining missing
            missing_after = {col: df[col].isna().sum() for col in cols_to_fix}
            logger.info(f"Missing values after: {missing_after}")

        return df

    except Exception as e:
        import traceback
        logger.error(f"Error handling missing values: {e}")
        logger.error(traceback.format_exc())
        return df


def create_normalized_load_features(df, forecast_delay_days=14):
    """
    Create load features normalized by meter counts and other ratios to remove
    data leakage from raw load values while preserving useful information.
    Modified to respect forecast delay of 14 days.
   
    Args:
        df: DataFrame with load and meter count columns
        forecast_delay_days: Number of days delay in data availability
       
    Returns:
        DataFrame with normalized load features
    """
    try:
        if df.empty:
            return df
       
        logger.info("Creating normalized load features")
       
        # Make a copy to avoid warnings
        df = df.copy()
       
        # Check for required columns
        required_cols = ['loadlal', 'genlal', 'loadmetercount', 'genmetercount', 'metercount']
        missing_cols = [col for col in required_cols if col not in df.columns]
       
        if missing_cols:
            logger.warning(f"Cannot create normalized features. Missing columns: {missing_cols}")
            return df
           
        # Create features safe for forecasting (avoid data leakage)
       
        # 1. Per-meter metrics (normalize by meter count)
        df['load_per_meter'] = df['loadlal'] / df['loadmetercount'].replace(0, np.nan)
        df['gen_per_meter'] = df['genlal'] / df['genmetercount'].replace(0, np.nan)
        df['net_load_per_meter'] = df['lossadjustedload'] / df['metercount'].replace(0, np.nan)
       
        # Fill NaN values with 0
        df['load_per_meter'] = df['load_per_meter'].fillna(0)
        df['gen_per_meter'] = df['gen_per_meter'].fillna(0)
        df['net_load_per_meter'] = df['net_load_per_meter'].fillna(0)
       
        # 2. Export/import indicators (these will be lagged later)
        df['is_net_export'] = (df['loadlal'] < np.abs(df['genlal'])).astype(int)
       
        # Create categorical export level
        # Create bins for export levels (0=import, 1=small export, 2=medium export, 3=large export)
        exportlat_ratio = -df['genlal'] / df['loadlal'].replace(0, np.nan)
        exportlat_ratio = exportlat_ratio.fillna(0)
       
        # Define export categories: none, low (<25%), medium (25-75%), high (>75%)
        conditions = [
            (exportlat_ratio <= 0),                # Import (no export)
            (exportlat_ratio > 0) & (exportlat_ratio <= 0.25),   # Low export
            (exportlat_ratio > 0.25) & (exportlat_ratio <= 0.75), # Medium export
            (exportlat_ratio > 0.75)                # High export
        ]
        values = [0, 1, 2, 3]
        df['export_level'] = np.select(conditions, values, default=0)
       
        # 3. Generation to load ratios (daily and hourly patterns)
        df['hourly_gen_ratio'] = np.abs(df['genlal'] / df['loadlal'].replace(0, np.nan))
        df['hourly_gen_ratio'] = df['hourly_gen_ratio'].fillna(0).clip(0, 10)  # Cap at 10x
       
        # Daily gen/load ratio - using historical data only
        # Group by date with appropriate shift to respect forecast delay
        shifted_genlal = df['genlal'].shift(forecast_delay_days * 24)
        shifted_loadlal = df['loadlal'].shift(forecast_delay_days * 24)
       
        if 'date' not in df.columns:
            df['date'] = df['datetime'].dt.date
           
        # Calculate historical daily ratios
        df['shifted_date'] = df['datetime'].dt.date - timedelta(days=forecast_delay_days)
        daily_gen = df.groupby('shifted_date')['genlal'].sum().abs()
        daily_load = df.groupby('shifted_date')['loadlal'].sum()
        daily_ratio = daily_gen / daily_load.replace(0, np.nan)
        daily_ratio = daily_ratio.fillna(0).clip(0, 5)  # Cap at 5x
       
        # Convert to DataFrame and merge
        daily_ratio_df = pd.DataFrame({'shifted_date': daily_ratio.index, 'daily_gen_load_ratio': daily_ratio.values})
        df = df.merge(daily_ratio_df, on='shifted_date', how='left')
       
        # Fill any missing values
        df['daily_gen_load_ratio'] = df['daily_gen_load_ratio'].fillna(0)
       
        # Clean up temporary column
        df = df.drop(columns=['shifted_date'])
       
        # Flag these features as requiring historical data
        for col in ['load_per_meter', 'gen_per_meter', 'net_load_per_meter',
                    'is_net_export', 'export_level', 'hourly_gen_ratio', 'daily_gen_load_ratio']:
            df[f'{col}_needs_lag'] = df[col]
           
        logger.info(f"Created normalized load features (forecast delay: {forecast_delay_days} days)")
        return df
       
    except Exception as e:
        import traceback
        logger.error(f"Error creating normalized load features: {e}")
        logger.error(traceback.format_exc())
        return df


def create_meter_features(df):
    """
    Create features based on meter counts that are suitable for forecasting.
   
    Args:
        df: DataFrame with meter counts
       
    Returns:
        DataFrame with meter features
    """
    try:
        if df.empty:
            return df
       
        logger.info("Creating meter count features")
       
        # Required columns
        meter_cols = ['metercount', 'loadmetercount', 'genmetercount']
        if not all(col in df.columns for col in meter_cols):
            logger.warning(f"Cannot create meter features. Missing columns: {[col for col in meter_cols if col not in df.columns]}")
            return df
           
        # Make a copy to avoid warnings
        df = df.copy()
       
        # 1. Calculate ratio of generation meters to load meters
        df['gen_meter_ratio'] = df['genmetercount'] / df['loadmetercount'].replace(0, np.nan)
        df['gen_meter_ratio'] = df['gen_meter_ratio'].fillna(0)
       
        # 2. Calculate historical growth rate (will be used with lags)
        # These metrics are suitable for forecasting when used with lags
       
        # Day-of-week meter patterns
        df['dow_metercount'] = df.groupby(['dayofweek', 'hour'])['metercount'].transform('mean')
        df['dow_loadmetercount'] = df.groupby(['dayofweek', 'hour'])['loadmetercount'].transform('mean')
        df['dow_genmetercount'] = df.groupby(['dayofweek', 'hour'])['genmetercount'].transform('mean')
       
        # Calculate moving average of meter counts
        df['metercount_ma7d'] = df.groupby(['hour'])['metercount'].shift(14*24).transform(
            lambda x: x.rolling(window=7*24, min_periods=1).mean()
        )
       
        df['loadmetercount_ma7d'] = df.groupby(['hour'])['loadmetercount'].shift(14*24).transform(
            lambda x: x.rolling(window=7*24, min_periods=1).mean()
        )
       
        df['genmetercount_ma7d'] = df.groupby(['hour'])['genmetercount'].shift(14*24).transform(
            lambda x: x.rolling(window=7*24, min_periods=1).mean()
        )
       
        # Ratios to moving averages - measures deviation from typical pattern
        # These will be useful when lagged
        df['metercount_ratio_to_ma'] = df['metercount'] / df['metercount_ma7d'].replace(0, np.nan)
        df['loadmetercount_ratio_to_ma'] = df['loadmetercount'] / df['loadmetercount_ma7d'].replace(0, np.nan)
        df['genmetercount_ratio_to_ma'] = df['genmetercount'] / df['genmetercount_ma7d'].replace(0, np.nan)
       
        # Fill NaN values
        ratio_cols = ['metercount_ratio_to_ma', 'loadmetercount_ratio_to_ma', 'genmetercount_ratio_to_ma']
        for col in ratio_cols:
            df[col] = df[col].fillna(1.0)
       
        logger.info("Created meter count features")
        return df
       
    except Exception as e:
        import traceback
        logger.error(f"Error creating meter features: {e}")
        logger.error(traceback.format_exc())
        return df


def create_transition_features(df, forecast_delay_days=14):
    """
    Create features specific to handling transitions between export and import conditions.
    Modified to respect forecast delay of 14 days.
   
    Args:
        df: DataFrame with load and generation data
        forecast_delay_days: Number of days delay in data availability
       
    Returns:
        DataFrame with added transition features
    """
    try:
        if df.empty:
            return df
           
        logger.info(f"Creating transition features (forecast delay: {forecast_delay_days} days)")
       
        # Make a copy to avoid warnings
        df = df.copy()
       
        # Required fields
        if not all(col in df.columns for col in ['loadlal', 'genlal']):
            logger.warning("Cannot create transition features - missing required columns")
            return df
       
        # 1. Calculate net load (should already exist as lossadjustedload, but confirm)
        if 'lossadjustedload' not in df.columns:
            df['lossadjustedload'] = df['loadlal'] - np.abs(df['genlal'])
       
        # 2. Create export/import flags
        df['is_net_export'] = (df['lossadjustedload'] < 0).astype(int)
        df['is_net_import'] = (df['lossadjustedload'] >= 0).astype(int)
       
        # 3. Create historical export/import pattern features
        # These features use only historically available data
       
        # Calculate typical export/import patterns by hour and day of week
        # Here we use a groupby with the delay built in
        df['historical_hour'] = df['hour']
        df['historical_dayofweek'] = df['dayofweek']
       
        # Calculate probability of export by hour and day of week using shifted data
        export_prob_by_hour = df.groupby('historical_hour')['is_net_export'].shift(forecast_delay_days * 24).rolling(
            window=30 * 24, min_periods=24).mean()
        df['historical_export_probability_by_hour'] = export_prob_by_hour
       
        export_prob_by_hour_dow = df.groupby(['historical_hour', 'historical_dayofweek'])['is_net_export'].shift(
            forecast_delay_days * 24).rolling(window=4 * 24, min_periods=1).mean()
        df['historical_export_probability_by_hour_dow'] = export_prob_by_hour_dow
       
        # Fill NaN values
        df['historical_export_probability_by_hour'] = df['historical_export_probability_by_hour'].fillna(0.5)
        df['historical_export_probability_by_hour_dow'] = df['historical_export_probability_by_hour_dow'].fillna(0.5)
       
        # 4. Net load cross-zero: Add features to identify when system crosses from net export to import or vice versa
        # Calculate how close to zero the net load is (feature will be useful when lagged)
        df['net_load_proximity_to_zero'] = np.abs(df['lossadjustedload']).clip(0, 100000)
       
        # Identify potential transition periods based on time patterns
        morning_transition_hours = [7, 8, 9, 10]  # Typically when solar starts producing
        evening_transition_hours = [16, 17, 18, 19]  # Typically when solar decreases
       
        df['is_morning_transition_period'] = df['hour'].isin(morning_transition_hours).astype(int)
        df['is_evening_transition_period'] = df['hour'].isin(evening_transition_hours).astype(int)
       
        # 5. Create transition tendency features based on historical data
        # Calculate the frequency of transitions by hour using shifted data
        hour_transitions = df.groupby('hour')['is_net_export'].shift(forecast_delay_days * 24).diff().abs()
        transition_freq_by_hour = df.groupby('hour')[hour_transitions.name].transform('mean')
        df['historical_transition_frequency_by_hour'] = transition_freq_by_hour
       
        # Identify hours with high transition probability
        df['high_transition_probability_hour'] = (df['historical_transition_frequency_by_hour'] >
                                                df['historical_transition_frequency_by_hour'].mean()).astype(int)
       
        # Clean up temporary columns
        df = df.drop(columns=['historical_hour', 'historical_dayofweek'])
       
        # Flag these features as requiring historical data
        for col in ['is_net_export', 'is_net_import', 'net_load_proximity_to_zero']:
            df[f'{col}_needs_lag'] = df[col]
           
        logger.info("Created transition features")
        return df
       
    except Exception as e:
        import traceback
        logger.error(f"Error creating transition features: {e}")
        logger.error(traceback.format_exc())
        return df


def create_forecast_dataframe(forecast_date, config, forecast_delay_days=14):
    """
    Create a dataframe for forecasting load on a specific date with all
    required features based on the specified data delay.
   
    Args:
        forecast_date: Date to forecast for (datetime.date or string YYYY-MM-DD)
        forecast_delay_days: Number of days delay in data availability (default: 14)
       
    Returns:
        DataFrame with all required features ready for model prediction
    """
    try:
        import pytz
       
        logger.info(f"Creating forecast dataframe for {forecast_date} (data delay: {forecast_delay_days} days)")
       
        # Convert forecast_date to datetime if it's a string
        if isinstance(forecast_date, str):
            forecast_date = pd.to_datetime(forecast_date).date()
        elif isinstance(forecast_date, datetime):
            forecast_date = forecast_date.date()
           
        # Create 24 rows for the forecast date (one for each hour)
        forecast_hours = []
        san_diego_tz = pytz.timezone('America/Los_Angeles')
        for hour in range(24):
            # Create a datetime with the specified hour
            dt = datetime.combine(forecast_date, dt_time(hour=hour))
            # Localize to San Diego timezone
            dt = san_diego_tz.localize(dt)
            # Append to list
            forecast_hours.append(dt)

        # Convert to pandas timestamps and remove timezone for consistency
        forecast_hours = pd.DatetimeIndex([
            pd.Timestamp(dt).tz_localize(None) for dt in forecast_hours
        ])

        # Create base dataframe with datetime
        forecast_df = pd.DataFrame({'datetime': forecast_hours})

        # logger.info(f"list of columns in forecast dataframe: {list(forecast_df.columns)}")
               
        # Add date
        forecast_df['date'] = forecast_df['datetime'].dt.date
       
        # Add time features
        forecast_df['hour'] = forecast_df['datetime'].dt.hour
        forecast_df['dayofweek'] = forecast_df['datetime'].dt.dayofweek
        forecast_df['month'] = forecast_df['datetime'].dt.month
        forecast_df['year'] = forecast_df['datetime'].dt.year
        forecast_df['day_of_year'] = forecast_df['datetime'].dt.dayofyear

        # logger.info(f"list of columns in forecast dataframe after date time features: {list(forecast_df.columns)}")
       
        # Add cyclical encodings
        forecast_df['hour_sin'] = np.sin(2 * np.pi * forecast_df['hour'] / 24)
        forecast_df['hour_cos'] = np.cos(2 * np.pi * forecast_df['hour'] / 24)
        forecast_df['dow_sin'] = np.sin(2 * np.pi * forecast_df['dayofweek'] / 7)
        forecast_df['dow_cos'] = np.cos(2 * np.pi * forecast_df['dayofweek'] / 7)
        forecast_df['month_sin'] = np.sin(2 * np.pi * forecast_df['month'] / 12)
        forecast_df['month_cos'] = np.cos(2 * np.pi * forecast_df['month'] / 12)
        forecast_df['day_of_year_sin'] = np.sin(2 * np.pi * forecast_df['day_of_year'] / 365.25)
        forecast_df['day_of_year_cos'] = np.cos(2 * np.pi * forecast_df['day_of_year'] / 365.25)

        # logger.info(f"list of columns in forecast dataframe after cyclical encodings: {list(forecast_df.columns)}")
       
        # Create calendar features
        forecast_df['is_weekend'] = (forecast_df['dayofweek'] >= 5).astype(int)
        forecast_df['is_business_hour'] = ((forecast_df['hour'] >= 8) &
                                        (forecast_df['hour'] < 18) &
                                        (forecast_df['dayofweek'] < 5)).astype(int)
        forecast_df['is_evening'] = ((forecast_df['hour'] >= 18) &
                                   (forecast_df['hour'] < 22)).astype(int)
        forecast_df['is_night'] = ((forecast_df['hour'] >= 22) |
                                 (forecast_df['hour'] < 6)).astype(int)
        forecast_df['is_morning'] = ((forecast_df['hour'] >= 6) &
                                    (forecast_df['hour'] < 11)).astype(int)
        forecast_df['is_afternoon'] = ((forecast_df['hour'] >= 11) &
                                     (forecast_df['hour'] < 18)).astype(int)
        # logger.info(f"list of columns in forecast dataframe with calendar features: {list(forecast_df.columns)}")
       
        # Day type features
        forecast_df['is_monday'] = (forecast_df['dayofweek'] == 0).astype(int)
        forecast_df['is_friday'] = (forecast_df['dayofweek'] == 4).astype(int)
        forecast_df['is_saturday'] = (forecast_df['dayofweek'] == 5).astype(int)
        forecast_df['is_sunday'] = (forecast_df['dayofweek'] == 6).astype(int)

        # logger.info(f"list of columns in forecast dataframe with day type features: {list(forecast_df.columns)}")
       
        # Add holiday features (if holidays package is available)
        try:
            import holidays
            us_holidays = holidays.US()
            forecast_df['is_holiday'] = forecast_df['datetime'].dt.date.isin(us_holidays).astype(int)
           
            # Day before/after holiday
            holiday_dates = [holiday_date for holiday_date in us_holidays.keys()]
            day_before_holiday_dates = [holiday_date - timedelta(days=1) for holiday_date in holiday_dates]
            day_after_holiday_dates = [holiday_date + timedelta(days=1) for holiday_date in holiday_dates]
           
            forecast_df['is_day_before_holiday'] = forecast_df['datetime'].dt.date.isin(
                day_before_holiday_dates).astype(int)
            forecast_df['is_day_after_holiday'] = forecast_df['datetime'].dt.date.isin(
                day_after_holiday_dates).astype(int)
        except ImportError:
            forecast_df['is_holiday'] = 0
            forecast_df['is_day_before_holiday'] = 0
            forecast_df['is_day_after_holiday'] = 0

        # logger.info(f"list of columns in forecast dataframe with holiday features: {list(forecast_df.columns)}")
       
        # Month start/end
        forecast_df['is_month_start'] = forecast_df['datetime'].dt.is_month_start.astype(int)
        forecast_df['is_month_end'] = forecast_df['datetime'].dt.is_month_end.astype(int)

        # logger.info(f"list of columns in forecast dataframe with month start-end features: {list(forecast_df.columns)}")

        morning_peak_hours = config.get('MORNING_PEAK_HOURS')
        solar_period_hours = config.get('SOLAR_PERIOD_HOURS')
        evening_ramp_hours = config.get('EVENING_RAMP_HOURS')
        evening_peak_hours = config.get('EVENING_PEAK_HOURS')
       
        # Add duck curve period flags
        forecast_df['is_morning_peak'] = ((forecast_df['hour'] >= morning_peak_hours[0]) &
                                         (forecast_df['hour'] <= morning_peak_hours[1])).astype(int)
        forecast_df['is_solar_period'] = ((forecast_df['hour'] >= solar_period_hours[0]) &
                                         (forecast_df['hour'] <= solar_period_hours[1])).astype(int)
        forecast_df['is_evening_ramp'] = ((forecast_df['hour'] >= evening_ramp_hours[0]) &
                                         (forecast_df['hour'] <= evening_ramp_hours[1])).astype(int)
        forecast_df['is_evening_peak'] = ((forecast_df['hour'] >= evening_peak_hours[0]) &
                                         (forecast_df['hour'] <= evening_peak_hours[1])).astype(int)

        # logger.info(f"list of columns in forecast dataframe with duck curve features: {list(forecast_df.columns)}")

        # Solar peak indicator
        forecast_df['is_solar_peak'] = forecast_df['hour'].isin([11, 12, 13, 14]).astype(int)
       
        # Add time window features
        forecast_df['is_early_morning'] = ((forecast_df['hour'] >= 5) &
                                          (forecast_df['hour'] <= 8)).astype(int)
        forecast_df['is_morning_ramp'] = ((forecast_df['hour'] >= 8) &
                                         (forecast_df['hour'] <= 11)).astype(int)
        forecast_df['is_midday'] = ((forecast_df['hour'] >= 11) &
                                   (forecast_df['hour'] <= 15)).astype(int)
        forecast_df['is_afternoon_decline'] = ((forecast_df['hour'] >= 15) &
                                              (forecast_df['hour'] <= 17)).astype(int)
        forecast_df['is_evening_ramp_refined'] = ((forecast_df['hour'] >= 17) &
                                                 (forecast_df['hour'] <= 20)).astype(int)
        forecast_df['is_night_period'] = ((forecast_df['hour'] >= 20) |
                                         (forecast_df['hour'] <= 5)).astype(int)

        # logger.info(f"list of columns in forecast dataframe with time window features: {list(forecast_df.columns)}")
       
        # Add position within periods
        period_defs = [
            ('early_morning', 5, 8),
            ('morning_ramp', 8, 11),
            ('midday', 11, 15),
            ('afternoon_decline', 15, 17),
            ('evening_ramp', 17, 20),
            ('night_period', 20, 5)
        ]
       
        for period_name, start, end in period_defs:
            if start < end:
                # Normal period within same day
                mask = (forecast_df['hour'] >= start) & (forecast_df['hour'] <= end)
                forecast_df[f'{period_name}_position'] = 0
                forecast_df.loc[mask, f'{period_name}_position'] = (
                    forecast_df.loc[mask, 'hour'] - start) / (end - start)
            else:
                # Period spanning midnight
                mask = (forecast_df['hour'] >= start) | (forecast_df['hour'] <= end)
                forecast_df[f'{period_name}_position'] = 0
               
                # Before midnight
                mask_before = (forecast_df['hour'] >= start)
                if mask_before.any():
                    forecast_df.loc[mask_before, f'{period_name}_position'] = \
                        (forecast_df.loc[mask_before, 'hour'] - start) / ((24 - start) + end)
               
                # After midnight
                mask_after = (forecast_df['hour'] <= end)
                if mask_after.any():
                    forecast_df.loc[mask_after, f'{period_name}_position'] = \
                        ((forecast_df.loc[mask_after, 'hour'] + 24) - start) / ((24 - start) + end)

        # logger.info(f"list of columns in forecast dataframe with period information: {list(forecast_df.columns)}")
       
        # Hour of day position
        forecast_df['hour_of_day_position'] = forecast_df['hour'] / 24.0
       
        # Add specific ramp hour features
        forecast_df['morning_ramp_hour'] = 0
        morning_mask = (forecast_df['hour'] >= 8) & (forecast_df['hour'] <= 11)
        forecast_df.loc[morning_mask, 'morning_ramp_hour'] = forecast_df.loc[morning_mask, 'hour'] - 8
       
        forecast_df['evening_ramp_hour'] = 0
        evening_mask = (forecast_df['hour'] >= 17) & (forecast_df['hour'] <= 20)
        forecast_df.loc[evening_mask, 'evening_ramp_hour'] = forecast_df.loc[evening_mask, 'hour'] - 17

        # logger.info(f"list of columns in forecast dataframe with ramp hour features: {list(forecast_df.columns)}")

        # Identify potential transition periods based on time patterns
        morning_transition_hours = [7, 8, 9, 10]  # Typically when solar starts producing
        evening_transition_hours = [16, 17, 18, 19]  # Typically when solar decreases
       
        forecast_df['is_morning_transition_period'] = forecast_df['hour'].isin(morning_transition_hours).astype(int)
        forecast_df['is_evening_transition_period'] = forecast_df['hour'].isin(evening_transition_hours).astype(int)

        # logger.info(f"list of columns in forecast dataframe with transition period features: {list(forecast_df.columns)}")
       
        # Set default values for metadata
        forecast_df['loadprofile'] = config.get('CUSTOMER_PROFILE', 'RES')
        forecast_df['submission'] = 'Forecast'

        # logger.info(f"list of columns in forecast dataframe with metadata features: {list(forecast_df.columns)}")

        # Now we need to add all the lag features from historical data
        # Calculate the dates we need historical data for
        # Base lag days from the forecast delay
        lag_days = [forecast_delay_days]  # Minimum delay
       
        # Add 1 day to minimum delay
        lag_days.append(forecast_delay_days + 1)
       
        # Add weekly lags (from minimum delay + multiples of 7)
        lag_days.extend([
            forecast_delay_days + 7,     # 21 days if delay is 14
            forecast_delay_days + 14,    # 28 days if delay is 14
            forecast_delay_days + 21     # 35 days if delay is 14
        ])
       
        # Convert to hours
        lag_hours = [days * 24 for days in lag_days]
       
        # Calculate dates for each lag
        lag_dates = [forecast_date - timedelta(days=days) for days in lag_days]
       
        # Map lags to user-friendly names for logging
        lag_map = {
            lag_hours[0]: f"{forecast_delay_days}d",
            lag_hours[1]: f"{forecast_delay_days+1}d",
            lag_hours[2]: f"{forecast_delay_days+7}d",
            lag_hours[3]: f"{forecast_delay_days+14}d",
            lag_hours[4]: f"{forecast_delay_days+21}d"
        }
       
        # Log the lag dates we're querying
        logger.info(f"Will query historical data for lag dates:")
        for i, date in enumerate(lag_dates):
            logger.info(f"  {lag_map[lag_hours[i]]}: {date.strftime('%Y-%m-%d')}")
       
        # Query historical data for all lag dates
        # Construct date range to include all days we need
        start_date = min(lag_dates) - timedelta(days=1)  # Add buffer day
        end_date = max(lag_dates) + timedelta(days=1)    # Add buffer day
       
        # Query for historical data
        logger.info(f"Querying data from {start_date} to {end_date}")
       
        # Query data for the entire date range at once
        current_date = datetime.now()
        historical_df = query_data(
            config,
            current_date=datetime.strptime(str(forecast_date), '%Y-%m-%d') if isinstance(forecast_date, str) else forecast_date,
            load_profile=config.get('LOAD_PROFILE', 'RES'),
            rate_group_filter=None,  # Will use config.RATE_GROUP_FILTER_CLAUSE automatically
            use_cache=True
        )
       
        # Check if we got data
        if historical_df.empty:
            logger.error("No historical data available for creating forecast dataframe")
            return None
       
        logger.info(f"Historical data shape: {historical_df.shape}")
        logger.info(f"Date range: {historical_df['datetime'].min()} to {historical_df['datetime'].max()}")
       
        # Ensure datetime is present and properly formatted
        historical_df = convert_column_types(historical_df)
       
        # Make sure we have datetime
        if 'datetime' not in historical_df.columns:
            if 'tradedate' in historical_df.columns and 'tradetime' in historical_df.columns:
                historical_df['datetime'] = pd.to_datetime(
                    historical_df['tradedate'].astype(str) + ' ' +
                    historical_df['tradetime'].astype(str)
                )
            else:
                logger.error("Cannot create datetime from historical data")
                return None
       
        # Filter to the date range we need
        start_dt = datetime.combine(start_date, dt_time.min)
        end_dt = datetime.combine(end_date, dt_time.max)
       
        historical_df = historical_df[
            (historical_df['datetime'] >= start_dt) &
            (historical_df['datetime'] <= end_dt)
        ]
       
        logger.info(f"Retrieved {len(historical_df)} historical records")

        historical_df = aggregate_timeseries(historical_df)
        # logger.info(f"After timestamp aggregation: {len(historical_df)} rows")

        historical_df = handle_missing(historical_df)
        # logger.info(f"After handling missing data: {len(historical_df)} rows")
       
        # Process data to create additional features
        # These are needed to create the lag features
        if 'hour' not in historical_df.columns:
            historical_df['hour'] = historical_df['datetime'].dt.hour
       
        if 'dayofweek' not in historical_df.columns:
            historical_df['dayofweek'] = historical_df['datetime'].dt.dayofweek
       
        if 'date' not in historical_df.columns:
            historical_df['date'] = historical_df['datetime'].dt.date

        # logger.info(f"list of columns: {list(historical_df.columns)}")
       
        # Create normalized load features
        historical_df = create_normalized_load_features(historical_df, forecast_delay_days)

        # logger.info(f"list of columns in historical_df after create_normalized_load_features: {list(historical_df.columns)}")
       
        # Create transition features
        historical_df = create_transition_features(historical_df, forecast_delay_days)

        # logger.info(f"list of columns in historical_df after create_transition_features: {list(historical_df.columns)}")
       
        # Create additional meter features
        historical_df = create_meter_features(historical_df)

        # logger.info(f"list of columns in historical_df after create_meter_features: {list(historical_df.columns)}")
       
        # Now map the historical features to the forecast dataframe based on the lags
        # For each lag days, extract the load data and add it to the forecast dataframe
       
        # Create lag-to-name mapping for features
        lag_feature_map = {
            lag_hours[0]: '336h',  # 14 days * 24 hours  (assuming delay is 14 days)
            lag_hours[1]: '360h',  # 15 days * 24 hours
            lag_hours[2]: '504h',  # 21 days * 24 hours
            lag_hours[3]: '672h',  # 28 days * 24 hours
            lag_hours[4]: '840h'   # 35 days * 24 hours
        }
       
        # For each lag and hour, add the historical value to the forecast dataframe
        for lag_hours_val, lag_suffix in lag_feature_map.items():
            # Calculate the lag date
            lag_days_val = lag_hours_val // 24
           
            # Create a mapping dictionary from (hour, dayofweek) to historical data
            lag_date = forecast_date - timedelta(days=lag_days_val)
           
            # Filter historical data for the lag date
            lag_data = historical_df[historical_df['datetime'].dt.date == lag_date]
           
            if lag_data.empty:
                logger.warning(f"No historical data found for {lag_date}")
                continue
           
            # Create mapping dictionaries for each feature we want to lag
            for feature in ['lossadjustedload', 'is_net_export', 'is_net_import', 'load_per_meter', 'gen_per_meter', 'genlal', 'loadlal']:
                if feature in historical_df.columns:
                    # Create hour to value mapping
                    hour_to_value = dict(zip(lag_data['hour'], lag_data[feature]))
                   
                    # Apply mapping to forecast dataframe
                    feature_name = f"{feature}_lag_{lag_suffix}"
                    forecast_df[feature_name] = forecast_df['hour'].map(hour_to_value)
                   
                    logger.info(f"Added lag feature: {feature_name}")
        # logger.info(f"list of columns in forecast_df after adding lag features: {list(forecast_df.columns)}")

        # Create same day-of-week and hour features (load_same_dow_hour_Nw)
        for week_back in range(2, 6):  # 2-5 weeks back
            lag_days_val = week_back * 7
            lag_date = forecast_date - timedelta(days=lag_days_val)
           
            # Filter historical data for the same day of week
            same_dow = forecast_date.weekday()
            lag_data = historical_df[
                (historical_df['datetime'].dt.date == lag_date) &
                (historical_df['dayofweek'] == same_dow)
            ]
           
            if not lag_data.empty:
                # Create hour to load mapping
                hour_to_load = dict(zip(lag_data['hour'], lag_data['lossadjustedload']))
               
                # Apply mapping to forecast dataframe
                feature_name = f"load_same_dow_hour_{week_back}w"
                forecast_df[feature_name] = forecast_df['hour'].map(hour_to_load)
               
                logger.info(f"Added same-day-of-week feature: {feature_name}")

        # logger.info(f"list of columns after adding same-day-of-week features: {list(forecast_df.columns)}")

        # Create moving average features
        # For simplicity, use the most recent lag as base
        if f"lossadjustedload_lag_{lag_feature_map[lag_hours[0]]}" in forecast_df.columns:
            base_lag_feature = f"lossadjustedload_lag_{lag_feature_map[lag_hours[0]]}"
            forecast_df['load_ma_7d'] = forecast_df[base_lag_feature]
           
            # Add 3-week back moving average if available
            if f"lossadjustedload_lag_{lag_feature_map[lag_hours[2]]}" in forecast_df.columns:
                week3_lag_feature = f"lossadjustedload_lag_{lag_feature_map[lag_hours[2]]}"
                forecast_df['load_ma_7d_lag_3w'] = forecast_df[week3_lag_feature]

        # logger.info(f"list of columns after adding moving average features: {list(forecast_df.columns)}")

        # Create growth rate features
        # Week-over-week growth (14d to 21d)
        if all(f in forecast_df.columns for f in [
            f"lossadjustedload_lag_{lag_feature_map[lag_hours[0]]}",
            f"lossadjustedload_lag_{lag_feature_map[lag_hours[2]]}"
        ]):
            forecast_df['load_growth_w2w'] = (
                forecast_df[f"lossadjustedload_lag_{lag_feature_map[lag_hours[0]]}"] /
                forecast_df[f"lossadjustedload_lag_{lag_feature_map[lag_hours[2]]}"].replace(0, np.nan) - 1
            )
            forecast_df['load_growth_w2w'] = forecast_df['load_growth_w2w'].fillna(0).clip(-1, 1)
       
        # Previous week-over-week growth (21d to 28d)
        if all(f in forecast_df.columns for f in [
            f"lossadjustedload_lag_{lag_feature_map[lag_hours[2]]}",
            f"lossadjustedload_lag_{lag_feature_map[lag_hours[3]]}"
        ]):
            forecast_df['load_growth_prev_w2w'] = (
                forecast_df[f"lossadjustedload_lag_{lag_feature_map[lag_hours[2]]}"] /
                forecast_df[f"lossadjustedload_lag_{lag_feature_map[lag_hours[3]]}"].replace(0, np.nan) - 1
            )
            forecast_df['load_growth_prev_w2w'] = forecast_df['load_growth_prev_w2w'].fillna(0).clip(-1, 1)
 
        # logger.info(f"list of columns after adding growth features: {list(forecast_df.columns)}")

        # # Get the final list of available features
        # available_features = mark_forecast_available_features(forecast_df, forecast_delay_days)
       
        # Add pattern-based features from historical data
        # Get hourly pattern data
        # Historical general patterns by hour and day of week
        hr_dow_patterns = historical_df.groupby(['hour', 'dayofweek'])[
            ['is_net_export', 'metercount', 'loadmetercount', 'genmetercount']
        ].mean().reset_index()

        # logger.info(f"HR_DOW_PATTERNS: {hr_dow_patterns.head(5)}")
        # logger.info(f"HR_DOW_PATTERNS COLUMNS: {list(hr_dow_patterns.columns)}")
        # logger.info(f"FORECAST_DF COLUMNS BEFORE: {list(forecast_df.columns)}")
       
        # Merge the patterns with forecast data
        forecast_df = forecast_df.merge(
            hr_dow_patterns,
            on=['hour', 'dayofweek'],
            how='left',
            suffixes=('', '_pattern')
        )

        forecast_df = forecast_df.rename(columns={
            'is_net_export': 'is_net_export_pattern',
            'metercount': 'metercount_pattern',
            'loadmetercount': 'loadmetercount_pattern',
            'genmetercount': 'genmetercount_pattern'
        })

        # logger.info(f"FORECAST_DF COLUMNS AFTER 1: {list(forecast_df.columns)}")
       
        # Rename pattern columns
        pattern_cols = [col for col in forecast_df.columns if col.endswith('_pattern')]
        # logger.info(f"PATTERN_COLS: {pattern_cols}")
        for old_col in pattern_cols:
            new_col = old_col.replace('_pattern', '')
            forecast_df = forecast_df.rename(columns={old_col: f'dow_{new_col}'})

        # logger.info(f"FORECAST_DF COLUMNS AFTER 2: {list(forecast_df.columns)}")
       
        # Create transition period likelihood
        hour_transition_pattern = historical_df.groupby('hour')['is_net_export'].diff().abs().fillna(0)
        historical_df['transition'] = hour_transition_pattern
        transition_freq = historical_df.groupby('hour')['transition'].mean().reset_index()
        transition_freq.columns = ['hour', 'historical_transition_frequency_by_hour']
       
        # Merge transition frequency
        forecast_df = forecast_df.merge(transition_freq, on='hour', how='left')
       
        # Create high transition probability flag
        if 'historical_transition_frequency_by_hour' in forecast_df.columns:
            mean_transition = forecast_df['historical_transition_frequency_by_hour'].mean()
            forecast_df['high_transition_probability_hour'] = (
                forecast_df['historical_transition_frequency_by_hour'] > mean_transition
            ).astype(int)
       
        # Calculate export probability by hour
        export_prob = historical_df.groupby('hour')['is_net_export'].mean().reset_index()
        export_prob.columns = ['hour', 'historical_export_probability_by_hour']
       
        # Merge export probability
        forecast_df = forecast_df.merge(export_prob, on='hour', how='left')
       
        logger.info(f"Created forecast dataframe with {len(forecast_df)} rows and {len(forecast_df.columns)} columns")
        return forecast_df
   
    except Exception as e:
        logger.error(f"Error creating forecast dataframe: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
