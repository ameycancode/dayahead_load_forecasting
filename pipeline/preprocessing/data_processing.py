# data_processing.py
# Enhanced data processing functions for energy forecasting pipeline

import logging
import os
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import boto3
from pyathena import connect
from pyathena.pandas.util import as_pandas

from configs import config

# Get logger
logger = logging.getLogger(__name__)


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
        
        # Get ALL query results with pagination - THIS IS THE KEY FIX
        df = get_all_paginated_results(redshift_data_client, query_id)
        
        logger.info(f"Query completed successfully. Retrieved {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error executing query via Data API: {str(e)}")
        logger.error(traceback.format_exc())
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

def get_all_paginated_results(redshift_data_client, query_id):
    """
    Get ALL results with proper pagination
    This is the main fix for your missing rows
    """
    all_records = []
    column_metadata = None
    next_token = None
    page_count = 0
    
    try:
        while True:
            page_count += 1
            logger.info(f"Fetching results page {page_count}...")
            
            # Prepare request parameters
            request_params = {'Id': query_id}
            if next_token:
                request_params['NextToken'] = next_token
            
            # Get results page
            result_response = redshift_data_client.get_statement_result(**request_params)
            
            # Get column metadata from first page only
            if column_metadata is None:
                column_metadata = result_response.get('ColumnMetadata', [])
                logger.info(f"Query has {len(column_metadata)} columns")
            
            # Get records from this page
            page_records = result_response.get('Records', [])
            all_records.extend(page_records)
            
            logger.info(f"Page {page_count}: Retrieved {len(page_records)} records (Total: {len(all_records)})")
            
            # Check if there are more pages
            next_token = result_response.get('NextToken')
            if not next_token:
                logger.info(f" Pagination complete. Total pages: {page_count}, Total records: {len(all_records)}")
                break
        
        # Convert all records to DataFrame
        df = convert_data_api_result_to_dataframe(column_metadata, all_records)
        return df
        
    except Exception as e:
        logger.error(f"Error in paginated result retrieval: {str(e)}")

def convert_data_api_result_to_dataframe(column_metadata, all_records):
    """
    Convert paginated results to DataFrame
    """
    try:
        # Get column names
        column_names = [col['name'] for col in column_metadata]
        
        logger.info(f"Converting {len(all_records)} records with {len(column_names)} columns")
        
        if not all_records:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=column_names)
        
        # Convert records to list of lists (same logic as your original)
        data_rows = []
        for record in all_records:
            row = []
            for field in record:
                # Extract value based on type (keeping your original logic)
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
        
        logger.info(f" DataFrame created: {len(df)} rows, {len(column_names)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error converting paginated results to DataFrame: {str(e)}")
        raise

def validate_row_counts(query, expected_count=None):
    """
    Add this function to validate row counts and test the pagination fix
    """
    try:
        # Get count first
        count_query = f"SELECT COUNT(*) FROM ({query}) AS subquery"
        
        count_df = execute_redshift_query_via_data_api(
            count_query,
            database=config.REDSHIFT_DATABASE,
            cluster_identifier=config.REDSHIFT_CLUSTER_IDENTIFIER,
            db_user=config.REDSHIFT_DB_USER,
            region=config.REDSHIFT_REGION
        )
        
        actual_count = count_df.iloc[0, 0]
        logger.info(f" Query COUNT(*) result: {actual_count}")
        
        if expected_count:
            logger.info(f" Expected count: {expected_count}")
            if actual_count != expected_count:
                logger.warning(f" Count mismatch! Expected {expected_count}, got {actual_count}")
        
        # Now get actual data
        data_df = execute_redshift_query_via_data_api(
            query,
            database=config.REDSHIFT_DATABASE,
            cluster_identifier=config.REDSHIFT_CLUSTER_IDENTIFIER,
            db_user=config.REDSHIFT_DB_USER,
            region=config.REDSHIFT_REGION
        )
        
        retrieved_count = len(data_df)
        logger.info(f" Retrieved data rows: {retrieved_count}")
        
        if retrieved_count == actual_count:
            logger.info(" Row counts match! Pagination working correctly.")
        else:
            logger.error(f" Still missing rows! Expected {actual_count}, got {retrieved_count}")
            
        return data_df, actual_count, retrieved_count
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


# Updated query_data function to use 3-year logic by default
def query_data(
    current_date=None,
    load_profile=None,
    use_cache=None,
    query_limit=None,
):
    """
    Query data from Redshift or Athena (based on config) with 3-year logic:
    1. Fetch ALL available Final submission data (no date filter)
    2. Find latest Final date and fetch Initial data to fill gaps
    3. Ensure 3-year data period for training
    """
    try:
        if current_date is None:
            current_date = datetime.now()

        # Check database type from config
        database_type = getattr(config, 'DATABASE_TYPE', 'redshift')
        
        if database_type == 'redshift':
            logger.info("Using Redshift data source with 3-year logic")
            return query_data_redshift(current_date, load_profile, query_limit)
        else:
            logger.info("Using Athena data source (legacy)")
            return query_data_athena(current_date, load_profile, use_cache, query_limit)
            
    except Exception as e:
        logger.error(f"Error in query_data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def query_data_redshift(current_date, load_profile, query_limit=None):
    """
    Query data from Redshift with 3-year logic:
    1. Fetch ALL available Final submission data (no date filter)
    2. Determine latest Final date available
    3. Fetch Initial data from day after latest Final to current_date - days_delay
    4. Ensure 3-year data period for training
    
    Args:
        current_date: Current date (June 17, 2025)
        load_profile: Load profile to filter by
        query_limit: Optional query limit for testing
        
    Returns:
        DataFrame with combined data covering 3 years
    """
    try:
        # Use defaults from config if not provided
        load_profile = load_profile or config.DEFAULT_LOAD_PROFILE
        days_delay = config.INITIAL_SUBMISSION_DELAY  # 14 days
        
        # Get rate group filter clause
        rate_group_filter = None
        if hasattr(config, "RATE_GROUP_FILTER_CLAUSE"):
            rate_group_filter = config.RATE_GROUP_FILTER_CLAUSE
            logger.info(f"Using dynamic RATE_GROUP_FILTER_CLAUSE: {rate_group_filter}")
        elif hasattr(config, "RATE_GROUP_FILTER"):
            rate_group_filter = f"rategroup LIKE '{config.RATE_GROUP_FILTER}'"
            logger.warning(f"Fallback to legacy RATE_GROUP_FILTER: {rate_group_filter}")

        logger.info(f"Querying Redshift data as of: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"Load profile: {load_profile}")
        logger.info(f"Days delay: {days_delay}")

        # Step 1: Get the latest available Final submission date (no date filter)
        latest_final_date = get_latest_final_submission_date(load_profile, rate_group_filter)
        logger.info(f"Latest Final submission date found: {latest_final_date}")
        
        # Step 2: Calculate data collection boundaries
        # Initial data cutoff = current_date - days_delay (June 3, 2025)
        initial_cutoff_date = current_date - timedelta(days=days_delay)
        
        # Ensure we have 3 years of data ending at initial_cutoff_date
        three_years_start = initial_cutoff_date - timedelta(days=3*365)  # June 3, 2022
        
        logger.info(f"Data collection period: {three_years_start.strftime('%Y-%m-%d')} to {initial_cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"Latest Final data available until: {latest_final_date.strftime('%Y-%m-%d')}")
        logger.info(f"Initial data period: {(latest_final_date + timedelta(days=1)).strftime('%Y-%m-%d')} to {initial_cutoff_date.strftime('%Y-%m-%d')}")

        # Step 3: Query Final data (ALL available data up to latest Final date, but within 3-year window)
        final_start_date = max(three_years_start, latest_final_date - timedelta(days=3*365))
        final_df = query_redshift_final_data(
            start_date=final_start_date,
            end_date=latest_final_date,
            load_profile=load_profile, 
            rate_group_filter=rate_group_filter, 
            query_limit=query_limit
        )
        final_df = convert_column_types(final_df)
        logger.info(f"Retrieved {len(final_df)} rows of Final data from {final_start_date} to {latest_final_date}")

        # Step 4: Query Initial data (from day after latest Final to initial_cutoff_date)
        initial_start_date = latest_final_date + timedelta(days=1)
        
        if initial_start_date <= initial_cutoff_date:
            initial_df = query_redshift_initial_data(
                start_date=initial_start_date,
                end_date=initial_cutoff_date,
                load_profile=load_profile,
                rate_group_filter=rate_group_filter,
                query_limit=query_limit
            )
            initial_df = convert_column_types(initial_df)
            logger.info(f"Retrieved {len(initial_df)} rows of Initial data from {initial_start_date} to {initial_cutoff_date}")
        else:
            initial_df = pd.DataFrame()
            logger.info("No Initial data needed - latest Final data is very recent")

        # Step 5: Combine datasets
        if final_df.empty and initial_df.empty:
            logger.warning("No data retrieved from either Final or Initial submissions")
            return pd.DataFrame()
        elif final_df.empty:
            logger.warning("No Final submission data available, using only Initial data")
            combined_df = initial_df
        elif initial_df.empty:
            logger.info("Using only Final submission data")
            combined_df = final_df
        else:
            combined_df = pd.concat([final_df, initial_df], ignore_index=True)
            logger.info(f"Combined data: {len(combined_df)} rows")

        return combined_df

    except Exception as e:
        logger.error(f"Error querying Redshift data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()



def query_redshift_final_data(start_date, end_date, load_profile, rate_group_filter, query_limit=None):
    """
    Query Redshift for Final submission data within date range
    """
    try:
        # Build WHERE clause
        where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.SUBMISSION_TYPE_FINAL}'"
        ]

        if rate_group_filter:
            where_clauses.append(rate_group_filter)
            logger.info(f"Added rate group filter clause: {rate_group_filter}")

        # Add date range filter
        where_clauses.append(f"tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'")
        where_clauses.append(f"tradedatelocal <= '{end_date.strftime('%Y-%m-%d')}'")

        where_clause = " AND ".join(where_clauses)
        
        # Get schema and table from config
        schema_name = getattr(config, 'REDSHIFT_INPUT_SCHEMA', 'edp_cust_dev')
        table_name = getattr(config, 'REDSHIFT_INPUT_TABLE', 'caiso_sqmd')

        # Build query for Redshift
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

        if query_limit is not None and query_limit > 0:
            query += f" LIMIT {query_limit}"

        logger.info(f"Executing Final data query from {start_date} to {end_date}")
        final_df, expected_count, retrieved_count = validate_row_counts(query)
        
        logger.info(f"Final data - Expected: {expected_count}, Retrieved: {retrieved_count}")
        return final_df
        
    except Exception as e:
        logger.error(f"Error querying Redshift Final data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def query_redshift_initial_data(start_date, end_date, load_profile, rate_group_filter, query_limit=None):
    """
    Query Redshift for Initial submission data within date range
    """
    try:
        # Build WHERE clause for Initial data
        where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.SUBMISSION_TYPE_INITIAL}'"
        ]

        if rate_group_filter:
            where_clauses.append(rate_group_filter)

        # Add date range filter
        where_clauses.append(f"tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'")
        where_clauses.append(f"tradedatelocal <= '{end_date.strftime('%Y-%m-%d')}'")

        where_clause = " AND ".join(where_clauses)
        
        # Get schema and table from config
        schema_name = getattr(config, 'REDSHIFT_INPUT_SCHEMA', 'edp_cust_dev')
        table_name = getattr(config, 'REDSHIFT_INPUT_TABLE', 'caiso_sqmd')

        # Build query for Redshift
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

        if query_limit is not None and query_limit > 0:
            query += f" LIMIT {query_limit}"

        logger.info(f"Executing Initial data query from {start_date} to {end_date}")
        initial_df, expected_count, retrieved_count = validate_row_counts(query)

        logger.info(f"Initial data - Expected: {expected_count}, Retrieved: {retrieved_count}")
        return initial_df
        
    except Exception as e:
        logger.error(f"Error querying Redshift Initial data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_latest_final_submission_date(load_profile, rate_group_filter):
    """
    Get the latest date for which Final submission data is available
    
    Returns:
        datetime: Latest Final submission date
    """
    try:
        # Build WHERE clause for Final data
        where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.SUBMISSION_TYPE_FINAL}'"
        ]

        if rate_group_filter:
            where_clauses.append(rate_group_filter)

        where_clause = " AND ".join(where_clauses)
        
        # Get schema and table from config
        schema_name = getattr(config, 'REDSHIFT_INPUT_SCHEMA', 'edp_cust_dev')
        table_name = getattr(config, 'REDSHIFT_INPUT_TABLE', 'caiso_sqmd')

        # Query to get latest Final submission date
        query = f"""
        SELECT MAX(tradedatelocal) as latest_final_date
        FROM {schema_name}.{table_name}
        WHERE {where_clause}
        """

        logger.info(f"Getting latest Final submission date with query: {query}")
        
        result_df = execute_redshift_query_via_data_api(
            query,
            database=config.REDSHIFT_DATABASE,
            cluster_identifier=config.REDSHIFT_CLUSTER_IDENTIFIER,
            db_user=config.REDSHIFT_DB_USER,
            region=config.REDSHIFT_REGION
        )
        
        if result_df.empty or pd.isna(result_df.iloc[0, 0]):
            # Fallback to current date minus final delay if no data found
            fallback_date = datetime.now() - timedelta(days=config.FINAL_SUBMISSION_DELAY)
            logger.warning(f"No Final submission data found, using fallback date: {fallback_date}")
            return fallback_date
        
        latest_date = pd.to_datetime(result_df.iloc[0, 0])
        return latest_date.to_pydatetime()
        
    except Exception as e:
        logger.error(f"Error getting latest Final submission date: {str(e)}")
        # Fallback to current date minus final delay
        fallback_date = datetime.now() - timedelta(days=config.FINAL_SUBMISSION_DELAY)
        logger.warning(f"Using fallback date due to error: {fallback_date}")
        return fallback_date


def query_data_athena(
    current_date=None,
    load_profile=None,
    # rate_group_filter=None,
    use_cache=None,
    query_limit=None,
):
    """
    Query data from Athena combining Final and Initial submissions

    Args:
        current_date: Current date (defaults to today)
        load_profile: Load profile to filter by (default from config)
        # rate_group_filter: Rate group filter pattern (default from config)

    Returns:
        DataFrame with combined data
    """
    # Original implementation - unchanged
    try:
        if current_date is None:
            current_date = datetime.now()

        # Determine whether to use cache
        if use_cache is None:
            use_cache = config.USE_CSV_CACHE

        # Define cache paths
        os.makedirs(config.CSV_CACHE_DIR, exist_ok=True)
        final_cache_path = os.path.join(
            config.CSV_CACHE_DIR, "final_submission_data.csv"
        )
        initial_cache_path = os.path.join(
            config.CSV_CACHE_DIR, "initial_submission_data.csv"
        )

        # Check if cache files exist and are recent enough
        use_final_cache = os.path.exists(final_cache_path) and use_cache
        use_initial_cache = os.path.exists(initial_cache_path) and use_cache

        # Check cache age if files exist
        if use_final_cache or use_initial_cache:
            # Check final cache age
            if use_final_cache:
                final_mod_time = datetime.fromtimestamp(
                    os.path.getmtime(final_cache_path)
                )
                final_age_days = (datetime.now() - final_mod_time).days
                use_final_cache = final_age_days <= config.CSV_CACHE_DAYS
                logger.info(f"Final cache age: {final_age_days} days")

            # Check initial cache age
            if use_initial_cache:
                initial_mod_time = datetime.fromtimestamp(
                    os.path.getmtime(initial_cache_path)
                )
                initial_age_days = (datetime.now() - initial_mod_time).days
                use_initial_cache = initial_age_days <= config.CSV_CACHE_DAYS
                logger.info(f"Initial cache age: {initial_age_days} days")

        # Load from cache if available and recent
        final_df = pd.DataFrame()
        initial_df = pd.DataFrame()

        # Use defaults from config if not provided
        load_profile = load_profile or config.DEFAULT_LOAD_PROFILE
        rate_group_filter = None
        # rate_group_filter = (
        #     rate_group_filter
        #     if rate_group_filter is not None
        #     else config.DEFAULT_RATE_GROUP_FILTER
        # )

        if rate_group_filter is None and hasattr(config, "RATE_GROUP_FILTER_CLAUSE"):
            rate_group_filter = config.RATE_GROUP_FILTER_CLAUSE

        logger.info(f"Querying data as of: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"Load profile: {load_profile}")
        logger.info(f"Rate group filter: {rate_group_filter}")

        # Calculate cutoff dates based on submission type delays
        initial_cutoff_date = current_date - timedelta(
            days=config.INITIAL_SUBMISSION_DELAY
        )
        final_cutoff_date = current_date - timedelta(days=config.FINAL_SUBMISSION_DELAY)

        logger.info(f"Initial cutoff: {initial_cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"Final cutoff: {final_cutoff_date.strftime('%Y-%m-%d')}")

        if use_final_cache:
            logger.info(f"Loading Final submission data from cache: {final_cache_path}")
            final_df = pd.read_csv(final_cache_path)
            # Convert string columns to appropriate types
            final_df = convert_column_types(final_df)
        else:
            # Calculate dates and query Athena for Final data
            final_cutoff_date = current_date - timedelta(
                days=config.FINAL_SUBMISSION_DELAY
            )
            logger.info(
                f"Querying Athena for Final submission data (cutoff: {final_cutoff_date})"
            )

            # Connect to Athena and make query
            final_df = query_athena_final_data(
                current_date, load_profile, rate_group_filter, query_limit
            )

            # Convert types
            final_df = convert_column_types(final_df)

            # Save to cache if data was retrieved
            if not final_df.empty:
                logger.info(
                    f"Saving Final submission data to cache: {final_cache_path}"
                )
                final_df.to_csv(final_cache_path, index=False)

        if use_initial_cache:
            logger.info(
                f"Loading Initial submission data from cache: {initial_cache_path}"
            )
            initial_df = pd.read_csv(initial_cache_path)
            # Convert string columns to appropriate types
            initial_df = convert_column_types(initial_df)
        else:
            # Query Athena for Initial data
            initial_cutoff_date = current_date - timedelta(
                days=config.INITIAL_SUBMISSION_DELAY
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
                current_date,
                latest_final_date,
                load_profile,
                rate_group_filter,
                query_limit,
            )

            # Convert types
            initial_df = convert_column_types(initial_df)

            # Save to cache if data was retrieved
            if not initial_df.empty:
                logger.info(
                    f"Saving Initial submission data to cache: {initial_cache_path}"
                )
                initial_df.to_csv(initial_cache_path, index=False)

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
        logger.error(f"Error querying data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def query_athena_final_data(
    current_date, load_profile, rate_group_filter, query_limit=None
):
    """Query Athena for Final submission data"""
    try:
        # Connect to Athena
        conn = connect(
            s3_staging_dir=config.ATHENA_STAGING_DIR, region_name=config.AWS_REGION
        )

        final_cutoff_date = current_date - timedelta(days=config.FINAL_SUBMISSION_DELAY)

        # Build WHERE clause for filtering
        where_clauses = [
            f"loadprofile = '{load_profile}'",
            f"submission = '{config.SUBMISSION_TYPE_FINAL}'",  # Using Final submission type as default
        ]

        # Add rate group filter if provided
        if rate_group_filter:
            where_clauses.append(rate_group_filter)

        where_clause = " AND ".join(where_clauses)

        # Get latest Final submission date
        cursor = conn.cursor()
        latest_final_query = f"""
        SELECT MAX(tradedate) as max_date
        FROM {config.ATHENA_DATABASE}.{config.ATHENA_TABLE}
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
            config.ATHENA_DATABASE, config.ATHENA_TABLE, final_where
        )

        if query_limit is not None and query_limit > 0:
            final_query += f" LIMIT {query_limit}"

        logger.info(f"Executing Athena query for Final submission data: {final_query}")
        cursor.execute(final_query)
        final_df = as_pandas(cursor)
        logger.info(f"Retrieved {len(final_df)} rows of Final data")

        return final_df
    except Exception as e:
        logger.error(f"Error querying Athena for Final data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def query_athena_initial_data(
    current_date, latest_final_date, load_profile, rate_group_filter, query_limit=None
):
    """Query Athena for Initial submission data"""
    # Original implementation - unchanged
    try:
        # Connect to Athena
        conn = connect(
            s3_staging_dir=config.ATHENA_STAGING_DIR, region_name=config.AWS_REGION
        )

        initial_cutoff_date = current_date - timedelta(
            days=config.INITIAL_SUBMISSION_DELAY
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
            f"submission = '{config.SUBMISSION_TYPE_INITIAL}'",  # Using Initial instead of config default
        ]

        # Add rate group filter if provided
        if rate_group_filter:
            init_where_clauses.append(rate_group_filter)

        # Add date constraints for Initial data
        init_where_clauses.append(
            f"CAST(tradedate AS DATE) <= CAST('{initial_cutoff_date.strftime('%Y-%m-%d')}' AS DATE)"
        )
        init_where_clauses.append(
            f"CAST(tradedate AS DATE) > CAST('{latest_final_date.strftime('%Y-%m-%d')}' AS DATE)"
        )

        init_where_clause = " AND ".join(init_where_clauses)

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
            config.ATHENA_DATABASE, config.ATHENA_TABLE, init_where_clause
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
        logger.error(f"Error querying Athena for Initial data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def convert_column_types(df):
    """
    Convert DataFrame columns to appropriate types.

    Args:
        df: DataFrame with string columns

    Returns:
        DataFrame with proper column types
    """
    # Original implementation - unchanged
    if df.empty:
        return df

    # Make a copy to avoid warnings
    df = df.copy()

    # Define column type mapping
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

    # Convert date columns
    if "tradedate" in df.columns:
        try:
            df["tradedate"] = pd.to_datetime(df["tradedate"])
        except:
            logger.warning("Error converting tradedate to datetime")

    # Create datetime column if not present but component columns are
    if (
        "datetime" not in df.columns
        and "tradedate" in df.columns
        and "tradetime" in df.columns
    ):
        try:
            df["datetime"] = pd.to_datetime(
                df["tradedate"].astype(str) + " " + df["tradetime"].astype(str)
            )
        except:
            logger.warning("Error creating datetime column")

    return df


def preprocess_raw(df):
    """
    Basic preprocessing of raw data

    Args:
        df: Raw DataFrame

    Returns:
        Preprocessed DataFrame
    """
    try:
        if df.empty:
            return df

        # Create datetime
        df["datetime"] = pd.to_datetime(
            df["tradedate"].astype(str) + " " + df["tradetime"].astype(str)
        )

        # Sort by datetime
        df = df.sort_values("datetime")

        # Add generation ratio
        df["generation_ratio"] = np.abs(df["genlal"] / df["loadlal"].replace(0, np.nan))
        df["generation_ratio"] = df["generation_ratio"].fillna(0)

        # Add date parts
        df["hour"] = df["datetime"].dt.hour
        df["day"] = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["year"] = df["datetime"].dt.year
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["date"] = df["datetime"].dt.date  # Add date column for easier grouping

        return df

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        logger.error(traceback.format_exc())
        return df


def standardize_timestamps(df):
    """
    Standardize timestamps to hour-beginning format and handle timezone issues.
    
    Args:
        df: DataFrame with datetime column
        
    Returns:
        DataFrame with standardized datetimes
    """
    try:
        if df.empty or 'datetime' not in df.columns:
            return df
            
        # Make a copy
        df = df.copy()
        
        # Ensure datetimes are in hour-beginning format (HH:00:00)
        df['datetime'] = df['datetime'].dt.floor('h')
        
        # Update derived time columns
        df['hour'] = df['datetime'].dt.hour
        if 'date' not in df.columns:
            df['date'] = df['datetime'].dt.date
            
        return df
    except Exception as e:
        logger.error(f"Error standardizing timestamps: {e}")
        return df


def find_meter_threshold(df):
    """
    Analyze meter count to find appropriate threshold

    Args:
        df: DataFrame with meter counts

    Returns:
        (DataFrame, meter_stats)
    """
    # Original implementation - unchanged
    try:
        if df.empty:
            return df, {}

        logger.info("Analyzing meter counts")

        # Create date column if needed
        if "date" not in df.columns:
            df["date"] = df["datetime"].dt.date

        # Group by date
        daily_stats = (
            df.groupby("date")
            .agg({"metercount": ["min", "mean", "max", "std"]})
            .reset_index()
        )

        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        meter_percentiles = {
            str(p): float(df["metercount"].quantile(p / 100)) for p in percentiles
        }

        # Look for rapid growth in meter counts
        daily_stats["meter_diff"] = daily_stats["metercount"]["mean"].diff()

        # Find significant jumps (> 1 std above mean diff)
        mean_diff = daily_stats["meter_diff"].mean()
        std_diff = daily_stats["meter_diff"].std()
        jump_threshold = mean_diff + std_diff

        jumps = daily_stats[daily_stats["meter_diff"] > jump_threshold]

        if len(jumps) > 0:
            # Use biggest jump as threshold
            biggest_jump = jumps.sort_values("meter_diff", ascending=False).iloc[0]
            jump_date = biggest_jump["date"]
            jump_meters = biggest_jump["metercount"]["mean"]

            logger.info(f"Found meter jump on {jump_date}: {jump_meters}")

            threshold_method = "jump"
            recommended = jump_meters
        else:
            # Use 25th percentile as fallback
            threshold_method = "percentile"
            recommended = meter_percentiles["25"]

        # Round to nice number
        recommended = round(recommended / 100) * 100

        meter_stats = {
            "percentiles": meter_percentiles,
            "recommended_threshold": float(recommended),
            "threshold_method": threshold_method,
        }

        logger.info(f"Recommended meter threshold: {recommended}")

        return df, meter_stats

    except Exception as e:
        logger.error(f"Error analyzing meters: {e}")
        logger.error(traceback.format_exc())
        return df, {}


def aggregate_timeseries(df):
    """
    Aggregate data by timestamp

    Args:
        df: DataFrame with datetime column

    Returns:
        Aggregated DataFrame
    """
    # Original implementation - unchanged
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
        agg_dict = {col: "sum" for col in sum_cols}

        # Categorical columns - take first
        cat_cols = ["loadprofile", "submission"]
        for col in cat_cols:
            if col in df.columns:
                agg_dict[col] = "first"

        # Aggregate
        before_count = len(df)
        agg_df = df.groupby("datetime").agg(agg_dict).reset_index()
        logger.info(f"Aggregated rows: {before_count} → {len(agg_df)}")

        # Check for missing hours
        min_date = agg_df["datetime"].min()
        max_date = agg_df["datetime"].max()
        expected = pd.date_range(start=min_date, end=max_date, freq="h")

        missing = set(expected) - set(agg_df["datetime"])
        if missing:
            logger.warning(f"Found {len(missing)} missing hourly intervals")

            # Add placeholder rows
            missing_rows = []
            for dt in missing:
                row = {"datetime": dt}
                for col in agg_df.columns:
                    if col != "datetime":
                        row[col] = np.nan
                missing_rows.append(row)

            if missing_rows:
                missing_df = pd.DataFrame(missing_rows)
                logger.info(f"MISSING DF HEAD 10: {missing_df.head()}")
                agg_df = pd.concat([agg_df, missing_df], ignore_index=True)
                agg_df = agg_df.sort_values("datetime")
                logger.info(f"Added {len(missing_rows)} placeholders")

        # Recalculate stats after aggregation
        if "loadlal" in agg_df.columns and "genlal" in agg_df.columns:
            agg_df["generation_ratio"] = np.abs(
                agg_df["genlal"] / agg_df["loadlal"].replace(0, np.nan)
            )
            agg_df["generation_ratio"] = agg_df["generation_ratio"].fillna(0)

        return agg_df

    except Exception as e:
        logger.error(f"Error in aggregation: {e}")
        logger.error(traceback.format_exc())
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
        logger.error(f"Error handling missing values: {e}")
        logger.error(traceback.format_exc())
        return df


def handle_outliers(df):
    """
    Handle outliers in the target variable

    Args:
        df: DataFrame with potential outliers

    Returns:
        DataFrame with handled outliers
    """
    # Original implementation - unchanged
    try:
        if df.empty:
            return df

        if "lossadjustedload" not in df.columns:
            return df

        logger.info("Handling outliers")

        # IQR method
        q1 = df["lossadjustedload"].quantile(0.25)
        q3 = df["lossadjustedload"].quantile(0.75)
        iqr = q3 - q1

        # Define bounds
        lower = q1 - config.OUTLIER_IQR_FACTOR * iqr
        upper = q3 + config.OUTLIER_IQR_FACTOR * iqr

        # Find outliers
        outliers = df[
            (df["lossadjustedload"] < lower) | (df["lossadjustedload"] > upper)
        ]
        pct_outliers = len(outliers) / len(df) * 100
        logger.info(f"Found {len(outliers)} outliers ({pct_outliers:.2f}%)")

        # Cap extreme outliers (3+ IQR)
        extreme_lower = q1 - config.EXTREME_IQR_FACTOR * iqr
        extreme_upper = q3 + config.EXTREME_IQR_FACTOR * iqr

        # Cap extreme values
        low_mask = df["lossadjustedload"] < extreme_lower
        high_mask = df["lossadjustedload"] > extreme_upper

        extreme_count = low_mask.sum() + high_mask.sum()
        if extreme_count > 0:
            logger.info(f"Capping {extreme_count} extreme values")
            df.loc[low_mask, "lossadjustedload"] = extreme_lower
            df.loc[high_mask, "lossadjustedload"] = extreme_upper

        return df

    except Exception as e:
        logger.error(f"Error handling outliers: {e}")
        logger.error(traceback.format_exc())
        return df


def analyze_duck_curve(df):
    """
    Analyze duck curve patterns

    Args:
        df: DataFrame with hourly load data

    Returns:
        (DataFrame, duck_curve_metrics)
    """
    # Original implementation - unchanged
    try:
        if df.empty:
            return df, {}

        logger.info("Analyzing duck curve")

        # Add hour if not present
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour

        # Calculate hourly averages
        hourly = df.groupby("hour")["lossadjustedload"].mean().reset_index()

        # Find peaks and troughs
        morning_peak_df = hourly[
            (hourly["hour"] >= config.MORNING_PEAK_HOURS[0])
            & (hourly["hour"] <= config.MORNING_PEAK_HOURS[1])
        ]
        solar_trough_df = hourly[
            (hourly["hour"] >= config.SOLAR_PERIOD_HOURS[0])
            & (hourly["hour"] <= config.SOLAR_PERIOD_HOURS[1])
        ]
        evening_peak_df = hourly[
            (hourly["hour"] >= config.EVENING_PEAK_HOURS[0])
            & (hourly["hour"] <= config.EVENING_PEAK_HOURS[1])
        ]

        # Extract metrics, handling empty cases
        if not morning_peak_df.empty:
            morning_peak = morning_peak_df["lossadjustedload"].max()
            morning_peak_hour = (
                morning_peak_df.loc[
                    morning_peak_df["lossadjustedload"].idxmax(), "hour"
                ]
                if not morning_peak_df["lossadjustedload"].empty
                else None
            )
        else:
            morning_peak, morning_peak_hour = None, None

        if not solar_trough_df.empty:
            solar_trough = solar_trough_df["lossadjustedload"].min()
            solar_trough_hour = (
                solar_trough_df.loc[
                    solar_trough_df["lossadjustedload"].idxmin(), "hour"
                ]
                if not solar_trough_df["lossadjustedload"].empty
                else None
            )
        else:
            solar_trough, solar_trough_hour = None, None

        if not evening_peak_df.empty:
            evening_peak = evening_peak_df["lossadjustedload"].max()
            evening_peak_hour = (
                evening_peak_df.loc[
                    evening_peak_df["lossadjustedload"].idxmax(), "hour"
                ]
                if not evening_peak_df["lossadjustedload"].empty
                else None
            )
        else:
            evening_peak, evening_peak_hour = None, None

        # Calculate duck curve ratio safely
        if (
            solar_trough is not None
            and evening_peak is not None
            and solar_trough != 0
            and abs(solar_trough) > 1e-10
        ):
            duck_ratio = evening_peak / solar_trough
        else:
            duck_ratio = None

        # Store metrics
        metrics = {
            "morning_peak": float(morning_peak) if morning_peak is not None else None,
            "morning_peak_hour": (
                int(morning_peak_hour) if morning_peak_hour is not None else None
            ),
            "solar_trough": float(solar_trough) if solar_trough is not None else None,
            "solar_trough_hour": (
                int(solar_trough_hour) if solar_trough_hour is not None else None
            ),
            "evening_peak": float(evening_peak) if evening_peak is not None else None,
            "evening_peak_hour": (
                int(evening_peak_hour) if evening_peak_hour is not None else None
            ),
            "duck_ratio": float(duck_ratio) if duck_ratio is not None else None,
        }

        # Log findings
        logger.info(
            f"Duck curve metrics: "
            f"morning peak = {metrics['morning_peak']} at hour {metrics['morning_peak_hour']}, "
            f"solar trough = {metrics['solar_trough']} at hour {metrics['solar_trough_hour']}, "
            f"evening peak = {metrics['evening_peak']} at hour {metrics['evening_peak_hour']}, "
            f"ratio = {metrics['duck_ratio']}"
        )

        # Add duck curve period flags to df
        df["is_morning_peak"] = (
            (df["hour"] >= config.MORNING_PEAK_HOURS[0])
            & (df["hour"] <= config.MORNING_PEAK_HOURS[1])
        ).astype(int)
        df["is_solar_period"] = (
            (df["hour"] >= config.SOLAR_PERIOD_HOURS[0])
            & (df["hour"] <= config.SOLAR_PERIOD_HOURS[1])
        ).astype(int)
        df["is_evening_ramp"] = (
            (df["hour"] >= config.EVENING_RAMP_HOURS[0])
            & (df["hour"] <= config.EVENING_RAMP_HOURS[1])
        ).astype(int)
        df["is_evening_peak"] = (
            (df["hour"] >= config.EVENING_PEAK_HOURS[0])
            & (df["hour"] <= config.EVENING_PEAK_HOURS[1])
        ).astype(int)

        return df, metrics

    except Exception as e:
        logger.error(f"Error in duck curve analysis: {e}")
        logger.error(traceback.format_exc())
        return df, {}


def create_features(df):
    """
    Create core features

    Args:
        df: DataFrame with datetime information

    Returns:
        DataFrame with added features
    """
    try:
        if df.empty:
            return df

        logger.info("Creating core features")

        # Make a copy
        df = df.copy()

        # Extract hour if not present
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour

        if "dayofweek" not in df.columns:
            df["dayofweek"] = df["datetime"].dt.dayofweek

        if "month" not in df.columns:
            df["month"] = df["datetime"].dt.month

        if "year" not in df.columns:
            df["year"] = df["datetime"].dt.year

        if "day_of_year" not in df.columns:
            df["day_of_year"] = df["datetime"].dt.dayofyear
            
        if "date" not in df.columns:
            df["date"] = df["datetime"].dt.date

        # Feature dict
        features = {}

        # Cyclical encodings
        features["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        features["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        features["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        features["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        features["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        # Calendar features
        features["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        # Enhanced calendar features
        features["is_business_hour"] = ((df["hour"] >= 8) & (df["hour"] < 18) & 
                                      (df["dayofweek"] < 5)).astype(int)
        features["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
        features["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
        features["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 11)).astype(int)
        features["is_afternoon"] = ((df["hour"] >= 11) & (df["hour"] < 18)).astype(int)

        try:
            import holidays

            us_holidays = holidays.US()
            features["is_holiday"] = (
                df["datetime"].dt.date.isin(us_holidays).astype(int)
            )
            
            # Create features for days around holidays
            # Holiday proximity features
            features["is_day_before_holiday"] = np.zeros(len(df), dtype=int)
            features["is_day_after_holiday"] = np.zeros(len(df), dtype=int)
            
            for holiday_date in us_holidays:
                day_before = holiday_date - timedelta(days=1)
                day_after = holiday_date + timedelta(days=1)
                
                features["is_day_before_holiday"] += (df["datetime"].dt.date == day_before).astype(int)
                features["is_day_after_holiday"] += (df["datetime"].dt.date == day_after).astype(int)
                
        except ImportError:
            features["is_holiday"] = 0
            features["is_day_before_holiday"] = 0
            features["is_day_after_holiday"] = 0

        # Month start/end
        features["is_month_start"] = df["datetime"].dt.is_month_start.astype(int)
        features["is_month_end"] = df["datetime"].dt.is_month_end.astype(int)

        # If duck curve periods not already added
        if "is_morning_peak" not in df.columns:
            features["is_morning_peak"] = (
                (df["hour"] >= config.MORNING_PEAK_HOURS[0])
                & (df["hour"] <= config.MORNING_PEAK_HOURS[1])
            ).astype(int)
            features["is_solar_period"] = (
                (df["hour"] >= config.SOLAR_PERIOD_HOURS[0])
                & (df["hour"] <= config.SOLAR_PERIOD_HOURS[1])
            ).astype(int)
            features["is_evening_ramp"] = (
                (df["hour"] >= config.EVENING_RAMP_HOURS[0])
                & (df["hour"] <= config.EVENING_RAMP_HOURS[1])
            ).astype(int)
            features["is_evening_peak"] = (
                (df["hour"] >= config.EVENING_PEAK_HOURS[0])
                & (df["hour"] <= config.EVENING_PEAK_HOURS[1])
            ).astype(int)

        # Solar peak indicator
        features["is_solar_peak"] = df["hour"].isin([11, 12, 13, 14]).astype(int)
        
        # Add day type features (weekday vs weekend)
        features["is_monday"] = (df["dayofweek"] == 0).astype(int)
        features["is_friday"] = (df["dayofweek"] == 4).astype(int)
        features["is_saturday"] = (df["dayofweek"] == 5).astype(int)
        features["is_sunday"] = (df["dayofweek"] == 6).astype(int)

        # Add features to dataframe
        for name, values in features.items():
            df[name] = values

        logger.info(f"Added {len(features)} core features")

        return df

    except Exception as e:
        logger.error(f"Error creating features: {e}")
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
        logger.error(f"Error creating transition features: {e}")
        logger.error(traceback.format_exc())
        return df


def create_time_window_features(df):
    """
    Create features specific to different time windows and periods of the day.
    
    Args:
        df: DataFrame with datetime information
        
    Returns:
        DataFrame with time window features
    """
    try:
        if df.empty:
            return df
            
        logger.info("Creating time window features")
        
        # Make a copy
        df = df.copy()
        
        # Required columns
        if 'datetime' not in df.columns:
            logger.warning("Cannot create time window features - missing datetime column")
            return df
            
        # Make sure hour column exists
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        
        # 1. Define time windows with more precision
        # Early morning (5-8)
        df['is_early_morning'] = ((df['hour'] >= 5) & (df['hour'] <= 8)).astype(int)
        
        # Morning ramp (8-11) - Solar starting to generate
        df['is_morning_ramp'] = ((df['hour'] >= 8) & (df['hour'] <= 11)).astype(int)
        
        # Midday (11-15) - Peak solar
        df['is_midday'] = ((df['hour'] >= 11) & (df['hour'] <= 15)).astype(int)
        
        # Afternoon decline (15-17) - Solar starting to decrease
        df['is_afternoon_decline'] = ((df['hour'] >= 15) & (df['hour'] <= 17)).astype(int)
        
        # Evening ramp (17-20) - Sharp increase in net load
        df['is_evening_ramp_refined'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        
        # Night (20-5) - Low solar, stable load
        df['is_night_period'] = ((df['hour'] >= 20) | (df['hour'] <= 5)).astype(int)
        
        # 2. Add hourly position within each period
        # For each time window, create a relative position feature
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
                mask = (df['hour'] >= start) & (df['hour'] <= end)
                df[f'{period_name}_position'] = 0
                df.loc[mask, f'{period_name}_position'] = (df.loc[mask, 'hour'] - start) / (end - start)
            else:
                # Period spanning midnight
                mask = (df['hour'] >= start) | (df['hour'] <= end)
                df[f'{period_name}_position'] = 0
                
                # Before midnight
                mask_before = (df['hour'] >= start)
                if mask_before.any():
                    df.loc[mask_before, f'{period_name}_position'] = \
                        (df.loc[mask_before, 'hour'] - start) / ((24 - start) + end)
                
                # After midnight
                mask_after = (df['hour'] <= end)
                if mask_after.any():
                    df.loc[mask_after, f'{period_name}_position'] = \
                        ((df.loc[mask_after, 'hour'] + 24) - start) / ((24 - start) + end)
        
        # 3. Create hour-of-day position (0-1)
        df['hour_of_day_position'] = df['hour'] / 24.0
        
        # 4. Add specific features for ramp periods
        # Morning ramp rate (load decrease rate during solar increase)
        df['morning_ramp_hour'] = 0
        morning_mask = (df['hour'] >= 8) & (df['hour'] <= 11)
        df.loc[morning_mask, 'morning_ramp_hour'] = df.loc[morning_mask, 'hour'] - 8
        
        # Evening ramp rate (load increase rate during solar decrease)
        df['evening_ramp_hour'] = 0
        evening_mask = (df['hour'] >= 17) & (df['hour'] <= 20)
        df.loc[evening_mask, 'evening_ramp_hour'] = df.loc[evening_mask, 'hour'] - 17
        
        logger.info("Created time window features")
        return df
    
    except Exception as e:
        logger.error(f"Error creating time window features: {e}")
        logger.error(traceback.format_exc())
        return df


def create_enhanced_lag_features(df, forecast_delay_days=14):
    """
    Create focused lag features for energy forecasting, limited to the most
    important predictors to avoid excessive dimensionality.
   
    Args:
        df: DataFrame with time series data
        forecast_delay_days: Number of days delay in data availability
       
    Returns:
        DataFrame with essential lag features
    """
    try:
        if df.empty:
            return df
           
        logger.info("Creating focused lag features")
       
        # Make a copy of the DataFrame to avoid modifying the original
        # This also addresses the fragmentation issue for any previous operations
        df = df.copy()
       
        # Convert days to hours
        forecast_lag_hours = forecast_delay_days * 24
       
        # Important lag periods for energy forecasting
        # 14, 15 days (previous 2 weeks, same day/hour)
        # 21 days (3 weeks prior, same day/hour)
        # 28 days (4 weeks prior, same day/hour)
        lag_hours = [
            forecast_lag_hours,         # 14 days - minimum available data
            forecast_lag_hours + 24,    # 15 days
            forecast_lag_hours + 7*24,  # 21 days (14d + 1 week)
            forecast_lag_hours + 14*24  # 28 days (14d + 2 weeks)
        ]
       
        # Create a dictionary to collect all new features at once
        # This avoids adding columns one by one to the DataFrame, preventing fragmentation
        new_features = {}
       
        # 1. Target variable lags - the most important features
        if 'lossadjustedload' in df.columns:
            # Base lags (same hour, previous weeks)
            for lag in lag_hours:
                new_features[f'lossadjustedload_lag_{lag}h'] = df['lossadjustedload'].shift(lag)
               
            # Same day-of-week and hour lags are very important for energy forecasting
            # These capture weekly patterns precisely
            if 'dayofweek' in df.columns and 'hour' in df.columns:
                # Pre-calculate grouped shifts all at once for efficiency
                for week_back in range(2, 6):  # 2-5 weeks back
                    lag_name = f'load_same_dow_hour_{week_back}w'
                    # Calculate as a Series first, then add to features dictionary
                    new_features[lag_name] = df.groupby(['dayofweek', 'hour'])[
                        'lossadjustedload'].shift(week_back * 7 * 24)
       
        # 2. Export/import pattern lags - critical for solar net metering
        if 'is_net_export' in df.columns:
            # Lag of export status
            for lag in lag_hours:
                new_features[f'is_net_export_lag_{lag}h'] = df['is_net_export'].shift(lag)

        if 'is_net_import' in df.columns:
            # Lag of export status
            for lag in lag_hours:
                new_features[f'is_net_import_lag_{lag}h'] = df['is_net_import'].shift(lag)
               
        # 3. Per-meter metrics lags
        # These are important for capturing system efficiency patterns
        for col in ['load_per_meter', 'gen_per_meter']:
            if col in df.columns:
                # Just use primary lags to avoid too many features
                for lag in [lag_hours[0], lag_hours[2]]:  # 14d and 21d
                    new_features[f'{col}_lag_{lag}h'] = df[col].shift(lag)
       
        # 4. Period load averages (key for duck curve)
        if all(col in df.columns for col in ['morning_load_avg', 'midday_load_avg', 'evening_load_avg']):
            # Week-over-week changes in duck curve shape
            # 2 and 3 weeks back are most useful forecast horizons
            for days_back in [forecast_delay_days, forecast_delay_days + 7]:
                hours_back = days_back * 24
                # Calculate grouped shifts
                ratio_name = f'evening_to_midday_ratio_lag_{hours_back}h'
                new_features[ratio_name] = df.groupby('hour')['evening_to_midday_ratio'].shift(hours_back)
       
        # 5. Growth rates between available lags
        # These capture trends without adding too many features
        if 'lossadjustedload' in df.columns:
            # 14-day to 21-day growth (week-over-week from earliest available)
            new_features['load_growth_w2w'] = (
                df['lossadjustedload'].shift(lag_hours[0]) /
                df['lossadjustedload'].shift(lag_hours[2]).replace(0, np.nan) - 1
            )
           
            # 21-day to 28-day growth (previous week-over-week)
            new_features['load_growth_prev_w2w'] = (
                df['lossadjustedload'].shift(lag_hours[2]) /
                df['lossadjustedload'].shift(lag_hours[3]).replace(0, np.nan) - 1
            )
       
        # Handle NaNs in growth rates
        for col in ['load_growth_w2w', 'load_growth_prev_w2w']:
            if col in new_features:
                new_features[col] = new_features[col].fillna(0).clip(-1, 1)
       
        # 6. Weekly average lags - smoothed signals are valuable for forecasting
        if 'lossadjustedload' in df.columns:
            # Calculate 7-day moving average starting from forecast delay
            new_features['load_ma_7d'] = df['lossadjustedload'].shift(forecast_lag_hours).rolling(
                window=7*24, min_periods=1).mean()
               
            # Calculate 7-day moving average from 21 days back (captures past seasonality)
            new_features['load_ma_7d_lag_3w'] = df['lossadjustedload'].shift(forecast_lag_hours + 7*24).rolling(
                window=7*24, min_periods=1).mean()
       
        # Add all the new features at once using pd.concat()
        # This is the key optimization that prevents DataFrame fragmentation
        new_features_df = pd.DataFrame(new_features, index=df.index)
        result_df = pd.concat([df, new_features_df], axis=1)
       
        # Handle any duplicate columns that might have been created
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
       
        # Log feature count
        logger.info(f"Created {len(new_features)} focused lag features")
       
        return result_df
   
    except Exception as e:
        logger.error(f"Error creating focused lag features: {e}")
        logger.error(traceback.format_exc())
        return df


def mark_forecast_available_features(df, forecast_delay_days=14):
    """
    Mark features as available or not at forecast time with given delay.
    Handles various lag notation patterns and time-based features.
   
    Args:
        df: DataFrame with features
        forecast_delay_days: Number of days delay in data availability
       
    Returns:
        List of column names that are available at forecast time
    """
    logger.info(f"DF COLUMNS: {list(df.columns)}")

    # Core datetime features are always available - these must be present in the DataFrame
    datetime_features = ['datetime']  # Keep 'datetime' separate to ensure it's always included
    logger.info(f"DATETIME FEATURES: {datetime_features}")
    
    time_features = ['hour', 'dayofweek', 'month', 'year', 'date']
    logger.info(f"TIME FEATURES: {time_features}")
   
    # Calendar-based features are always available
    calendar_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in
                      ('is_', 'hour_', 'dow_', 'month_', 'day_of_year'))]
    calendar_features = [feat for feat in calendar_features if 'import' not in feat and 'export' not in feat]
    logger.info(f"Calendar FEATURES: {calendar_features}")
   
    # Position features are derived from hour and always available
    position_features = [col for col in df.columns if any(col.endswith(suffix) for suffix in
                       ('_position', '_hour'))]
    logger.info(f"POSITION FEATURES: {position_features}")
   
    # Metadata features are always available but they do not play any role in profile specific model development
    metadata_features = [col for col in df.columns if col in ['loadprofile', 'submission']]
    logger.info(f"METADATA FEATURES: {metadata_features}")
   
    # # Combine all always-available features
    # always_available = list(set(datetime_features +
    #                          [col for col in time_features if col in df.columns] +
    #                          calendar_features +
    #                          position_features +
    #                          metadata_features))
   
    # Features with sufficient lag - handle multiple lag notation patterns
    lag_available = []
   
    # Calculate the minimum hours threshold
    min_hours_threshold = forecast_delay_days * 24
   
    for col in df.columns:
        # Check for features with "_lag_" in name
        if '_lag_' in col:
            # Handle different lag notations
            lag_part = col.split('_lag_')[1]
           
            # Case 1: Hour lag notation (e.g., "load_lag_336h")
            if lag_part.endswith('h'):
                try:
                    lag_hours = int(lag_part.replace('h', ''))
                    if lag_hours >= min_hours_threshold:
                        lag_available.append(col)
                except ValueError:
                    # Not a numeric hour format
                    pass
           
            # Case 2: Day lag notation (e.g., "load_lag_14d")
            elif lag_part.endswith('d'):
                try:
                    lag_days = int(lag_part.replace('d', ''))
                    if lag_days >= forecast_delay_days:
                        lag_available.append(col)
                except ValueError:
                    # Not a numeric day format
                    pass
           
            # Case 3: Week lag notation (e.g., "load_lag_2w" or "load_ma_7d_lag_3w")
            elif lag_part.endswith('w'):
                try:
                    lag_weeks = int(lag_part.replace('w', ''))
                    if lag_weeks >= forecast_delay_days / 7:
                        lag_available.append(col)
                except ValueError:
                    # Not a numeric week format
                    pass

    logger.info(f"LAG FEATURES: {lag_available}")
    
    # Historical pattern features (based on day of week, etc.)
    historical_pattern_features = []
    for col in df.columns:
        # Features with week numbers (2w, 3w, 4w, etc.)
        if any(f"_{i}w" in col for i in range(2, 10)):
            for i in range(2, 10):
                if f"_{i}w" in col:
                    if i >= (forecast_delay_days / 7):
                        historical_pattern_features.append(col)
                    break

    logger.info(f"HIST PATTERN FEATURES: {historical_pattern_features}")
  
    # Additional explicitly available features (known to be based on historical data)
    explicit_available = [col for col in df.columns if any(col.endswith(suffix) for suffix in
                        ('_ma_7d', '_ma_14d', '_growth_w2w', '_growth_prev_w2w'))]
    logger.info(f"EXPLICIT FEATURES: {explicit_available}") 
    
    # Combine all available features
    all_available = []
    # all_available = datetime_features  # Always include datetime first
    all_available.extend(list(set(datetime_features + time_features +
                                  calendar_features + position_features +
                                  lag_available + historical_pattern_features +
                                  explicit_available)))

    logger.info(f"ALL AVAILABLE FEATURES 1: {all_available}")
   
    # Filter out any duplicates and ensure they're actually in the DataFrame
    all_available = [col for col in dict.fromkeys(all_available) if col in df.columns]
    
    logger.info(f"ALL AVAILABLE FEATURES 2: {all_available}")
   
    # Make sure datetime is included
    if 'datetime' in df.columns and 'datetime' not in all_available:
        all_available.insert(0, 'datetime')
   
    # All other features are potentially not available
    not_available = [col for col in df.columns if col not in all_available
                    and not col.endswith('_LEAKAGE') and not col.endswith('_needs_lag')]

    logger.info("================ FEATURE ANALYSIS ================")
       
    logger.info(f"Always Available Features are: {sorted(set(time_features + calendar_features + position_features + metadata_features))}")
    logger.info(f"Lag Available Features are: {sorted(set(lag_available))}")
    logger.info(f"Historical Pattern Features are: {sorted(set(historical_pattern_features))}")
    logger.info(f"Explicit Available Features are: {sorted(set(explicit_available))}")
    logger.info(f"NOT Available Features are: {sorted(set(not_available))}")
    logger.info(f"Available features at forecast time: {len(all_available)} of {len(df.columns)}")
    logger.info(f"All Available Features are: {sorted(set(all_available))}")
    logger.info(f"Features not available at forecast time: {len(not_available)}")
   
    return all_available


def process_data_for_forecasting(df, meter_threshold, stats, load_profile='RES'):
    """
    Apply all preprocessing steps to prepare data for forecasting.
    This function applies the entire pipeline of data preprocessing
    designed to avoid data leakage while keeping as much useful information
    as possible.
    
    Args:
        df: Raw DataFrame with energy data
        
    Returns:
        Processed DataFrame ready for forecasting
    """
    try:
        if df.empty:
            return df
            
        logger.info("Starting full data processing pipeline for forecasting")
        
        # 1. Convert column types
        df = convert_column_types(df)
        
        # 2. Basic preprocessing
        df = preprocess_raw(df)

        # Analyze meters and set threshold
        df, meter_stats = find_meter_threshold(df)
        stats["meter_analysis"] = meter_stats

        if meter_threshold is None:
            meter_threshold = meter_stats.get(
                "recommended_threshold", config.DEFAULT_METER_THRESHOLD
            )
            logger.info(f"Using meter threshold: {meter_threshold}")

        # Filter by meter count
        before_count = len(df)
        if load_profile == 'RES':
            logger.info(f"Filtering meters with count >= {meter_threshold}")
            df = df[df["metercount"] >= meter_threshold]
            logger.info(
                f"Filtered meters >= {meter_threshold}: {len(df)} rows (removed {before_count - len(df)})"
            )
        
        # 3. Standardize timestamps
        df = standardize_timestamps(df)
        
        # 4. Aggregate time series
        df = aggregate_timeseries(df)
        
        # 5. Handle missing values
        df = handle_missing(df)
        
        # 6. Handle outliers
        df = handle_outliers(df)
        
        # 7. Analyze duck curve and add duck curve flags
        df, duck_metrics = analyze_duck_curve(df)
        stats["duck_curve"] = duck_metrics
        
        # 8. Create basic features
        df = create_features(df)
        
        # 9. Create normalized load features
        df = create_normalized_load_features(df)
        
        # 10. Create meter features
        df = create_meter_features(df)
        
        # 11. Create transition features
        df = create_transition_features(df)
        
        # 12. Create time window features
        df = create_time_window_features(df)
        
        # 13. Create enhanced lag features
        df = create_enhanced_lag_features(df)
        
        # Filter out load features that would cause data leakage
        # These should not be used directly in a forecasting model
        leakage_cols = [
            'baseload', 'lossadjustedload', 'loadbl', 'loadlal', 
            'genbl', 'genlal'
        ]
        
        # Keep these columns in the dataframe but flag them as leakage
        leakage_flag_cols = []
        for col in leakage_cols:
            if col in df.columns:
                leakage_flag_cols.append(col + '_LEAKAGE')
                df[col + '_LEAKAGE'] = df[col]
                # df = df.drop(columns=[col])  # Don't drop, just rename to flag
        
        if leakage_flag_cols:
            logger.warning(f"Flagged {len(leakage_flag_cols)} columns as potential data leakage. "
                          f"These should not be used directly for forecasting: {leakage_flag_cols}")
                          
        # Don't drop the target column, it's needed for training
        if 'lossadjustedload' in leakage_flag_cols and 'lossadjustedload' not in df.columns:
            df['lossadjustedload'] = df['lossadjustedload_LEAKAGE']
            
        logger.info(f"Completed data processing pipeline with {len(df.columns)} features")
        logger.info(f"List of features are: {df.columns.tolist()}")
        return df, stats
        
    except Exception as e:
        logger.error(f"Error in full data processing pipeline: {e}")
        logger.error(traceback.format_exc())
        return df, stats


def create_lags(df, lag_days=None):
    """
    Legacy lag creation function. Use create_enhanced_lag_features instead
    for more advanced lag features.

    Args:
        df: DataFrame with time series data
        lag_days: List of lag days to create (default: from config)

    Returns:
        DataFrame with lag features
    """
    try:
        if df.empty:
            return df

        # Use config defaults if not provided
        if lag_days is None:
            lag_days = config.EXTENDED_LAG_DAYS

        logger.info(f"Creating lag features for {lag_days} days")

        # Make a copy
        df = df.copy()

        # Target and lag columns
        target = "lossadjustedload"
        lag_dict = {}

        # Convert days to hours
        lag_hours = [days * 24 for days in lag_days]

        # Create target lags
        for lag in lag_hours:
            lag_dict[f"{target}_lag_{lag}h"] = df[target].shift(lag)

        # Generation lags if available
        if "genlal" in df.columns:
            for lag in lag_hours:
                lag_dict[f"genlal_lag_{lag}h"] = df["genlal"].shift(lag)

        # Consumption lags if available
        if "loadlal" in df.columns:
            for lag in lag_hours:
                lag_dict[f"loadlal_lag_{lag}h"] = df["loadlal"].shift(lag)

        # Meter count lags
        for meter_col in ['metercount', 'loadmetercount', 'genmetercount']:
            if meter_col in df.columns:
                for lag in lag_hours:
                    lag_dict[f"{meter_col}_lag_{lag}h"] = df[meter_col].shift(lag)

        # Add features to dataframe
        for name, values in lag_dict.items():
            df[name] = values

        logger.info(f"Added {len(lag_dict)} lag features")

        return df

    except Exception as e:
        logger.error(f"Error creating lag features: {e}")
        logger.error(traceback.format_exc())
        return df
