#!/usr/bin/env python3
"""
Athena Infrastructure Setup Script for Energy Load Forecasting Pipeline - Final Version
Creates Athena table with automatic partition discovery for daily prediction files
"""
import boto3
import time
import os
import json

def setup_athena_infrastructure():
    """Setup complete Athena infrastructure for energy forecasting"""
    
    # Get configuration from environment variables with validation
    try:
        database_name = os.environ['ATHENA_DATABASE']
        table_name = os.environ['ATHENA_TABLE']
        results_location = os.environ['ATHENA_RESULTS_LOCATION']
        data_location = os.environ['ATHENA_DATA_LOCATION']
        env_name = os.environ['ENV_NAME']
        s3_bucket = os.environ['S3_BUCKET']
    except KeyError as e:
        print(f' Missing required environment variable: {e}')
        return 'failed'
    
    # Validate that S3 locations are properly formatted
    if not data_location.startswith('s3://') or '///' in data_location:
        print(f' Invalid data location format: {data_location}')
        return 'failed'
    
    if not results_location.startswith('s3://') or '///' in results_location:
        print(f' Invalid results location format: {results_location}')
        return 'failed'

    print(f'Setting up Athena infrastructure for {env_name} environment')
    print(f'Database: {database_name}')
    print(f'Table: {table_name}')
    print(f'Data location: {data_location}')
    print(f'Results location: {results_location}')
    print(f'S3 bucket: {s3_bucket}')
    
    try:
        athena_client = boto3.client('athena')
        s3_client = boto3.client('s3')
    except Exception as e:
        print(f' Error creating AWS clients: {str(e)}')
        return 'failed'
    
    try:
        # Step 1: Create S3 directories
        print('Step 1: Setting up S3 directories...')
        create_s3_directories(s3_client, data_location, results_location)
        
        # Step 2: Create Athena table with automatic partition discovery
        print('Step 2: Creating Athena table with automatic partition discovery...')
        create_athena_table_with_auto_partitions(athena_client, database_name, table_name, data_location, results_location)
        
        # Step 3: Save configuration for Lambda
        print('Step 3: Saving configuration...')
        save_athena_config(s3_client, s3_bucket, database_name, table_name, results_location, data_location, env_name)
        
        print(' Athena infrastructure setup completed successfully!')
        return 'success'
        
    except Exception as e:
        print(f' Error setting up Athena infrastructure: {str(e)}')
        import traceback
        traceback.print_exc()
        return 'failed'

def create_s3_directories(s3_client, data_location, results_location):
    """Create necessary S3 directories"""
    
    # Extract bucket and prefixes from locations
    data_bucket = data_location.replace('s3://', '').split('/')[0]
    data_prefix = '/'.join(data_location.replace(f's3://{data_bucket}/', '').split('/'))
    if data_prefix and not data_prefix.endswith('/'):
        data_prefix += '/'
    
    results_bucket = results_location.replace('s3://', '').split('/')[0]
    results_prefix = '/'.join(results_location.replace(f's3://{results_bucket}/', '').split('/'))
    if results_prefix and not results_prefix.endswith('/'):
        results_prefix += '/'
    
    directories = []
    if data_prefix:
        directories.append((data_bucket, data_prefix))
    if results_prefix:
        directories.append((results_bucket, results_prefix))
    
    for bucket, prefix in directories:
        try:
            s3_client.put_object(Bucket=bucket, Key=prefix)
            print(f' Created S3 directory: s3://{bucket}/{prefix}')
        except Exception as e:
            print(f' Directory may already exist: s3://{bucket}/{prefix} - {str(e)}')

def create_athena_table_with_auto_partitions(athena_client, database_name, table_name, data_location, results_location):
    """Create Athena table with automatic partition discovery"""
    
    # Check if table already exists
    if table_exists(athena_client, database_name, table_name, results_location):
        print(f' Table {database_name}.{table_name} already exists - checking configuration...')
        verify_table_configuration(athena_client, database_name, table_name, results_location)
        return
    
    print(f'Creating new table: {database_name}.{table_name}')
    
    # Create table with automatic partition projection
    create_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {database_name}.{table_name} (
        forecast_datetime string,
        predicted_lossadjustedload double,
        run_id string,
        model_version string,
        run_user string,
        created_at string
    )
    PARTITIONED BY (
        load_profile string,
        load_segment string,
        year int,
        month int,
        day int
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    STORED AS TEXTFILE
    LOCATION '{data_location}'
    TBLPROPERTIES (
        'has_encrypted_data'='false',
        'skip.header.line.count'='1',
        'projection.enabled'='true',
        'projection.load_profile.type'='enum',
        'projection.load_profile.values'='RES,MEDCI,SMLCOM',
        'projection.load_segment.type'='enum',
        'projection.load_segment.values'='SOLAR,NONSOLAR',
        'projection.year.type'='integer',
        'projection.year.range'='2024,2030',
        'projection.month.type'='integer',
        'projection.month.range'='1,12',
        'projection.month.digits'='2',
        'projection.day.type'='integer',
        'projection.day.range'='1,31',
        'projection.day.digits'='2',
        'storage.location.template'='s3://{data_location.replace("s3://", "").rstrip("/")}/load_profile=${{load_profile}}/load_segment=${{load_segment}}/year=${{year}}/month=${{month}}/day=${{day}}/'
    )"""
    
    try:
        response = athena_client.start_query_execution(
            QueryString=create_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={'OutputLocation': results_location}
        )
        
        wait_for_query_completion(athena_client, response['QueryExecutionId'])
        print(f' Table {database_name}.{table_name} created successfully')
        
        # Test the table configuration
        print('Testing automatic partition discovery...')
        test_partition_discovery(athena_client, database_name, table_name, results_location)
        
    except Exception as e:
        print(f' Error creating table: {str(e)}')
        raise

def table_exists(athena_client, database_name, table_name, results_location):
    """Check if Athena table exists"""
    
    try:
        query = f"SHOW TABLES IN {database_name} LIKE '{table_name}'"
        
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={'OutputLocation': results_location}
        )
        
        query_execution_id = response['QueryExecutionId']
        wait_for_query_completion(athena_client, query_execution_id)
        
        # Get query results
        results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        
        # Check if table name appears in results
        for row in results.get('ResultSet', {}).get('Rows', []):
            for col in row.get('Data', []):
                if col.get('VarCharValue') == table_name:
                    return True
        
        return False
        
    except Exception as e:
        print(f' Could not check if table exists: {str(e)}')
        return False

def verify_table_configuration(athena_client, database_name, table_name, results_location):
    """Verify existing table has the correct configuration"""
    
    try:
        # Check table properties
        query = f"SHOW TBLPROPERTIES {database_name}.{table_name}"
        
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={'OutputLocation': results_location}
        )
        
        query_execution_id = response['QueryExecutionId']
        wait_for_query_completion(athena_client, query_execution_id)
        
        # Get query results to check properties
        results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        
        properties = {}
        for row in results.get('ResultSet', {}).get('Rows', [])[1:]:  # Skip header row
            row_data = row.get('Data', [])
            if len(row_data) >= 2:
                prop_name = row_data[0].get('VarCharValue', '')
                prop_value = row_data[1].get('VarCharValue', '')
                if prop_name:
                    properties[prop_name] = prop_value
        
        # Check key properties
        if properties.get('projection.enabled') == 'true':
            print(' Partition projection is enabled')
        else:
            print(' Warning: Partition projection is not enabled on existing table')
            
        if 'storage.location.template' in properties:
            print(' Storage location template is configured')
        else:
            print(' Warning: Storage location template is missing on existing table')
            
        if properties.get('projection.month.digits') == '2':
            print(' Month zero-padding is configured correctly')
        else:
            print(' Warning: Month zero-padding may not be configured correctly')
            
        if properties.get('projection.day.digits') == '2':
            print(' Day zero-padding is configured correctly')
        else:
            print(' Warning: Day zero-padding may not be configured correctly')
            
    except Exception as e:
        print(f' Could not verify table configuration: {str(e)}')

def test_partition_discovery(athena_client, database_name, table_name, results_location):
    """Test if partition discovery is working"""
    
    try:
        # Test query to check if any data is discoverable
        test_query = f"""
        SELECT 
            load_profile, load_segment, year, month, day,
            COUNT(*) as record_count
        FROM {database_name}.{table_name} 
        WHERE year >= 2025 
        GROUP BY load_profile, load_segment, year, month, day
        ORDER BY year, month, day
        LIMIT 10
        """
        
        response = athena_client.start_query_execution(
            QueryString=test_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={'OutputLocation': results_location}
        )
        
        wait_for_query_completion(athena_client, response['QueryExecutionId'])
        
        # Get results to see if data was found
        results = athena_client.get_query_results(QueryExecutionId=response['QueryExecutionId'])
        rows = results.get('ResultSet', {}).get('Rows', [])
        
        if len(rows) > 1:  # More than header row
            print(f' Partition discovery test successful - found {len(rows)-1} partitions with data')
            for row in rows[1:4]:  # Show first 3 data rows
                row_data = [col.get('VarCharValue', '') for col in row.get('Data', [])]
                print(f'   Found partition: {"/".join(row_data[:5])} with {row_data[5]} records')
        else:
            print(' Partition discovery test - no data found yet (this is normal if no prediction files exist)')
        
    except Exception as e:
        print(f' Could not test partition discovery: {str(e)}')

def wait_for_query_completion(athena_client, query_execution_id, max_wait_time=300):
    """Wait for Athena query to complete"""
    
    waited_time = 0
    while waited_time < max_wait_time:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response['QueryExecution']['Status']['State']
        
        if status in ['SUCCEEDED']:
            return
        elif status in ['FAILED', 'CANCELLED']:
            error_msg = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
            raise Exception(f'Query failed with status {status}: {error_msg}')
        
        time.sleep(5)
        waited_time += 5
    
    raise Exception(f'Query timed out after {max_wait_time} seconds')

def save_athena_config(s3_client, bucket, database_name, table_name, results_location, data_location, env_name):
    """Save Athena configuration for Lambda to use"""
    
    config = {
        'athena_database': database_name,
        'athena_table': table_name,
        'environment': env_name,
        'data_location': data_location,
        'results_location': results_location,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'table_full_name': f'{database_name}.{table_name}',
        'partition_structure': 'load_profile={load_profile}/load_segment={load_segment}/year={year}/month={month:02d}/day={day:02d}/',
        'automatic_partition_discovery': True,
        'supports_zero_padded_partitions': True
    }
    
    config_key = f'athena-config/{env_name}/config.json'
    
    s3_client.put_object(
        Bucket=bucket,
        Key=config_key,
        Body=json.dumps(config, indent=2),
        ContentType='application/json'
    )
    
    print(f' Configuration saved to s3://{bucket}/{config_key}')
    print(f'Configuration details:')
    print(f'  Database: {database_name}')
    print(f'  Table: {table_name}')
    print(f'  Full table name: {database_name}.{table_name}')
    print(f'  Data location: {data_location}')
    print(f'  Results location: {results_location}')
    print(f'  Automatic partition discovery: Enabled')
    print(f'  Zero-padded partitions: Supported (month=05, day=30)')

if __name__ == '__main__':
    result = setup_athena_infrastructure()
    print(f'Setup result: {result}')
