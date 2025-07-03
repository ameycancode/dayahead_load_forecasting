#!/usr/bin/env python3
"""
Redshift Infrastructure Setup Script for Energy Load Forecasting Pipeline
Creates tables, views, and accuracy calculation views with configurable parameters
Uses Redshift Data API to avoid GitHub Actions connection issues
"""
import boto3
import time
import os
import json
import traceback

# Configuration for accuracy views
ACCURACY_CONFIG = {
    # Submission lag days (business days)
    'INITIAL_SUBMISSION_LAG_DAYS': 14,
    'FINAL_SUBMISSION_LAG_DAYS': 48,
   
    # Data retention for accuracy views (1 year)
    'ACCURACY_RETENTION_MONTHS': 12,
   
    # Solar segment rate group prefixes (configurable)
    'SOLAR_RATE_GROUP_PREFIXES': ['NEM', 'SBP'],  # Easy to extend
   
    # Customer segment values (configurable for case handling)
    'CUSTOMER_SEGMENTS': {
        'solar': 'SOLAR',      # Database value for solar customers
        'nonsolar': 'NONSOLAR' # Database value for non-solar customers
    },
   
    # Accuracy metrics configuration
    'ACCURACY_METRICS': {
        'common': ['mape', 'rmse', 'mae', 'r2'],
        'solar_additional': ['smape', 'wape']  # Additional metrics for solar
    }
}

def setup_redshift_infrastructure():
    """Setup Redshift table and views with proper verification"""
   
    # Get configuration from environment variables
    try:
        cluster_identifier = os.environ['REDSHIFT_CLUSTER_IDENTIFIER']
        database = os.environ['REDSHIFT_DATABASE']
        db_user = os.environ['REDSHIFT_DB_USER']
        region = os.environ['REDSHIFT_REGION']
        operational_schema = os.environ['REDSHIFT_OPERATIONAL_SCHEMA']
        operational_table = os.environ['REDSHIFT_OPERATIONAL_TABLE']
        bi_schema = os.environ['REDSHIFT_BI_SCHEMA']
        bi_view = os.environ['REDSHIFT_BI_VIEW']
       
        # Input schema and table for accuracy calculations
        input_schema = os.environ.get('REDSHIFT_INPUT_SCHEMA', 'edp_cust_dev')
        input_table = os.environ.get('REDSHIFT_INPUT_TABLE', 'caiso_sqmd')
       
        s3_staging_bucket = os.environ['S3_STAGING_BUCKET']
        s3_staging_prefix = os.environ['S3_STAGING_PREFIX']
        redshift_iam_role = os.environ['REDSHIFT_IAM_ROLE']
        env_name = os.environ['ENV_NAME']
    except KeyError as e:
        print(f' Missing required environment variable: {e}')
        return 'failed'
   
    print(f' Setting up Redshift table and views for {env_name} environment')
    print(f'   Cluster: {cluster_identifier}')
    print(f'   Database: {database}')
    print(f'   Operational Schema: {operational_schema} (assuming exists)')
    print(f'   BI Schema: {bi_schema} (assuming exists)')
    print(f'   Input Schema: {input_schema} (for accuracy calculations)')
    print(f'   Table: {operational_schema}.{operational_table}')
    print(f'   Main View: {bi_schema}.{bi_view}')
   
    try:
        # Step 1: Verify cluster exists and get basic info
        print('Step 1: Verifying Redshift cluster...')
        try:
            cluster_info = verify_cluster_exists(cluster_identifier, region)
        except Exception as e:
            print(f'❌ CRITICAL: Cluster verification failed: {str(e)}')
            return 'failed'
       
        # Step 2: Verify schemas exist (CRITICAL CHECK)
        print('Step 2: Verifying schemas exist...')
        try:
            verify_schemas_exist(cluster_identifier, database, db_user, region,
                                 operational_schema, bi_schema)
        except Exception as e:
            print(f'❌ CRITICAL: Schema verification failed: {str(e)}')
            return 'failed'
       
        # Step 3: Create operational table using Data API
        print('Step 3: Creating operational table...')
        try:
            create_operational_table_via_data_api(
                cluster_identifier, database, db_user, region,
                operational_schema, operational_table
            )
        except Exception as e:
            print(f'❌ CRITICAL: Operational table creation failed: {str(e)}')
            return 'failed'
       
        # Step 4: Verify table was actually created
        print('Step 4: Verifying table creation...')
        try:
            verify_table_creation(cluster_identifier, database, db_user, region,
                                 operational_schema, operational_table)
        except Exception as e:
            print(f'❌ CRITICAL: Table verification failed: {str(e)}')
            return 'failed'
       
        # Step 5: Create BI views using Data API
        print('Step 5: Creating BI views and accuracy views...')
        try:
            success_count, failure_count, failed_views = create_bi_views_via_data_api(
                cluster_identifier, database, db_user, region,
                operational_schema, operational_table, bi_schema, bi_view,
                input_schema, input_table
            )
           
            # Evaluate view creation results
            if failure_count > 0:
                print(f'❌ CRITICAL: {failure_count} views failed to create: {failed_views}')
                return 'failed'
            elif success_count == 0:
                print('❌ CRITICAL: No views were created successfully')
                return 'failed'
            else:
                print(f'✅ All {success_count} views created successfully')
        except Exception as e:
            print(f'❌ CRITICAL: BI views creation failed: {str(e)}')
            return 'failed'
       
        # Step 6: Verify views were actually created
        print('Step 6: Verifying view creation...')
        try:
            missing_views = verify_views_creation(cluster_identifier, database, db_user, region,
                                                  bi_schema, bi_view)
            if missing_views:
                print(f'❌ CRITICAL: Required views missing: {missing_views}')
                return 'failed'
        except Exception as e:
            print(f'❌ CRITICAL: View verification failed: {str(e)}')
            return 'failed'
       
        # Step 7: Create S3 staging directories
        print('Step 7: Setting up S3 staging...')
        try:
            create_s3_staging_directories(s3_staging_bucket, s3_staging_prefix)
        except Exception as e:
            print(f'❌ CRITICAL: S3 staging setup failed: {str(e)}')
            return 'failed'
       
        # Step 8: Save configuration for Lambda
        print('Step 8: Saving configuration...')
        try:
            save_redshift_config(s3_staging_bucket, cluster_identifier, database,
                            operational_schema, operational_table, bi_schema, bi_view,
                            redshift_iam_role, env_name)
        except Exception as e:
            print(f'❌ CRITICAL: Configuration save failed: {str(e)}')
            return 'failed'
       
        print('✅ Redshift infrastructure setup completed successfully!')
        return 'success'
   
    except KeyError as e:
        print(f'❌ CRITICAL: Missing required environment variable: {e}')
        return 'failed'
       
    except Exception as e:
        print(f' Error setting up Redshift infrastructure: {str(e)}')
        traceback.print_exc()
        return 'failed'

def verify_cluster_exists(cluster_identifier, region):
    """Verify cluster exists and get basic info"""
    try:
        redshift_client = boto3.client('redshift', region_name=region)
       
        cluster_response = redshift_client.describe_clusters(
            ClusterIdentifier=cluster_identifier
        )
        cluster = cluster_response['Clusters'][0]
       
        print(f'   Cluster found: {cluster_identifier}')
        print(f'   Status: {cluster["ClusterStatus"]}')
        print(f'   Publicly accessible: {cluster.get("PubliclyAccessible", False)}')
        print(f'   Endpoint: {cluster["Endpoint"]["Address"]}:{cluster["Endpoint"]["Port"]}')
       
        if cluster['ClusterStatus'] != 'available':
            raise Exception(f"Cluster is not available. Status: {cluster['ClusterStatus']}")
       
        return cluster
       
    except Exception as e:
        print(f'   Error verifying cluster: {str(e)}')
        raise

def verify_schemas_exist(cluster_identifier, database, db_user, region,
                        operational_schema, bi_schema):
    """
    Verify that both schemas exist using information_schema.tables
    Since information_schema.schemata might be empty in some Redshift configurations
    """
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
       
        # Use information_schema.tables to check schema existence
        # This approach works better with Redshift configurations
        schema_check_sql = f"""
        SELECT DISTINCT table_schema
        FROM information_schema.tables
        WHERE table_schema IN ('{operational_schema}', '{bi_schema}')
        ORDER BY table_schema
        """

        print(f'SCHEMA CHECK SQL QUERY: {schema_check_sql}')
       
        print(f'   Checking schemas using information_schema.tables: {operational_schema}, {bi_schema}')
       
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=schema_check_sql
        )
       
        query_id = response['Id']
        wait_for_query_completion(redshift_data_client, query_id, 'schema verification')
       
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=query_id)
       
        existing_schemas = []
        for row in result_response.get('Records', []):
            if row and len(row) > 0 and 'stringValue' in row[0]:
                existing_schemas.append(row[0]['stringValue'])
       
        print(f'   Found schemas in tables: {existing_schemas}')
       
        # Alternative check: Try to query pg_namespace directly (more reliable)
        namespace_check_sql = f"""
        SELECT nspname
        FROM pg_namespace
        WHERE nspname IN ('{operational_schema}', '{bi_schema}')
        ORDER BY nspname
        """

        print(f'NAMESPACE CHECK SQL: {namespace_check_sql}')
       
        print(f'   Double-checking schemas using pg_namespace...')
       
        namespace_response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=namespace_check_sql
        )
       
        namespace_query_id = namespace_response['Id']
        wait_for_query_completion(redshift_data_client, namespace_query_id, 'pg_namespace schema check')
       
        # Get pg_namespace results
        namespace_result = redshift_data_client.get_statement_result(Id=namespace_query_id)
       
        pg_schemas = []
        for row in namespace_result.get('Records', []):
            if row and len(row) > 0 and 'stringValue' in row[0]:
                pg_schemas.append(row[0]['stringValue'])
       
        print(f'   Found schemas in pg_namespace: {pg_schemas}')
       
        # Check if both required schemas exist (use pg_namespace as authoritative)
        required_schemas = [operational_schema, bi_schema]
        missing_schemas = [schema for schema in required_schemas if schema not in pg_schemas]
       
        if missing_schemas:
            print(f'   Missing required schemas: {missing_schemas}')
           
            # Show available schemas for debugging
            all_schemas_sql = "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' AND nspname != 'information_schema' ORDER BY nspname"

            print(f'ALL SCHEMAS SQL QUERY: {all_schemas_sql}')
           
            all_schemas_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=all_schemas_sql
            )
           
            all_schemas_query_id = all_schemas_response['Id']
            wait_for_query_completion(redshift_data_client, all_schemas_query_id, 'list all schemas')
           
            all_schemas_result = redshift_data_client.get_statement_result(Id=all_schemas_query_id)
           
            available_schemas = []
            for row in all_schemas_result.get('Records', []):
                if row and len(row) > 0 and 'stringValue' in row[0]:
                    available_schemas.append(row[0]['stringValue'])
           
            print(f'   Available schemas in cluster: {available_schemas}')
            raise Exception(f"Missing required schemas: {missing_schemas}. Available: {available_schemas}")
       
        print(f'   All required schemas exist: {required_schemas}')
       
        # Check permissions on schemas
        check_permissions(redshift_data_client, cluster_identifier, database, db_user,
                         operational_schema, bi_schema)
       
    except Exception as e:
        print(f'   Schema verification failed: {str(e)}')
        raise

def check_permissions(redshift_data_client, cluster_identifier, database, db_user,
                     operational_schema, bi_schema):
    """Check if user has necessary permissions on schemas"""
    try:
        print(f'   Checking permissions for user {db_user}...')
       
        # Check CREATE permission on operational schema
        perm_check_sql = f"""
        SELECT
            HAS_SCHEMA_PRIVILEGE('{operational_schema}', 'CREATE') as can_create_in_operational,
            HAS_SCHEMA_PRIVILEGE('{bi_schema}', 'CREATE') as can_create_in_bi,
            HAS_SCHEMA_PRIVILEGE('{operational_schema}', 'USAGE') as can_use_operational,
            HAS_SCHEMA_PRIVILEGE('{bi_schema}', 'USAGE') as can_use_bi
        """
       
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=perm_check_sql
        )
       
        query_id = response['Id']
        wait_for_query_completion(redshift_data_client, query_id, 'permission check')
       
        result_response = redshift_data_client.get_statement_result(Id=query_id)
       
        if result_response.get('Records'):
            perms = result_response['Records'][0]
            can_create_operational = perms[0].get('booleanValue', False)
            can_create_bi = perms[1].get('booleanValue', False)
            can_use_operational = perms[2].get('booleanValue', False)
            can_use_bi = perms[3].get('booleanValue', False)
           
            print(f'   Permission check results:')
            print(f'      CREATE in {operational_schema}: {can_create_operational}')
            print(f'      CREATE in {bi_schema}: {can_create_bi}')
            print(f'      USAGE in {operational_schema}: {can_use_operational}')
            print(f'      USAGE in {bi_schema}: {can_use_bi}')
           
            if not can_create_operational:
                print(f'   Warning: No CREATE permission in {operational_schema}')
            if not can_create_bi:
                print(f'   Warning: No CREATE permission in {bi_schema}')
               
        else:
            print(f'   Could not retrieve permission information')
           
    except Exception as e:
        print(f'   Permission check failed (may not be critical): {str(e)}')
        # Don't raise here as permissions check is informational

def create_operational_table_via_data_api(cluster_identifier, database, db_user, region,
                                         operational_schema, operational_table):
    """Create operational table using Redshift Data API with explicit verification"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
       
        # Check if table already exists
        check_table_sql = f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = '{operational_schema}'
        AND table_name = '{operational_table}'
        """
       
        print(f'   Checking if table {operational_schema}.{operational_table} exists...')
        print(f'   QUERY: {check_table_sql}')
       
        check_response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=check_table_sql
        )
       
        # Wait for check query to complete
        check_query_id = check_response['Id']
        wait_for_query_completion(redshift_data_client, check_query_id, 'table existence check')
       
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=check_query_id)
        table_exists = int(result_response['Records'][0][0]['longValue']) > 0
       
        if table_exists:
            print(f'   Table {operational_schema}.{operational_table} already exists')
           
            # Verify table structure
            verify_table_sql = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{operational_schema}'
            AND table_name = '{operational_table}'
            ORDER BY ordinal_position
            LIMIT 10
            """

            print(f'Verify table structure Query: {verify_table_sql}')
           
            verify_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=verify_table_sql
            )
           
            verify_query_id = verify_response['Id']
            wait_for_query_completion(redshift_data_client, verify_query_id, 'table structure verification')
           
            structure_result = redshift_data_client.get_statement_result(Id=verify_query_id)
            columns = structure_result['Records']
           
            print(f'   Table has {len(columns)} columns')
            for i, col in enumerate(columns[:5]):  # Show first 5 columns
                col_name = col[0]['stringValue']
                col_type = col[1]['stringValue']
                print(f'      {i+1}. {col_name}: {col_type}')
           
            if len(columns) > 5:
                print(f'      ... and {len(columns) - 5} more columns')
               
        else:
            # Create new table
            print(f'   Creating table {operational_schema}.{operational_table}...')
           
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {operational_schema}.{operational_table} (
                forecast_datetime TIMESTAMP NOT NULL,
                predicted_lossadjustedload DECIMAL(12,4),
                run_id VARCHAR(100),
                model_version VARCHAR(50),
                run_user VARCHAR(50),
                created_at TIMESTAMP DEFAULT GETDATE(),
                load_profile VARCHAR(20),
                load_segment VARCHAR(20),
                year INTEGER,
                month INTEGER,
                day INTEGER,
               
                -- Additional metadata for better tracking
                forecast_horizon_hours INTEGER,
                model_confidence DECIMAL(5,4)
            )
            DISTKEY(load_profile)
            SORTKEY(forecast_datetime, load_profile, load_segment);
            """

            print(f"Create TABLE SQL: {create_table_sql}")
           
            create_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=create_table_sql
            )
           
            create_query_id = create_response['Id']
            wait_for_query_completion(redshift_data_client, create_query_id, 'table creation')
           
            print(f'   Table {operational_schema}.{operational_table} created successfully')
           
            # Add table comment
            comment_sql = f"""
            COMMENT ON TABLE {operational_schema}.{operational_table} IS
            'Day-ahead load forecasts generated by ML models. Updated by prediction Lambda functions.'
            """
           
            comment_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=comment_sql
            )
           
            comment_query_id = comment_response['Id']
            wait_for_query_completion(redshift_data_client, comment_query_id, 'table comment')
       
    except Exception as e:
        print(f'   Error with operational table: {str(e)}')
        raise

def verify_table_creation(cluster_identifier, database, db_user, region,
                         operational_schema, operational_table):
    """Explicitly verify that the table was actually created"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
       
        # Method 1: Check using information_schema.tables
        verify_sql = f"""
        SELECT
            table_schema,
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = '{operational_schema}'
        AND table_name = '{operational_table}'
        """
       
        print(f'   Verifying table {operational_schema}.{operational_table} using information_schema.tables...')
       
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=verify_sql
        )
       
        query_id = response['Id']
        wait_for_query_completion(redshift_data_client, query_id, 'table verification')
       
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=query_id)
        records = result_response.get('Records', [])
       
        if records:
            table_info = records[0]
            found_schema = table_info[0]['stringValue']
            found_table = table_info[1]['stringValue']
            found_type = table_info[2]['stringValue']
           
            print(f'   VERIFIED: Table found!')
            print(f'      Schema: {found_schema}')
            print(f'      Table: {found_table}')
            print(f'      Type: {found_type}')
           
            # Method 2: Get column count
            column_check_sql = f"""
            SELECT COUNT(*) as column_count
            FROM information_schema.columns
            WHERE table_schema = '{operational_schema}'
            AND table_name = '{operational_table}'
            """

            print(f'Column Check SQL QUERY: {column_check_sql}')
           
            col_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=column_check_sql
            )
           
            col_query_id = col_response['Id']
            wait_for_query_completion(redshift_data_client, col_query_id, 'column count check')
           
            col_result = redshift_data_client.get_statement_result(Id=col_query_id)
            column_count = int(col_result['Records'][0][0]['longValue'])
           
            print(f'      Columns: {column_count}')
           
            if column_count < 5:
                print(f'   Warning: Table has only {column_count} columns, expected more')
            else:
                print(f'   Table structure looks good with {column_count} columns')
               
        else:
            print(f'   Table {operational_schema}.{operational_table} NOT FOUND after creation!')
           
            # Debug: Show what tables DO exist in the schema
            debug_sql = f"""
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = '{operational_schema}'
            ORDER BY table_name
            """

            print(f'DEBUG SQL QUERY: {debug_sql}')
           
            debug_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=debug_sql
            )
           
            debug_query_id = debug_response['Id']
            wait_for_query_completion(redshift_data_client, debug_query_id, 'debug table list')
           
            debug_result = redshift_data_client.get_statement_result(Id=debug_query_id)
           
            print(f'   Tables that DO exist in {operational_schema}:')
            for row in debug_result.get('Records', []):
                table_name = row[0]['stringValue']
                table_type = row[1]['stringValue']
                print(f'      - {table_name} ({table_type})')
           
            raise Exception(f"Table verification failed - table not found after creation")
       
    except Exception as e:
        print(f'   Table verification failed: {str(e)}')
        raise

def get_solar_filter_clause():
    """Generate configurable solar filter clause based on rate group prefixes"""
    prefixes = ACCURACY_CONFIG['SOLAR_RATE_GROUP_PREFIXES']
    conditions = [f"rategroup LIKE '{prefix}%'" for prefix in prefixes]
    return f"({' OR '.join(conditions)})"

def get_customer_segment_case_statement():
    """Generate CASE statement for customer segments with proper case handling"""
    solar_filter = get_solar_filter_clause()
    solar_value = ACCURACY_CONFIG['CUSTOMER_SEGMENTS']['solar']
    nonsolar_value = ACCURACY_CONFIG['CUSTOMER_SEGMENTS']['nonsolar']
   
    return f"""
    CASE
        WHEN {solar_filter} THEN '{solar_value}'
        ELSE '{nonsolar_value}'
    END as load_segment
    """

def create_bi_views_via_data_api(cluster_identifier, database, db_user, region,
                               operational_schema, operational_table, bi_schema, bi_view,
                               input_schema, input_table):
    """Create BI views using Redshift Data API"""
    success_count = 0
    failure_count = 0
    failed_views = []
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
       
        print(f'   Using solar segments: SOLAR={ACCURACY_CONFIG["CUSTOMER_SEGMENTS"]["solar"]}, NONSOLAR={ACCURACY_CONFIG["CUSTOMER_SEGMENTS"]["nonsolar"]}')
        print(f'   Using solar filter: {get_solar_filter_clause()}')

        views_to_create = [
            {
                'name': bi_view,
                'description': 'Main BI view for all forecast reporting with business-friendly labels',
                'sql': get_main_bi_view_sql(operational_schema, operational_table, bi_schema, bi_view),
                'critical': True # CRITICAL - must succeed
            },
            {
                'name': 'vw_current_dayahead_forecasts',
                'description': 'Latest forecasts only - for real-time dashboards',
                'sql': get_current_forecasts_view_sql(operational_schema, operational_table, bi_schema),
                'critical': True # CRITICAL - must succeed
            },
            {
                'name': 'vw_forecast_summary_by_profile',
                'description': 'Daily aggregated forecasts - for executive dashboards',
                'sql': get_summary_view_sql(operational_schema, operational_table, bi_schema),
                'critical': True # CRITICAL - must succeed
            },
            {
                'name': 'vw_initial_submission_actuals',
                'description': 'Initial submission actual loads with configurable solar filters (1-year retention)',
                'sql': get_initial_actuals_view_sql(input_schema, input_table, bi_schema),
                'critical': False # NON-CRITICAL - accuracy feature
            },
            {
                'name': 'vw_final_submission_actuals',
                'description': 'Final submission actual loads with configurable solar filters (1-year retention)',
                'sql': get_final_actuals_view_sql(input_schema, input_table, bi_schema),
                'critical': False # NON-CRITICAL - accuracy feature
            },
            {
                'name': 'vw_forecast_accuracy_metrics',
                'description': 'Comprehensive forecast accuracy calculations with sMAPE and WAPE for solar',
                'sql': get_forecast_accuracy_view_sql(bi_schema),
                'critical': False # NON-CRITICAL - accuracy feature
            },
            {
                'name': 'vw_accuracy_summary_dashboard',
                'description': 'Aggregated accuracy metrics for reporting dashboards',
                'sql': get_accuracy_summary_view_sql(bi_schema),
                'critical': False # NON-CRITICAL - accuracy feature
            }
        ]
       
        for view_info in views_to_create:
            view_name = view_info['name']
            is_critical = view_info.get('critical', False)
            print(f' Creating view {bi_schema}.{view_name} ({"CRITICAL" if is_critical else "OPTIONAL"})...')
           
            try:
                # Create view
                create_response = redshift_data_client.execute_statement(
                    ClusterIdentifier=cluster_identifier,
                    Database=database,
                    DbUser=db_user,
                    Sql=view_info['sql']
                )
               
                create_query_id = create_response['Id']
                wait_for_query_completion(redshift_data_client, create_query_id, f'view {view_info["name"]} creation')
               
                print(f'   ✅ View {bi_schema}.{view_info["name"]} created successfully')
                success_count += 1
               
                # Add view comment
                try:
                    comment_sql = f"COMMENT ON VIEW {bi_schema}.{view_info['name']} IS '{view_info['description']}'"
                   
                    comment_response = redshift_data_client.execute_statement(
                        ClusterIdentifier=cluster_identifier,
                        Database=database,
                        DbUser=db_user,
                        Sql=comment_sql
                    )
                   
                    comment_query_id = comment_response['Id']
                    wait_for_query_completion(redshift_data_client, comment_query_id, f'view {view_info["name"]} comment')
                except Exception as comment_error:
                    print(f' ⚠️         Warning: Failed to add comment to {view_name}: {str(comment_error)}')
               
            except Exception as view_error:
                failure_count += 1
                failed_views.append(view_name)

                if is_critical:
                    print(f' CRITICAL VIEW FAILURE: {view_name} - {str(view_error)}')
                    # For critical views, we should fail immediately
                    raise Exception(f'Critical view {view_name} failed: {str(view_error)}')
                else:
                    print(f' Optional view failed: {view_name} - {str(view_error)}')
                    print(' Continuing with remaining views...')
                    print(f'   Failed to create view {view_info["name"]}: {str(view_error)}')
                    # Continue with other views, don't fail completely
                    continue
       
        print(f' BI view creation completed: {success_count} succeeded, {failure_count} failed')
 
        # Check if we have minimum required views
        if success_count < 3: # Must have at least the 3 critical views
            raise Exception(f'Insufficient views created: {success_count}/3 critical views succeeded')
       
        return success_count, failure_count, failed_views
       
    except Exception as e:
        print(f' ❌ CRITICAL ERROR in view creation: {str(e)}')
        raise

def get_main_bi_view_sql(operational_schema, operational_table, bi_schema, bi_view):
    """Get SQL for main BI view with latest record logic"""
    return f"""
    CREATE OR REPLACE VIEW {bi_schema}.{bi_view} AS
    WITH latest_predictions AS (
        SELECT
            forecast_datetime,
            load_profile,
            load_segment,
            MAX(created_at) as latest_created_at
        FROM {operational_schema}.{operational_table}
        WHERE predicted_lossadjustedload IS NOT NULL
        GROUP BY forecast_datetime, load_profile, load_segment
    )
    SELECT
        -- Core forecast data
        f.forecast_datetime,
        f.predicted_lossadjustedload,
        f.load_profile,
        f.load_segment,
       
        -- Metadata
        f.run_id,
        f.model_version,
        f.run_user,
        f.created_at,
       
        -- Date dimensions for easier reporting
        f.year,
        f.month,
        f.day,
        EXTRACT(hour FROM f.forecast_datetime) as forecast_hour,
        EXTRACT(dow FROM f.forecast_datetime) as day_of_week,
        EXTRACT(doy FROM f.forecast_datetime) as day_of_year,
        EXTRACT(quarter FROM f.forecast_datetime) as forecast_quarter,
       
        -- Business-friendly calculated fields
        CASE
            WHEN EXTRACT(dow FROM f.forecast_datetime) IN (0,6) THEN 'Weekend'
            ELSE 'Weekday'
        END as day_type,
       
        CASE
            WHEN EXTRACT(hour FROM f.forecast_datetime) BETWEEN 6 AND 9 THEN 'Morning Peak'
            WHEN EXTRACT(hour FROM f.forecast_datetime) BETWEEN 10 AND 15 THEN 'Midday'
            WHEN EXTRACT(hour FROM f.forecast_datetime) BETWEEN 16 AND 21 THEN 'Evening Peak'
            ELSE 'Off Peak'
        END as time_period,
       
        -- Customer type labels
        CASE f.load_profile
            WHEN 'RES' THEN 'Residential'
            WHEN 'MEDCI' THEN 'Medium Commercial'
            WHEN 'SMLCOM' THEN 'Small Commercial'
            ELSE f.load_profile
        END as customer_type,
       
        CASE f.load_segment
            WHEN 'SOLAR' THEN 'Solar Customers'
            WHEN 'NONSOLAR' THEN 'Non-Solar Customers'
            ELSE f.load_segment
        END as customer_segment_desc,
       
        -- Forecast metadata
        DATEDIFF(day, f.created_at::date, f.forecast_datetime::date) as forecast_horizon_days,
        DATEDIFF(hour, f.created_at, GETDATE()) as hours_since_forecast
       
    FROM {operational_schema}.{operational_table} f
    INNER JOIN latest_predictions lp ON (
        f.forecast_datetime = lp.forecast_datetime
        AND f.load_profile = lp.load_profile
        AND f.load_segment = lp.load_segment
        AND f.created_at = lp.latest_created_at
    )
    WHERE f.predicted_lossadjustedload IS NOT NULL
    """

def get_current_forecasts_view_sql(operational_schema, operational_table, bi_schema):
    """Get SQL for current forecasts view with latest record logic"""
    return f"""
    CREATE OR REPLACE VIEW {bi_schema}.vw_current_dayahead_forecasts AS
    WITH latest_forecasts AS (
        SELECT
            forecast_datetime,
            load_profile,
            load_segment,
            MAX(created_at) as latest_created_at
        FROM {operational_schema}.{operational_table}
        WHERE forecast_datetime >= GETDATE()::date  -- Future forecasts only
        AND predicted_lossadjustedload IS NOT NULL
        GROUP BY forecast_datetime, load_profile, load_segment
    )
    SELECT
        f.forecast_datetime,
        f.predicted_lossadjustedload,
        f.load_profile,
        f.load_segment,
        f.run_id,
        f.model_version,
        f.created_at,
       
        -- Time dimensions
        EXTRACT(hour FROM f.forecast_datetime) as hour,
        EXTRACT(dow FROM f.forecast_datetime) as day_of_week,
        DATE_TRUNC('day', f.forecast_datetime)::date as forecast_date,
       
        -- Business labels
        CASE f.load_profile
            WHEN 'RES' THEN 'Residential'
            WHEN 'MEDCI' THEN 'Medium Commercial'
            WHEN 'SMLCOM' THEN 'Small Commercial'
            ELSE f.load_profile
        END as customer_type,
       
        CASE f.load_segment
            WHEN 'SOLAR' THEN 'Solar'
            WHEN 'NONSOLAR' THEN 'Non-Solar'
            ELSE f.load_segment
        END as segment
       
    FROM {operational_schema}.{operational_table} f
    INNER JOIN latest_forecasts lf ON (
        f.forecast_datetime = lf.forecast_datetime
        AND f.load_profile = lf.load_profile
        AND f.load_segment = lf.load_segment
        AND f.created_at = lf.latest_created_at
    )
    ORDER BY f.forecast_datetime, f.load_profile, f.load_segment
    """

def get_summary_view_sql(operational_schema, operational_table, bi_schema):
    """Get SQL for summary view with latest record logic"""
    return f"""
    CREATE OR REPLACE VIEW {bi_schema}.vw_forecast_summary_by_profile AS
    WITH latest_forecasts AS (
        SELECT
            forecast_datetime,
            load_profile,
            load_segment,
            MAX(created_at) as latest_created_at
        FROM {operational_schema}.{operational_table}
        WHERE forecast_datetime >= DATEADD(month, -3, GETDATE())  -- Last 3 months
        AND predicted_lossadjustedload IS NOT NULL
        GROUP BY forecast_datetime, load_profile, load_segment
    )
    SELECT
        DATE_TRUNC('day', f.forecast_datetime)::date as forecast_date,
        f.load_profile,
        f.load_segment,
       
        -- Daily aggregations
        AVG(f.predicted_lossadjustedload) as avg_daily_load,
        MIN(f.predicted_lossadjustedload) as min_daily_load,
        MAX(f.predicted_lossadjustedload) as max_daily_load,
        SUM(f.predicted_lossadjustedload) as total_daily_load,
        COUNT(*) as forecast_count,
       
        -- Peak period analysis
        AVG(CASE WHEN EXTRACT(hour FROM f.forecast_datetime) BETWEEN 17 AND 21
            THEN f.predicted_lossadjustedload END) as avg_evening_peak_load,
        AVG(CASE WHEN EXTRACT(hour FROM f.forecast_datetime) BETWEEN 10 AND 15
            THEN f.predicted_lossadjustedload END) as avg_midday_load,
       
        -- Metadata
        MAX(f.created_at) as latest_forecast_time,
        MAX(f.run_id) as latest_run_id,
       
        -- Business dimensions
        CASE f.load_profile
            WHEN 'RES' THEN 'Residential'
            WHEN 'MEDCI' THEN 'Medium Commercial'
            WHEN 'SMLCOM' THEN 'Small Commercial'
            ELSE f.load_profile
        END as customer_type
       
    FROM {operational_schema}.{operational_table} f
    INNER JOIN latest_forecasts lf ON (
        f.forecast_datetime = lf.forecast_datetime
        AND f.load_profile = lf.load_profile
        AND f.load_segment = lf.load_segment
        AND f.created_at = lf.latest_created_at
    )
    GROUP BY
        DATE_TRUNC('day', f.forecast_datetime)::date,
        f.load_profile,
        f.load_segment
    """

def get_initial_actuals_view_sql(input_schema, input_table, bi_schema):
    """Get SQL for initial submission actuals view - SIMPLIFIED VERSION"""
    return f"""
    CREATE OR REPLACE VIEW {bi_schema}.vw_initial_submission_actuals AS
    SELECT
        DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp) as actual_datetime,
        loadprofile as load_profile,
        CASE
            WHEN (UPPER(rategroup) LIKE 'NEM%' OR UPPER(rategroup) LIKE 'SBP%') THEN 'SOLAR'
            ELSE 'NONSOLAR'
        END as load_segment,
        SUM(lossadjustedload) as actual_lossadjustedload,
        SUM(metercount) as total_meters,
        COUNT(*) as record_count,
        MAX(createddate) as latest_created,
       
        -- Date dimensions
        EXTRACT(year FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as year,
        EXTRACT(month FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as month,
        EXTRACT(day FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as day,
        EXTRACT(hour FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as hour,
        EXTRACT(dow FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as day_of_week,
       
        -- Business labels
        CASE loadprofile
            WHEN 'RES' THEN 'Residential'
            WHEN 'MEDCI' THEN 'Medium Commercial'
            WHEN 'SMLCOM' THEN 'Small Commercial'
            ELSE loadprofile
        END as customer_type,
       
        'Initial' as submission_type
       
    FROM {input_schema}.{input_table}
    WHERE submission = 'Initial'
    AND tradedatelocal >= DATEADD(month, -12, GETDATE())
    AND tradedatelocal <= GETDATE()
    AND lossadjustedload IS NOT NULL
    AND loadprofile IN ('RES', 'MEDCI', 'SMLCOM')
    GROUP BY
        DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp),
        loadprofile,
        CASE
            WHEN (UPPER(rategroup) LIKE 'NEM%' OR UPPER(rategroup) LIKE 'SBP%') THEN 'SOLAR'
            ELSE 'NONSOLAR'
        END
    ORDER BY actual_datetime, loadprofile, load_segment
    """

def get_final_actuals_view_sql(input_schema, input_table, bi_schema):
    """Get SQL for final submission actuals view - SIMPLIFIED VERSION"""
    return f"""
    CREATE OR REPLACE VIEW {bi_schema}.vw_final_submission_actuals AS
    SELECT
        DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp) as actual_datetime,
        loadprofile as load_profile,
        CASE
            WHEN (UPPER(rategroup) LIKE 'NEM%' OR UPPER(rategroup) LIKE 'SBP%') THEN 'SOLAR'
            ELSE 'NONSOLAR'
        END as load_segment,
        SUM(lossadjustedload) as actual_lossadjustedload,
        SUM(metercount) as total_meters,
        COUNT(*) as record_count,
        MAX(createddate) as latest_created,
       
        -- Date dimensions
        EXTRACT(year FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as year,
        EXTRACT(month FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as month,
        EXTRACT(day FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as day,
        EXTRACT(hour FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as hour,
        EXTRACT(dow FROM DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp)) as day_of_week,
       
        -- Business labels
        CASE loadprofile
            WHEN 'RES' THEN 'Residential'
            WHEN 'MEDCI' THEN 'Medium Commercial'
            WHEN 'SMLCOM' THEN 'Small Commercial'
            ELSE loadprofile
        END as customer_type,
       
        'Final' as submission_type
       
    FROM {input_schema}.{input_table}
    WHERE submission = 'Final'
    AND tradedatelocal >= DATEADD(month, -12, GETDATE())
    AND tradedatelocal <= GETDATE()
    AND lossadjustedload IS NOT NULL
    AND loadprofile IN ('RES', 'MEDCI', 'SMLCOM')
    GROUP BY
        DATEADD(hour, tradehourstartlocal::integer, tradedatelocal::timestamp),
        loadprofile,
        CASE
            WHEN (UPPER(rategroup) LIKE 'NEM%' OR UPPER(rategroup) LIKE 'SBP%') THEN 'SOLAR'
            ELSE 'NONSOLAR'
        END
    ORDER BY actual_datetime, loadprofile, load_segment
    """

def get_forecast_accuracy_view_sql(bi_schema):
    """Get SQL for comprehensive forecast accuracy calculations - CORRECTED"""
    solar_segment = ACCURACY_CONFIG['CUSTOMER_SEGMENTS']['solar']
   
    return f"""
    CREATE OR REPLACE VIEW {bi_schema}.vw_forecast_accuracy_metrics AS
    WITH forecast_accuracy_base AS (
        -- Join predictions with Final actuals (preferred)
        SELECT
            f.forecast_datetime,
            f.load_profile,
            f.load_segment,
            f.predicted_lossadjustedload as forecast_value,
            a.actual_lossadjustedload as actual_value,
            f.run_id,
            f.model_version,
            f.customer_type,
            a.submission_type,
           
            -- Date dimensions
            f.year,
            f.month,
            f.day,
            f.forecast_hour,
            f.day_of_week,
            f.day_type,
            f.time_period,
           
            -- Forecast metadata
            f.forecast_horizon_days,
            f.created_at as forecast_created_at,
            a.latest_created as actual_created_at,
           
            -- Error calculations
            ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) as absolute_error,
            (f.predicted_lossadjustedload - a.actual_lossadjustedload) as error,
            POWER(f.predicted_lossadjustedload - a.actual_lossadjustedload, 2) as squared_error,
           
            -- MAPE components (for all segments)
            CASE
                WHEN a.actual_lossadjustedload != 0 THEN
                    ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) / ABS(a.actual_lossadjustedload) * 100
                ELSE NULL
            END as mape_component,
           
            -- sMAPE components (calculated for all but used only for solar)
            CASE
                WHEN (ABS(f.predicted_lossadjustedload) + ABS(a.actual_lossadjustedload)) != 0 THEN
                    ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) /
                    ((ABS(f.predicted_lossadjustedload) + ABS(a.actual_lossadjustedload)) / 2) * 100
                ELSE NULL
            END as smape_component,
           
            -- WAPE components (calculated for all but used only for solar)
            ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) as wape_numerator,
            ABS(a.actual_lossadjustedload) as wape_denominator,
           
            1 as record_count
           
        FROM {bi_schema}.vw_dayahead_load_forecasts f
        LEFT JOIN {bi_schema}.vw_final_submission_actuals a ON (
            f.forecast_datetime = a.actual_datetime
            AND f.load_profile = a.load_profile
            AND f.load_segment = a.load_segment
        )
        WHERE f.forecast_datetime <= GETDATE()  -- Only past forecasts
        AND a.actual_lossadjustedload IS NOT NULL  -- Must have actual data
       
        UNION ALL
       
        -- Join predictions with Initial actuals (when Final not available)
        SELECT
            f.forecast_datetime,
            f.load_profile,
            f.load_segment,
            f.predicted_lossadjustedload as forecast_value,
            a.actual_lossadjustedload as actual_value,
            f.run_id,
            f.model_version,
            f.customer_type,
            a.submission_type,
           
            -- Date dimensions
            f.year,
            f.month,
            f.day,
            f.forecast_hour,
            f.day_of_week,
            f.day_type,
            f.time_period,
           
            -- Forecast metadata
            f.forecast_horizon_days,
            f.created_at as forecast_created_at,
            a.latest_created as actual_created_at,
           
            -- Error calculations
            ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) as absolute_error,
            (f.predicted_lossadjustedload - a.actual_lossadjustedload) as error,
            POWER(f.predicted_lossadjustedload - a.actual_lossadjustedload, 2) as squared_error,
           
            -- MAPE components (for all segments)
            CASE
                WHEN a.actual_lossadjustedload != 0 THEN
                    ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) / ABS(a.actual_lossadjustedload) * 100
                ELSE NULL
            END as mape_component,
           
            -- sMAPE components (calculated for all but used only for solar)
            CASE
                WHEN (ABS(f.predicted_lossadjustedload) + ABS(a.actual_lossadjustedload)) != 0 THEN
                    ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) /
                    ((ABS(f.predicted_lossadjustedload) + ABS(a.actual_lossadjustedload)) / 2) * 100
                ELSE NULL
            END as smape_component,
           
            -- WAPE components (calculated for all but used only for solar)
            ABS(f.predicted_lossadjustedload - a.actual_lossadjustedload) as wape_numerator,
            ABS(a.actual_lossadjustedload) as wape_denominator,
           
            1 as record_count
           
        FROM {bi_schema}.vw_dayahead_load_forecasts f
        LEFT JOIN {bi_schema}.vw_initial_submission_actuals a ON (
            f.forecast_datetime = a.actual_datetime
            AND f.load_profile = a.load_profile
            AND f.load_segment = a.load_segment
        )
        WHERE f.forecast_datetime <= GETDATE()  -- Only past forecasts
        AND a.actual_lossadjustedload IS NOT NULL  -- Must have actual data
        -- Only use Initial if Final is not available
        AND NOT EXISTS (
            SELECT 1 FROM {bi_schema}.vw_final_submission_actuals fa
            WHERE fa.actual_datetime = f.forecast_datetime
            AND fa.load_profile = f.load_profile
            AND fa.load_segment = f.load_segment
        )
    )
    SELECT
        forecast_datetime,
        load_profile,
        load_segment,
        customer_type,
        forecast_value,
        actual_value,
        submission_type,
        run_id,
        model_version,
       
        -- Date dimensions
        year,
        month,
        day,
        forecast_hour,
        day_of_week,
        day_type,
        time_period,
       
        -- Forecast metadata
        forecast_horizon_days,
        forecast_created_at,
        actual_created_at,
       
        -- Individual errors (available for all segments)
        error,
        absolute_error,
        squared_error,
        mape_component,
       
        -- Solar-specific components (only meaningful for solar segment)
        CASE
            WHEN load_segment = '{solar_segment}' THEN smape_component
            ELSE NULL
        END as smape_component,
       
        CASE
            WHEN load_segment = '{solar_segment}' THEN wape_numerator
            ELSE NULL
        END as wape_numerator,
       
        CASE
            WHEN load_segment = '{solar_segment}' THEN wape_denominator
            ELSE NULL
        END as wape_denominator,
       
        -- Record metadata
        record_count
       
    FROM forecast_accuracy_base
    """

def get_accuracy_summary_view_sql(bi_schema):
    """Get SQL for accuracy summary dashboard view - CORRECTED"""
    solar_segment = ACCURACY_CONFIG['CUSTOMER_SEGMENTS']['solar']
   
    return f"""
    CREATE OR REPLACE VIEW {bi_schema}.vw_accuracy_summary_dashboard AS
    SELECT
        load_profile,
        load_segment,
        customer_type,
        submission_type,
       
        -- Time groupings
        year,
        month,
        day_type,
        time_period,
       
        -- Aggregate metrics
        COUNT(*) as total_forecasts,
        AVG(forecast_value) as avg_forecast,
        AVG(actual_value) as avg_actual,
       
        -- Standard metrics (available for all segments)
        AVG(absolute_error) as mae,  -- Mean Absolute Error
        SQRT(AVG(squared_error)) as rmse,  -- Root Mean Square Error
        AVG(mape_component) as mape,  -- Mean Absolute Percentage Error
       
        -- Additional metrics for solar only (using proper case comparison)
        CASE
            WHEN load_segment = '{solar_segment}' THEN AVG(smape_component)
            ELSE NULL
        END as smape,  -- Symmetric MAPE (for solar only)
       
        CASE
            WHEN load_segment = '{solar_segment}' THEN
                CASE
                    WHEN SUM(wape_denominator) != 0 THEN
                        SUM(wape_numerator) / SUM(wape_denominator) * 100
                    ELSE NULL
                END
            ELSE NULL
        END as wape,  -- Weighted Absolute Percentage Error (for solar only)
       
        -- R-squared calculation (for all segments)
        CASE
            WHEN VAR_POP(actual_value) != 0 THEN
                1 - (SUM(squared_error) / (COUNT(*) * VAR_POP(actual_value)))
            ELSE NULL
        END as r_squared,
       
        -- Percentile errors (for all segments)
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY absolute_error) as median_absolute_error,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY absolute_error) as p95_absolute_error,
       
        -- Forecast metadata
        MIN(forecast_created_at) as earliest_forecast,
        MAX(forecast_created_at) as latest_forecast,
        COUNT(DISTINCT run_id) as distinct_runs
       
    FROM {bi_schema}.vw_forecast_accuracy_metrics
    GROUP BY
        load_profile,
        load_segment,
        customer_type,
        submission_type,
        year,
        month,
        day_type,
        time_period
    HAVING COUNT(*) >= 10  -- Minimum sample size for reliable metrics
    """

def wait_for_query_completion(redshift_data_client, query_id, operation_name, max_wait_seconds=300):
    """Wait for Redshift Data API query to complete with better error handling"""
    print(f'   Waiting for {operation_name} to complete (Query ID: {query_id})')
   
    waited = 0
    while waited < max_wait_seconds:
        try:
            status_response = redshift_data_client.describe_statement(Id=query_id)
            status = status_response['Status']
           
            if status == 'FINISHED':
                print(f'   {operation_name} completed successfully')
                return
            elif status == 'FAILED':
                error_msg = status_response.get('Error', 'Unknown error')
                print(f'   {operation_name} failed: {error_msg}')
               
                # Try to get more details
                if 'QueryString' in status_response:
                    query_string = status_response['QueryString']
                    print(f'   Failed query: {query_string}...')
               
                raise Exception(f'{operation_name} failed: {error_msg}')
            elif status == 'ABORTED':
                raise Exception(f'{operation_name} was aborted')
           
            # Still running, wait a bit more
            time.sleep(5)
            waited += 5
           
            if waited % 30 == 0:  # Log progress every 30 seconds
                print(f'   Still waiting for {operation_name}... ({waited}s elapsed, status: {status})')
       
        except Exception as e:
            if 'failed:' in str(e) or 'aborted' in str(e):
                raise  # Re-raise operation failures
            else:
                print(f'   Error checking status: {str(e)}')
                time.sleep(5)
                waited += 5
                continue
   
    raise Exception(f'{operation_name} timed out after {max_wait_seconds} seconds')

def verify_views_creation(cluster_identifier, database, db_user, region,
                         bi_schema, main_view_name):
    """Explicitly verify that views were actually created including accuracy views"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
       
        # Check all views in BI schema using information_schema.views
        verify_sql = f"""
        SELECT
            table_schema,
            table_name,
            view_definition
        FROM information_schema.views
        WHERE table_schema = '{bi_schema}'
        ORDER BY table_name
        """
       
        print(f'   Verifying views in {bi_schema} schema...')
       
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=verify_sql
        )
       
        query_id = response['Id']
        wait_for_query_completion(redshift_data_client, query_id, 'views verification')
       
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=query_id)
        records = result_response.get('Records', [])
       
        # Critical views that MUST exist
        critical_views = [
            main_view_name,
            'vw_current_dayahead_forecasts',
            'vw_forecast_summary_by_profile'
        ]
       
        # Optional views (accuracy features)
        optional_views = [
            'vw_initial_submission_actuals',
            'vw_final_submission_actuals',
            'vw_forecast_accuracy_metrics',
            'vw_accuracy_summary_dashboard'
        ]
       
        found_views = []
       
        if records:
            found_views = [record[1]['stringValue'] for record in records]
            print(f' Found {len(found_views)} views in {bi_schema}:')
            for view_name in found_views:
                print(f' - {view_name}')
        else:
            print(f'   No views found in schema {bi_schema}!')
           
            # Debug: Check if schema exists and show all schemas with views
            debug_sql = """
            SELECT DISTINCT table_schema, COUNT(*) as view_count
            FROM information_schema.views
            GROUP BY table_schema
            ORDER BY table_schema
            """

            print(f'DEBUG SQL QUERY: {debug_sql}')
           
            debug_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=debug_sql
            )
           
            debug_query_id = debug_response['Id']
            wait_for_query_completion(redshift_data_client, debug_query_id, 'debug schemas with views')
           
            debug_result = redshift_data_client.get_statement_result(Id=debug_query_id)
           
            print(f'   Schemas that DO have views:')
            for row in debug_result.get('Records', []):
                schema_name = row[0]['stringValue']
                view_count = int(row[1]['longValue'])
                print(f'      - {schema_name}: {view_count} views')
       
        # Check critical views
        missing_critical = [v for v in critical_views if v not in found_views]
        missing_optional = [v for v in optional_views if v not in found_views]

        if missing_critical:
            print(f' ❌ CRITICAL VIEWS MISSING: {missing_critical}')
            return missing_critical
        else:
            print(f' ✅ All critical views found')
       
        if missing_optional:
            print(f' ⚠️ Optional views missing: {missing_optional}')
            print(f' ℹ️ Accuracy features will not be available')
        else:
            print(f' ✅ All optional views found')
       
        return [] # No critical views missing
       
    except Exception as e:
        print(f'   Views verification failed: {str(e)}')

def create_s3_staging_directories(s3_staging_bucket, s3_staging_prefix):
    """Create S3 staging directories for Redshift COPY operations"""
    try:
        s3_client = boto3.client('s3')
       
        # Create staging directories
        directories = [
            f"{s3_staging_prefix}/forecasts/",
            f"{s3_staging_prefix}/forecasts/load_profile=RES/",
            f"{s3_staging_prefix}/forecasts/load_profile=MEDCI/",
            f"{s3_staging_prefix}/forecasts/load_profile=SMLCOM/",
            f"{s3_staging_prefix}/archive/"
        ]
       
        for directory in directories:
            try:
                s3_client.put_object(Bucket=s3_staging_bucket, Key=directory)
                print(f'   Created S3 directory: s3://{s3_staging_bucket}/{directory}')
            except Exception as e:
                print(f'   Directory may already exist: s3://{s3_staging_bucket}/{directory}')
       
    except Exception as e:
        print(f'   S3 staging setup failed: {str(e)}')
        raise
def save_redshift_config(s3_staging_bucket, cluster_identifier, database,
                        operational_schema, operational_table, bi_schema, bi_view,
                        redshift_iam_role, env_name):
    """Save Redshift configuration including accuracy views"""
    try:
        config = {
            'infrastructure_type': 'redshift',
            'cluster_identifier': cluster_identifier,
            'database': database,
            'environment': env_name,
            'operational': {
                'schema': operational_schema,
                'table': operational_table,
                'full_table_name': f'{operational_schema}.{operational_table}'
            },
            'bi': {
                'schema': bi_schema,
                'main_view': bi_view,
                'current_forecasts_view': 'vw_current_dayahead_forecasts',
                'summary_view': 'vw_forecast_summary_by_profile',
                'initial_actuals_view': 'vw_initial_submission_actuals',
                'final_actuals_view': 'vw_final_submission_actuals',
                'accuracy_metrics_view': 'vw_forecast_accuracy_metrics',
                'accuracy_summary_view': 'vw_accuracy_summary_dashboard'
            },
            's3_staging': {
                'bucket': s3_staging_bucket,
                'prefix': f'redshift-staging/{env_name}',
                'copy_iam_role': redshift_iam_role
            },
            'accuracy_config': {
                'solar_rate_group_prefixes': ACCURACY_CONFIG['SOLAR_RATE_GROUP_PREFIXES'],
                'retention_months': ACCURACY_CONFIG['ACCURACY_RETENTION_MONTHS'],
                'metrics': ACCURACY_CONFIG['ACCURACY_METRICS'],
                'initial_lag_days': ACCURACY_CONFIG['INITIAL_SUBMISSION_LAG_DAYS'],
                'final_lag_days': ACCURACY_CONFIG['FINAL_SUBMISSION_LAG_DAYS']
            },
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'features': {
                'uses_data_api': True,
                'schemas_pre_exist': True,
                'supports_partitioned_data': True,
                'accuracy_views_enabled': True,
                'latest_record_logic': True,
                'solar_metrics_enabled': True
            }
        }
       
        config_key = f'redshift-config/{env_name}/config.json'
       
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=s3_staging_bucket,
            Key=config_key,
            Body=json.dumps(config, indent=2),
            ContentType='application/json'
        )
       
        print(f'   Configuration saved to s3://{s3_staging_bucket}/{config_key}')
        print(f'   Setup summary:')
        print(f'      Database: {database}')
        print(f'      Operational table: {operational_schema}.{operational_table}')
        print(f'      Main BI view: {bi_schema}.{bi_view}')
        print(f'      Accuracy views: 4 additional views created')
        print(f'      Solar filter: {get_solar_filter_clause()}')
        print(f'      S3 staging: s3://{s3_staging_bucket}/redshift-staging/{env_name}/')
       
    except Exception as e:
        print(f'   Configuration save failed: {str(e)}')
        raise

def get_required_environment_variables():
    """List of all environment variables needed for enhanced setup"""
    return [
        # Existing variables
        'REDSHIFT_CLUSTER_IDENTIFIER',
        'REDSHIFT_DATABASE',
        'REDSHIFT_DB_USER',
        'REDSHIFT_REGION',
        'REDSHIFT_OPERATIONAL_SCHEMA',  # e.g., 'edp_forecasting_dev'
        'REDSHIFT_OPERATIONAL_TABLE',   # e.g., 'dayahead_load_forecasts'
        'REDSHIFT_BI_SCHEMA',          # e.g., 'edp_bi_dev'
        'REDSHIFT_BI_VIEW',            # e.g., 'vw_dayahead_load_forecasts'
        'S3_STAGING_BUCKET',
        'S3_STAGING_PREFIX',
        'REDSHIFT_IAM_ROLE',
        'ENV_NAME',
       
        # New variables for accuracy views
        'REDSHIFT_INPUT_SCHEMA',       # e.g., 'edp_cust_dev' (where caiso_sqmd is)
        'REDSHIFT_INPUT_TABLE'         # e.g., 'caiso_sqmd'
    ]

if __name__ == '__main__':
    print("=== Enhanced Redshift Infrastructure Setup ===")
    print(f"Solar Rate Group Prefixes: {ACCURACY_CONFIG['SOLAR_RATE_GROUP_PREFIXES']}")
    print(f"Accuracy Retention: {ACCURACY_CONFIG['ACCURACY_RETENTION_MONTHS']} months")
    print("Required Environment Variables:")
    for var in get_required_environment_variables():
        print(f"  - {var}")
   
    # Run the setup
    try:
        result = setup_redshift_infrastructure()
        print(f'Setup result: {result}')
       
        if result == 'success':
            print('✅ SUCCESS: All critical infrastructure components created')
            exit(0)
        else:
            print('❌ FAILURE: Critical infrastructure setup failed')
            exit(1)
       
    except Exception as e:
        print(f'❌ FATAL ERROR: {str(e)}')
        traceback.print_exc()
        exit(1)
