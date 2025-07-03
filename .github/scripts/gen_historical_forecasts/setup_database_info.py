#!/usr/bin/env python3
"""
Database Infrastructure Information Setup Script

This script configures database connection information for either Athena or Redshift
based on the environment and database type selected.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_athena_info(environment, s3_bucket):
    """Setup Athena database information"""
    logger.info("Configuring Athena infrastructure information...")
   
    athena_database = f"sdcp_edp_{environment}"
    athena_table = "dayahead_load_forecasts"
    athena_results_location = f"s3://{s3_bucket}/athena-results/{environment}/"
    athena_data_location = f"s3://{s3_bucket}/dayahead_load_forecasts/"
   
    # Set GitHub outputs
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"athena_database={athena_database}\n")
        f.write(f"athena_table={athena_table}\n")
        f.write(f"athena_results_location={athena_results_location}\n")
        f.write(f"athena_data_location={athena_data_location}\n")
       
        # Set empty Redshift outputs
        f.write(f"redshift_cluster=\n")
        f.write(f"redshift_database=\n")
        f.write(f"redshift_operational_schema=\n")
        f.write(f"redshift_operational_table=\n")
   
    logger.info("✅ Athena configuration set:")
    logger.info(f"  Database: {athena_database}")
    logger.info(f"  Table: {athena_table}")
    logger.info(f"  Results location: {athena_results_location}")
    logger.info(f"  Data location: {athena_data_location}")
   
    return {
        'database': athena_database,
        'table': athena_table,
        'results_location': athena_results_location,
        'data_location': athena_data_location
    }


def setup_redshift_info(environment, redshift_cluster_prefix, redshift_database,
                       redshift_operational_schema_prefix, redshift_operational_table):
    """Setup Redshift database information"""
    logger.info("Configuring Redshift infrastructure information...")
   
    redshift_cluster_identifier = f"{redshift_cluster_prefix}-{environment}"
    redshift_operational_schema = f"{redshift_operational_schema_prefix}_{environment}"
   
    # Set GitHub outputs
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"redshift_cluster={redshift_cluster_identifier}\n")
        f.write(f"redshift_database={redshift_database}\n")
        f.write(f"redshift_operational_schema={redshift_operational_schema}\n")
        f.write(f"redshift_operational_table={redshift_operational_table}\n")
       
        # Set empty Athena outputs
        f.write(f"athena_database=\n")
        f.write(f"athena_table=\n")
        f.write(f"athena_results_location=\n")
        f.write(f"athena_data_location=\n")
   
    logger.info("✅ Redshift configuration set:")
    logger.info(f"  Cluster: {redshift_cluster_identifier}")
    logger.info(f"  Database: {redshift_database}")
    logger.info(f"  Schema: {redshift_operational_schema}")
    logger.info(f"  Table: {redshift_operational_table}")
   
    return {
        'cluster': redshift_cluster_identifier,
        'database': redshift_database,
        'schema': redshift_operational_schema,
        'table': redshift_operational_table
    }


def main():
    """Main database setup function"""
    try:
        logger.info("Starting database infrastructure information setup...")
       
        # Get inputs from environment variables
        database_type = os.environ.get('DATABASE_TYPE')
        environment = os.environ.get('ENVIRONMENT')
        s3_bucket = os.environ.get('S3_BUCKET')
       
        logger.info(f"Database type: {database_type}")
        logger.info(f"Environment: {environment}")
        logger.info(f"S3 bucket: {s3_bucket}")
       
        # Validate required inputs
        if not database_type or not environment:
            raise ValueError("DATABASE_TYPE and ENVIRONMENT are required")
       
        # Set database type in outputs
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"database_type={database_type}\n")
       
        if database_type == "athena":
            if not s3_bucket:
                raise ValueError("S3_BUCKET is required for Athena configuration")
           
            athena_info = setup_athena_info(environment, s3_bucket)
            logger.info("Athena infrastructure information configured successfully")
           
        elif database_type == "redshift":
            # Get Redshift-specific environment variables
            redshift_cluster_prefix = os.environ.get('REDSHIFT_CLUSTER_PREFIX')
            redshift_database = os.environ.get('REDSHIFT_DATABASE')
            redshift_operational_schema_prefix = os.environ.get('REDSHIFT_OPERATIONAL_SCHEMA_PREFIX')
            redshift_operational_table = os.environ.get('REDSHIFT_OPERATIONAL_TABLE')
           
            # Validate Redshift inputs
            required_redshift_vars = [
                ('REDSHIFT_CLUSTER_PREFIX', redshift_cluster_prefix),
                ('REDSHIFT_DATABASE', redshift_database),
                ('REDSHIFT_OPERATIONAL_SCHEMA_PREFIX', redshift_operational_schema_prefix),
                ('REDSHIFT_OPERATIONAL_TABLE', redshift_operational_table)
            ]
           
            for var_name, var_value in required_redshift_vars:
                if not var_value:
                    raise ValueError(f"{var_name} is required for Redshift configuration")
           
            redshift_info = setup_redshift_info(
                environment,
                redshift_cluster_prefix,
                redshift_database,
                redshift_operational_schema_prefix,
                redshift_operational_table
            )
            logger.info("Redshift infrastructure information configured successfully")
           
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
       
        logger.info("Database infrastructure setup completed successfully")
       
    except Exception as e:
        logger.error(f"❌ Database setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
