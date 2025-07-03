#!/usr/bin/env python3
"""
Final Summary Generation Script with Better Debugging
"""

import json
import os
import sys
import glob
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_file_system():
    """Debug function to understand what files are available"""
    logger.info("=== FILE SYSTEM DEBUGGING ===")
    logger.info(f"Current working directory: {os.getcwd()}")
   
    # List all files in current directory
    try:
        all_files = os.listdir('.')
        logger.info(f"All files in current directory ({len(all_files)} total):")
        for file in sorted(all_files):
            file_size = os.path.getsize(file) if os.path.isfile(file) else 0
            logger.info(f"  {'[FILE]' if os.path.isfile(file) else '[DIR] '} {file} ({file_size} bytes)")
    except Exception as e:
        logger.error(f"Error listing directory: {str(e)}")
   
    # Look for JSON files specifically
    json_files = glob.glob("*.json")
    logger.info(f"JSON files found ({len(json_files)} total): {json_files}")
   
    # Look for prediction summary files specifically
    prediction_files = glob.glob("prediction_summary_*.json")
    logger.info(f"Prediction summary files found ({len(prediction_files)} total): {prediction_files}")
   
    # Look for cleanup files specifically
    cleanup_files = glob.glob("cleanup_log_*.json")
    logger.info(f"Cleanup log files found ({len(cleanup_files)} total): {cleanup_files}")
   
    # Look for any files with prediction in the name
    any_prediction_files = glob.glob("*prediction*")
    logger.info(f"Any files with 'prediction' in name ({len(any_prediction_files)} total): {any_prediction_files}")
   
    # Look for any files with summary in the name
    any_summary_files = glob.glob("*summary*")
    logger.info(f"Any files with 'summary' in name ({len(any_summary_files)} total): {any_summary_files}")
   
    logger.info("=== END FILE SYSTEM DEBUGGING ===")


def collect_combination_results():
    """Collect results from all combination summary files with debugging"""
    logger.info("=== COLLECTING COMBINATION RESULTS ===")
    results = []
   
    # Try multiple possible file patterns
    patterns = [
        "prediction_summary_*.json",
        "*prediction_summary*.json",
        "*prediction*.json",
        "*/prediction_summary_*.json",  # In case files are in subdirectories
        "**/prediction_summary_*.json"  # Recursive search
    ]
   
    all_found_files = []
    for pattern in patterns:
        found_files = glob.glob(pattern, recursive=True)
        if found_files:
            logger.info(f"Pattern '{pattern}' found {len(found_files)} files: {found_files}")
            all_found_files.extend(found_files)
        else:
            logger.info(f"Pattern '{pattern}' found no files")
   
    # Remove duplicates
    unique_files = list(set(all_found_files))
    logger.info(f"Total unique prediction files found: {len(unique_files)}")
   
    # Try to load each file
    for summary_file in unique_files:
        try:
            logger.info(f"Attempting to load: {summary_file}")
            with open(summary_file, 'r') as f:
                summary = json.load(f)
           
            # Validate it's a prediction summary file
            if 'combination' in summary or 'total_dates' in summary:
                results.append(summary)
                combination = summary.get('combination', summary.get('load_profile', 'unknown'))
                logger.info(f"✅ Successfully loaded results for {combination}")
            else:
                logger.warning(f"⚠️ File {summary_file} doesn't look like a prediction summary")
               
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in {summary_file}: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error loading {summary_file}: {str(e)}")
   
    logger.info(f"=== COLLECTION COMPLETE: {len(results)} combination results loaded ===")
    return results


def collect_cleanup_results():
    """Collect cleanup results from all combinations with debugging"""
    logger.info("=== COLLECTING CLEANUP RESULTS ===")
    cleanup_results = []
   
    # Try multiple possible patterns
    patterns = [
        "cleanup_log_*.json",
        "*cleanup*.json",
        "*/cleanup_log_*.json",
        "**/cleanup_log_*.json"
    ]
   
    all_found_files = []
    for pattern in patterns:
        found_files = glob.glob(pattern, recursive=True)
        if found_files:
            logger.info(f"Cleanup pattern '{pattern}' found {len(found_files)} files: {found_files}")
            all_found_files.extend(found_files)
        else:
            logger.info(f"Cleanup pattern '{pattern}' found no files")
   
    # Remove duplicates
    unique_files = list(set(all_found_files))
    logger.info(f"Total unique cleanup files found: {len(unique_files)}")
   
    # Try to load each file
    for cleanup_file in unique_files:
        try:
            logger.info(f"Attempting to load cleanup file: {cleanup_file}")
            with open(cleanup_file, 'r') as f:
                cleanup = json.load(f)
           
            # Validate it's a cleanup file
            if 'cleanup_status' in cleanup or 'endpoint_name' in cleanup:
                cleanup_results.append(cleanup)
                combination = cleanup.get('combination', 'unknown')
                logger.info(f"✅ Successfully loaded cleanup results for {combination}")
            else:
                logger.warning(f"⚠️ File {cleanup_file} doesn't look like a cleanup log")
               
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in {cleanup_file}: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error loading {cleanup_file}: {str(e)}")
   
    logger.info(f"=== CLEANUP COLLECTION COMPLETE: {len(cleanup_results)} cleanup results loaded ===")
    return cleanup_results


def calculate_overall_statistics(results):
    """Calculate overall statistics across all combinations"""
    if not results:
        logger.warning("No results provided to calculate_overall_statistics")
        return {
            'total_combinations': 0,
            'total_dates': 0,
            'total_predictions': 0,
            'total_successful_dates': 0,
            'total_failed_dates': 0,
            'total_records_inserted': 0,
            'total_duration': 0,
            'overall_success_rate': 0,
            'combinations_with_failures': 0,
            'used_existing_endpoints': 0,
            'total_time_saved': 0
        }
   
    logger.info(f"Calculating statistics from {len(results)} combinations")
   
    # Debug: Show what keys are available in results
    if results:
        sample_result = results[0]
        logger.info(f"Sample result keys: {list(sample_result.keys())}")
   
    total_dates = sum(r.get('total_dates', 0) for r in results)
    total_predictions = sum(r.get('total_predictions', 0) for r in results)
    total_successful_dates = sum(r.get('successful_dates', 0) for r in results)
    total_failed_dates = sum(r.get('failed_dates', 0) for r in results)
    total_records_inserted = sum(r.get('total_records_inserted', 0) for r in results)
    total_duration = sum(r.get('total_duration', 0) for r in results)
    overall_success_rate = (total_successful_dates / total_dates * 100) if total_dates > 0 else 0
    combinations_with_failures = sum(1 for r in results if r.get('failed_dates', 0) > 0)
    used_existing_endpoints = sum(1 for r in results if r.get('used_existing_endpoint', False))
   
    # Calculate time savings
    total_time_saved = 0
    for r in results:
        if r.get('used_existing_endpoint') and r.get('total_dates', 0) > 0:
            dates_count = r.get('total_dates', 0)
            actual_duration = r.get('total_duration', 0)
            estimated_without_optimization = dates_count * 360  # 6 minutes per date
            time_saved = max(0, estimated_without_optimization - actual_duration)
            total_time_saved += time_saved
   
    stats = {
        'total_combinations': len(results),
        'total_dates': total_dates,
        'total_predictions': total_predictions,
        'total_successful_dates': total_successful_dates,
        'total_failed_dates': total_failed_dates,
        'total_records_inserted': total_records_inserted,
        'total_duration': total_duration,
        'overall_success_rate': overall_success_rate,
        'combinations_with_failures': combinations_with_failures,
        'used_existing_endpoints': used_existing_endpoints,
        'total_time_saved': total_time_saved
    }
   
    logger.info(f"Calculated statistics: {stats}")
    return stats


def calculate_cleanup_statistics(cleanup_results):
    """Calculate cleanup statistics for cost optimization tracking"""
    if not cleanup_results:
        logger.warning("No cleanup results provided to calculate_cleanup_statistics")
        return {
            'total_endpoints_processed': 0,
            'successfully_cleaned': 0,
            'cleanup_failures': 0,
            'cost_optimized_combinations': 0
        }
   
    logger.info(f"Calculating cleanup statistics from {len(cleanup_results)} cleanup operations")
   
    total_endpoints = len(cleanup_results)
    successfully_cleaned = sum(1 for c in cleanup_results if c.get('cleanup_status') == 'success')
    cleanup_failures = sum(1 for c in cleanup_results if c.get('cleanup_status') != 'success')
    cost_optimized = sum(1 for c in cleanup_results if c.get('cost_optimized', False))
   
    cleanup_stats = {
        'total_endpoints_processed': total_endpoints,
        'successfully_cleaned': successfully_cleaned,
        'cleanup_failures': cleanup_failures,
        'cost_optimized_combinations': cost_optimized
    }
   
    logger.info(f"Calculated cleanup statistics: {cleanup_stats}")
    return cleanup_stats


def determine_overall_status(stats):
    """Determine overall execution status"""
    if stats['total_failed_dates'] == 0 and stats['total_successful_dates'] > 0:
        return "🎉 COMPLETE SUCCESS"
    elif stats['overall_success_rate'] >= 90:
        return "✅ EXCELLENT SUCCESS"
    elif stats['overall_success_rate'] >= 70:
        return "🟢 GOOD SUCCESS"
    elif stats['overall_success_rate'] >= 50:
        return "🟡 PARTIAL SUCCESS"
    else:
        return "❌ FAILED"


def generate_combinations_table(results):
    """Generate markdown table of combination results"""
    if not results:
        return "No combination results available.\n"
   
    table = "\n| Combination | Total Dates | Successful | Failed | Success Rate | Duration | Endpoint Optimization |\n"
    table += "|-------------|-------------|------------|--------|--------------|----------|----------------------|\n"
   
    for result in sorted(results, key=lambda x: x.get('combination', '')):
        combination = result.get('combination', 'Unknown')
        total_dates = result.get('total_dates', 0)
        successful = result.get('successful_dates', 0)
        failed = result.get('failed_dates', 0)
        success_rate = result.get('success_rate', 0)
        duration = result.get('total_duration', 0)
        used_existing = result.get('used_existing_endpoint', False)
       
        # Format duration
        duration_str = f"{duration:.0f}s" if duration > 0 else "N/A"
       
        # Format optimization status
        optimization_status = "✅ Optimized" if used_existing else "❌ Not Optimized"
       
        table += f"| {combination} | {total_dates} | {successful} | {failed} | {success_rate:.1f}% | {duration_str} | {optimization_status} |\n"
   
    return table


def generate_cleanup_table(cleanup_results):
    """Generate markdown table of cleanup results"""
    if not cleanup_results:
        return "No cleanup results available.\n"
   
    table = "\n| Combination | Endpoint Name | Cleanup Status | Cost Optimized | Timestamp |\n"
    table += "|-------------|---------------|----------------|----------------|-----------|\n"
   
    for cleanup in sorted(cleanup_results, key=lambda x: x.get('combination', '')):
        combination = cleanup.get('combination', 'Unknown')
        endpoint_name = cleanup.get('endpoint_name', 'Unknown')
        cleanup_status = cleanup.get('cleanup_status', 'unknown')
        cost_optimized = cleanup.get('cost_optimized', False)
        timestamp = cleanup.get('cleanup_timestamp', 'Unknown')
       
        # Format status
        status_icon = "✅" if cleanup_status == 'success' else "❌"
        cost_icon = "💰" if cost_optimized else "⚠️"
       
        # Truncate endpoint name for better table formatting
        short_endpoint = endpoint_name.replace('-energy-ml-endpoint-', '-...-') if len(endpoint_name) > 30 else endpoint_name
       
        table += f"| {combination} | {short_endpoint} | {status_icon} {cleanup_status} | {cost_icon} {cost_optimized} | {timestamp[:16]} |\n"
   
    return table


def generate_database_query_section():
    """Generate database query section for verification"""
    environment = os.environ.get('ENVIRONMENT', 'unknown')
    redshift_cluster = os.environ.get('REDSHIFT_CLUSTER', 'unknown')
    redshift_database = os.environ.get('REDSHIFT_DATABASE', 'unknown')
    redshift_schema = os.environ.get('REDSHIFT_OPERATIONAL_SCHEMA', 'edp_forecasting_dev')
    redshift_table = os.environ.get('REDSHIFT_OPERATIONAL_TABLE', 'dayahead_load_forecasts')
   
    query_section = f"""
## Database Verification

### Redshift Query to Verify Results
```sql
-- Count predictions by combination and date
SELECT
    load_profile,
    load_segment,
    DATE(forecast_datetime) as forecast_date,
    COUNT(*) as prediction_count,
    MIN(forecast_datetime) as first_prediction,
    MAX(forecast_datetime) as last_prediction,
    run_user
FROM {redshift_schema}.{redshift_table}
WHERE DATE(forecast_datetime) >= DATE('{os.environ.get("PREDICTION_DATES", "[]")[1:11] if os.environ.get("PREDICTION_DATES", "[]").startswith('[') else "2025-05-01"}')
  AND run_user LIKE '%historical%'
GROUP BY load_profile, load_segment, DATE(forecast_datetime), run_user
ORDER BY load_profile, load_segment, forecast_date;

-- Summary by combination
SELECT
    load_profile,
    load_segment,
    COUNT(DISTINCT DATE(forecast_datetime)) as unique_dates,
    COUNT(*) as total_predictions,
    MIN(DATE(forecast_datetime)) as first_date,
    MAX(DATE(forecast_datetime)) as last_date
FROM {redshift_schema}.{redshift_table}
WHERE run_user LIKE '%historical%'
GROUP BY load_profile, load_segment
ORDER BY load_profile, load_segment;
```

### Redshift Infrastructure
- **Cluster:** {redshift_cluster}
- **Database:** {redshift_database}
- **Schema:** {redshift_schema}
- **Table:** {redshift_table}
"""
   
    return query_section


def generate_troubleshooting_section(results, stats, cleanup_stats):
    """Generate troubleshooting section based on results"""
    troubleshooting = """
## Troubleshooting Guide

### Delete/Recreate Approach Status
"""
   
    # Endpoint optimization analysis
    if stats['used_existing_endpoints'] == stats['total_combinations']:
        troubleshooting += "✅ **All combinations used endpoint optimization successfully**\n\n"
    elif stats['used_existing_endpoints'] > 0:
        troubleshooting += f"⚠️ **Partial optimization**: {stats['used_existing_endpoints']}/{stats['total_combinations']} combinations used existing endpoints\n\n"
    else:
        troubleshooting += "❌ **No endpoint optimization detected** - check setup_historical_endpoints job\n\n"
   
    # Cleanup analysis
    if cleanup_stats['cost_optimized_combinations'] == cleanup_stats['total_endpoints_processed']:
        troubleshooting += "✅ **All endpoints cleaned up successfully** - zero ongoing costs\n\n"
    elif cleanup_stats['cleanup_failures'] > 0:
        troubleshooting += f"⚠️ **Cleanup issues**: {cleanup_stats['cleanup_failures']} endpoints may still be running\n"
        troubleshooting += "💰 **Action required**: Manual cleanup to avoid ongoing costs\n\n"
   
    # Time savings analysis
    if stats['total_time_saved'] > 0:
        hours_saved = stats['total_time_saved'] / 3600
        troubleshooting += f"🚀 **Time savings achieved**: {stats['total_time_saved']:.0f} seconds ({hours_saved:.1f} hours)\n"
        troubleshooting += f"📊 **Optimization efficiency**: {stats['used_existing_endpoints']}/{stats['total_combinations']} combinations optimized\n\n"
   
    if stats['total_failed_dates'] > 0:
        troubleshooting += """
### Failed Predictions Analysis
1. **Endpoint Issues**: Verify endpoints were created in setup_historical_endpoints
2. **Configuration Issues**: Check S3 endpoint configurations are accessible
3. **Lambda Timeouts**: Increase Lambda timeout if predictions are timing out
4. **Rate Limiting**: Adjust `max_parallel_requests` and `request_delay_seconds`

"""
   
    troubleshooting += """
### Manual Cleanup Commands (if needed)
```bash
# Check for any remaining endpoints
aws sagemaker list-endpoints --status-equals InService --query 'Endpoints[?contains(EndpointName, `energy-ml-endpoint`)].EndpointName'

# Delete specific endpoint if still running
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>

# Verify deletion
aws sagemaker describe-endpoint --endpoint-name <endpoint-name>
```

### Performance Analysis
- **Endpoint Optimization Rate**: {:.1f}% ({}/{} combinations)
- **Average Time per Date**: {:.1f} seconds
- **Cost Optimization Success**: {:.1f}% ({}/{} endpoints)
""".format(
        (stats['used_existing_endpoints'] / stats['total_combinations'] * 100) if stats['total_combinations'] > 0 else 0,
        stats['used_existing_endpoints'],
        stats['total_combinations'],
        (stats['total_duration'] / stats['total_dates']) if stats['total_dates'] > 0 else 0,
        (cleanup_stats['cost_optimized_combinations'] / cleanup_stats['total_endpoints_processed'] * 100) if cleanup_stats['total_endpoints_processed'] > 0 else 0,
        cleanup_stats['cost_optimized_combinations'],
        cleanup_stats['total_endpoints_processed']
    )
   
    return troubleshooting

def main():
    """Main summary generation function with debugging"""
    try:
        logger.info("=== STARTING FINAL SUMMARY GENERATION ===")
       
        # Debug file system first
        debug_file_system()
       
        # Get configuration from environment variables
        environment = os.environ.get('ENVIRONMENT', 'unknown')
        database_type = os.environ.get('DATABASE_TYPE', 'redshift')
        combinations_matrix = json.loads(os.environ.get('COMBINATIONS_MATRIX', '[]'))
        prediction_dates = json.loads(os.environ.get('PREDICTION_DATES', '[]'))
        total_predictions = os.environ.get('TOTAL_PREDICTIONS', '0')
        max_parallel = os.environ.get('MAX_PARALLEL_REQUESTS', '3')
        request_delay = os.environ.get('REQUEST_DELAY_SECONDS', '2')
        github_run_id = os.environ.get('GITHUB_RUN_ID', 'unknown')
        github_actor = os.environ.get('GITHUB_ACTOR', 'unknown')
        github_repository = os.environ.get('GITHUB_REPOSITORY', 'unknown')
        github_ref_name = os.environ.get('GITHUB_REF_NAME', 'unknown')
       
        logger.info(f"Environment: {environment}")
        logger.info(f"Expected combinations: {len(combinations_matrix)}")
        logger.info(f"Expected dates: {len(prediction_dates)}")
       
        # Collect results from all combinations
        results = collect_combination_results()
        cleanup_results = collect_cleanup_results()
       
        if not results:
            logger.error("❌ No prediction results found!")
            logger.error("This indicates that prediction generation failed for all combinations.")
            logger.error("Please check:")
            logger.error("  1. Artifact download patterns in YAML")
            logger.error("  2. Artifact upload in generate_historical_predictions job")
            logger.error("  3. File naming consistency")
            logger.error("  4. Job dependencies and execution order")
           
            # Create a minimal error summary
            error_summary_file = f"error_summary_{environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(error_summary_file, 'w') as f:
                f.write("# Historical Forecasting Error Summary\n\n")
                f.write("## ❌ No Prediction Results Found\n\n")
                f.write(f"- **Environment:** {environment}\n")
                f.write(f"- **Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
                f.write(f"- **GitHub Workflow:** {github_run_id}\n\n")
                f.write("## Issue\n")
                f.write("No prediction summary files were found. This indicates:\n")
                f.write("1. Artifact download failed\n")
                f.write("2. Prediction generation failed for all combinations\n")
                f.write("3. File naming or path issues\n\n")
                f.write("## Next Steps\n")
                f.write("1. Check individual job logs\n")
                f.write("2. Verify artifact upload/download patterns\n")
                f.write("3. Check file system permissions\n")
           
            logger.info(f"Created error summary: {error_summary_file}")
            return False
       
        logger.info(f"Successfully collected results from {len(results)} combinations and {len(cleanup_results)} cleanup operations")
       
        # Calculate overall statistics
        stats = calculate_overall_statistics(results)
        cleanup_stats = calculate_cleanup_statistics(cleanup_results)
        overall_status = determine_overall_status(stats)
       
        # Generate summary filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"historical_forecasting_summary_{environment}_{timestamp}.md"
       
        logger.info(f"Generating summary file: {summary_file}")
       
        # Create comprehensive summary (simplified version for debugging)
        with open(summary_file, 'w') as f:
            f.write("# Historical Energy Load Forecasting Summary\n")
            f.write("## Delete/Recreate Endpoint Management Approach\n\n")
           
            # Execution Overview
            f.write("## Execution Overview\n")
            f.write(f"- **Environment:** {environment}\n")
            f.write(f"- **Database Type:** {database_type}\n")
            f.write(f"- **Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"- **GitHub Workflow:** {github_run_id}\n")
            f.write(f"- **Triggered By:** {github_actor}\n")
            f.write(f"- **Repository:** {github_repository}\n")
            f.write(f"- **Branch:** {github_ref_name}\n")
            f.write(f"- **Overall Status:** {overall_status}\n")
            f.write(f"- **Cost Optimization Approach:** Delete/Recreate Endpoints\n\n")
           
            f.write("## Configuration\n")
            f.write(f"- **Total Combinations:** {len(combinations_matrix)}\n")
            f.write(f"- **Total Dates:** {len(prediction_dates)}\n")
            f.write(f"- **Expected Predictions:** {total_predictions}\n")
            f.write(f"- **Max Parallel Requests:** {max_parallel}\n")
            f.write(f"- **Request Delay:** {request_delay} seconds\n")
            f.write(f"- **Endpoint Management:** Delete/Recreate for Cost Optimization\n\n")
           
            # Date Range
            if prediction_dates:
                f.write("## Date Range Processed\n")
                f.write(f"- **Start Date:** {min(prediction_dates)}\n")
                f.write(f"- **End Date:** {max(prediction_dates)}\n")
                f.write(f"- **Total Days:** {len(prediction_dates)}\n\n")

            # Results Summary
            f.write("## Results Summary\n")
            f.write(f"- **Total Combinations Processed:** {stats['total_combinations']}\n")
            f.write(f"- **Total Dates Processed:** {stats['total_dates']}\n")
            f.write(f"- **Total Predictions Generated:** {stats['total_predictions']}\n")
            f.write(f"- **Successful Date Predictions:** {stats['total_successful_dates']}\n")
            f.write(f"- **Failed Date Predictions:** {stats['total_failed_dates']}\n")
            f.write(f"- **Overall Success Rate:** {stats['overall_success_rate']:.1f}%\n\n")
           
            # Cost Optimization Summary
            f.write("## Cost Optimization Summary\n")
            f.write(f"- **Endpoint Optimization Success:** {stats['used_existing_endpoints']}/{stats['total_combinations']} combinations\n")
            f.write(f"- **Time Saved via Optimization:** {stats['total_time_saved']:.0f} seconds\n")
            f.write(f"- **Endpoints Successfully Cleaned:** {cleanup_stats['successfully_cleaned']}/{cleanup_stats['total_endpoints_processed']}\n")
            f.write(f"- **Cost Optimized Combinations:** {cleanup_stats['cost_optimized_combinations']}/{cleanup_stats['total_endpoints_processed']}\n\n")
           
            # Combination Results Table
            f.write("## Combination Results\n")
            f.write(generate_combinations_table(results))
            f.write("\n")
           
            # Cleanup Results Table
            f.write("## Endpoint Cleanup Results\n")
            f.write(generate_cleanup_table(cleanup_results))
            f.write("\n")
           
            # Database Query Section
            f.write(generate_database_query_section())
            f.write("\n")
           
            # Troubleshooting
            f.write(generate_troubleshooting_section(results, stats, cleanup_stats))
           
            # Footer
            f.write(f"\n---\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} by Historical Forecasting Pipeline*\n")
            f.write(f"*Cost Optimization: Delete/Recreate Endpoint Management*\n")
       
        logger.info(f"Summary generated successfully: {summary_file}")
       
        # Log key metrics to console
        print(f"\n🎯 HISTORICAL FORECASTING SUMMARY")
        print(f"📊 Overall Status: {overall_status}")
        print(f"📈 Success Rate: {stats['overall_success_rate']:.1f}%")
        print(f"🚀 Time Saved: {stats['total_time_saved']:.0f}s")
        print(f"💰 Cost Optimization: {cleanup_stats['cost_optimized_combinations']}/{cleanup_stats['total_endpoints_processed']} endpoints")
        print(f"📄 Detailed Report: {summary_file}")
       
        return True
       
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
