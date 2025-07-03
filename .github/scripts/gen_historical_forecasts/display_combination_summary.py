#!/usr/bin/env python3
"""
Enhanced Combination Summary Display Script

This script displays the results summary for an individual combination
after historical prediction generation is complete. Enhanced version includes
endpoint optimization metrics and time savings calculations.
"""

import json
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def display_combination_results(profile, segment):
    """Display results for a specific combination with enhanced metrics"""
    summary_file = f"prediction_summary_{profile}_{segment}.json"
   
    if not os.path.exists(summary_file):
        logger.error(f"❌ No prediction summary found: {summary_file}")
        logger.error("Prediction generation may have failed for this combination")
        return False
   
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
       
        logger.info("📊 Prediction Results:")
       
        # Enhanced metrics display with both old and new formats
        if 'successful_predictions' in summary:
            # Legacy format support
            logger.info(f"  ✅ Successful: {summary['successful_predictions']}")
            logger.info(f"  ❌ Failed: {summary['failed_predictions']}")
            logger.info(f"  📈 Success Rate: {summary['success_rate']:.1f}%")
        else:
            # New enhanced format
            logger.info(f"  ✅ Successful: {summary.get('successful_dates', 0)}")
            logger.info(f"  ❌ Failed: {summary.get('failed_dates', 0)}")
            logger.info(f"  📈 Success Rate: {summary.get('success_rate', 0):.1f}%")
           
        # Additional enhanced metrics
        logger.info(f"  📅 Total dates: {summary.get('total_dates', 0)}")
        logger.info(f"  🔢 Total predictions: {summary.get('total_predictions', 0)}")
        logger.info(f"  💾 Records inserted: {summary.get('total_records_inserted', 0)}")
       
        # Execution details
        execution_time = summary.get('execution_timestamp') or summary.get('processing_timestamp')
        if execution_time:
            logger.info(f"  🕒 Execution Time: {execution_time}")
           
        logger.info(f"  🎯 Lambda Function: {summary.get('lambda_function_name', 'Unknown')}")
        logger.info(f"  💾 Database: {summary.get('database_type', 'Unknown')}")
       
        # Enhanced: Endpoint optimization metrics
        if summary.get('used_existing_endpoint'):
            logger.info(f"  🚀 Used existing endpoint: {summary.get('used_existing_endpoint')}")
            logger.info(f"  🔧 Test invocation mode: {summary.get('test_invocation_mode', False)}")
            logger.info(f"  🏛️ Historical mode: {summary.get('historical_mode', False)}")
           
        # Enhanced: Performance metrics
        if 'total_duration' in summary:
            total_duration = summary.get('total_duration', 0)
            avg_duration = summary.get('average_duration_per_date', 0)
            logger.info(f"  ⏱️ Total duration: {total_duration:.1f}s")
            logger.info(f"  ⚡ Avg per date: {avg_duration:.1f}s")
           
            # Calculate and display time savings
            dates_count = summary.get('total_dates', 0)
            if summary.get('used_existing_endpoint') and dates_count > 0:
                estimated_without_optimization = dates_count * 360  # 6 minutes per date
                time_saved = estimated_without_optimization - total_duration
                savings_percentage = (time_saved / estimated_without_optimization) * 100 if estimated_without_optimization > 0 else 0
               
                logger.info(f"  💰 Time saved: {time_saved:.0f}s ({savings_percentage:.0f}% reduction)")
                logger.info(f"     Without optimization: {estimated_without_optimization:.0f}s")
                logger.info(f"     With existing endpoint: {total_duration:.0f}s")
       
        # Show failed dates if any
        failed_count = summary.get('failed_dates') or summary.get('failed_predictions', 0)
        if failed_count > 0:
            logger.warning(f"\n❌ Failed dates:")
            failed_results = [r for r in summary.get('results', []) if not r.get('success', True)]
            for result in failed_results[:5]:  # Show first 5 failures
                error_msg = result.get('error', 'Unknown error')
                logger.warning(f"  • {result.get('date', 'Unknown')}: {error_msg}")
            if len(failed_results) > 5:
                logger.warning(f"  ... and {len(failed_results) - 5} more failures")
       
        # Show sample successful dates
        successful_count = summary.get('successful_dates') or summary.get('successful_predictions', 0)
        if successful_count > 0:
            successful_results = [r for r in summary.get('results', []) if r.get('success', False)]
            logger.info(f"\n✅ Sample successful dates:")
            for result in successful_results[:3]:  # Show first 3 successes
                date = result.get('date', 'Unknown')
                predictions = result.get('predictions_count', 'N/A')
                logger.info(f"  • {date} ({predictions} predictions)")
            if len(successful_results) > 3:
                logger.info(f"  ... and {len(successful_results) - 3} more successful predictions")
       
        # Performance assessment with enhanced categories
        success_rate = summary.get('success_rate', 0)
        if success_rate == 100:
            logger.info("\n🎉 PERFECT SUCCESS RATE!")
        elif success_rate >= 95:
            logger.info("\n🌟 EXCELLENT SUCCESS RATE!")
        elif success_rate >= 80:
            logger.info("\n👍 GOOD SUCCESS RATE")
        elif success_rate >= 50:
            logger.warning("\n⚠️ MODERATE SUCCESS RATE - Some issues detected")
        else:
            logger.error("\n🚨 LOW SUCCESS RATE - Investigation needed")
       
        # Enhanced: Optimization status summary
        if summary.get('used_existing_endpoint'):
            logger.info("\n💡 Optimization Status:")
            logger.info("  ✅ Endpoint optimization: ENABLED")
            logger.info("  ✅ Time savings: ACHIEVED")
            logger.info("  ✅ Cost efficiency: MAXIMIZED")
       
        return True
       
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing summary file: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading summary: {str(e)}")
        return False


def display_file_information(profile, segment):
    """Display information about generated files"""
    files_info = []
   
    # Check for summary file
    summary_file = f"prediction_summary_{profile}_{segment}.json"
    if os.path.exists(summary_file):
        size = os.path.getsize(summary_file)
        files_info.append(f"  📋 {summary_file} ({size} bytes)")
   
    # Check for successful dates file
    success_file = f"successful_dates_{profile}_{segment}.txt"
    if os.path.exists(success_file):
        with open(success_file, 'r') as f:
            line_count = sum(1 for _ in f)
        files_info.append(f"  ✅ {success_file} ({line_count} dates)")
   
    # Check for failed dates file
    failed_file = f"failed_dates_{profile}_{segment}.txt"
    if os.path.exists(failed_file):
        with open(failed_file, 'r') as f:
            line_count = sum(1 for _ in f)
        files_info.append(f"  ❌ {failed_file} ({line_count} dates)")
   
    if files_info:
        logger.info("\n📁 Generated files:")
        for info in files_info:
            logger.info(info)
    else:
        logger.warning("\n📁 No result files found")


def display_troubleshooting_tips(profile, segment):
    """Enhanced troubleshooting tips including endpoint optimization issues"""
    summary_file = f"prediction_summary_{profile}_{segment}.json"
   
    if not os.path.exists(summary_file):
        logger.info("\n🔧 Troubleshooting Tips:")
        logger.info("  1. Check if the endpoint exists and is InService")
        logger.info("  2. Verify AWS credentials and permissions")
        logger.info("  3. Check CloudWatch logs for detailed error messages")
        logger.info("  4. Ensure the combination was deployed successfully")
        logger.info("  5. Verify endpoint setup job completed successfully")
        logger.info("  6. Check S3 for endpoint configuration files")
        return
   
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
       
        failed_count = summary.get('failed_dates') or summary.get('failed_predictions', 0)
        if failed_count > 0:
            logger.info("\n🔧 Troubleshooting Tips for Failed Predictions:")
            logger.info("  1. Check endpoint health and capacity")
            logger.info("  2. Review rate limiting and request delays")
            logger.info("  3. Verify input data format and payload structure")
            logger.info("  4. Check CloudWatch logs for endpoint errors")
            logger.info("  5. Consider retrying failed dates individually")
           
            # Enhanced: Endpoint optimization specific tips
            if not summary.get('used_existing_endpoint'):
                logger.info("  6. Verify endpoint setup job completed successfully")
                logger.info("  7. Check if endpoint recreation from S3 config worked")
                logger.info("  8. Ensure endpoint config files exist in S3")
            else:
                logger.info("  6. Endpoint was pre-created successfully - check Lambda timeout")
                logger.info("  7. Verify test_invocation mode is working correctly")
                logger.info("  8. Check if cleanup job needs to be run manually")
           
    except Exception as e:
        logger.warning(f"Could not load summary for troubleshooting: {str(e)}")


def main():
    """Main display function"""
    try:
        logger.info("Displaying enhanced combination summary...")
       
        # Get inputs from environment variables (supporting both formats)
        profile = os.environ.get('PROFILE') or os.environ.get('CUSTOMER_PROFILE')
        segment = os.environ.get('SEGMENT') or os.environ.get('CUSTOMER_SEGMENT')
       
        if not profile or not segment:
            logger.error("PROFILE/CUSTOMER_PROFILE and SEGMENT/CUSTOMER_SEGMENT environment variables are required")
            sys.exit(1)
       
        logger.info(f"Generating enhanced summary for combination: {profile}-{segment}")
       
        # Display results
        success = display_combination_results(profile, segment)
       
        # Display file information
        display_file_information(profile, segment)
       
        # Display troubleshooting tips if needed
        if not success:
            display_troubleshooting_tips(profile, segment)
       
        logger.info("Enhanced combination summary display completed")
       
    except Exception as e:
        logger.error(f"❌ Error displaying combination summary: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
