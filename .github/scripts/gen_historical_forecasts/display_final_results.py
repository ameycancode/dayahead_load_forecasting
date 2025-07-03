#!/usr/bin/env python3
"""
Final Results Display Script

This script displays the final results and statistics for the entire
historical forecasting execution.
"""

import json
import glob
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_all_results():
    """Collect results from all combination summary files"""
    results = []
   
    for summary_file in glob.glob("prediction_summary_*.json"):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            results.append(summary)
        except Exception as e:
            logger.warning(f"Could not load {summary_file}: {str(e)}")
   
    return results


def display_combination_breakdown(results):
    """Display detailed breakdown by combination"""
    if not results:
        logger.warning("No combination results to display")
        return
   
    logger.info("📊 Detailed Results by Combination:")
   
    # Sort results by combination name for consistent display
    sorted_results = sorted(results, key=lambda x: x['combination'])
   
    for result in sorted_results:
        combo = result['combination']
        successful = result['successful_predictions']
        total = result['total_predictions']
        rate = result['success_rate']
       
        # Choose emoji based on success rate
        if rate == 100:
            emoji = "🎉"
        elif rate >= 95:
            emoji = "✅"
        elif rate >= 80:
            emoji = "👍"
        elif rate >= 50:
            emoji = "⚠️"
        else:
            emoji = "❌"
       
        logger.info(f"  {emoji} {combo}: {successful}/{total} successful ({rate:.1f}%)")


def display_overall_statistics(results):
    """Display overall execution statistics"""
    if not results:
        logger.error("No results available for statistics")
        return
   
    # Calculate totals
    total_combinations = len(results)
    total_predictions = sum(r['total_predictions'] for r in results)
    total_successful = sum(r['successful_predictions'] for r in results)
    total_failed = sum(r['failed_predictions'] for r in results)
   
    overall_success_rate = (total_successful / total_predictions * 100) if total_predictions > 0 else 0
    combinations_with_failures = sum(1 for r in results if r['failed_predictions'] > 0)
    perfect_combinations = sum(1 for r in results if r['success_rate'] == 100)
   
    logger.info("\n🎯 Overall Execution Statistics:")
    logger.info(f"  📋 Combinations Processed: {total_combinations}")
    logger.info(f"  🔢 Total Predictions: {total_predictions}")
    logger.info(f"  ✅ Successful: {total_successful}")
    logger.info(f"  ❌ Failed: {total_failed}")
    logger.info(f"  📈 Success Rate: {overall_success_rate:.1f}%")
    logger.info(f"  🎯 Perfect Combinations: {perfect_combinations}/{total_combinations}")
    logger.info(f"  ⚠️ Combinations with Failures: {combinations_with_failures}")


def display_performance_insights(results):
    """Display performance insights and analysis"""
    if not results:
        return
   
    logger.info("\n📊 Performance Insights:")
   
    # Find best and worst performers
    best_combo = max(results, key=lambda x: x['success_rate'])
    worst_combo = min(results, key=lambda x: x['success_rate'])
   
    logger.info(f"  🌟 Best Performer: {best_combo['combination']} ({best_combo['success_rate']:.1f}%)")
    logger.info(f"  📉 Needs Attention: {worst_combo['combination']} ({worst_combo['success_rate']:.1f}%)")
   
    # Analyze by profile and segment
    profile_stats = {}
    segment_stats = {}
   
    for result in results:
        combo_parts = result['combination'].split('-')
        if len(combo_parts) == 2:
            profile, segment = combo_parts
           
            if profile not in profile_stats:
                profile_stats[profile] = {'total': 0, 'successful': 0}
            if segment not in segment_stats:
                segment_stats[segment] = {'total': 0, 'successful': 0}
           
            profile_stats[profile]['total'] += result['total_predictions']
            profile_stats[profile]['successful'] += result['successful_predictions']
            segment_stats[segment]['total'] += result['total_predictions']
            segment_stats[segment]['successful'] += result['successful_predictions']
   
    # Display profile performance
    logger.info("\n  📋 Performance by Customer Profile:")
    for profile, stats in profile_stats.items():
        rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        logger.info(f"    • {profile}: {stats['successful']}/{stats['total']} ({rate:.1f}%)")
   
    # Display segment performance
    logger.info("\n  🔋 Performance by Customer Segment:")
    for segment, stats in segment_stats.items():
        rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        logger.info(f"    • {segment}: {stats['successful']}/{stats['total']} ({rate:.1f}%)")


def display_execution_status(results):
    """Display overall execution status and recommendations"""
    if not results:
        logger.error("\n❌ NO RESULTS FOUND")
        logger.error("  Possible issues:")
        logger.error("  • Prediction generation scripts failed")
        logger.error("  • No combinations were processed")
        logger.error("  • Summary files were not created")
        return
   
    # Calculate overall success rate
    total_predictions = sum(r['total_predictions'] for r in results)
    total_successful = sum(r['successful_predictions'] for r in results)
    total_failed = sum(r['failed_predictions'] for r in results)
   
    if total_predictions == 0:
        logger.error("\n❌ NO PREDICTIONS WERE ATTEMPTED")
        return
   
    success_rate = (total_successful / total_predictions) * 100
   
    logger.info("\n🎯 EXECUTION STATUS:")
   
    if total_failed == 0:
        logger.info("✅ COMPLETE SUCCESS - ALL PREDICTIONS GENERATED!")
        logger.info("  🎉 Perfect execution across all combinations")
        logger.info("  📊 Ready for BI dashboard integration")
        logger.info("  ✨ No further action required")
       
    elif success_rate >= 95:
        logger.info("🌟 EXCELLENT SUCCESS RATE!")
        logger.info("  👍 Nearly perfect execution")
        logger.info(f"  📈 {success_rate:.1f}% success rate")
        logger.info("  🔍 Minor issues to investigate")
       
    elif success_rate >= 80:
        logger.info("👍 GOOD SUCCESS RATE")
        logger.info("  ✅ Most predictions completed successfully")
        logger.info(f"  📈 {success_rate:.1f}% success rate")
        logger.info("  🔧 Some troubleshooting recommended")
       
    elif success_rate >= 50:
        logger.warning("⚠️ MODERATE SUCCESS RATE")
        logger.warning("  📉 Significant issues detected")
        logger.warning(f"  📊 {success_rate:.1f}% success rate")
        logger.warning("  🚨 Investigation required")
       
    else:
        logger.error("🚨 LOW SUCCESS RATE - ATTENTION REQUIRED")
        logger.error("  ❌ Major issues with prediction generation")
        logger.error(f"  📉 {success_rate:.1f}% success rate")
        logger.error("  🔧 Immediate troubleshooting needed")


def display_next_steps(results):
    """Display recommended next steps based on results"""
    if not results:
        return
   
    total_predictions = sum(r['total_predictions'] for r in results)
    total_successful = sum(r['successful_predictions'] for r in results)
    total_failed = sum(r['failed_predictions'] for r in results)
    success_rate = (total_successful / total_predictions) * 100 if total_predictions > 0 else 0
   
    logger.info("\n📋 RECOMMENDED NEXT STEPS:")
   
    if total_failed == 0:
        logger.info("  1. ✅ Verify data in database using provided SQL queries")
        logger.info("  2. 📊 Update BI dashboards to include historical forecasts")
        logger.info("  3. 🔍 Validate forecast accuracy against known values")
        logger.info("  4. 📚 Document successful execution for future reference")
       
    elif success_rate >= 80:
        logger.info("  1. 🔍 Investigate failed predictions using individual logs")
        logger.info("  2. ✅ Verify successful predictions in database")
        logger.info("  3. 🔄 Consider retrying failed dates individually")
        logger.info("  4. 📊 Proceed with BI dashboard updates for successful data")
       
    else:
        logger.info("  1. 🚨 Priority: Review CloudWatch logs for error patterns")
        logger.info("  2. 🔧 Check endpoint health and capacity")
        logger.info("  3. ⚙️ Validate configuration and rate limiting settings")
        logger.info("  4. 👥 Consider escalating to platform team if needed")
   
    logger.info("\n📁 ARTIFACTS AVAILABLE:")
    logger.info("  • Individual prediction summaries per combination")
    logger.info("  • Comprehensive execution summary (markdown)")
    logger.info("  • Success/failure date lists")
    logger.info("  • Database verification queries")


def display_file_summary():
    """Display summary of generated files"""
    summary_files = glob.glob("prediction_summary_*.json")
    success_files = glob.glob("successful_dates_*.txt")
    failed_files = glob.glob("failed_dates_*.txt")
   
    logger.info("\n📁 Generated Files Summary:")
    logger.info(f"  📋 Summary files: {len(summary_files)}")
    logger.info(f"  ✅ Success files: {len(success_files)}")
    logger.info(f"  ❌ Failure files: {len(failed_files)}")
   
    # Show total file sizes
    total_size = 0
    for file_pattern in ["prediction_summary_*.json", "successful_dates_*.txt", "failed_dates_*.txt"]:
        for file_path in glob.glob(file_pattern):
            total_size += os.path.getsize(file_path)
   
    logger.info(f"  💾 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")


def main():
    """Main results display function"""
    try:
        logger.info("Displaying final historical forecasting results...")
       
        # Collect all results
        results = collect_all_results()
       
        if not results:
            logger.error("❌ No prediction results found!")
            logger.error("This indicates that prediction generation failed for all combinations.")
            sys.exit(1)
       
        # Display various result summaries
        display_combination_breakdown(results)
        display_overall_statistics(results)
        display_performance_insights(results)
        display_execution_status(results)
        display_next_steps(results)
        display_file_summary()
       
        logger.info("\n=== HISTORICAL FORECASTING RESULTS DISPLAY COMPLETE ===")
       
    except Exception as e:
        logger.error(f"❌ Error displaying final results: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
