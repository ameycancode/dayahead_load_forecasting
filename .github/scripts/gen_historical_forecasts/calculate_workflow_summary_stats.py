#!/usr/bin/env python3
"""
Calculate Workflow Summary Statistics Script

This script calculates statistics for the final workflow summary display.
"""

import json
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_summary_stats():
    """Calculate summary statistics for GitHub workflow summary"""
    try:
        # Get data from environment variables
        combinations_json = os.environ.get('COMBINATIONS_MATRIX', '[]')
        prediction_dates_json = os.environ.get('PREDICTION_DATES', '[]')
       
        combinations = json.loads(combinations_json)
        prediction_dates = json.loads(prediction_dates_json)
       
        combo_count = len(combinations)
        dates_count = len(prediction_dates)
       
        # Create date range info
        date_range_info = "N/A"
        if prediction_dates:
            if len(prediction_dates) == 1:
                date_range_info = prediction_dates[0]
            else:
                date_range_info = f"{prediction_dates[0]} to {prediction_dates[-1]}"
       
        # Output for GitHub workflow summary
        print(f"SUMMARY_COMBO_COUNT={combo_count}")
        print(f"SUMMARY_DATES_COUNT={dates_count}")
        print(f"SUMMARY_DATE_RANGE={date_range_info}")
       
        # Log the information
        logger.info(f"Summary statistics calculated:")
        logger.info(f"  Combinations: {combo_count}")
        logger.info(f"  Dates: {dates_count}")
        logger.info(f"  Date range: {date_range_info}")
       
        # Set GitHub outputs if available
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"summary_combo_count={combo_count}\n")
                f.write(f"summary_dates_count={dates_count}\n")
                f.write(f"summary_date_range={date_range_info}\n")
       
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON data: {str(e)}")
        print("SUMMARY_COMBO_COUNT=0")
        print("SUMMARY_DATES_COUNT=0")
        print("SUMMARY_DATE_RANGE=N/A")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error calculating summary stats: {str(e)}")
        print("SUMMARY_COMBO_COUNT=0")
        print("SUMMARY_DATES_COUNT=0")
        print("SUMMARY_DATE_RANGE=N/A")
        sys.exit(1)


def main():
    """Main function"""
    try:
        calculate_summary_stats()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
