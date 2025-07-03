#!/usr/bin/env python3
"""
Count Combinations and Dates Script

This script counts the number of combinations and dates for display purposes.
"""

import json
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_and_output():
    """Count combinations and dates, output to both stdout and GitHub outputs"""
    try:
        # Get data from environment variables
        combinations_json = os.environ.get('COMBINATIONS_MATRIX', '[]')
        prediction_dates_json = os.environ.get('PREDICTION_DATES', '[]')
       
        combinations = json.loads(combinations_json)
        prediction_dates = json.loads(prediction_dates_json)
       
        combo_count = len(combinations)
        dates_count = len(prediction_dates)
       
        # Output counts for use in workflow
        print(f"COMBO_COUNT={combo_count}")
        print(f"DATES_COUNT={dates_count}")
       
        # Also log for visibility
        logger.info(f"Combinations count: {combo_count}")
        logger.info(f"Dates count: {dates_count}")
       
        # Set GitHub outputs if available
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"combo_count={combo_count}\n")
                f.write(f"dates_count={dates_count}\n")
       
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON data: {str(e)}")
        print("COMBO_COUNT=0")
        print("DATES_COUNT=0")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error counting combinations and dates: {str(e)}")
        print("COMBO_COUNT=0")
        print("DATES_COUNT=0")
        sys.exit(1)


def main():
    """Main function"""
    try:
        count_and_output()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
