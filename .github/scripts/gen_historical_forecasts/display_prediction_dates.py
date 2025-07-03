#!/usr/bin/env python3
"""
Display Prediction Dates Script

This script displays the prediction dates for dry run validation.
"""

import json
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def display_prediction_dates():
    """Display prediction dates in a formatted way"""
    try:
        # Get prediction dates from environment variables
        prediction_dates_json = os.environ.get('PREDICTION_DATES', '[]')
        prediction_dates = json.loads(prediction_dates_json)
       
        if not prediction_dates:
            logger.warning("No prediction dates found")
            return
       
        logger.info("Prediction dates:")
       
        # Show first 10 dates
        for date in prediction_dates[:10]:
            logger.info(f"  • {date}")
       
        # Show ellipsis and count if more than 10
        if len(prediction_dates) > 10:
            logger.info(f"  ... and {len(prediction_dates) - 10} more dates")
       
        logger.info(f"\nTotal dates: {len(prediction_dates)}")
       
        if len(prediction_dates) > 1:
            logger.info(f"Date range: {prediction_dates[0]} to {prediction_dates[-1]}")
       
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing prediction dates: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error displaying prediction dates: {str(e)}")
        sys.exit(1)


def main():
    """Main function"""
    try:
        display_prediction_dates()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
