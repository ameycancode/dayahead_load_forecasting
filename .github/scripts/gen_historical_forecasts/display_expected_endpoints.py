#!/usr/bin/env python3
"""
Display Expected Endpoints Script

This script displays the expected endpoint names for dry run validation.
"""

import json
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def display_expected_endpoints():
    """Display expected endpoint names based on combinations matrix"""
    try:
        # Get combinations matrix and environment from environment variables
        combinations_json = os.environ.get('COMBINATIONS_MATRIX', '[]')
        environment = os.environ.get('ENVIRONMENT', 'unknown')
       
        combinations = json.loads(combinations_json)
       
        if not combinations:
            logger.warning("No combinations found in matrix")
            return
       
        logger.info("Expected endpoints to be used:")
       
        for combo in combinations:
            endpoint_name = f"{environment}-energy-ml-endpoint-{combo['profile']}-{combo['segment']}"
            logger.info(f"  • {endpoint_name}")
       
        logger.info(f"\nTotal endpoints: {len(combinations)}")
       
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing combinations matrix: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error displaying expected endpoints: {str(e)}")
        sys.exit(1)


def main():
    """Main function"""
    try:
        display_expected_endpoints()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
