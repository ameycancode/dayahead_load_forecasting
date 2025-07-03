#!/usr/bin/env python3
"""
Create cleanup log for monitoring and cost tracking.
Generates JSON log file with cleanup results and cost optimization status.
"""
import json
import os
import sys
from datetime import datetime

def create_cleanup_log():
    """Create cleanup log entry with all relevant information"""
    try:
        # Get required environment variables
        endpoint_name = os.environ.get('ENDPOINT_NAME')
        customer_profile = os.environ.get('CUSTOMER_PROFILE')
        customer_segment = os.environ.get('CUSTOMER_SEGMENT')
        environment = os.environ.get('ENVIRONMENT')
        cleanup_status = os.environ.get('CLEANUP_STATUS')
        predictions_status = os.environ.get('PREDICTIONS_STATUS')
       
        # Validate required variables
        if not all([endpoint_name, customer_profile, customer_segment, environment]):
            print("Error: Missing required environment variables")
            print(f"  ENDPOINT_NAME: {endpoint_name}")
            print(f"  CUSTOMER_PROFILE: {customer_profile}")
            print(f"  CUSTOMER_SEGMENT: {customer_segment}")
            print(f"  ENVIRONMENT: {environment}")
            sys.exit(1)
       
        # Create log entry
        cleanup_log = {
            "endpoint_name": endpoint_name,
            "combination": f"{customer_profile}-{customer_segment}",
            "environment": environment,
            "cleanup_status": cleanup_status or "unknown",
            "predictions_status": predictions_status or "unknown",
            "cleanup_timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "historical_mode": True,
            "cost_optimized": cleanup_status == 'success',
            "metadata": {
                "customer_profile": customer_profile,
                "customer_segment": customer_segment,
                "endpoint_type": "historical_forecasting",
                "cleanup_method": "automated_workflow",
                "cost_optimization_approach": "delete_after_historical_batch"
            }
        }
       
        # Create log file name
        log_filename = f"cleanup_log_{customer_profile}_{customer_segment}.json"
       
        # Write log file
        with open(log_filename, 'w') as f:
            json.dump(cleanup_log, f, indent=2)
       
        print(f"✅ Cleanup log created: {log_filename}")
        print(f"   Endpoint: {endpoint_name}")
        print(f"   Combination: {customer_profile}-{customer_segment}")
        print(f"   Cleanup status: {cleanup_status}")
        print(f"   Predictions status: {predictions_status}")
        print(f"   Cost optimized: {cleanup_log['cost_optimized']}")
       
        return True
       
    except Exception as e:
        print(f"Error creating cleanup log: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_cleanup_log()
    sys.exit(0 if success else 1)
