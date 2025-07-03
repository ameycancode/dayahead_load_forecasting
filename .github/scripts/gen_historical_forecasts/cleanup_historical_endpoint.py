import boto3
import time
import os
import sys
from datetime import datetime

def get_endpoint_status(sagemaker_client, endpoint_name):
    """Get current endpoint status"""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except sagemaker_client.exceptions.ClientError as e:
        if 'ValidationException' in str(e):
            return 'NotFound'
        raise e

def delete_endpoint_for_cleanup(sagemaker_client, endpoint_name):
    """Delete endpoint for cost optimization after historical predictions"""
    try:
        current_status = get_endpoint_status(sagemaker_client, endpoint_name)
        print(f"Current endpoint status: {current_status}")
       
        if current_status == 'NotFound':
            print("✅ Endpoint already deleted or doesn't exist")
            return True
       
        if current_status == 'Deleting':
            print("⏳ Endpoint already being deleted - waiting for completion...")
            return wait_for_endpoint_deletion(sagemaker_client, endpoint_name)
       
        if current_status in ['InService', 'OutOfService', 'Failed']:
            print(f"🗑️ Deleting endpoint: {endpoint_name}")
           
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print("✅ Endpoint deletion initiated")
           
            # Wait for deletion to complete
            return wait_for_endpoint_deletion(sagemaker_client, endpoint_name)
       
        else:
            print(f"⚠️ Endpoint in unexpected state: {current_status}")
            force_cleanup = os.environ.get('FORCE_CLEANUP', 'false').lower() == 'true'
           
            if force_cleanup:
                print("🔨 Force cleanup enabled - attempting deletion anyway...")
                try:
                    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                    print("✅ Force deletion initiated")
                    return wait_for_endpoint_deletion(sagemaker_client, endpoint_name)
                except Exception as force_e:
                    print(f"❌ Force deletion failed: {str(force_e)}")
                    return False
            else:
                print("❌ Cleanup skipped due to unexpected endpoint state")
                return False
       
    except Exception as e:
        print(f"❌ Error during endpoint cleanup: {str(e)}")
        return False

def wait_for_endpoint_deletion(sagemaker_client, endpoint_name):
    """Wait for endpoint deletion to complete"""
    cleanup_timeout = int(os.environ.get('CLEANUP_TIMEOUT', '300'))
    print(f"Waiting for endpoint deletion to complete (max {cleanup_timeout}s)...")
   
    start_time = time.time()
    while time.time() - start_time < cleanup_timeout:
        try:
            status = get_endpoint_status(sagemaker_client, endpoint_name)
            elapsed = time.time() - start_time
           
            if status == 'NotFound':
                print(f"✅ Endpoint successfully deleted ({elapsed:.0f}s)")
                return True
            elif status == 'Deleting':
                print(f"⏳ Still deleting... ({elapsed:.0f}s)")
            else:
                print(f"⚠️ Unexpected status during deletion: {status} ({elapsed:.0f}s)")
           
            time.sleep(10)
           
        except Exception as e:
            print(f"Error checking deletion status: {str(e)}")
            time.sleep(10)
   
    print(f"⏰ Deletion timeout after {cleanup_timeout}s")
   
    # Check final status
    try:
        final_status = get_endpoint_status(sagemaker_client, endpoint_name)
        if final_status == 'NotFound':
            print("✅ Endpoint deleted successfully (confirmed after timeout)")
            return True
        else:
            print(f"⚠️ Endpoint still exists with status: {final_status}")
            print("💰 Manual cleanup may be required to avoid ongoing costs")
            return False
    except:
        print("✅ Endpoint likely deleted (cannot verify)")
        return True

def log_cleanup_summary(endpoint_name, success, predictions_status):
    """Log cleanup summary with cost optimization impact"""
    print(f"\n=== CLEANUP SUMMARY FOR {endpoint_name} ===")
    print(f"Endpoint: {endpoint_name}")
    print(f"Cleanup success: {'✅ Yes' if success else '❌ No'}")
    print(f"Predictions status: {predictions_status}")
    print(f"Cleanup timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   
    if success:
        print("💰 Cost optimization: Endpoint deleted - zero ongoing costs")
        print("🔄 Next historical run will recreate endpoint as needed")
    else:
        print("⚠️ Cost warning: Endpoint may still be running and incurring costs")
        print("🔧 Manual cleanup recommended: aws sagemaker delete-endpoint --endpoint-name " + endpoint_name)
   
    print("=== END CLEANUP SUMMARY ===")

def main():
    """Main cleanup function"""
    try:
        endpoint_name = os.environ['ENDPOINT_NAME']
        predictions_status = os.environ.get('PREDICTIONS_STATUS', 'unknown')
       
        print(f"Starting cleanup for endpoint: {endpoint_name}")
        print(f"Historical predictions status: {predictions_status}")
       
        # Initialize SageMaker client
        sagemaker_client = boto3.client('sagemaker')
       
        # Cleanup endpoint
        success = delete_endpoint_for_cleanup(sagemaker_client, endpoint_name)
       
        # Log summary
        log_cleanup_summary(endpoint_name, success, predictions_status)
       
        return success
       
    except Exception as e:
        print(f"❌ Error in cleanup: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
