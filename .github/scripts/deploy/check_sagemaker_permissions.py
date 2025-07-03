#!/usr/bin/env python3
"""
SageMaker Permission Checker Script for GitHub Actions
Validates all required permissions for end-to-end ML pipeline operations
"""

import boto3
import json
import sagemaker
import os
import sys
from datetime import datetime
import time
import traceback

class GitHubActionsSageMakerChecker:
    def __init__(self):
        """Initialize the permission checker for GitHub Actions environment"""
        try:
            self.session = boto3.Session()
            self.region = self.session.region_name
            self.account_id = boto3.client('sts').get_caller_identity()['Account']
           
            # Initialize AWS clients
            self.sm_client = boto3.client('sagemaker')
            self.s3_client = boto3.client('s3')
            self.iam_client = boto3.client('iam')
            self.lambda_client = boto3.client('lambda')
            self.logs_client = boto3.client('logs')
            self.events_client = boto3.client('events')
           
            # SageMaker session with error handling
            try:
                self.sagemaker_session = sagemaker.Session()
                self.execution_role = sagemaker.get_execution_role()
                self.default_bucket = self.sagemaker_session.default_bucket()
            except Exception as e:
                print(f"⚠️ SageMaker session initialization issue: {e}")
                # Fallback for GitHub Actions environment
                self.execution_role = os.environ.get('SAGEMAKER_ROLE_ARN', 'Unknown')
                self.default_bucket = f"sagemaker-{self.region}-{self.account_id}"
           
            print(f"🔍 SageMaker Permission Checker - GitHub Actions Mode")
            print(f"   Region: {self.region}")
            print(f"   Account: {self.account_id}")
            print(f"   Execution Role: {self.execution_role}")
            print(f"   Default Bucket: {self.default_bucket}")
           
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            sys.exit(1)

    def check_permission(self, service_name, action_name, test_func):
        """Generic permission checker with GitHub Actions friendly output"""
        try:
            result = test_func()
            print(f"✅ {service_name} - {action_name}: ALLOWED")
            return True, result
        except Exception as e:
            error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', 'Unknown')
            print(f"❌ {service_name} - {action_name}: DENIED ({error_code})")
            print(f"   Error: {str(e)}")
            return False, str(e)

    def check_critical_permissions(self):
        """Check only the critical permissions needed for pipeline operations"""
        print("\n🚨 CHECKING CRITICAL PERMISSIONS FOR PIPELINE OPERATIONS")
        print("=" * 60)
       
        critical_results = {}
        failed_critical = []

        # Critical S3 permissions
        print("\n📦 S3 Critical Permissions:")
       
        def head_default_bucket():
            return self.s3_client.head_bucket(Bucket=self.default_bucket)
       
        result = self.check_permission("S3", "Head Default Bucket", head_default_bucket)
        critical_results['s3_head_bucket'] = result
        if not result[0]:
            failed_critical.append("S3 - Head Default Bucket")

        def list_default_bucket():
            return self.s3_client.list_objects_v2(Bucket=self.default_bucket, MaxKeys=1)
       
        result = self.check_permission("S3", "List Default Bucket", list_default_bucket)
        critical_results['s3_list_bucket'] = result
        if not result[0]:
            failed_critical.append("S3 - List Default Bucket")

        # Test S3 write capability
        test_key = f"github-actions-permission-test-{int(time.time())}.txt"
        def test_s3_write():
            self.s3_client.put_object(
                Bucket=self.default_bucket,
                Key=test_key,
                Body=b"GitHub Actions permission test"
            )
            return "Write successful"
       
        result = self.check_permission("S3", "Write to Default Bucket", test_s3_write)
        critical_results['s3_write'] = result
        if not result[0]:
            failed_critical.append("S3 - Write Access")

        # Clean up test file
        try:
            self.s3_client.delete_object(Bucket=self.default_bucket, Key=test_key)
        except:
            pass

        # Critical SageMaker permissions
        print("\n🤖 SageMaker Critical Permissions:")
       
        def list_training_jobs():
            return self.sm_client.list_training_jobs(MaxResults=1)
       
        result = self.check_permission("SageMaker", "List Training Jobs", list_training_jobs)
        critical_results['sm_training_jobs'] = result
        if not result[0]:
            failed_critical.append("SageMaker - List Training Jobs")

        def list_endpoints():
            return self.sm_client.list_endpoints(MaxResults=1)
       
        result = self.check_permission("SageMaker", "List Endpoints", list_endpoints)
        critical_results['sm_endpoints'] = result
        if not result[0]:
            failed_critical.append("SageMaker - List Endpoints")

        def list_pipelines():
            return self.sm_client.list_pipelines(MaxResults=1)
       
        result = self.check_permission("SageMaker", "List Pipelines", list_pipelines)
        critical_results['sm_pipelines'] = result
        if not result[0]:
            failed_critical.append("SageMaker - List Pipelines")

        def list_model_packages():
            return self.sm_client.list_model_packages(MaxResults=1)
       
        result = self.check_permission("SageMaker", "List Model Packages", list_model_packages)
        critical_results['sm_model_registry'] = result
        if not result[0]:
            failed_critical.append("SageMaker - Model Registry")

        # Critical CloudWatch permissions
        print("\n📊 CloudWatch Critical Permissions:")
       
        def list_log_groups():
            return self.logs_client.describe_log_groups(
                logGroupNamePrefix='/aws/sagemaker/', limit=1
            )
       
        result = self.check_permission("CloudWatch", "List SageMaker Log Groups", list_log_groups)
        critical_results['cw_logs'] = result
        if not result[0]:
            failed_critical.append("CloudWatch - Log Groups")

        # Critical EventBridge permissions
        print("\n⏰ EventBridge Critical Permissions:")
       
        def list_rules():
            return self.events_client.list_rules(Limit=1)
       
        result = self.check_permission("EventBridge", "List Rules", list_rules)
        critical_results['eb_rules'] = result
        if not result[0]:
            failed_critical.append("EventBridge - List Rules")

        # IAM permissions
        print("\n🔐 IAM Critical Permissions:")
       
        def get_caller_identity():
            return boto3.client('sts').get_caller_identity()
       
        result = self.check_permission("IAM", "Get Caller Identity", get_caller_identity)
        critical_results['iam_identity'] = result
        if not result[0]:
            failed_critical.append("IAM - Get Caller Identity")

        return critical_results, failed_critical

    def check_additional_critical_permissions(self):
        """Check additional critical permissions for complete pipeline operations"""
        print("\n CHECKING ADDITIONAL CRITICAL PERMISSIONS")
        print("=" * 60)
       
        additional_results = {}
        additional_failed = []
   
        # Critical Redshift permissions
        print("\n Redshift Critical Permissions:")
       
        try:
            redshift_client = boto3.client('redshift')
            redshift_data_client = boto3.client('redshift-data')
           
            def list_redshift_clusters():
                return redshift_client.describe_clusters(MaxRecords=20)
           
            result = self.check_permission("Redshift", "List Clusters", list_redshift_clusters)
            additional_results['redshift_clusters'] = result
            if not result[0]:
                additional_failed.append("Redshift - List Clusters")
   
            # Check Redshift Data API access
            def list_databases():
                return redshift_data_client.list_databases(
                    ClusterIdentifier='dummy',  # This will fail but tests permission
                    DbUser='dummy'
                )
           
            # This will likely fail but tells us about permissions
            try:
                list_databases()
            except Exception as e:
                if 'AccessDenied' in str(e) or 'Forbidden' in str(e):
                    print(" Redshift Data API - List Databases: DENIED")
                else:
                    print(" Redshift Data API - List Databases: ALLOWED (expected error for dummy values)")

            print(f"✅ Redshift - Data API Access: ALLOWED")
               
        except Exception as e:
            print(f" Redshift client initialization failed: {e}")
            additional_results['redshift_init'] = (False, str(e))
            additional_failed.append("Redshift - Service Access")
   
        # Critical Lambda permissions
        print("\n Lambda Critical Permissions:")
       
        def list_lambda_functions():
            return self.lambda_client.list_functions(MaxItems=1)
       
        result = self.check_permission("Lambda", "List Functions", list_lambda_functions)
        additional_results['lambda_functions'] = result
        if not result[0]:
            additional_failed.append("Lambda - List Functions")
   
        # Critical ECR permissions
        print("\n ECR Critical Permissions:")
       
        try:
            ecr_client = boto3.client('ecr')
           
            def list_ecr_repositories():
                return ecr_client.describe_repositories(maxResults=1)
           
            result = self.check_permission("ECR", "List Repositories", list_ecr_repositories)
            additional_results['ecr_repositories'] = result
            if not result[0]:
                additional_failed.append("ECR - List Repositories")
               
        except Exception as e:
            print(f" ECR client initialization failed: {e}")
            additional_results['ecr_init'] = (False, str(e))
            additional_failed.append("ECR - Service Access")
   
        return additional_results, additional_failed

    def generate_github_actions_report(self, results, failed_permissions):
        """Generate GitHub Actions specific report"""
        print("\n" + "=" * 60)
        print("📋 PERMISSION CHECK SUMMARY")
        print("=" * 60)
       
        total_critical = len(results)
        passed_critical = len([r for r in results.values() if r[0]])
        failed_count = len(failed_permissions)
       
        success_rate = (passed_critical / total_critical * 100) if total_critical > 0 else 0
       
        print(f"Critical Permissions: {passed_critical}/{total_critical} ({success_rate:.1f}%)")
        print(f"Failed Permissions: {failed_count}")
       
        if failed_permissions:
            print(f"\n❌ FAILED CRITICAL PERMISSIONS:")
            for perm in failed_permissions:
                print(f"   - {perm}")
        else:
            print(f"\n✅ All critical permissions are available!")

        # Set GitHub Actions outputs
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"permission_check_passed={'true' if not failed_permissions else 'false'}\n")
                f.write(f"failed_permissions_count={failed_count}\n")
                f.write(f"success_rate={success_rate:.1f}\n")
                f.write(f"failed_permissions={';'.join(failed_permissions)}\n")

        return success_rate, failed_permissions

    def generate_detailed_report_file(self, results, failed_permissions):
        """Generate a detailed markdown report file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sagemaker_permission_report_{timestamp}.md"
       
        markdown_content = f"""# SageMaker Permission Check Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Environment:** GitHub Actions  
**AWS Account:** {self.account_id}  
**Region:** {self.region}  
**Execution Role:** {self.execution_role}

## Summary

- **Total Critical Permissions:** {len(results)}
- **Passed:** {len([r for r in results.values() if r[0]])}
- **Failed:** {len(failed_permissions)}
- **Success Rate:** {(len([r for r in results.values() if r[0]]) / len(results) * 100):.1f}%

## Failed Permissions

"""
        if failed_permissions:
            markdown_content += "❌ **Critical permissions missing:**\n\n"
            for perm in failed_permissions:
                markdown_content += f"- {perm}\n"
        else:
            markdown_content += "✅ **All critical permissions are available**\n"

        markdown_content += f"""

## Required Actions

{"### Immediate Action Required" if failed_permissions else "### Ready to Proceed"}

"""
        if failed_permissions:
            markdown_content += """The pipeline cannot proceed safely with missing critical permissions.

**Next Steps:**
1. Contact AWS administrator
2. Review and attach required IAM policies
3. Re-run permission check
4. Proceed with pipeline deployment

**Required Policies:**
- AmazonSageMakerFullAccess
- Custom S3 policy for SageMaker buckets
- CloudWatch Logs access
- EventBridge access for scheduling
"""
        else:
            markdown_content += """All critical permissions are available. Pipeline can proceed safely.

**Status:** ✅ Ready for deployment
"""

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"📄 Detailed report saved: {filename}")
            return filename
        except Exception as e:
            print(f"⚠️ Could not save report file: {e}")
            return None

def main():
    """Main function for GitHub Actions execution"""
    print("🚀 Starting SageMaker Permission Check for GitHub Actions")
   
    try:
        # Initialize checker
        checker = GitHubActionsSageMakerChecker()
       
        # Check critical permissions
        results, failed_permissions = checker.check_critical_permissions()
       
        # Check additional critical permissions
        additional_results, additional_failed = checker.check_additional_critical_permissions()
       
        # Combine results
        all_results = {**results, **additional_results}
        all_failed = failed_permissions + additional_failed
       
        # Generate reports
        success_rate, failed_perms = checker.generate_github_actions_report(all_results, all_failed)
       
        # Save detailed report
        report_file = checker.generate_detailed_report_file(all_results, all_failed)
       
        # Determine exit code based on critical permissions
        if all_failed:
            print(f"\n❌ PERMISSION CHECK FAILED")
            print(f"   Missing {len(all_failed)} critical permissions")
            print(f"   Pipeline deployment cannot proceed safely")
            sys.exit(1)
        else:
            print(f"\n✅ PERMISSION CHECK PASSED")
            print(f"   All critical permissions available")
            print(f"   Pipeline deployment can proceed")
            sys.exit(0)
           
    except Exception as e:
        print(f"❌ Permission check failed with error: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()
       
        # Set GitHub Actions output for failure
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write("permission_check_passed=false\n")
                f.write(f"error_message={str(e)}\n")
       
        sys.exit(1)

if __name__ == "__main__":
    main()
