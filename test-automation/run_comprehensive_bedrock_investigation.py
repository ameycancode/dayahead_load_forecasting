#!/usr/bin/env python3
"""
Debug version of ClaudeClient to investigate Bedrock response structure
and verify correct API version for Claude 3.5 Sonnet v2
"""

import sys
import json
import logging
import boto3
from datetime import datetime
from typing import Tuple, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_bedrock_investigation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Track token usage for cost monitoring"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    timestamp: str = ""
    
    def calculate_cost(self):
        """Calculate cost based on Claude 3.5 Sonnet v2 pricing"""
        input_cost = self.input_tokens * 0.003 / 1000  # $0.003 per 1K input tokens
        output_cost = self.output_tokens * 0.015 / 1000  # $0.015 per 1K output tokens
        self.total_cost = input_cost + output_cost
        return self.total_cost


class BedrockDebugClient:
    """
    Debug version of Claude client to investigate response structure
    and verify API version compatibility
    """
    
    def __init__(self, region_name: str = "us-west-2"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        
        # Claude 3.5 Sonnet v2 model ID (released October 2024)
        self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        
        # API version - this is correct for Bedrock Anthropic models
        # This version supports Claude 3.5 Sonnet v2
        self.api_version = "bedrock-2023-05-31"
        
        logger.info(f"Initializing Bedrock client with:")
        logger.info(f"  Model ID: {self.model_id}")
        logger.info(f"  API Version: {self.api_version}")
        logger.info(f"  Region: {region_name}")
    
    def investigate_response_structure(self, prompt: str = "Hello, what is 2+2?") -> Dict[str, Any]:
        """
        Make a simple call to investigate the complete response structure
        """
        logger.info(" Investigating Bedrock response structure...")
        
        try:
            # Prepare request body
            request_body = {
                "anthropic_version": self.api_version,
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            logger.info(f" Request body: {json.dumps(request_body, indent=2)}")
            
            # Make the API call
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            logger.info(" RAW RESPONSE INVESTIGATION:")
            logger.info("=" * 60)
            
            # Print all top-level response keys
            logger.info(f" Top-level response keys: {list(response.keys())}")
            
            # Investigate each key
            for key, value in response.items():
                logger.info(f" Key: '{key}'")
                logger.info(f"   Type: {type(value)}")
                
                if key == 'body':
                    # Don't print the body directly as it's a stream
                    logger.info(f"   Value: <StreamingBody object>")
                    logger.info(f"   Methods: {[method for method in dir(value) if not method.startswith('_')]}")
                else:
                    logger.info(f"   Value: {value}")
            
            # Read and parse the response body
            response_body_raw = response['body'].read()
            logger.info(f" Response body (raw bytes length): {len(response_body_raw)}")
            
            # Parse JSON response
            response_body = json.loads(response_body_raw)
            
            logger.info(" PARSED RESPONSE BODY INVESTIGATION:")
            logger.info("=" * 60)
            
            # Print all keys in response body
            logger.info(f" Response body keys: {list(response_body.keys())}")
            
            # Investigate each key in the response body
            for key, value in response_body.items():
                logger.info(f" Body key: '{key}'")
                logger.info(f"   Type: {type(value)}")
                
                if key == 'usage':
                    # Deep dive into usage statistics
                    logger.info(f"   USAGE DETAILS:")
                    if isinstance(value, dict):
                        for usage_key, usage_value in value.items():
                            logger.info(f"      {usage_key}: {usage_value} (type: {type(usage_value)})")
                    else:
                        logger.info(f"   Value: {value}")
                        
                elif key == 'content':
                    # Deep dive into content structure
                    logger.info(f"   CONTENT DETAILS:")
                    if isinstance(value, list):
                        logger.info(f"      Content is a list with {len(value)} items")
                        for i, item in enumerate(value):
                            logger.info(f"      Item {i}: {type(item)}")
                            if isinstance(item, dict):
                                logger.info(f"        Keys: {list(item.keys())}")
                                for content_key, content_value in item.items():
                                    if content_key == 'text':
                                        logger.info(f"        {content_key}: '{content_value[:100]}...' (truncated)")
                                    else:
                                        logger.info(f"        {content_key}: {content_value}")
                    else:
                        logger.info(f"   Value: {value}")
                        
                elif key == 'stop_reason':
                    logger.info(f"    Stop reason: {value}")
                    
                elif key == 'stop_sequence':
                    logger.info(f"   Stop sequence: {value}")
                    
                else:
                    # Print other keys as-is
                    if isinstance(value, (str, int, float, bool)):
                        logger.info(f"   Value: {value}")
                    elif isinstance(value, dict):
                        logger.info(f"   Dict keys: {list(value.keys())}")
                    elif isinstance(value, list):
                        logger.info(f"   List length: {len(value)}")
                    else:
                        logger.info(f"   Value: {str(value)[:200]}...")
            
            return {
                'full_response': response,
                'parsed_body': response_body,
                'investigation_complete': True
            }
            
        except Exception as e:
            logger.error(f" Error during response investigation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'investigation_complete': False
            }
    
    def call_claude_with_token_debugging(self, prompt: str, max_tokens: int = 1000) -> Tuple[str, TokenUsage, Dict]:
        """
        Call Claude with detailed token usage debugging
        """
        logger.info(" Making Claude API call with token debugging...")
        
        try:
            # Prepare request
            request_body = {
                "anthropic_version": self.api_version,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            }
            
            # Log request details
            logger.info(f" Request details:")
            logger.info(f"   Model ID: {self.model_id}")
            logger.info(f"   Max tokens requested: {max_tokens}")
            logger.info(f"   Prompt length (chars): {len(prompt)}")
            logger.info(f"   Estimated input tokens: {len(prompt.split()) * 1.3:.0f}")
            
            # Make API call
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract content
            content = ""
            if 'content' in response_body and len(response_body['content']) > 0:
                content = response_body['content'][0].get('text', '')
            
            # Extract and analyze token usage
            usage_info = response_body.get('usage', {})
            
            logger.info(f" Response details:")
            logger.info(f"   Content length (chars): {len(content)}")
            logger.info(f"   Stop reason: {response_body.get('stop_reason', 'unknown')}")
            
            # Debug token usage
            logger.info(f" TOKEN USAGE ANALYSIS:")
            logger.info("=" * 40)
            
            if usage_info:
                logger.info(f" Usage info found in response:")
                for key, value in usage_info.items():
                    logger.info(f"   {key}: {value}")
                
                # Create TokenUsage object
                token_usage = TokenUsage(
                    input_tokens=usage_info.get('input_tokens', 0),
                    output_tokens=usage_info.get('output_tokens', 0),
                    timestamp=datetime.now().isoformat()
                )
                
                # Calculate cost
                cost = token_usage.calculate_cost()
                
                logger.info(f" COST CALCULATION:")
                logger.info(f"   Input tokens: {token_usage.input_tokens} × $0.003/1K = ${token_usage.input_tokens * 0.003 / 1000:.6f}")
                logger.info(f"   Output tokens: {token_usage.output_tokens} × $0.015/1K = ${token_usage.output_tokens * 0.015 / 1000:.6f}")
                logger.info(f"   Total cost: ${cost:.6f}")
                
            else:
                logger.warning("  No usage info found in response!")
                logger.warning("Available response keys:", list(response_body.keys()))
                
                # Fallback estimation
                estimated_input = int(len(prompt.split()) * 1.3)
                estimated_output = int(len(content.split()) * 1.3)
                
                token_usage = TokenUsage(
                    input_tokens=estimated_input,
                    output_tokens=estimated_output,
                    timestamp=datetime.now().isoformat()
                )
                
                cost = token_usage.calculate_cost()
                
                logger.info(f" ESTIMATED TOKEN USAGE (fallback):")
                logger.info(f"   Estimated input tokens: {estimated_input}")
                logger.info(f"   Estimated output tokens: {estimated_output}")
                logger.info(f"   Estimated cost: ${cost:.6f}")
            
            return content, token_usage, response_body
            
        except Exception as e:
            logger.error(f" Error in Claude API call: {e}")
            return "", TokenUsage(), {}
    
    def verify_model_availability(self) -> bool:
        """
        Verify that Claude 3.5 Sonnet v2 is available in the current region
        """
        logger.info(" Verifying model availability...")
        
        try:
            # Use Bedrock client to list available models
            bedrock_client = boto3.client("bedrock", region_name=self.client._client_config.region_name)
            
            # List foundation models
            response = bedrock_client.list_foundation_models()
            
            # Look for our specific model
            claude_models = []
            for model in response.get('modelSummaries', []):
                if 'claude' in model.get('modelId', '').lower():
                    claude_models.append({
                        'modelId': model.get('modelId'),
                        'modelName': model.get('modelName'),
                        'providerName': model.get('providerName'),
                        'status': model.get('modelLifecycle', {}).get('status')
                    })
            
            logger.info(f" Available Claude models in region:")
            for model in claude_models:
                status_emoji = " " if model['status'] == 'ACTIVE' else " "
                logger.info(f"   {status_emoji} {model['modelId']} - {model['modelName']}")
            
            # Check if our target model is available
            target_model_available = any(
                model['modelId'] == self.model_id and model['status'] == 'ACTIVE'
                for model in claude_models
            )
            
            if target_model_available:
                logger.info(f" Target model {self.model_id} is available and active!")
            else:
                logger.warning(f"  Target model {self.model_id} not found or not active")
                logger.info("Available alternatives:")
                for model in claude_models:
                    if model['status'] == 'ACTIVE':
                        logger.info(f"   - {model['modelId']}")
            
            return target_model_available
            
        except Exception as e:
            logger.error(f" Error checking model availability: {e}")
            return False
    
    def test_api_versions(self) -> Dict[str, Any]:
        """
        Test different API versions to find the correct one for Claude 3.5 Sonnet v2
        """
        logger.info(" Testing different API versions...")
        
        api_versions_to_test = [
            "bedrock-2023-05-31",  # Current version
            "bedrock-2024-06-01",  # Possible newer version
            "bedrock-2024-10-01",  # Possible version for Claude 3.5 Sonnet v2
        ]
        
        test_prompt = "What is the capital of France? Answer in one word."
        results = {}
        
        for api_version in api_versions_to_test:
            logger.info(f" Testing API version: {api_version}")
            
            try:
                request_body = {
                    "anthropic_version": api_version,
                    "max_tokens": 50,
                    "messages": [
                        {
                            "role": "user",
                            "content": test_prompt
                        }
                    ]
                }
                
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                
                results[api_version] = {
                    'success': True,
                    'response_keys': list(response_body.keys()),
                    'has_usage': 'usage' in response_body,
                    'usage_keys': list(response_body.get('usage', {}).keys()) if 'usage' in response_body else [],
                    'content_received': bool(response_body.get('content'))
                }
                
                logger.info(f"   Success with {api_version}")
                if 'usage' in response_body:
                    usage = response_body['usage']
                    logger.info(f"   Token usage: input={usage.get('input_tokens')}, output={usage.get('output_tokens')}")
                
            except Exception as e:
                results[api_version] = {
                    'success': False,
                    'error': str(e)
                }
                logger.info(f"   Failed with {api_version}: {e}")
        
        # Summary
        logger.info(" API VERSION TEST SUMMARY:")
        logger.info("=" * 50)
        
        working_versions = [v for v, r in results.items() if r.get('success')]
        if working_versions:
            logger.info(f" Working API versions: {working_versions}")
            
            # Find the best version (one with usage info)
            best_version = None
            for version in working_versions:
                if results[version].get('has_usage'):
                    best_version = version
                    break
            
            if best_version:
                logger.info(f" Recommended API version: {best_version} (has token usage)")
            else:
                logger.info(f"  No version provides token usage info")
        else:
            logger.info(" No working API versions found")
        
        return results


def run_comprehensive_bedrock_investigation():
    """
    Run a comprehensive investigation of Bedrock API responses
    """
    print(" COMPREHENSIVE BEDROCK CLAUDE INVESTIGATION")
    print("=" * 70)
    
    # Initialize debug client
    debug_client = BedrockDebugClient()
    
    print("\n  VERIFYING MODEL AVAILABILITY")
    print("-" * 50)
    model_available = debug_client.verify_model_availability()
    
    if not model_available:
        print("  Target model not available. Proceeding with investigation anyway...")
    
    print("\n  TESTING API VERSIONS")
    print("-" * 50)
    api_test_results = debug_client.test_api_versions()
    
    print("\n  INVESTIGATING RESPONSE STRUCTURE")
    print("-" * 50)
    investigation_results = debug_client.investigate_response_structure()
    
    print("\n  TESTING TOKEN USAGE EXTRACTION")
    print("-" * 50)
    test_prompt = "Explain the importance of unit testing in Python development in exactly 100 words."
    content, token_usage, raw_response = debug_client.call_claude_with_token_debugging(test_prompt)
    
    print(f"\n Generated content preview: '{content[:100]}...'")
    print(f" Final token usage: {token_usage.input_tokens} input, {token_usage.output_tokens} output")
    print(f" Final cost: ${token_usage.total_cost:.6f}")
    
    print("\n  RECOMMENDATIONS")
    print("-" * 50)
    
    # Analyze results and provide recommendations
    working_versions = [v for v, r in api_test_results.items() if r.get('success')]
    
    if working_versions:
        # Find version with token usage
        version_with_usage = None
        for version in working_versions:
            if api_test_results[version].get('has_usage'):
                version_with_usage = version
                break
        
        if version_with_usage:
            print(f" USE API VERSION: {version_with_usage}")
            print(f" TOKEN USAGE: Available and accurate")
            print(f" COST CALCULATION: Reliable")
        else:
            print(f"  USE API VERSION: {working_versions[0]} (first working)")
            print(f" TOKEN USAGE: May need estimation")
            print(f"  COST CALCULATION: Use fallback estimation")
    else:
        print(" No working API versions found!")
        print(" CHECK: AWS credentials, region, model access")
    
    print(f"\n MODEL ID CONFIRMED: {debug_client.model_id}")
    print(f" PRICING ASSUMPTION: $0.003/1K input, $0.015/1K output tokens")
    
    return {
        'model_available': model_available,
        'api_test_results': api_test_results,
        'investigation_results': investigation_results,
        'token_usage': token_usage,
        'working_versions': working_versions
    }


if __name__ == "__main__":
    # Run the comprehensive investigation
    results = run_comprehensive_bedrock_investigation()
    
    print("\n INVESTIGATION COMPLETE!")
    print("Use the recommendations above to configure your ClaudeClient correctly.")
