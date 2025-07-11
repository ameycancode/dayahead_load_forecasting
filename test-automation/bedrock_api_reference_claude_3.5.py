"""
Bedrock API Reference for Claude 3.5 Sonnet v2

This file provides the correct API configuration and explains the relationship
between API versions and Claude model versions.
"""

# ============================================================================
# CLAUDE 3.5 SONNET V2 - CORRECT CONFIGURATION
# ============================================================================

"""
✅ CORRECT MODEL ID: anthropic.claude-3-5-sonnet-20241022-v2:0
✅ CORRECT API VERSION: bedrock-2023-05-31
✅ PRICING: $0.003/1K input tokens, $0.015/1K output tokens

IMPORTANT CLARIFICATIONS:
1. The API version "bedrock-2023-05-31" is CORRECT for Claude 3.5 Sonnet v2
2. This API version is the Bedrock messaging format version, NOT the model version
3. Claude 3.5 Sonnet v2 (released October 2024) uses the same API format
4. The pricing we're using is correct for Claude 3.5 Sonnet v2
"""

# Correct configuration
CLAUDE_3_5_SONNET_V2_CONFIG = {
    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "api_version": "bedrock-2023-05-31",  # This is the messaging format version
    "pricing": {
        "input_tokens_per_1k": 0.003,    # $0.003 per 1K input tokens
        "output_tokens_per_1k": 0.015    # $0.015 per 1K output tokens
    },
    "region_availability": [
        "us-east-1", 
        "us-west-2", 
        "eu-west-1", 
        "ap-southeast-1"
    ]
}

# ============================================================================
# EXPECTED RESPONSE STRUCTURE FROM BEDROCK
# ============================================================================

"""
When you call invoke_model with Claude 3.5 Sonnet v2, you should get:

📥 TOP-LEVEL RESPONSE KEYS:
- 'ResponseMetadata': AWS service metadata
- 'contentType': 'application/json'
- 'body': StreamingBody object containing the actual response

📋 PARSED BODY STRUCTURE:
{
    "id": "msg_01ABC123...",                    # Unique message ID
    "type": "message",                          # Always "message"
    "role": "assistant",                        # Always "assistant"
    "content": [                                # Array of content blocks
        {
            "type": "text",
            "text": "The actual response text..."
        }
    ],
    "model": "claude-3-5-sonnet-20241022",      # Model used
    "stop_reason": "end_turn",                  # Why generation stopped
    "stop_sequence": null,                      # Stop sequence if used
    "usage": {                                  # ✅ TOKEN USAGE INFO
        "input_tokens": 123,                    # Actual input token count
        "output_tokens": 456                    # Actual output token count
    }
}

🔑 KEY POINTS:
- The 'usage' key contains exact token counts
- These are the actual tokens used, not estimates
- Use these for precise cost calculation
"""

# ============================================================================
# SAMPLE IMPLEMENTATION WITH CORRECT TOKEN EXTRACTION
# ============================================================================

import json
import boto3
from datetime import datetime

class CorrectClaudeClient:
    """
    Correct implementation for Claude 3.5 Sonnet v2 with accurate token tracking
    """
    
    def __init__(self, region_name: str = "us-west-2"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        self.api_version = "bedrock-2023-05-31"  # This is CORRECT
        
    def call_claude_correct(self, prompt: str, max_tokens: int = 4000):
        """
        Correct implementation with proper token extraction
        """
        try:
            # Request body with correct API version
            request_body = {
                "anthropic_version": self.api_version,  # CORRECT: bedrock-2023-05-31
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Make API call
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response body
            response_body = json.loads(response['body'].read())
            
            # Extract content
            content = ""
            if response_body.get('content') and len(response_body['content']) > 0:
                content = response_body['content'][0].get('text', '')
            
            # Extract token usage (THIS IS THE KEY PART)
            usage = response_body.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            
            # Calculate exact cost
            input_cost = input_tokens * 0.003 / 1000
            output_cost = output_tokens * 0.015 / 1000
            total_cost = input_cost + output_cost
            
            # Print debugging info
            print(f"📊 Token Usage:")
            print(f"   Input tokens: {input_tokens}")
            print(f"   Output tokens: {output_tokens}")
            print(f"   Input cost: ${input_cost:.6f}")
            print(f"   Output cost: ${output_cost:.6f}")
            print(f"   Total cost: ${total_cost:.6f}")
            
            return content, {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_cost': total_cost,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return "", {'input_tokens': 0, 'output_tokens': 0, 'total_cost': 0.0}

# ============================================================================
# DEBUGGING FUNCTION TO PRINT ALL RESPONSE KEYS
# ============================================================================

def debug_bedrock_response():
    """
    Function to debug and print all response keys
    """
    client = CorrectClaudeClient()
    
    print("🔍 DEBUGGING BEDROCK RESPONSE STRUCTURE")
    print("=" * 60)
    
    try:
        # Make a simple call
        response = client.client.invoke_model(
            modelId=client.model_id,
            body=json.dumps({
                "anthropic_version": client.api_version,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, what is 2+2?"}]
            })
        )
        
        print("📥 TOP-LEVEL RESPONSE KEYS:")
        for key in response.keys():
            print(f"   - {key}: {type(response[key])}")
        
        # Read and parse body
        body_content = response['body'].read()
        parsed_body = json.loads(body_content)
        
        print(f"\n📋 PARSED BODY KEYS:")
        for key in parsed_body.keys():
            print(f"   - {key}: {type(parsed_body[key])}")
        
        print(f"\n📊 USAGE DETAILS:")
        if 'usage' in parsed_body:
            usage = parsed_body['usage']
            for key, value in usage.items():
                print(f"   - {key}: {value}")
        else:
            print("   ❌ No 'usage' key found!")
        
        print(f"\n📝 CONTENT STRUCTURE:")
        if 'content' in parsed_body:
            content = parsed_body['content']
            print(f"   Type: {type(content)}")
            if isinstance(content, list) and len(content) > 0:
                print(f"   First item keys: {list(content[0].keys())}")
        
        print(f"\n🔧 FULL RESPONSE KEYS BREAKDOWN:")
        def print_dict_structure(d, indent=0):
            spaces = "  " * indent
            for key, value in d.items():
                if isinstance(value, dict):
                    print(f"{spaces}- {key}: (dict)")
                    print_dict_structure(value, indent + 1)
                elif isinstance(value, list):
                    print(f"{spaces}- {key}: (list, length={len(value)})")
                    if value and isinstance(value[0], dict):
                        print(f"{spaces}  First item keys: {list(value[0].keys())}")
                else:
                    print(f"{spaces}- {key}: {type(value).__name__}")
        
        print_dict_structure(parsed_body)
        
        return parsed_body
        
    except Exception as e:
        print(f"❌ Error in debugging: {e}")
        return None

# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def quick_test():
    """
    Quick test to verify everything works
    """
    print("🧪 QUICK TEST OF CLAUDE 3.5 SONNET V2")
    print("=" * 50)
    
    client = CorrectClaudeClient()
    
    # Test with a simple prompt
    prompt = "What is the capital of France? Answer in exactly 5 words."
    content, usage = client.call_claude_correct(prompt)
    
    print(f"✅ Response: {content}")
    print(f"💰 Cost: ${usage['total_cost']:.6f}")
    
    return content, usage

# ============================================================================
# API VERSION COMPATIBILITY MATRIX
# ============================================================================

API_COMPATIBILITY_MATRIX = {
    "bedrock-2023-05-31": {
        "claude_3_haiku": "✅ Supported",
        "claude_3_sonnet": "✅ Supported", 
        "claude_3_opus": "✅ Supported",
        "claude_3_5_sonnet":