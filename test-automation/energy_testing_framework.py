#!/usr/bin/env python3
"""
Agentic AI Unit Test Generation Framework for Energy Forecasting Codebase

A cost-optimized, sequential multi-agent system for automated unit test generation
achieving 85%+ code coverage while maintaining budget constraints.
"""

import ast
import json
import logging
import os
import re
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_testing_framework.log'),
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
        """Calculate cost based on Claude 3.5 Sonnet v2 pricing (December 2024)"""
        # Updated pricing for Claude 3.5 Sonnet v2 via Bedrock
        input_cost = self.input_tokens * 0.003 / 1000   # $3.00 per 1M input tokens
        output_cost = self.output_tokens * 0.015 / 1000  # $15.00 per 1M output tokens
        self.total_cost = input_cost + output_cost
        return self.total_cost


@dataclass
class FileAnalysis:
    """Results from code analysis agent"""
    file_path: str
    complexity_score: int
    functions: List[Dict]
    classes: List[Dict]
    imports: List[str]
    aws_services: List[str]
    external_apis: List[str]
    dependencies: Dict[str, List[str]]
    estimated_test_lines: int
    criticality_score: int


@dataclass
class TestStrategy:
    """Results from strategy planning agent"""
    file_path: str
    test_approach: str
    mock_requirements: Dict[str, List[str]]
    test_cases_needed: List[Dict]
    coverage_targets: Dict[str, int]
    estimated_tokens: int


@dataclass
class GeneratedTests:
    """Results from test generation agent"""
    file_path: str
    test_content: str
    coverage_achieved: float
    quality_score: float
    tokens_used: TokenUsage


@dataclass
class ProcessingState:
    """Track processing state for incremental updates"""
    file_path: str
    last_modified: float
    last_processed: float
    current_coverage: float
    processing_status: str
    cost_spent: float


class BudgetExceededException(Exception):
    """Raised when budget limits are exceeded"""
    pass


class InsufficientProgressException(Exception):
    """Raised when quality improvement is insufficient"""
    pass


class BudgetMonitor:
    """Monitor and control budget usage"""
   
    def __init__(self, max_budget_usd: float):
        self.max_budget = max_budget_usd
        self.current_spend = 0.0
        self.token_costs = {
            'input_tokens': 0.003 / 1000,   # Claude 3.5 Sonnet v2: $3.00 per 1M tokens
            'output_tokens': 0.015 / 1000   # Claude 3.5 Sonnet v2: $15.00 per 1M tokens
        }
        self.usage_log = []
        logger.info(f"Budget monitor initialized: ${max_budget_usd:.2f} limit")
        logger.info(f"Pricing: ${self.token_costs['input_tokens']*1000:.3f}/1K input, ${self.token_costs['output_tokens']*1000:.3f}/1K output")
       
    def estimate_cost(self, estimated_tokens: int) -> float:
        """Estimate cost for given token count"""
        # Conservative estimate: 70% input, 30% output tokens
        input_tokens = int(estimated_tokens * 0.7)
        output_tokens = int(estimated_tokens * 0.3)
       
        cost = (input_tokens * self.token_costs['input_tokens'] +
                output_tokens * self.token_costs['output_tokens'])
        return cost
   
    def can_afford(self, estimated_tokens: int) -> bool:
        """Check if we can afford the estimated token usage"""
        estimated_cost = self.estimate_cost(estimated_tokens)
        return (self.current_spend + estimated_cost) <= self.max_budget
   
    def record_usage(self, token_usage: TokenUsage):
        """Record actual token usage"""
        token_usage.calculate_cost()
        self.current_spend += token_usage.total_cost
        self.usage_log.append(token_usage)

        # Log detailed cost breakdown for transparency
        logger.info(f"Token usage: {token_usage.input_tokens} input + {token_usage.output_tokens} output = ${token_usage.total_cost:.4f}")
        logger.info(f"Running total: ${self.current_spend:.2f} / ${self.max_budget:.2f} ({(self.current_spend/self.max_budget)*100:.1f}%)")

        # Warn when approaching budget limit
        remaining_budget = self.max_budget - self.current_spend
        if remaining_budget < self.max_budget * 0.1:  # Less than 10% remaining
            logger.warning(f"Budget warning: Only ${remaining_budget:.2f} remaining!")
       
        if self.current_spend > self.max_budget:
            raise BudgetExceededException(
                f"Budget exceeded: ${self.current_spend:.2f} > ${self.max_budget:.2f}"
            )
   
    def get_remaining_budget(self) -> float:
        """Get remaining budget"""
        return max(0, self.max_budget - self.current_spend)
   
    def get_usage_summary(self) -> Dict:
        """Get detailed usage summary"""
        total_input_tokens = sum(u.input_tokens for u in self.usage_log)
        total_output_tokens = sum(u.output_tokens for u in self.usage_log)
       
        return {
            'total_spent': self.current_spend,
            'remaining_budget': self.get_remaining_budget(),
            'budget_utilization': (self.current_spend / self.max_budget) * 100,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'average_cost_per_call': self.current_spend / max(1, len(self.usage_log)),
            'calls_made': len(self.usage_log)
        }


class StateManager:
    """Manage processing state for incremental updates"""
   
    def __init__(self, state_file: str = "test_generation_state.json"):
        self.state_file = Path(state_file)
        self.state: Dict[str, ProcessingState] = self.load_state()
   
    def load_state(self) -> Dict[str, ProcessingState]:
        """Load processing state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return {
                    path: ProcessingState(**state_data)
                    for path, state_data in data.items()
                }
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
        return {}
   
    def save_state(self):
        """Save processing state to file"""
        try:
            data = {
                path: asdict(state)
                for path, state in self.state.items()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
   
    def should_process_file(self, file_path: str, target_coverage: float = 85.0) -> bool:
        """Determine if file should be processed"""
        file_path = str(Path(file_path).resolve())
       
        # Check if file exists in state
        if file_path not in self.state:
            return True
       
        state = self.state[file_path]
       
        # Skip if already has sufficient coverage
        if state.current_coverage >= target_coverage:
            logger.info(f"Skipping {file_path}: already has {state.current_coverage:.1f}% coverage")
            return False
       
        # Check if file was modified since last processing
        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > state.last_processed:
                logger.info(f"File {file_path} modified since last processing")
                return True
        except OSError:
            logger.warning(f"Cannot access file modification time: {file_path}")
            return True
       
        # Process if previous attempt failed
        if state.processing_status == "failed":
            return True
       
        return False
   
    def update_file_state(self, file_path: str, coverage: float,
                         status: str, cost: float):
        """Update processing state for a file"""
        file_path = str(Path(file_path).resolve())
       
        try:
            current_mtime = os.path.getmtime(file_path)
        except OSError:
            current_mtime = time.time()
       
        self.state[file_path] = ProcessingState(
            file_path=file_path,
            last_modified=current_mtime,
            last_processed=time.time(),
            current_coverage=coverage,
            processing_status=status,
            cost_spent=cost
        )
        self.save_state()


class ClaudeClient:
    """AWS Bedrock Claude client with token tracking"""
   
    def __init__(self, region_name: str = "us-west-2"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
   
    def call_claude(self, prompt: str, max_tokens: int = 4000) -> Tuple[str, TokenUsage]:
        """Call Claude with token tracking"""
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "top_p": 0.9
                })
            )
           
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
           
            # Extract actual token usage from Bedrock response
            # Bedrock returns usage in the response metadata
            actual_input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            actual_output_tokens = response_body.get('usage', {}).get('output_tokens', 0)

            # If usage not in main response, check if it's in the response metadata
            if actual_input_tokens == 0 or actual_output_tokens == 0:
                # Some Bedrock responses put usage in different locations
                metadata = response.get('ResponseMetadata', {})
                usage_info = metadata.get('HTTPHeaders', {}).get('x-amzn-bedrock-usage', {})
                if usage_info:
                    actual_input_tokens = usage_info.get('inputTokens', actual_input_tokens)
                    actual_output_tokens = usage_info.get('outputTokens', actual_output_tokens)

            # Create token usage with actual values
            token_usage = TokenUsage(
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                timestamp=datetime.now().isoformat()
            )

            # Log if we had to fall back to estimates
            if actual_input_tokens == 0 or actual_output_tokens == 0:
                logger.warning("Could not get actual token usage from Bedrock response, using estimates")
                # Fall back to rough estimates only if actual usage unavailable
                token_usage.input_tokens = max(actual_input_tokens, int(len(prompt) / 4))  # ~4 chars per token
                token_usage.output_tokens = max(actual_output_tokens, int(len(content) / 4))
           
            # Debug: Log actual vs estimated tokens
            logger.info(f"Bedrock token usage - Input: {token_usage.input_tokens}, Output: {token_usage.output_tokens}")
            logger.info(f"Prompt length: {len(prompt)} chars, Response length: {len(content)} chars")
            logger.info(f"Cost calculated: ${token_usage.calculate_cost():.4f}")
           
            return content, token_usage
           
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            # Return empty response with estimated usage for error tracking
            return "", TokenUsage(
                input_tokens=self._estimate_tokens(prompt),
                output_tokens=0,
                timestamp=datetime.now().isoformat()
            )
   
    def test_token_accuracy(self, test_prompt="Hello, how are you?"):
        """Test method to verify token counting accuracy"""
        logger.info("Testing token counting accuracy...")
   
        response, token_usage = self.call_claude(test_prompt, max_tokens=100)
   
        # Manual verification
        estimated_input = len(test_prompt) / 4  # Rough estimate: 4 chars per token
        estimated_output = len(response) / 4
   
        logger.info(f"Test prompt: '{test_prompt}'")
        logger.info(f"Actual input tokens: {token_usage.input_tokens}")
        logger.info(f"Estimated input tokens: {estimated_input:.0f}")
        logger.info(f"Actual output tokens: {token_usage.output_tokens}")
        logger.info(f"Estimated output tokens: {estimated_output:.0f}")
        logger.info(f"Total cost: ${token_usage.total_cost:.4f}")
   
        return token_usage


    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for Claude 3.5 Sonnet v2"""
        # Claude 3.5 Sonnet v2 uses approximately 4 characters per token
        # This is more accurate than word-based estimation
        return max(1, len(text) // 4)


class BaseAgent(ABC):
    """Base class for all agents"""
   
    def __init__(self, claude_client: ClaudeClient, budget_monitor: BudgetMonitor):
        self.claude_client = claude_client
        self.budget_monitor = budget_monitor
        self.name = self.__class__.__name__
   
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process agent-specific task"""
        pass
   
    def _call_claude_with_budget_check(self, prompt: str, max_tokens: int = 4000) -> str:
        """Call Claude with budget verification"""
        # Estimate token usage
        estimated_tokens = len(prompt.split()) * 1.3 + max_tokens
       
        if not self.budget_monitor.can_afford(int(estimated_tokens)):
            raise BudgetExceededException(
                f"{self.name} cannot afford estimated {estimated_tokens} tokens"
            )
       
        response, token_usage = self.claude_client.call_claude(prompt, max_tokens)
        self.budget_monitor.record_usage(token_usage)
       
        logger.info(f"{self.name} used {token_usage.input_tokens} input + "
                   f"{token_usage.output_tokens} output tokens (${token_usage.total_cost:.4f})")
       
        return response


class CodeAnalysisAgent(BaseAgent):
    """Analyze code structure and dependencies"""
   
    def process(self, file_path: str) -> FileAnalysis:
        """Analyze a Python file for testing requirements"""
        logger.info(f"Analyzing code structure: {file_path}")
       
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return self._create_empty_analysis(file_path)
       
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return self._create_empty_analysis(file_path)
       
        # Extract basic information
        functions = self._extract_functions(tree)
        classes = self._extract_classes(tree)
        imports = self._extract_imports(tree)
       
        # Analyze for AWS services and external APIs
        aws_services = self._detect_aws_services(content)
        external_apis = self._detect_external_apis(content)
       
        # Calculate complexity and criticality
        complexity_score = self._calculate_complexity(tree, content)
        criticality_score = self._calculate_criticality(file_path, aws_services, functions, classes)
       
        # Use Claude for detailed dependency analysis
        analysis_prompt = self._create_analysis_prompt(content, file_path)
        claude_analysis = self._call_claude_with_budget_check(analysis_prompt, 2000)
       
        # Parse Claude's analysis
        dependencies = self._parse_dependencies(claude_analysis)
        estimated_test_lines = self._estimate_test_lines(functions, classes, complexity_score)
       
        return FileAnalysis(
            file_path=file_path,
            complexity_score=complexity_score,
            functions=functions,
            classes=classes,
            imports=imports,
            aws_services=aws_services,
            external_apis=external_apis,
            dependencies=dependencies,
            estimated_test_lines=estimated_test_lines,
            criticality_score=criticality_score
        )
   
    def _create_empty_analysis(self, file_path: str) -> FileAnalysis:
        """Create empty analysis for failed files"""
        return FileAnalysis(
            file_path=file_path,
            complexity_score=0,
            functions=[],
            classes=[],
            imports=[],
            aws_services=[],
            external_apis=[],
            dependencies={},
            estimated_test_lines=0,
            criticality_score=0
        )
   
    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function definitions"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'line_number': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'decorators': [ast.dump(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node)
                })
        return functions
   
    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class definitions"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'line_number': item.lineno,
                            'args': [arg.arg for arg in item.args.args],
                            'is_property': any('property' in ast.dump(d) for d in item.decorator_list)
                        })
               
                classes.append({
                    'name': node.name,
                    'line_number': node.lineno,
                    'bases': [ast.dump(base) for base in node.bases],
                    'methods': methods,
                    'decorators': [ast.dump(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node)
                })
        return classes
   
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
   
    def _detect_aws_services(self, content: str) -> List[str]:
        """Detect AWS services used in the code"""
        aws_patterns = {
            'boto3': r'boto3\.client\([\'"]([^\'"]+)[\'"]',
            's3': r'[\'"]s3[\'"]',
            'sagemaker': r'[\'"]sagemaker[\'"]',
            'redshift': r'[\'"]redshift[\'"]',
            'athena': r'[\'"]athena[\'"]',
            'bedrock': r'[\'"]bedrock[\'"]',
            'redshift-data': r'[\'"]redshift-data[\'"]'
        }
       
        detected_services = set()
        for service, pattern in aws_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected_services.add(service)
       
        return list(detected_services)
   
    def _detect_external_apis(self, content: str) -> List[str]:
        """Detect external APIs used in the code"""
        api_patterns = {
            'openmeteo': r'open-meteo|openmeteo',
            'requests': r'requests\.',
            'urllib': r'urllib\.',
            'http': r'http\.'
        }
       
        detected_apis = set()
        for api, pattern in api_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected_apis.add(api)
       
        return list(detected_apis)
   
    def _calculate_complexity(self, tree: ast.AST, content: str) -> int:
        """Calculate code complexity score"""
        complexity = 0
       
        # Count various complexity indicators
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += len(node.args.args)  # Parameter complexity
            elif isinstance(node, ast.ClassDef):
                complexity += 2  # Classes add complexity
       
        # Add line count factor
        line_count = len(content.splitlines())
        complexity += line_count // 100  # 1 point per 100 lines
       
        return complexity
   
    def _calculate_criticality(self, file_path: str, aws_services: List[str],
                             functions: List[Dict], classes: List[Dict]) -> int:
        """Calculate file criticality score"""
        criticality = 0
       
        # Path-based criticality
        path_lower = file_path.lower()
        if 'config' in path_lower:
            criticality += 10
        if 'pipeline' in path_lower or 'orchestration' in path_lower:
            criticality += 8
        if 'model' in path_lower or 'training' in path_lower:
            criticality += 6
       
        # AWS service usage increases criticality
        criticality += len(aws_services) * 2
       
        # Function and class count
        criticality += len(functions) + len(classes) * 2
       
        return criticality
   
    def _create_analysis_prompt(self, content: str, file_path: str) -> str:
        """Create prompt for Claude analysis"""
        # Truncate content if too long to save tokens
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated for analysis]"
       
        return f"""
Analyze this Python file for unit testing requirements:

File: {file_path}

```python
{content}
```

Please provide a JSON response with the following structure:
{{
    "dependencies": {{
        "internal_modules": ["list of internal module dependencies"],
        "external_libraries": ["list of external library dependencies"],
        "aws_services": ["list of AWS services used"],
        "configuration_dependencies": ["list of config dependencies"]
    }},
    "testing_challenges": ["list of potential testing challenges"],
    "mock_requirements": ["list of components that need mocking"],
    "test_complexity": "low|medium|high"
}}

Focus on identifying dependencies that will need mocking for unit tests.
"""
   
    def _parse_dependencies(self, claude_response: str) -> Dict[str, List[str]]:
        """Parse dependencies from Claude's response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', claude_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('dependencies', {})
        except json.JSONDecodeError:
            logger.warning("Failed to parse Claude's dependency analysis")
       
        return {}
   
    def _estimate_test_lines(self, functions: List[Dict], classes: List[Dict],
                           complexity_score: int) -> int:
        """Estimate number of test lines needed"""
        # Base estimate: 10 lines per function, 20 lines per class
        base_lines = len(functions) * 10 + len(classes) * 20
       
        # Complexity multiplier
        complexity_multiplier = 1 + (complexity_score / 100)
       
        return int(base_lines * complexity_multiplier)


class TestStrategyAgent(BaseAgent):
    """Plan testing strategy and approach"""
   
    def process(self, analysis: FileAnalysis) -> TestStrategy:
        """Create testing strategy based on code analysis"""
        logger.info(f"Planning test strategy for: {analysis.file_path}")
       
        strategy_prompt = self._create_strategy_prompt(analysis)
        claude_response = self._call_claude_with_budget_check(strategy_prompt, 3000)
       
        # Parse strategy from Claude's response
        strategy_data = self._parse_strategy_response(claude_response)
       
        return TestStrategy(
            file_path=analysis.file_path,
            test_approach=strategy_data.get('approach', 'standard'),
            mock_requirements=strategy_data.get('mock_requirements', {}),
            test_cases_needed=strategy_data.get('test_cases', []),
            coverage_targets=strategy_data.get('coverage_targets', {}),
            estimated_tokens=strategy_data.get('estimated_tokens', 5000)
        )
   
    def _create_strategy_prompt(self, analysis: FileAnalysis) -> str:
        """Create strategy planning prompt"""
        return f"""
Create a comprehensive testing strategy for this Python file:

File: {analysis.file_path}
Complexity Score: {analysis.complexity_score}
Functions: {len(analysis.functions)}
Classes: {len(analysis.classes)}
AWS Services: {analysis.aws_services}
External APIs: {analysis.external_apis}

Functions to test:
{json.dumps(analysis.functions, indent=2)}

Classes to test:
{json.dumps(analysis.classes, indent=2)}

Dependencies:
{json.dumps(analysis.dependencies, indent=2)}

Please provide a comprehensive testing strategy in JSON format:
{{
    "approach": "standard|aws_heavy|config_focused|ml_pipeline",
    "mock_requirements": {{
        "aws_services": ["list of AWS services to mock"],
        "external_apis": ["list of external APIs to mock"],
        "internal_modules": ["list of internal modules to mock"],
        "file_system": ["list of file operations to mock"]
    }},
    "test_cases": [
        {{
            "type": "unit|integration|edge_case|error_handling",
            "target": "function_or_class_name",
            "description": "what this test validates",
            "mock_setup": ["required mocks for this test"],
            "expected_coverage": 85
        }}
    ],
    "coverage_targets": {{
        "overall": 85,
        "functions": 90,
        "classes": 80,
        "error_handling": 70
    }},
    "testing_priorities": ["ordered list of testing priorities"],
    "estimated_tokens": 5000
}}

Focus on achieving 85%+ code coverage while maintaining practical, maintainable tests.
For energy forecasting domain, pay special attention to:
- Configuration validation
- Data processing pipelines
- AWS service integrations
- Time series operations
- Error handling and edge cases
"""
   
    def _parse_strategy_response(self, claude_response: str) -> Dict:
        """Parse strategy response from Claude"""
        try:
            json_match = re.search(r'\{.*\}', claude_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse Claude's strategy response")
       
        # Return default strategy
        return {
            'approach': 'standard',
            'mock_requirements': {},
            'test_cases': [],
            'coverage_targets': {'overall': 85},
            'estimated_tokens': 5000
        }


class TestGeneratorAgent(BaseAgent):
    """Generate actual unit test code"""
   
    def process(self, analysis: FileAnalysis, strategy: TestStrategy) -> GeneratedTests:
        """Generate unit tests based on analysis and strategy"""
        logger.info(f"Generating tests for: {analysis.file_path}")
       
        # Read the actual source code
        try:
            with open(analysis.file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read source file: {e}")
            return self._create_empty_test_result(analysis.file_path)
       
        # Generate tests using Claude
        generation_prompt = self._create_generation_prompt(analysis, strategy, source_code)
        claude_response = self._call_claude_with_budget_check(generation_prompt, 6000)
       
        # Extract test code from response
        test_content = self._extract_test_code(claude_response)
       
        # Estimate coverage (basic estimation)
        coverage_achieved = self._estimate_coverage(test_content, analysis)
       
        # Calculate quality score
        quality_score = self._calculate_quality_score(test_content)
       
        return GeneratedTests(
            file_path=analysis.file_path,
            test_content=test_content,
            coverage_achieved=coverage_achieved,
            quality_score=quality_score,
            tokens_used=TokenUsage()  # Will be populated by budget monitor
        )
   
    def _create_empty_test_result(self, file_path: str) -> GeneratedTests:
        """Create empty test result for failed generation"""
        return GeneratedTests(
            file_path=file_path,
            test_content="# Test generation failed",
            coverage_achieved=0.0,
            quality_score=0.0,
            tokens_used=TokenUsage()
        )
   
    def _create_generation_prompt(self, analysis: FileAnalysis, strategy: TestStrategy,
                                source_code: str) -> str:
        """Create test generation prompt"""
        # Truncate source code if too long
        if len(source_code) > 10000:
            source_code = source_code[:10000] + "\n... [truncated]"
       
        return f"""
Generate comprehensive unit tests for this Python file:

File: {analysis.file_path}

Source Code:
```python
{source_code}
```

Testing Strategy:
{json.dumps(asdict(strategy), indent=2)}

Requirements:
1. Use pytest framework
2. Achieve 85%+ code coverage
3. Follow Black formatting standards
4. Pass Flake8 linting
5. Include proper mocking for external dependencies
6. Add comprehensive docstrings
7. Test edge cases and error conditions

For AWS services, use boto3 mocking patterns like:
```python
import boto3
from moto import mock_s3, mock_sagemaker
import pytest

@mock_s3
def test_s3_operation():
    # Test implementation
    pass
```

For energy forecasting domain, focus on:
- Configuration validation and dynamic loading
- Data processing with various input scenarios
- Time series operations and edge cases
- Error handling for external service failures
- Integration with AWS services (S3, SageMaker, Redshift)

Generate ONLY the test file content. Include:
1. All necessary imports
2. Test fixtures
3. Mock setups
4. Comprehensive test cases
5. Proper error handling tests

The output should be a complete, runnable pytest file.
"""
   
    def _extract_test_code(self, claude_response: str) -> str:
        """Extract test code from Claude's response"""
        # Look for code blocks
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, claude_response, re.DOTALL)
       
        if matches:
            return matches[0]
       
        # If no code blocks found, look for test patterns
        lines = claude_response.split('\n')
        test_lines = []
        in_test_code = False
       
        for line in lines:
            if line.strip().startswith(('import ', 'def test_', 'class Test', '@pytest', '@mock')):
                in_test_code = True
           
            if in_test_code:
                test_lines.append(line)
       
        if test_lines:
            return '\n'.join(test_lines)
       
        # Fallback: return the entire response
        return claude_response
   
    def _estimate_coverage(self, test_content: str, analysis: FileAnalysis) -> float:
        """Estimate test coverage based on generated content"""
        if not test_content or test_content.strip() == "# Test generation failed":
            return 0.0
       
        # Count test functions
        test_functions = len(re.findall(r'def test_\w+', test_content))
       
        # Count functions and methods in source
        total_functions = len(analysis.functions)
        for cls in analysis.classes:
            total_functions += len(cls.get('methods', []))
       
        if total_functions == 0:
            return 50.0  # Default for config files
       
        # Estimate coverage based on test to function ratio
        coverage_ratio = test_functions / total_functions
        estimated_coverage = min(95.0, coverage_ratio * 100)
       
        # Adjust based on content quality
        if 'mock' in test_content.lower():
            estimated_coverage += 10
        if 'edge' in test_content.lower() or 'error' in test_content.lower():
            estimated_coverage += 5
       
        return min(95.0, estimated_coverage)
   
    def _calculate_quality_score(self, test_content: str) -> float:
        """Calculate test quality score"""
        if not test_content or test_content.strip() == "# Test generation failed":
            return 0.0
       
        quality_score = 0.0
       
        # Check for pytest patterns
        if 'import pytest' in test_content:
            quality_score += 20
       
        # Check for proper test structure
        if re.search(r'def test_\w+.*:', test_content):
            quality_score += 20
       
        # Check for mocking
        if any(pattern in test_content for pattern in ['@mock', 'mock_', 'Mock()']):
            quality_score += 20
       
        # Check for assertions
        assert_count = len(re.findall(r'assert ', test_content))
        quality_score += min(20, assert_count * 2)
       
        # Check for docstrings
        if '"""' in test_content or "'''" in test_content:
            quality_score += 10
       
        # Check for fixtures
        if '@pytest.fixture' in test_content:
            quality_score += 10
       
        return min(100.0, quality_score)


class QualityValidatorAgent(BaseAgent):
    """Validate test quality and coverage"""
   
    def process(self, generated_tests: GeneratedTests, target_coverage: float = 85.0) -> Tuple[bool, Dict[str, Any]]:
        """Validate generated tests for quality and coverage"""
        logger.info(f"Validating test quality for: {generated_tests.file_path}")
       
        validation_results = {
            'coverage_check': False,
            'quality_check': False,
            'formatting_check': False,
            'linting_check': False,
            'syntax_check': False,
            'issues': [],
            'suggestions': []
        }
       
        # Check syntax
        try:
            compile(generated_tests.test_content, '<test>', 'exec')
            validation_results['syntax_check'] = True
        except SyntaxError as e:
            validation_results['issues'].append(f"Syntax error: {e}")
       
        # Check coverage estimate
        if generated_tests.coverage_achieved >= target_coverage:
            validation_results['coverage_check'] = True
        else:
            validation_results['issues'].append(
                f"Coverage {generated_tests.coverage_achieved:.1f}% below target {target_coverage}%"
            )
       
        # Check quality score
        if generated_tests.quality_score >= 70:
            validation_results['quality_check'] = True
        else:
            validation_results['issues'].append(
                f"Quality score {generated_tests.quality_score:.1f} below threshold 70"
            )
       
        # Validate with Claude for detailed analysis
        if generated_tests.test_content and len(generated_tests.test_content) > 100:
            validation_prompt = self._create_validation_prompt(generated_tests)
            claude_response = self._call_claude_with_budget_check(validation_prompt, 2000)
           
            claude_validation = self._parse_validation_response(claude_response)
            validation_results.update(claude_validation)
       
        # Overall validation
        all_passed = all([
            validation_results['coverage_check'],
            validation_results['quality_check'],
            validation_results['syntax_check']
        ])
       
        return all_passed, validation_results
   
    def _create_validation_prompt(self, generated_tests: GeneratedTests) -> str:
        """Create validation prompt for Claude"""
        return f"""
Validate this generated test file for quality and completeness:

File being tested: {generated_tests.file_path}
Estimated coverage: {generated_tests.coverage_achieved:.1f}%
Quality score: {generated_tests.quality_score:.1f}

Test Content:
```python
{generated_tests.test_content[:5000]}  # Truncated for analysis
```

Please analyze and provide feedback in JSON format:
{{
    "formatting_issues": ["list of formatting issues"],
    "missing_test_cases": ["list of missing test scenarios"],
    "mock_quality": "good|needs_improvement|poor",
    "coverage_gaps": ["areas not covered by tests"],
    "suggestions": ["suggestions for improvement"],
    "overall_assessment": "excellent|good|needs_work|poor"
}}

Focus on:
1. Are all functions and classes tested?
2. Are mocks properly implemented?
3. Are edge cases covered?
4. Is error handling tested?
5. Are pytest best practices followed?
"""
   
    def _parse_validation_response(self, claude_response: str) -> Dict[str, Any]:
        """Parse validation response from Claude"""
        try:
            json_match = re.search(r'\{.*\}', claude_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse Claude's validation response")
       
        return {}


class IntegrationAgent(BaseAgent):
    """Integrate and finalize test generation"""
   
    def process(self, generated_tests: GeneratedTests, validation_results: Dict[str, Any],
                output_dir: str = "tests") -> Tuple[bool, str]:
        """Save generated tests and create integration reports"""
        logger.info(f"Integrating tests for: {generated_tests.file_path}")
       
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
       
        # Generate test file name
        source_file = Path(generated_tests.file_path)
        test_file_name = f"test_{source_file.stem}.py"
        test_file_path = output_path / test_file_name
       
        # Save test file
        try:
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(generated_tests.test_content)
           
            logger.info(f"Test file saved: {test_file_path}")
           
            # Apply Black formatting
            self._apply_black_formatting(test_file_path)
           
            # Run basic validation
            success = self._run_basic_validation(test_file_path)
           
            return success, str(test_file_path)
           
        except Exception as e:
            logger.error(f"Failed to save test file: {e}")
            return False, ""
   
    def _apply_black_formatting(self, test_file_path: Path):
        """Apply Black formatting to test file"""
        try:
            result = subprocess.run(
                ['black', str(test_file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"Applied Black formatting to {test_file_path}")
            else:
                logger.warning(f"Black formatting issues: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Black formatter not available or timed out")
   
    def _run_basic_validation(self, test_file_path: Path) -> bool:
        """Run basic validation on test file"""
        try:
            # Check syntax
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
           
            compile(content, str(test_file_path), 'exec')
            logger.info(f"Syntax validation passed for {test_file_path}")
            return True
           
        except SyntaxError as e:
            logger.error(f"Syntax validation failed: {e}")
            return False


class EnergyTestingFramework:
    """Main framework orchestrating all agents"""
   
    def __init__(self, budget_limit_usd: float, target_coverage: float = 85.0,
                 aws_region: str = "us-west-2"):
        self.budget_monitor = BudgetMonitor(budget_limit_usd)
        self.state_manager = StateManager()
        self.target_coverage = target_coverage
       
        # Initialize Claude client
        self.claude_client = ClaudeClient(aws_region)
       
        # Initialize agents
        self.code_analyzer = CodeAnalysisAgent(self.claude_client, self.budget_monitor)
        self.strategy_planner = TestStrategyAgent(self.claude_client, self.budget_monitor)
        self.test_generator = TestGeneratorAgent(self.claude_client, self.budget_monitor)
        self.quality_validator = QualityValidatorAgent(self.claude_client, self.budget_monitor)
        self.integration_agent = IntegrationAgent(self.claude_client, self.budget_monitor)
       
        self.results = {
            'processed_files': [],
            'skipped_files': [],
            'failed_files': [],
            'total_coverage_achieved': 0.0,
            'total_cost': 0.0
        }
   
    def process_file(self, file_path: str, output_dir: str = "tests") -> Dict[str, Any]:
        """Process a single file through the complete pipeline"""
        logger.info(f"Processing file: {file_path}")
       
        file_result = {
            'file_path': file_path,
            'status': 'failed',
            'coverage_achieved': 0.0,
            'cost_spent': 0.0,
            'test_file_path': '',
            'issues': []
        }
       
        try:
            # Check if file should be processed
            if not self.state_manager.should_process_file(file_path, self.target_coverage):
                file_result['status'] = 'skipped'
                self.results['skipped_files'].append(file_result)
                return file_result
           
            # Step 1: Code Analysis
            analysis = self.code_analyzer.process(file_path)
            if analysis.complexity_score == 0:
                file_result['issues'].append("Code analysis failed")
                self.results['failed_files'].append(file_result)
                return file_result
           
            # Step 2: Strategy Planning
            strategy = self.strategy_planner.process(analysis)
           
            # Step 3: Test Generation
            generated_tests = self.test_generator.process(analysis, strategy)
            if not generated_tests.test_content or len(generated_tests.test_content) < 100:
                file_result['issues'].append("Test generation failed")
                self.results['failed_files'].append(file_result)
                return file_result
           
            # Step 4: Quality Validation
            validation_passed, validation_results = self.quality_validator.process(
                generated_tests, self.target_coverage
            )
           
            # Step 5: Integration
            integration_success, test_file_path = self.integration_agent.process(
                generated_tests, validation_results, output_dir
            )
           
            # Update results
            file_result.update({
                'status': 'success' if validation_passed and integration_success else 'completed_with_issues',
                'coverage_achieved': generated_tests.coverage_achieved,
                'cost_spent': sum(usage.total_cost for usage in self.budget_monitor.usage_log[-10:]),  # Approximate
                'test_file_path': test_file_path,
                'issues': validation_results.get('issues', [])
            })
           
            # Update state
            self.state_manager.update_file_state(
                file_path,
                generated_tests.coverage_achieved,
                file_result['status'],
                file_result['cost_spent']
            )
           
            self.results['processed_files'].append(file_result)
           
        except BudgetExceededException as e:
            logger.error(f"Budget exceeded processing {file_path}: {e}")
            file_result['issues'].append(str(e))
            self.results['failed_files'].append(file_result)
            raise  # Re-raise to stop processing
           
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            file_result['issues'].append(str(e))
            self.results['failed_files'].append(file_result)
       
        return file_result
   
    def process_directory(self, directory_path: str, recursive: bool = False,
                         output_dir: str = "tests") -> Dict[str, Any]:
        """Process all Python files in a directory"""
        logger.info(f"Processing directory: {directory_path} (recursive={recursive})")
       
        # Discover Python files
        files = self._discover_python_files(directory_path, recursive)
       
        # Prioritize files by criticality and cost
        prioritized_files = self._prioritize_files(files)
       
        logger.info(f"Found {len(files)} Python files to process")
       
        # Process files in priority order
        for file_path in prioritized_files:
            try:
                # Check budget before processing each file
                if self.budget_monitor.get_remaining_budget() < 5.0:  # $5 minimum
                    logger.warning("Insufficient budget remaining, stopping processing")
                    break
               
                self.process_file(file_path, output_dir)
               
            except BudgetExceededException:
                logger.error("Budget exceeded, stopping directory processing")
                break
       
        # Calculate summary statistics
        total_files = len(self.results['processed_files'])
        total_coverage = sum(f['coverage_achieved'] for f in self.results['processed_files'])
        average_coverage = total_coverage / max(1, total_files)
       
        self.results.update({
            'total_coverage_achieved': average_coverage,
            'total_cost': self.budget_monitor.current_spend,
            'budget_remaining': self.budget_monitor.get_remaining_budget()
        })
       
        return self.results
   
    def _discover_python_files(self, directory_path: str, recursive: bool) -> List[str]:
        """Discover Python files in directory"""
        python_files = []
        directory = Path(directory_path)
       
        pattern = "**/*.py" if recursive else "*.py"
       
        for file_path in directory.glob(pattern):
            if file_path.is_file() and not file_path.name.startswith('test_'):
                python_files.append(str(file_path))
       
        return python_files
   
    def _prioritize_files(self, files: List[str]) -> List[str]:
        """Prioritize files by criticality and estimated cost"""
        file_priorities = []
       
        for file_path in files:
            # Quick analysis for prioritization
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
               
                # Calculate priority factors
                criticality = self._calculate_file_criticality(file_path, content)
                complexity = len(content.splitlines())
                estimated_cost = complexity * 0.01  # Rough estimate
               
                # Priority score (higher = more important)
                priority_score = criticality - (estimated_cost / 100)
               
                file_priorities.append((priority_score, file_path))
               
            except Exception:
                # If file can't be read, give it low priority
                file_priorities.append((0, file_path))
       
        # Sort by priority (highest first)
        file_priorities.sort(reverse=True)
       
        return [file_path for _, file_path in file_priorities]
   
    def _calculate_file_criticality(self, file_path: str, content: str) -> int:
        """Calculate file criticality for prioritization"""
        criticality = 0
        path_lower = file_path.lower()
       
        # Path-based criticality
        if 'config' in path_lower:
            criticality += 100
        elif 'pipeline' in path_lower or 'orchestration' in path_lower:
            criticality += 80
        elif 'model' in path_lower or 'training' in path_lower:
            criticality += 60
        elif 'preprocessing' in path_lower:
            criticality += 40
       
        # Content-based criticality
        if 'boto3' in content or 'aws' in content.lower():
            criticality += 20
        if 'class ' in content:
            criticality += 10
        if 'def ' in content:
            criticality += 5
       
        return criticality
   
    def generate_report(self, output_file: str = "test_generation_report.json"):
        """Generate comprehensive processing report"""
        report = {
            'summary': {
                'total_files_processed': len(self.results['processed_files']),
                'total_files_skipped': len(self.results['skipped_files']),
                'total_files_failed': len(self.results['failed_files']),
                'average_coverage': self.results.get('total_coverage_achieved', 0),
                'total_cost': self.budget_monitor.current_spend,
                'budget_utilization': (self.budget_monitor.current_spend / self.budget_monitor.max_budget) * 100,
                'processing_timestamp': datetime.now().isoformat()
            },
            'budget_details': self.budget_monitor.get_usage_summary(),
            'file_details': {
                'processed': self.results['processed_files'],
                'skipped': self.results['skipped_files'],
                'failed': self.results['failed_files']
            },
            'coverage_by_file': {
                f['file_path']: f['coverage_achieved']
                for f in self.results['processed_files']
            }
        }
       
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
       
        logger.info(f"Processing report saved to: {output_file}")
        return report


# Usage example and CLI interface
def main():
    """Main CLI interface"""
    import argparse
   
    parser = argparse.ArgumentParser(description="Energy Testing Framework - Automated Unit Test Generation")
    parser.add_argument("target", help="File or directory to process")
    parser.add_argument("--budget", type=float, default=50.0, help="Budget limit in USD (default: $50)")
    parser.add_argument("--coverage", type=float, default=85.0, help="Target coverage percentage (default: 85)")
    parser.add_argument("--output", default="tests", help="Output directory for tests (default: tests)")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--aws-region", default="us-west-2", help="AWS region for Bedrock (default: us-west-2)")
   
    args = parser.parse_args()
   
    # Initialize framework
    framework = EnergyTestingFramework(
        budget_limit_usd=args.budget,
        target_coverage=args.coverage,
        aws_region=args.aws_region
    )
   
    # Process target
    target_path = Path(args.target)
   
    if target_path.is_file():
        logger.info(f"Processing single file: {target_path}")
        result = framework.process_file(str(target_path), args.output)
        print(f"File processing result: {result}")
    elif target_path.is_dir():
        logger.info(f"Processing directory: {target_path}")
        results = framework.process_directory(str(target_path), args.recursive, args.output)
        print(f"Directory processing completed. Processed {len(results['processed_files'])} files.")
    else:
        print(f"Error: {target_path} is not a valid file or directory")
        return 1
   
    # Generate report
    report = framework.generate_report()
   
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total Cost: ${report['summary']['total_cost']:.2f}")
    print(f"Budget Utilization: {report['summary']['budget_utilization']:.1f}%")
    print(f"Average Coverage: {report['summary']['average_coverage']:.1f}%")
    print(f"Files Processed: {report['summary']['total_files_processed']}")
    print(f"Files Skipped: {report['summary']['total_files_skipped']}")
    print(f"Files Failed: {report['summary']['total_files_failed']}")
   
    return 0


if __name__ == "__main__":
    exit(main())
