"""
Energy Forecasting Performance Dashboard - Database Connection & Base Architecture
Comprehensive database connection and base architecture for day-ahead load forecasting analysis
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
from botocore.exceptions import ClientError
import redshift_connector
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyForecastingDashboard:
    """
    Comprehensive energy forecasting performance dashboard with secure AWS connectivity
    and segment-specific metrics handling for solar and non-solar customers.
    """
    
    def __init__(self, environment: str = None, enable_sample_data: bool = False):
        """
        Initialize dashboard with environment-aware configuration.
        
        Args:
            environment: Target environment (dev/qa/prod). Auto-detects if None.
            enable_sample_data: If True, uses sample data instead of database queries
        """
        self.environment = environment or self._detect_environment()
        self.enable_sample_data = enable_sample_data
        self.connection = None
        self.schema_name = self._get_schema_name()
        # Updated to match your actual secret naming convention
        self.secret_name = f"sdcp-{self.environment}/redshift_cred"
        
        # Performance thresholds by segment type (from evaluation.py)
        self.performance_thresholds = {
            'solar': {  # WAPE-based for solar segments
                'excellent': 15,
                'good': 25,
                'acceptable': 35
            },
            'non_solar': {  # MAPE-based for non-solar segments
                'excellent': 10,
                'good': 20,
                'acceptable': 30
            }
        }
        
        # Time period definitions (from evaluation.py)
        self.time_periods = {
            # Standard periods
            'morning_peak': (6, 9),
            'midday': (10, 15),
            'evening_peak': (16, 21),
            'off_peak': (22, 6),
            # Solar-specific periods
            'solar_ramp_up': (7, 10),
            'solar_peak': (11, 14),
            'duck_curve': (15, 18),
            'solar_evening_peak': (19, 21),
            # Commercial periods
            'business_hours': (8, 18),
            'peak_business': (10, 16)
        }
        
        # Transition threshold for near-zero analysis (from evaluation.py)
        self.transition_threshold = 20000
        
        logger.info(f"Initialized Energy Forecasting Dashboard for {self.environment} environment")
        logger.info(f"Sample data mode: {self.enable_sample_data}")
    
    def _detect_environment(self) -> str:
        """Auto-detect environment from various sources."""
        # Check environment variables
        env = os.getenv('ENVIRONMENT', os.getenv('ENV', 'dev')).lower()
        
        # Validate environment
        valid_envs = ['dev', 'qa', 'prod']
        if env not in valid_envs:
            logger.warning(f"Unknown environment '{env}', defaulting to 'dev'")
            env = 'dev'
        
        return env
    
    def _get_schema_name(self) -> str:
        """Get schema name based on environment."""
        schema_mapping = {
            'dev': 'edp_bi_dev',
            'qa': 'edp_bi_qa',
            'prod': 'edp_bi'
        }
        return schema_mapping.get(self.environment, 'edp_bi_dev')
    
    def _get_secret(self) -> Dict[str, str]:
        """Retrieve database credentials from AWS Secrets Manager using your configuration."""
        try:
            # Use the same initialization as in your notebook
            session = boto3.Session()
            client = session.client('secretsmanager', region_name='us-west-2')
            
            logger.info(f"Retrieving secret: {self.secret_name}")
            response = client.get_secret_value(SecretId=self.secret_name)
            
            secret = json.loads(response['SecretString'])
            logger.info("Successfully retrieved database credentials")
            return secret
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.error(f"Secret {self.secret_name} not found in AWS Secrets Manager")
            elif error_code == 'InvalidRequestException':
                logger.error(f"Invalid request for secret {self.secret_name}")
            elif error_code == 'InvalidParameterException':
                logger.error(f"Invalid parameter for secret {self.secret_name}")
            elif error_code == 'DecryptionFailureException':
                logger.error(f"Cannot decrypt secret {self.secret_name}")
            elif error_code == 'InternalServiceErrorException':
                logger.error(f"Internal service error when retrieving secret {self.secret_name}")
            else:
                logger.error(f"Unexpected error retrieving secret {self.secret_name}: {e}")
            
            if self.environment == 'dev':
                logger.warning("Using sample credentials for development")
                return self._get_sample_credentials()
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret {self.secret_name}: {e}")
            if self.environment == 'dev':
                logger.warning("Using sample credentials for development")
                return self._get_sample_credentials()
            raise
    
    def _get_sample_credentials(self) -> Dict[str, str]:
        """Provide sample credentials for development/testing based on your config."""
        return {
            'host': f'sdcp-edp-backend-{self.environment}.ctmqhtu8ojan.us-west-2.redshift.amazonaws.com',
            'port': '5439',
            'database': 'sdcp',
            'username': 'dev_user',
            'password': 'dev_password'
        }
    
    def connect_to_database(self) -> bool:
        """
        Establish secure connection to Redshift database using your proven configuration.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            credentials = self._get_secret()
            
            # Extract connection parameters matching your notebook structure
            username = credentials['username']
            password = credentials['password']
            database = credentials.get('database', 'sdcp')  # Default from your config
            port = int(credentials.get('port', '5439'))  # Default from your config
            host = credentials.get('host', f'sdcp-edp-backend-{self.environment}.ctmqhtu8ojan.us-west-2.redshift.amazonaws.com')
            
            logger.info(f"Connecting to: {host}:{port}/{database}")
            
            # Build connection parameters exactly like your notebook
            conn_params = {
                'host': host,
                'database': database,
                'user': username,
                'password': password,
                'port': port,
                'ssl': True,
                'timeout': 30
            }
            
            # Create connection using redshift_connector as in your notebook
            self.connection = redshift_connector.connect(**conn_params)
            
            # Test connection with a simple query
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            cursor.close()
            
            if result[0] == 1:
                logger.info(f"Successfully connected to Redshift cluster: sdcp-edp-backend-{self.environment}")
                logger.info(f"Database: {database}, Schema: {self.schema_name}")
                return True
            else:
                logger.error("Connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            if self.environment == 'dev':
                logger.warning("Connection failed, will use sample data for development")
                return False
            raise
    
    def execute_query(self, query: str, params: Tuple = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame using your proven method.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            pd.DataFrame: Query results
        """
        # If sample data is enabled, return sample data immediately
        if self.enable_sample_data:
            logger.info("Sample data mode enabled - returning sample data")
            return self._get_sample_data(query)
        
        try:
            if self.connection is None:
                if not self.connect_to_database():
                    logger.error("Database connection failed and sample data disabled")
                    return pd.DataFrame()  # Return empty DataFrame
            
            # Replace schema placeholder in query
            formatted_query = query.replace('{schema}', self.schema_name)
            
            logger.info(f"Executing query (first 100 chars): {formatted_query[:100]}...")
            
            # Use pandas to execute query and return DataFrame (like your notebook)
            df = pd.read_sql_query(formatted_query, self.connection, params=params)
            
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {formatted_query[:200] if 'formatted_query' in locals() else query[:200]}...")
            if self.enable_sample_data:
                logger.warning("Query failed, returning sample data")
                return self._get_sample_data(query)
            else:
                logger.error("Query failed and sample data disabled")
                return pd.DataFrame()  # Return empty DataFrame
    
    def test_connection(self) -> bool:
        """
        Test connection with a simple query like your notebook.
        
        Returns:
            bool: True if test successful, False otherwise
        """
        if self.connection is None:
            if not self.connect_to_database():
                logger.error("No active connection to test")
                return False
            
        try:
            test_query = """
            SELECT 
                CURRENT_DATE as current_date,
                CURRENT_USER as current_user,
                VERSION() as version
            """
            
            result_df = self.execute_query(test_query)
            
            if result_df is not None and len(result_df) > 0:
                logger.info("Connection test successful:")
                logger.info(f"   Date: {result_df.iloc[0]['current_date']}")
                logger.info(f"   User: {result_df.iloc[0]['current_user']}")
                logger.info(f"   Version: {result_df.iloc[0]['version'][:50]}...")
                return True
            else:
                logger.error("Connection test failed - no results returned")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_schema_info(self) -> pd.DataFrame:
        """
        Get information about available schemas and tables like your notebook.
        
        Returns:
            pandas.DataFrame: Schema information
        """
        if self.connection is None:
            if not self.connect_to_database():
                logger.error("No active connection")
                return pd.DataFrame()
            
        schema_query = f"""
        SELECT 
            schemaname,
            tablename,
            tableowner
        FROM pg_tables 
        WHERE schemaname = '{self.schema_name}'
        ORDER BY tablename
        """
        
        return self.execute_query(schema_query)
    
    def _get_sample_data(self, query: str) -> pd.DataFrame:
        """Generate sample data for testing when database is unavailable."""
        logger.info("Generating sample data for testing")
        
        # Determine data type based on query content
        if 'hourly' in query.lower() or 'vw_fr_hourly_forecast_metrics' in query.lower():
            return self._generate_hourly_sample_data()
        elif 'daily' in query.lower() or 'vw_fr_daily_forecast_summary' in query.lower():
            return self._generate_daily_sample_data()
        elif 'monthly' in query.lower() or 'vw_fr_monthly_performance_trends' in query.lower():
            return self._generate_monthly_sample_data()
        elif 'summary' in query.lower() or 'vw_fr_dashboard_summary' in query.lower():
            return self._generate_summary_sample_data()
        elif 'time_period' in query.lower() or 'vw_fr_time_period_performance' in query.lower():
            return self._generate_time_period_sample_data()
        else:
            return self._generate_summary_sample_data()
    
    def _generate_hourly_sample_data(self) -> pd.DataFrame:
        """Generate sample hourly forecast metrics data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        
        sample_data = []
        for dt in dates:
            for profile in ['RES', 'MEDCI', 'SMLCOM']:
                for segment in ['SOLAR', 'NONSOLAR']:
                    # Generate realistic solar vs non-solar patterns
                    is_solar = segment == 'SOLAR'
                    hour = dt.hour
                    
                    if is_solar:
                        # Solar patterns with duck curve effects
                        base_load = max(0, 50000 + 30000 * np.sin((hour - 6) * np.pi / 12))
                        if 11 <= hour <= 14:  # Solar peak - reduced net load
                            base_load *= 0.3
                        elif 15 <= hour <= 18:  # Duck curve ramp
                            base_load *= 1.5
                    else:
                        # Traditional load patterns
                        base_load = 40000 + 20000 * np.sin((hour - 6) * np.pi / 12)
                    
                    # Add noise and prediction error
                    actual_load = base_load * (1 + np.random.normal(0, 0.1))
                    predicted_load = actual_load * (1 + np.random.normal(0, 0.15 if is_solar else 0.1))
                    
                    # Calculate metrics
                    error = abs(actual_load - predicted_load)
                    pct_error = (error / max(abs(actual_load), 1)) * 100
                    
                    # Use WAPE for solar, MAPE for non-solar
                    primary_metric = pct_error if not is_solar else min(pct_error, 50)
                    
                    # Generate customer segment description
                    customer_type = 'Residential' if profile == 'RES' else ('Medium Commercial' if profile == 'MEDCI' else 'Small Commercial')
                    segment_desc = f"{customer_type} {'Solar' if is_solar else 'Non-Solar'}"
                    
                    sample_data.append({
                        'forecast_datetime': dt,
                        'load_profile': profile,
                        'load_segment': segment,
                        'customer_type': customer_type,
                        'customer_segment_desc': segment_desc,
                        'actual_lossadjustedload': round(actual_load, 2),
                        'predicted_lossadjustedload': round(predicted_load, 2),
                        'absolute_error': round(error, 2),
                        'percentage_error': round(pct_error, 2),
                        'primary_metric': round(primary_metric, 2),
                        'is_solar': is_solar,
                        'hour': hour,
                        'day_type': 'Weekend' if dt.weekday() >= 5 else 'Weekday',
                        'time_period': self._classify_time_period(hour, is_solar),
                        'performance_tier': self._classify_performance(primary_metric, is_solar)
                    })
        
        return pd.DataFrame(sample_data)
    
    def _generate_daily_sample_data(self) -> pd.DataFrame:
        """Generate sample daily summary data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        
        sample_data = []
        for dt in dates:
            for profile in ['RES', 'MEDCI', 'SMLCOM']:
                for segment in ['SOLAR', 'NONSOLAR']:
                    is_solar = segment == 'SOLAR'
                    
                    # Daily aggregated metrics
                    daily_volume = np.random.uniform(800000, 1200000)
                    primary_metric = np.random.uniform(10, 40) if is_solar else np.random.uniform(5, 25)
                    
                    # Generate customer segment description
                    customer_type = 'Residential' if profile == 'RES' else ('Medium Commercial' if profile == 'MEDCI' else 'Small Commercial')
                    segment_desc = f"{customer_type} {'Solar' if is_solar else 'Non-Solar'}"
                    
                    sample_data.append({
                        'forecast_date': dt.date(),
                        'load_profile': profile,
                        'load_segment': segment,
                        'customer_type': customer_type,
                        'customer_segment_desc': segment_desc,
                        'total_actual_mwh': round(daily_volume, 2),
                        'daily_primary_metric': round(primary_metric, 2),
                        'daily_performance_tier': self._classify_performance(primary_metric, is_solar),
                        'success_rate_pct': np.random.uniform(60, 90),
                        'weighted_business_impact_score': np.random.uniform(0.6, 0.95),
                        'is_solar': is_solar
                    })
        
        return pd.DataFrame(sample_data)
    
    def _generate_monthly_sample_data(self) -> pd.DataFrame:
        """Generate sample monthly trend data."""
        months = pd.date_range(start='2023-01-01', end='2024-01-31', freq='M')
        
        sample_data = []
        for dt in months:
            for profile in ['RES', 'MEDCI', 'SMLCOM']:
                for segment in ['SOLAR', 'NONSOLAR']:
                    is_solar = segment == 'SOLAR'
                    
                    # Monthly performance with improvement trend
                    base_metric = 30 if is_solar else 15
                    trend_improvement = (dt.year - 2023) * 12 + dt.month
                    monthly_metric = max(base_metric - trend_improvement * 0.5, 
                                       15 if is_solar else 8)
                    
                    # Generate customer segment description
                    customer_type = 'Residential' if profile == 'RES' else ('Medium Commercial' if profile == 'MEDCI' else 'Small Commercial')
                    segment_desc = f"{customer_type} {'Solar' if is_solar else 'Non-Solar'}"
                    
                    sample_data.append({
                        'year_month': dt.strftime('%Y-%m'),
                        'load_profile': profile,
                        'load_segment': segment,
                        'customer_type': customer_type,
                        'customer_segment_desc': segment_desc,
                        'monthly_primary_metric': round(monthly_metric, 2),
                        'monthly_performance_tier': self._classify_performance(monthly_metric, is_solar),
                        'metric_trend_direction': 'Improving',
                        'metric_change_pct': round(np.random.uniform(-2, -0.5), 2),
                        'overall_trend_classification': 'Improving',
                        'roi_potential': 'High ROI' if monthly_metric < 20 else 'Medium ROI',
                        'is_solar': is_solar
                    })
        
        return pd.DataFrame(sample_data)
    
    def _generate_summary_sample_data(self) -> pd.DataFrame:
        """Generate sample executive summary data."""
        summary_data = []
        
        for profile in ['RES', 'MEDCI', 'SMLCOM']:
            for segment in ['SOLAR', 'NONSOLAR']:
                is_solar = segment == 'SOLAR'
                primary_metric = np.random.uniform(15, 30) if is_solar else np.random.uniform(8, 18)
                
                # Generate customer segment description
                customer_type = 'Residential' if profile == 'RES' else ('Medium Commercial' if profile == 'MEDCI' else 'Small Commercial')
                segment_desc = f"{customer_type} {'Solar' if is_solar else 'Non-Solar'} Customers"
                
                summary_data.append({
                    'load_profile': profile,
                    'load_segment': segment,
                    'customer_type': customer_type,
                    'customer_segment_desc': segment_desc,
                    'current_primary_metric': round(primary_metric, 2),
                    'current_performance_tier': self._classify_performance(primary_metric, is_solar),
                    'deployment_recommendation': self._get_deployment_recommendation(primary_metric, is_solar),
                    'success_rate_pct': np.random.uniform(60, 90),
                    'avg_business_impact_score': np.random.uniform(0.7, 0.9),
                    'total_volume_mwh': np.random.uniform(5000, 15000),
                    'deployment_readiness_score': np.random.uniform(70, 95),
                    'business_priority': 'High Priority' if is_solar else 'Medium Priority',
                    'roi_assessment': 'High ROI' if primary_metric < 20 else 'Medium ROI',
                    'is_solar': is_solar,
                    'sample_size': np.random.randint(5000, 15000)
                })
        
        return pd.DataFrame(summary_data)
    
    def _generate_time_period_sample_data(self) -> pd.DataFrame:
        """Generate sample time period performance data."""
        periods = ['Morning Peak', 'Duck Curve', 'Evening Peak', 'Solar Peak', 'Business Hours']
        customer_segments = ['Residential Solar', 'Residential Non-Solar', 'Commercial Solar', 'Commercial Non-Solar']
        
        sample_data = []
        for period in periods:
            for segment in customer_segments:
                is_solar = 'Solar' in segment
                primary_metric = np.random.uniform(15, 35) if is_solar else np.random.uniform(8, 25)
                
                sample_data.append({
                    'time_period': period,
                    'customer_segment_desc': segment,
                    'period_primary_metric': round(primary_metric, 2),
                    'period_performance_tier': self._classify_performance(primary_metric, is_solar),
                    'success_rate_pct': np.random.uniform(65, 90),
                    'sample_size': np.random.randint(1000, 5000),
                    'business_priority_weight': np.random.uniform(0.1, 0.4),
                    'period_deployment_recommendation': self._get_deployment_recommendation(primary_metric, is_solar),
                    'is_solar': 1 if is_solar else 0,
                    'is_critical_period': 1 if period in ['Duck Curve', 'Evening Peak'] else 0
                })
        
        return pd.DataFrame(sample_data)
    
    def _classify_time_period(self, hour: int, is_solar: bool) -> str:
        """Classify hour into appropriate time period."""
        if is_solar:
            if 7 <= hour < 10:
                return 'Solar Ramp Up'
            elif 11 <= hour < 14:
                return 'Solar Peak'
            elif 15 <= hour < 18:
                return 'Duck Curve'
            elif 19 <= hour < 21:
                return 'Solar Evening Peak'
        
        # Standard periods
        if 6 <= hour < 9:
            return 'Morning Peak'
        elif 10 <= hour < 15:
            return 'Midday'
        elif 16 <= hour < 21:
            return 'Evening Peak'
        else:
            return 'Off Peak'
    
    def _classify_performance(self, metric_value: float, is_solar: bool) -> str:
        """Classify performance based on segment-appropriate thresholds."""
        thresholds = self.performance_thresholds['solar' if is_solar else 'non_solar']
        
        if metric_value <= thresholds['excellent']:
            return 'Excellent'
        elif metric_value <= thresholds['good']:
            return 'Good'
        elif metric_value <= thresholds['acceptable']:
            return 'Acceptable'
        else:
            return 'Needs Improvement'
    
    def _get_deployment_recommendation(self, metric_value: float, is_solar: bool) -> str:
        """Get deployment recommendation based on performance."""
        performance = self._classify_performance(metric_value, is_solar)
        
        recommendations = {
            'Excellent': 'Deploy Immediately',
            'Good': 'Deploy with Monitoring',
            'Acceptable': 'Enhance Before Deployment',
            'Needs Improvement': 'Significant Improvement Needed'
        }
        
        return recommendations.get(performance, 'Review Required')
    
    def close_connection(self):
        """Close database connection safely."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
                self.connection = None
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_to_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()


# =============================================================================
# UTILITY FUNCTIONS FOR METRIC CALCULATIONS
# =============================================================================

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error - better for zero-crossing data.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        WAPE value (lower is better)
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error - handles zero and near-zero values better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        sMAPE value (lower is better)
    """
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def transition_weighted_error(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 20000) -> float:
    """
    Error metric that gives higher weight to transition periods (near zero).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        threshold: Threshold to identify points near zero
        
    Returns:
        Weighted error value (lower is better)
    """
    # Identify transition points (near zero)
    near_zero_mask = np.abs(y_true) < threshold
    
    # Calculate RMSE for all points and transition points
    overall_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # If there are transition points, calculate their RMSE
    if np.any(near_zero_mask):
        transition_rmse = np.sqrt(np.mean((y_true[near_zero_mask] - y_pred[near_zero_mask]) ** 2))
    else:
        transition_rmse = 0
    
    # Weight overall and transition errors (60% overall, 40% transition)
    return 0.6 * overall_rmse + 0.4 * transition_rmse


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the dashboard initialization
    print("Testing Energy Forecasting Dashboard...")
    
    # Test with sample data mode
    print("\n1. Testing Sample Data Mode:")
    dashboard_sample = EnergyForecastingDashboard(environment='dev', enable_sample_data=True)
    
    sample_summary = dashboard_sample._generate_summary_sample_data()
    print(f"Generated {len(sample_summary)} summary records")
    print("\nSample summary data:")
    print(sample_summary.head())
    
    # Test with database mode (will use sample if connection fails)
    print("\n2. Testing Database Mode:")
    dashboard_db = EnergyForecastingDashboard(environment='dev', enable_sample_data=False)
    
    # Test connection
    try:
        connection_success = dashboard_db.connect_to_database()
        if connection_success:
            print(" Database connection successful!")
            
            # Test basic query
            test_success = dashboard_db.test_connection()
            if test_success:
                print(" Database query test successful!")
            else:
                print(" Database query test failed!")
        else:
            print(" Database connection failed!")
    except Exception as e:
        print(f" Database connection error: {e}")
    finally:
        dashboard_db.close_connection()
    
    print("\n3. Testing Metric Calculations:")
    # Test utility functions
    y_true = np.array([100, 200, 300, 10, 500])
    y_pred = np.array([110, 190, 320, 15, 480])
    
    wape_result = calculate_wape(y_true, y_pred)
    smape_result = calculate_smape(y_true, y_pred)
    transition_result = transition_weighted_error(y_true, y_pred, threshold=50)
    
    print(f"WAPE: {wape_result:.2f}%")
    print(f"sMAPE: {smape_result:.2f}%")
    print(f"Transition Weighted Error: {transition_result:.2f}")
    
    print("\nDashboard initialization and testing completed successfully!")
