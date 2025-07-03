"""
Configuration for energy forecasting pipeline - Domain Knowledge & Business Logic
Contains parameters that require deep understanding of energy forecasting business rules.
Operational parameters are now controlled via GitHub Actions environment variables.
"""
import os
import sys

# ============================================================================
# BUSINESS DOMAIN KNOWLEDGE (Keep in config.py)
# ============================================================================

# Query parameters - Business domain logic
LOAD_PROFILES = ["RES", "SMLCOM", "MEDCI", "A6", "LIGHT", "AGR"]
DEFAULT_LOAD_PROFILE = "RES"
SUBMISSION_TYPE_INITIAL = "Initial"
SUBMISSION_TYPE_FINAL = "Final"

# Feature configuration - Domain expertise
DEFAULT_LAG_DAYS = [14, 21]  # Default lag days for reduced feature set
EXTENDED_LAG_DAYS = [14, 21, 28, 35]  # Extended lag days

# Time periods for analysis - Business domain knowledge
MORNING_PEAK_HOURS = (6, 9)  # 6 AM - 9 AM
SOLAR_PERIOD_HOURS = (9, 16)  # 9 AM - 4 PM
EVENING_RAMP_HOURS = (14, 18)  # 2 PM - 6 PM
EVENING_PEAK_HOURS = (17, 21)  # 5 PM - 9 PM

# Processing parameters - Algorithm specifics
OUTLIER_IQR_FACTOR = 1.5  # Factor for outlier detection
EXTREME_IQR_FACTOR = 3.0  # Factor for extreme outlier capping
TEST_DAYS = 30  # Days to include in test set
VALIDATION_DAYS = 60  # Days to include in validation set
MAX_RMSE_THRESHOLD = 99999.0  # Maximum RMSE threshold for deployment

# Folder structure - Technical implementation
PREFIX_ROOT = ""  # "/home/sagemaker-user"
ROOT_FOLDER = f"{PREFIX_ROOT}/opt/ml"
PREPROCESSING_FOLDER = f"{ROOT_FOLDER}/processing"
PREPROCESSING_OUTPUT_FOLDER = f"{PREPROCESSING_FOLDER}/output"
PREPROCESSING_OUTPUT_TRAIN_FOLDER = f"{PREPROCESSING_OUTPUT_FOLDER}/train"
PREPROCESSING_OUTPUT_VAL_FOLDER = f"{PREPROCESSING_OUTPUT_FOLDER}/validation"
PREPROCESSING_OUTPUT_TEST_FOLDER = f"{PREPROCESSING_OUTPUT_FOLDER}/test"
PREPROCESSING_LOG_FOLDER = f"{PREPROCESSING_FOLDER}/logs"

# ============================================================================
# RATE GROUP FILTER CONFIGURATIONS (Business Logic)
# ============================================================================

RATE_GROUP_FILTERS = {
    "RES": {
        "SOLAR": {
            "include": ["NEM", "SBP"],  # List of patterns to include
            "exclude": [],             # List of patterns to exclude
            "operator": "LIKE",        # LIKE or = 
            "logic": "OR"              # AND or OR for multiple include patterns
        },
        "NONSOLAR": {
            "include": [],
            "exclude": ["NEM", "SBP"],
            "operator": "LIKE",
            "logic": "OR"
        }
    },
    "MEDCI": {
        "SOLAR": {
            "include": ["NEM", "SBP"],
            "exclude": [],
            "operator": "LIKE",
            "logic": "OR"
        },
        "NONSOLAR": {
            "include": [],
            "exclude": ["NEM", "SBP"],
            "operator": "LIKE",
            "logic": "OR"
        }
    },
    "SMLCOM": {
        "SOLAR": {
            "include": ["NEM", "SBP"],
            "exclude": [],
            "operator": "LIKE",
            "logic": "OR"
        },
        "NONSOLAR": {
            "include": [],
            "exclude": ["NEM", "SBP"],
            "operator": "LIKE",
            "logic": "OR"
        }
    }
}

# ============================================================================
# PROFILE AND SEGMENT SPECIFIC CONFIGURATIONS (Business Domain)
# ============================================================================

PROFILE_CONFIGS = {
    "RES": {
        "SOLAR": {
            "METER_THRESHOLD": 100,
            "USE_SOLAR_FEATURES": True,
            "LOAD_PROFILE": "RES",
            "MODEL_BASE_NAME": "res-solar",
            "BASE_JOB_NAME": "res-solar-load-forecasting"
        },
        "NONSOLAR": {
            "METER_THRESHOLD": 100,
            "USE_SOLAR_FEATURES": False,
            "LOAD_PROFILE": "RES",
            "MODEL_BASE_NAME": "res-nonsolar",
            "BASE_JOB_NAME": "res-nonsolar-load-forecasting"
        }
    },
    "MEDCI": {
        "SOLAR": {
            "METER_THRESHOLD": 50,
            "USE_SOLAR_FEATURES": True,
            "LOAD_PROFILE": "MEDCI",
            "MODEL_BASE_NAME": "medci-solar",
            "BASE_JOB_NAME": "medci-solar-load-forecasting"
        },
        "NONSOLAR": {
            "METER_THRESHOLD": 50,
            "USE_SOLAR_FEATURES": False,
            "LOAD_PROFILE": "MEDCI",
            "MODEL_BASE_NAME": "medci-nonsolar",
            "BASE_JOB_NAME": "medci-nonsolar-load-forecasting"
        }
    },
    "SMLCOM": {
        "SOLAR": {
            "METER_THRESHOLD": 30,
            "USE_SOLAR_FEATURES": True,
            "LOAD_PROFILE": "SMLCOM",
            "MODEL_BASE_NAME": "smlcom-solar",
            "BASE_JOB_NAME": "smlcom-solar-load-forecasting"
        },
        "NONSOLAR": {
            "METER_THRESHOLD": 30,
            "USE_SOLAR_FEATURES": False,
            "LOAD_PROFILE": "SMLCOM",
            "MODEL_BASE_NAME": "smlcom-nonsolar",
            "BASE_JOB_NAME": "smlcom-nonsolar-load-forecasting"
        }
    }
}

# ============================================================================
# ENHANCED CONFIGURATION LOADING (Integration with Environment Variables)
# ============================================================================

def get_config_for_profile_segment(profile="RES", segment="SOLAR"):
    """
    Get configuration for a specific customer profile and segment.
    Now integrates with environment variables from deploy.yml.
    
    Args:
        profile (str): Customer profile (RES, MEDCI, SMLCOM)
        segment (str): Customer segment (SOLAR, NONSOLAR)
        
    Returns:
        dict: Combined configuration dictionary
    """
    # Start with domain knowledge from this module
    config_dict = {}
    current_module = sys.modules[__name__]
    for key in dir(current_module):
        if key.isupper():
            config_dict[key] = getattr(current_module, key)
    
    # Add customer profile and segment identifiers
    config_dict["CUSTOMER_PROFILE"] = profile
    config_dict["CUSTOMER_SEGMENT"] = segment
   
    # Create dynamic prefix
    dynamic_prefix = f"{profile}-{segment.upper()}"
    config_dict["S3_PREFIX"] = dynamic_prefix

    # Apply profile and segment specific configurations (business logic)
    if profile in PROFILE_CONFIGS and segment in PROFILE_CONFIGS[profile]:
        profile_config = PROFILE_CONFIGS[profile][segment]
        config_dict.update(profile_config)
    
    # Build rate group filter clause (business logic)
    rate_group_filter = build_rate_group_filter(profile, segment)
    if rate_group_filter:
        config_dict["RATE_GROUP_FILTER_CLAUSE"] = rate_group_filter
    
    # ========================================================================
    # OPERATIONAL PARAMETERS FROM ENVIRONMENT (deploy.yml controls these)
    # ========================================================================
    
    # AWS Configuration (from deploy.yml)
    config_dict.update({
        "ENVIRONMENT": os.environ.get('ENVIRONMENT', 'dev'),
        "AWS_REGION": os.environ.get('AWS_REGION', 'us-west-2'),
        "S3_BUCKET": os.environ.get('S3_BUCKET', f'sdcp-{os.environ.get("ENVIRONMENT", "dev")}-sagemaker-energy-forecasting-data'),
    })
    
    # Database Configuration (from deploy.yml)
    database_type = os.environ.get('DATABASE_TYPE', 'redshift')
    config_dict.update({
        "DATABASE_TYPE": database_type,
        "REDSHIFT_CLUSTER_IDENTIFIER": os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER'),
        "REDSHIFT_DATABASE": os.environ.get('REDSHIFT_DATABASE', 'sdcp'),
        "REDSHIFT_DB_USER": os.environ.get('REDSHIFT_DB_USER', 'ds_service_user'),
        "REDSHIFT_REGION": os.environ.get('REDSHIFT_REGION', 'us-west-2'),
        "REDSHIFT_INPUT_SCHEMA": os.environ.get('REDSHIFT_INPUT_SCHEMA'),
        "REDSHIFT_INPUT_TABLE": os.environ.get('REDSHIFT_INPUT_TABLE', 'caiso_sqmd'),
        "REDSHIFT_OUTPUT_SCHEMA": os.environ.get('REDSHIFT_OUTPUT_SCHEMA'),
        "REDSHIFT_OUTPUT_TABLE": os.environ.get('REDSHIFT_OUTPUT_TABLE', 'dayahead_load_forecasts'),
        "REDSHIFT_BI_SCHEMA": os.environ.get('REDSHIFT_BI_SCHEMA', 'edp_forecasting'),
        "REDSHIFT_BI_VIEW_NAME": os.environ.get('REDSHIFT_BI_VIEW_NAME', 'vw_dayahead_forecasts'),
        "REDSHIFT_BI_VIEW": os.environ.get('REDSHIFT_BI_VIEW', 'vw_dayahead_forecasts'),
        "ATHENA_DATABASE": os.environ.get('ATHENA_DATABASE'),
        "ATHENA_TABLE": os.environ.get('ATHENA_TABLE', 'raw_agg_caiso_sqmd'),
    })
    
    # SageMaker Instance Configuration (from deploy.yml)
    config_dict.update({
        "PREPROCESSING_INSTANCE_TYPE": os.environ.get('PREPROCESSING_INSTANCE_TYPE', 'ml.m5.large'),
        "PREPROCESSING_INSTANCE_COUNT": int(os.environ.get('PREPROCESSING_INSTANCE_COUNT', '1')),
        "PREPROCESSING_FRAMEWORK_VERSION": os.environ.get('PREPROCESSING_FRAMEWORK_VERSION', '1.0-1'),
        "TRAINING_INSTANCE_TYPE": os.environ.get('TRAINING_INSTANCE_TYPE', 'ml.m5.large'),
        "TRAINING_INSTANCE_COUNT": int(os.environ.get('TRAINING_INSTANCE_COUNT', '1')),
        "INFERENCE_INSTANCE_TYPE": os.environ.get('INFERENCE_INSTANCE_TYPE', 'ml.m5.large'),
        "INFERENCE_INSTANCE_COUNT": int(os.environ.get('INFERENCE_INSTANCE_COUNT', '1')),
    })
    
    # Data Processing Configuration (from deploy.yml)
    config_dict.update({
        "INITIAL_SUBMISSION_DELAY": int(os.environ.get('INITIAL_SUBMISSION_DELAY', '14')),
        "FINAL_SUBMISSION_DELAY": int(os.environ.get('FINAL_SUBMISSION_DELAY', '48')),
        "DEFAULT_METER_THRESHOLD": int(os.environ.get('METER_THRESHOLD', '1000')),
        "USE_CSV_CACHE": os.environ.get('USE_CACHE', 'true').lower() == 'true',
    })
    
    # Update S3 bucket paths with the profile/segment specific prefix
    s3_bucket = config_dict["S3_BUCKET"]
    config_dict.update({
        "PREPROCESSING_S3_BUCKET": f"s3://{s3_bucket}/{dynamic_prefix}/processed",
        "TRAINED_S3_BUCKET_MODELS": f"s3://{s3_bucket}/{dynamic_prefix}/models",
        "EVALUATION_S3_BUCKET": f"s3://{s3_bucket}/{dynamic_prefix}/evaluation",
        "DEPLOY_S3_BUCKET": f"s3://{s3_bucket}/{dynamic_prefix}/deployment",
        "SCRIPTS_S3_BUCKET": f"s3://{s3_bucket}/{dynamic_prefix}/scripts",
        "FORECASTS_S3_BUCKET": f"s3://{s3_bucket}/{dynamic_prefix}/forecasts",
    })
    
    # Update dynamic S3 paths
    config_dict["PREPROCESSING_S3_BUCKET_TRAIN"] = f"{config_dict['PREPROCESSING_S3_BUCKET']}/training"
    config_dict["PREPROCESSING_S3_BUCKET_VAL"] = f"{config_dict['PREPROCESSING_S3_BUCKET']}/validation"
    config_dict["PREPROCESSING_S3_BUCKET_TEST"] = f"{config_dict['PREPROCESSING_S3_BUCKET']}/test"
    
    # Update default endpoint name
    config_dict["DEFAULT_ENDPOINT_NAME"] = f"{profile_config.get('MODEL_BASE_NAME', 'energy')}-forecasting-endpoint"
    
    # Base job names to include profile and segment
    config_dict["PREPROCESSING_BASE_JOB_NAME"] = f"{profile.lower()}-{segment}-load-preprocessing"
    
    # Database-specific settings
    if database_type == 'redshift':
        config_dict.update({
            "OUTPUT_METHOD": "redshift",
            "PRIMARY_DATABASE": config_dict["REDSHIFT_DATABASE"],
            "PRIMARY_SCHEMA": config_dict["REDSHIFT_INPUT_SCHEMA"],
            "PRIMARY_TABLE": config_dict["REDSHIFT_INPUT_TABLE"],
            "PRIMARY_FULL_TABLE_NAME": f"{config_dict['REDSHIFT_INPUT_SCHEMA']}.{config_dict['REDSHIFT_INPUT_TABLE']}",
        })
    else:  # athena
        config_dict.update({
            "OUTPUT_METHOD": "athena",
            "PRIMARY_DATABASE": config_dict["ATHENA_DATABASE"],
            "PRIMARY_SCHEMA": "public",
            "PRIMARY_TABLE": config_dict["ATHENA_TABLE"],
            "PRIMARY_FULL_TABLE_NAME": f"{config_dict['ATHENA_DATABASE']}.{config_dict['ATHENA_TABLE']}",
        })
    
    config_dict.update({
        # Weather Configuration for Forecasting
        "WEATHER_VARIABLES": [
            "temperature_2m", "apparent_temperature", "precipitation", "rain",
            "cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
            "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
            "shortwave_radiation", "windspeed_10m", "winddirection_10m",
            "windgusts_10m", "is_day", "relativehumidity_2m"
        ],
        
        # Location Configuration
        "DEFAULT_LATITUDE": 32.7157,
        "DEFAULT_LONGITUDE": -117.1611,
        "DEFAULT_TIMEZONE": "America/Los_Angeles",
        
        # Lambda Runtime Configuration
        "LOG_LEVEL": "INFO",
        
        # Output Configuration
        "OUTPUT_FULL_TABLE_NAME": f"{config_dict['REDSHIFT_OUTPUT_SCHEMA']}.{config_dict['REDSHIFT_OUTPUT_TABLE']}",
    })

    
    return config_dict

def build_rate_group_filter(profile, segment):
    """
    Build a SQL WHERE clause for rate group filtering based on profile and segment.
    
    Args:
        profile (str): Customer profile (RES, MEDCI, SMLCOM)
        segment (str): Customer segment (SOLAR, NONSOLAR)
        
    Returns:
        str: SQL WHERE clause for rate group filtering or empty string if no filtering needed
    """
    # Get filter config for this profile/segment
    if (profile not in RATE_GROUP_FILTERS or 
        segment not in RATE_GROUP_FILTERS[profile]):
        return ""  # No filtering
        
    filter_config = RATE_GROUP_FILTERS[profile][segment]
    
    # Include patterns
    include_patterns = filter_config.get("include", [])
    include_clause = ""
    
    if include_patterns:
        operator = filter_config.get("operator", "LIKE")
        logic = filter_config.get("logic", "OR")
        
        # Build include clause
        if operator == "LIKE":
            # Add % for LIKE operator
            clauses = [f"rategroup LIKE '{pattern}%'" for pattern in include_patterns]
        else:
            # Exact match
            clauses = [f"rategroup = '{pattern}'" for pattern in include_patterns]
            
        include_clause = f"({f' {logic} '.join(clauses)})"
    
    # Exclude patterns
    exclude_patterns = filter_config.get("exclude", [])
    exclude_clause = ""
    
    if exclude_patterns:
        operator = filter_config.get("operator", "LIKE")
        
        # Build exclude clause
        if operator == "LIKE":
            # Add % for LIKE operator
            clauses = [f"rategroup NOT LIKE '{pattern}%'" for pattern in exclude_patterns]
        else:
            # Exact match
            clauses = [f"rategroup != '{pattern}'" for pattern in exclude_patterns]
            
        exclude_clause = f" ({' AND '.join(clauses)})"
    
    # Combine include and exclude
    if include_clause and exclude_clause:
        return f"{include_clause} AND {exclude_clause}"
    elif include_clause:
        return include_clause
    elif exclude_clause:
        return exclude_clause
    else:
        return ""  # No filtering

# ============================================================================
# CUSTOMER SEGMENT CONFIGURATIONS (Complex Business Logic)
# ============================================================================

CUSTOMER_SEGMENTS = {
    "RES_SOLAR": {
        "profile": "RES",
        "has_solar": True,
        "evaluation_periods": {
            "morning_high": (6, 9),           # Before solar kicks in
            "solar_ramp_down": (9, 12),       # Solar generation increasing
            "midday_low": (10, 14),           # Peak solar/minimum net load
            "duck_curve_critical": (14, 18),  # Solar declining, demand rising
            "evening_peak": (17, 21),         # Peak net demand
            "night_baseload": (21, 6)         # Night hours
        },
        "metric_weights": {
            "duck_curve_critical": 0.35,      # Most critical transition
            "evening_peak": 0.25,             # High demand period
            "midday_low": 0.15,               # Net metering period
            "solar_ramp_down": 0.1,
            "morning_high": 0.1,
            "night_baseload": 0.05
        },
        "solar_specific_metrics": True,
        "duck_curve_analysis": True,
        "commercial_metrics": False
    },
    "RES_NONSOLAR": {
        "profile": "RES",
        "has_solar": False,
        "evaluation_periods": {
            "morning_consumption": (6, 10),   # Morning usage
            "midday_steady": (10, 14),        # Stable midday (no solar dip)
            "afternoon_build": (14, 17),      # Pre-peak buildup
            "evening_super_peak": (17, 21),   # CRITICAL: Massive evening peak
            "night_baseload": (21, 6)         # Night hours
        },
        "metric_weights": {
            "evening_super_peak": 0.6,        # Dominant pattern - highest weight
            "afternoon_build": 0.2,           # Pre-peak ramp accuracy
            "morning_consumption": 0.1,
            "midday_steady": 0.05,
            "night_baseload": 0.05
        },
        "solar_specific_metrics": False,
        "duck_curve_analysis": False,
        "commercial_metrics": False,
        "evening_peak_focus": True           # New flag for evening peak analysis
    },
    "MEDCI_SOLAR": {
        "profile": "MEDCI",
        "has_solar": True,
        "evaluation_periods": {
            "business_morning": (7, 10),
            "business_solar_peak": (10, 14),  # Business hours + solar offset
            "business_afternoon": (14, 17),   # Post-solar business hours
            "business_evening": (17, 19),     # End of business day
            "off_hours": (19, 7)              # Non-business hours
        },
        "metric_weights": {
            "business_solar_peak": 0.3,
            "business_afternoon": 0.3,        # Critical post-solar period
            "business_morning": 0.2,
            "business_evening": 0.15,
            "off_hours": 0.05
        },
        "solar_specific_metrics": True,
        "duck_curve_analysis": True,
        "commercial_metrics": True
    },
    "MEDCI_NONSOLAR": {
        "profile": "MEDCI", 
        "has_solar": False,
        "evaluation_periods": {
            "business_hours": (8, 18),
            "morning_transition": (6, 8),     # Earlier startup
            "evening_transition": (18, 20),   # Business shutdown
            "off_hours": (20, 6),             # Night + early morning
            "weekend": "weekend"              # Special weekend handling
        },
        "metric_weights": {
            "business_hours": 0.6,            # Increased weight
            "morning_transition": 0.15,
            "evening_transition": 0.15,
            "off_hours": 0.05,
            "weekend": 0.05
        },
        "solar_specific_metrics": False,
        "duck_curve_analysis": False,
        "commercial_metrics": True
    },
    "SMLCOM_SOLAR": {
        "profile": "SMLCOM",
        "has_solar": True,
        "evaluation_periods": {
            "business_morning": (8, 11),
            "business_solar_peak": (11, 15),  # Small commercial + solar
            "business_afternoon": (15, 17),   
            "business_end": (17, 18),
            "off_hours": (18, 8)
        },
        "metric_weights": {
            "business_solar_peak": 0.3,
            "business_afternoon": 0.3,
            "business_morning": 0.2,
            "business_end": 0.15,
            "off_hours": 0.05
        },
        "solar_specific_metrics": True,
        "duck_curve_analysis": True,
        "commercial_metrics": True
    },
    "SMLCOM_NONSOLAR": {
        "profile": "SMLCOM",
        "has_solar": False,
        "evaluation_periods": {
            "business_hours": (8, 17),        # Typical small business hours
            "morning_transition": (7, 8),   
            "evening_transition": (17, 19),
            "off_hours": (19, 7),
            "weekend": "weekend"
        },
        "metric_weights": {
            "business_hours": 0.6,
            "morning_transition": 0.15,
            "evening_transition": 0.15,
            "off_hours": 0.05,
            "weekend": 0.05
        },
        "solar_specific_metrics": False,
        "duck_curve_analysis": False,
        "commercial_metrics": True
    }
}

# Segment-specific time periods for evaluation
SEGMENT_EVALUATION_PERIODS = {
    "RES_SOLAR": {
        "morning_high": (6, 9),
        "solar_ramp_down": (9, 12),
        "midday_low": (10, 14),
        "duck_curve_critical": (14, 18),  # Most critical for solar
        "evening_peak": (17, 21),
        "night_baseload": (21, 6)
    },
    "RES_NONSOLAR": {
        "morning_consumption": (6, 10),
        "midday_steady": (10, 14),
        "afternoon_build": (14, 17),
        "evening_super_peak": (17, 21),  # Most critical for non-solar
        "night_baseload": (21, 6)
    },
    "MEDCI_SOLAR": {
        "business_morning": (7, 10),
        "business_solar_peak": (10, 14),
        "business_afternoon": (14, 17),
        "business_evening": (17, 19),
        "off_hours": (19, 7)
    },
    "MEDCI_NONSOLAR": {
        "business_hours": (8, 18),  # Extended based on actual patterns
        "morning_transition": (6, 8),
        "evening_transition": (18, 20),
        "off_hours": (20, 6)
    },
    "SMLCOM_SOLAR": {
        "business_morning": (8, 11),
        "business_solar_peak": (11, 15),
        "business_afternoon": (15, 17),
        "business_end": (17, 18),
        "off_hours": (18, 8)
    },
    "SMLCOM_NONSOLAR": {
        "business_hours": (8, 17),
        "morning_transition": (7, 8),
        "evening_transition": (17, 19),
        "off_hours": (19, 7)
    }
}

# Segment-specific metric weights based on observed patterns
SEGMENT_METRIC_WEIGHTS = {
    "RES_SOLAR": {
        "duck_curve_critical": 0.35,    # Highest - most complex transition
        "evening_peak": 0.25,
        "midday_low": 0.15,             # Net metering critical
        "solar_ramp_down": 0.1,
        "morning_high": 0.1,
        "night_baseload": 0.05
    },
    "RES_NONSOLAR": {
        "evening_super_peak": 0.6,      # Dominant pattern - massive evening surge
        "afternoon_build": 0.2,         # Pre-peak buildup accuracy
        "morning_consumption": 0.1,
        "midday_steady": 0.05,
        "night_baseload": 0.05
    },
    "MEDCI_SOLAR": {
        "business_solar_peak": 0.3,
        "business_afternoon": 0.3,
        "business_morning": 0.2,
        "business_evening": 0.15,
        "off_hours": 0.05
    },
    "MEDCI_NONSOLAR": {
        "business_hours": 0.6,          # Increased weight for flat business pattern
        "morning_transition": 0.15,
        "evening_transition": 0.15,
        "off_hours": 0.1
    },
    "SMLCOM_SOLAR": {
        "business_solar_peak": 0.3,
        "business_afternoon": 0.3,
        "business_morning": 0.2,
        "business_end": 0.15,
        "off_hours": 0.05
    },
    "SMLCOM_NONSOLAR": {
        "business_hours": 0.6,
        "morning_transition": 0.15,
        "evening_transition": 0.15,
        "off_hours": 0.1
    }
}

# Priority metrics for each segment (for model selection/comparison)
PRIORITY_METRICS = {
    "RES_SOLAR": ["rmse_duck_curve_critical", "mape_evening_peak", "rmse_midday_low", "r2_overall"],
    "RES_NONSOLAR": ["rmse_evening_super_peak", "mape_evening_super_peak", "rmse_afternoon_build", "r2_overall"],
    "MEDCI_SOLAR": ["rmse_business_solar_peak", "mape_business_afternoon", "rmse_business_morning", "r2_overall"],
    "MEDCI_NONSOLAR": ["rmse_business_hours", "mape_business_hours", "rmse_morning_transition", "r2_overall"],
    "SMLCOM_SOLAR": ["rmse_business_solar_peak", "mape_business_afternoon", "rmse_business_morning", "r2_overall"],
    "SMLCOM_NONSOLAR": ["rmse_business_hours", "mape_business_hours", "rmse_morning_transition", "r2_overall"]
}

# ============================================================================
# HELPER FUNCTIONS FOR CUSTOMER SEGMENT CONFIGURATIONS
# ============================================================================

def get_segment_config(customer_segment="RES_SOLAR"):
    """
    Get complete configuration for a specific customer segment.
    
    Args:
        customer_segment (str): Customer segment identifier
        
    Returns:
        dict: Configuration dictionary for the segment
    """
    base_config = {
        "evaluation_periods": SEGMENT_EVALUATION_PERIODS.get(customer_segment, {}),
        "metric_weights": SEGMENT_METRIC_WEIGHTS.get(customer_segment, {}),
        "priority_metrics": PRIORITY_METRICS.get(customer_segment, [])
    }
    
    # Add segment-specific flags
    if customer_segment.endswith("_SOLAR"):
        base_config.update({
            "has_solar": True,
            "solar_specific_metrics": True,
            "duck_curve_analysis": True
        })
    else:
        base_config.update({
            "has_solar": False,
            "solar_specific_metrics": False,
            "duck_curve_analysis": False
        })
    
    if customer_segment.startswith(("MEDCI", "SMLCOM")):
        base_config["commercial_metrics"] = True
    else:
        base_config["commercial_metrics"] = False
    
    if customer_segment == "RES_NONSOLAR":
        base_config["evening_peak_focus"] = True
    
    return base_config

def get_periods_to_evaluate(customer_segment="RES_SOLAR"):
    """Get evaluation periods specific to customer segment."""
    return SEGMENT_EVALUATION_PERIODS.get(customer_segment, SEGMENT_EVALUATION_PERIODS["RES_SOLAR"])

def get_evaluation_weights(customer_segment="RES_SOLAR"):
    """Get evaluation period weights for detailed analysis."""
    return SEGMENT_METRIC_WEIGHTS.get(customer_segment, SEGMENT_METRIC_WEIGHTS["RES_SOLAR"])

def get_metric_weights(customer_segment="RES_SOLAR"):
    """Get metric weights for hyperparameter optimization scoring."""
    return SEGMENT_METRIC_WEIGHTS.get(customer_segment, SEGMENT_METRIC_WEIGHTS["RES_SOLAR"])

def get_priority_metrics(customer_segment="RES_SOLAR"):
    """Get priority metrics for model selection and comparison."""
    return PRIORITY_METRICS.get(customer_segment, PRIORITY_METRICS["RES_SOLAR"])

def enhance_config_for_forecasting(config_dict):
    """Add forecasting-specific configurations"""
    database_type = config_dict["DATABASE_TYPE"]
    # Database-specific settings
    if database_type == 'redshift':
        config_dict.update({
            "OUTPUT_METHOD": "redshift",
        })
    else:  # athena
        config_dict.update({
            "OUTPUT_METHOD": "athena",
        })

    config_dict.update({
        # Weather Configuration for Forecasting
        "WEATHER_VARIABLES": [
            "temperature_2m", "apparent_temperature", "precipitation", "rain",
            "cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
            "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
            "shortwave_radiation", "windspeed_10m", "winddirection_10m",
            "windgusts_10m", "is_day", "relativehumidity_2m"
        ],
        
        # Location Configuration
        "DEFAULT_LATITUDE": 32.7157,
        "DEFAULT_LONGITUDE": -117.1611,
        "DEFAULT_TIMEZONE": "America/Los_Angeles",
        
        # Lambda Runtime Configuration
        "LOG_LEVEL": "INFO",
        
        # Output Configuration
        "OUTPUT_FULL_TABLE_NAME": f"{config_dict['REDSHIFT_OUTPUT_SCHEMA']}.{config_dict['REDSHIFT_OUTPUT_TABLE']}",
    })
    return config_dict

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Create default configuration for backward compatibility
DEFAULT_CONFIG = get_config_for_profile_segment("RES", "SOLAR")

# Make key config values available at module level for backward compatibility
globals().update({k: v for k, v in DEFAULT_CONFIG.items() if k.isupper()})