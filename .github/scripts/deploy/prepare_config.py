# .github/scripts/prepare_config.py
import json
import os
import sys
import argparse
from pathlib import Path

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create a JSON configuration file from the config module.')
    parser.add_argument('output', help='Output JSON file path')
    parser.add_argument('--profile', default='RES', help='Customer profile (RES, MEDCI, SMLCOM)')
    parser.add_argument('--segment', default='RES_SOLAR', help='Customer segment (solar, nonsolar)')
    args = parser.parse_args()
    
    output_file = args.output
    customer_profile = args.profile
    customer_segment = args.segment
    
    try:
        # Try to import the enhanced config module with get_config_for_profile_segment
        from configs import config
        
        # Check if the profile/segment-specific function exists
        if hasattr(config, 'get_config_for_profile_segment'):
            # Use the profile/segment-specific function
            config_dict = config.get_config_for_profile_segment(customer_profile, customer_segment)

            # Override with environment variables (from deploy.yml)
            env_overrides = {
                # Core settings
                'ENVIRONMENT': os.environ.get('ENVIRONMENT', config_dict.get('ENVIRONMENT')),
                'AWS_REGION': os.environ.get('AWS_REGION', config_dict.get('AWS_REGION')),
                'S3_BUCKET': os.environ.get('S3_BUCKET', config_dict.get('S3_BUCKET')),
                'DATABASE_TYPE': os.environ.get('DATABASE_TYPE', config_dict.get('DATABASE_TYPE')),
                
                # Data processing parameters
                'DAYS_DELAY': int(os.environ.get('DAYS_DELAY', config_dict.get('INITIAL_SUBMISSION_DELAY', 14))),
                'USE_REDUCED_FEATURES': os.environ.get('USE_REDUCED_FEATURES', 'false').lower() == 'true',
                'METER_THRESHOLD': int(os.environ.get('METER_THRESHOLD', config_dict.get('DEFAULT_METER_THRESHOLD', 1000))),
                'USE_WEATHER': os.environ.get('USE_WEATHER', 'true').lower() == 'true',
                'USE_SOLAR': os.environ.get('USE_SOLAR', 'true').lower() == 'true',
                'USE_CACHE': os.environ.get('USE_CACHE', 'true').lower() == 'true',
                'WEATHER_CACHE': os.environ.get('WEATHER_CACHE', 'true').lower() == 'true',
                
                # Training parameters
                'FEATURE_SEL_METHOD': os.environ.get('FEATURE_SEL_METHOD', 'importance'),
                'FEATURE_COUNT': int(os.environ.get('FEATURE_COUNT', 40)),
                'CORRELATION_THRESHOLD': float(os.environ.get('CORRELATION_THRESHOLD', 85)) / 100,
                'HPO_METHOD': os.environ.get('HPO_METHOD', 'bayesian'),
                'HPO_MAX_EVALS': int(os.environ.get('HPO_MAX_EVALS', 50)),
                'CV_FOLDS': int(os.environ.get('CV_FOLDS', 5)),
                'CV_GAP_DAYS': int(os.environ.get('CV_GAP_DAYS', os.environ.get('DAYS_DELAY', 14))),
                'ENABLE_MULTI_MODEL': os.environ.get('ENABLE_MULTI_MODEL', 'false').lower() == 'true',
                
                # Infrastructure settings
                'PREPROCESSING_INSTANCE_TYPE': os.environ.get('PREPROCESSING_INSTANCE_TYPE', 'ml.m5.large'),
                'TRAINING_INSTANCE_TYPE': os.environ.get('TRAINING_INSTANCE_TYPE', 'ml.m5.large'),
                'EVALUATION_INSTANCE_TYPE': os.environ.get('EVALUATION_INSTANCE_TYPE', 'ml.m5.large'),
                
                # Database settings
                'REDSHIFT_CLUSTER_IDENTIFIER': os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER'),
                'REDSHIFT_DATABASE': os.environ.get('REDSHIFT_DATABASE'),
                'REDSHIFT_INPUT_SCHEMA': os.environ.get('REDSHIFT_INPUT_SCHEMA'),
                'REDSHIFT_INPUT_TABLE': os.environ.get('REDSHIFT_INPUT_TABLE'),
            }
    
            # Update config dict with environment overrides
            config_dict.update({k: v for k, v in env_overrides.items() if v is not None})

            print(f"Using profile-specific configuration for {customer_profile}-{customer_segment}")
        else:
            # Fall back to the old approach - just get all uppercase variables
            print(f"Using standard configuration (no profile-specific function found)")
            config_dict = {key: getattr(config, key) for key in dir(config) if key.isupper()}
            
            # Add profile and segment manually
            config_dict['CUSTOMER_PROFILE'] = customer_profile
            config_dict['CUSTOMER_SEGMENT'] = customer_segment
            
            # Set S3 prefix to include profile and segment
            config_dict['S3_PREFIX_FULL'] = f"{config_dict.get('S3_PREFIX', 'res-load-forecasting')}/{customer_profile}/{customer_segment}"
            
            # Apply some hardcoded defaults based on profile and segment
            if customer_profile == 'RES':
                if customer_segment == 'solar':
                    config_dict['METER_THRESHOLD'] = 1000
                    config_dict['USE_SOLAR_FEATURES'] = True
                else:  # nonsolar
                    config_dict['METER_THRESHOLD'] = 1500
                    config_dict['USE_SOLAR_FEATURES'] = False
            elif customer_profile == 'MEDCI':
                if customer_segment == 'solar':
                    config_dict['METER_THRESHOLD'] = 500
                    config_dict['USE_SOLAR_FEATURES'] = True
                else:  # nonsolar
                    config_dict['METER_THRESHOLD'] = 750
                    config_dict['USE_SOLAR_FEATURES'] = False
            elif customer_profile == 'SMLCOM':
                if customer_segment == 'solar':
                    config_dict['METER_THRESHOLD'] = 300
                    config_dict['USE_SOLAR_FEATURES'] = True
                else:  # nonsolar
                    config_dict['METER_THRESHOLD'] = 400
                    config_dict['USE_SOLAR_FEATURES'] = False
            
            # Build rate group filter
            if customer_segment == 'solar':
                config_dict['RATE_GROUP_FILTER'] = "(NEM% OR SBP%)"
                # Generate SQL clause for the filter
                config_dict['RATE_GROUP_FILTER_CLAUSE'] = "(rategroup LIKE 'NEM%' OR rategroup LIKE 'SBP%')"
            else:  # nonsolar
                config_dict['RATE_GROUP_FILTER'] = "NOT (NEM% OR SBP%)"
                # Generate SQL clause for the filter
                config_dict['RATE_GROUP_FILTER_CLAUSE'] = "(rategroup NOT LIKE 'NEM%' AND rategroup NOT LIKE 'SBP%')"
        
        if hasattr(config, 'enhance_config_for_forecasting'):
            config_dict = config.enhance_config_for_forecasting(config_dict)

        # Convert non-serializable types
        serializable_config = {}
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                serializable_config[key] = list(value)
            else:
                serializable_config[key] = value
        
        # Write to JSON file
        with open(output_file, "w") as f:
            json.dump(serializable_config, f, indent=2)
        
        print(f"Configuration saved to {output_file}")
        print(f"Profile: {customer_profile}, Segment: {customer_segment}")
        
        # Print some key settings
        print(f"Key configuration settings:")
        important_keys = ['CUSTOMER_PROFILE', 'CUSTOMER_SEGMENT', 'METER_THRESHOLD', 
                         'USE_SOLAR_FEATURES', 'RATE_GROUP_FILTER', 'RATE_GROUP_FILTER_CLAUSE']
        
        for key in important_keys:
            if key in serializable_config:
                print(f"  {key}: {serializable_config[key]}")
        
        return output_file
        
    except Exception as e:
        print(f"Error creating configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
