# processing_wrapper.py
import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys


def log_debug_info():
    """Log detailed debug information to help verify execution."""
    print("\n=== DEBUG INFORMATION ===")
    print(f"Current time: {datetime.datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Environment variables:")
    for key, value in sorted(os.environ.items()):
        print(f"  {key}={value}")
    print(f"Directory contents:")
    for root, dirs, files in os.walk('/opt/ml/processing', topdown=True, followlinks=False):
        level = root.replace('/opt/ml/processing', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  {f}")
        if level >= 3:  # Limit depth to avoid too much output
            dirs[:] = []
    print("=== END DEBUG INFORMATION ===\n")


def parse_parameter_value(param_str):
    """
    Parse SageMaker parameter template expressions.
   
    This function handles expressions like:
    {"Std:Join": {"On": "", "Values": [{"Get": "Parameters.DaysDelay"}]}}
   
    Args:
        param_str: The parameter string that might contain a template expression
       
    Returns:
        The parsed parameter value as a string
    """
    # Default parameter values
    default_values = {
        'DaysDelay': '7',
        'UseReducedFeatures': 'True',
        'MeterThreshold': '500',
        'UseCache': 'True',
        'QueryLimit': '-1',
        'UseWeather': 'True',
        'UseSolar': 'True',
        'WeatherCache': 'True'
    }
   
    # If it's not a string, return as is
    if not isinstance(param_str, str):
        return param_str
   
    # Check if this is potentially a JSON template expression
    if param_str.strip().startswith('{') and 'Parameters.' in param_str:
        try:
            # Look for "Parameters.X" pattern in the string
            match = re.search(r'"Parameters\.(\w+)"', param_str)
            if match:
                param_name = match.group(1)
                # First check environment variables
                env_value = os.environ.get(param_name)
                if env_value is not None:
                    return env_value
                # Fallback to default values
                return default_values.get(param_name, param_str)
        except Exception as e:
            print(f"Error parsing template expression: {str(e)}")
            # Continue with original value
   
    # Return the original string if no match or error
    return param_str


def safe_repr(value):
    """Create a safe string representation for Python config files."""
    if isinstance(value, str):
        if "'" in value and '"' not in value:
            # Use double quotes if string contains single quotes
            return f'"{value}"'
        elif '"' in value and "'" not in value:
            # Use single quotes if string contains double quotes  
            return f"'{value}'"
        elif "'" in value and '"' in value:
            # Use triple quotes if string contains both
            return f'"""{value}"""'
        else:
            # Default to single quotes
            return f"'{value}'"
    else:
        return repr(value)


def main():
    """Wrapper script that sets up config and then calls the real preprocessing script."""
    print("Starting processing wrapper...")

    #  Check Python version
    import sys
    if sys.version_info < (3, 8):
        print(f"ERROR: Python 3.8+ is required, but found {sys.version}")
        print("Please update the SageMaker processor to use py_version='py39' or higher")
        sys.exit(1)
   
    # Log debug information at the start
    log_debug_info()

    # Install required packages first
    install_packages()

    # Parse additional arguments for file paths
    parser = argparse.ArgumentParser(description="Process energy load data")
    parser.add_argument("--config-path", type=str,
                        default="/opt/ml/processing/input/config/processing_config.json")
    parser.add_argument("--preprocessing-path", type=str,
                        default="/opt/ml/processing/input/code/preprocessing/preprocessing.py")
    parser.add_argument("--data-processing-path", type=str,
                        default="/opt/ml/processing/input/code/data_processing/data_processing.py")
    parser.add_argument("--solar-features-path", type=str,
                        default="/opt/ml/processing/input/code/solar_features/solar_features.py")
    parser.add_argument("--weather-features-path", type=str,
                        default="/opt/ml/processing/input/code/weather_features/weather_features.py")
   
    # Parse known args to get paths, but preserve all other args for preprocessing.py
    args, remaining = parser.parse_known_args()
   
    # Get the config file path
    config_path = args.config_path
    preprocessing_path = args.preprocessing_path
    data_processing_path = args.data_processing_path
    solar_features_path = args.solar_features_path
    weather_features_path = args.weather_features_path
   
    print(f"Using config path: {config_path}")
    print(f"Using preprocessing path: {preprocessing_path}")
    print(f"Using data_processing path: {data_processing_path}")
    print(f"Using solar_features path: {solar_features_path}")
    print(f"Using weather_features path: {weather_features_path}")

    # Check if files exist
    print(f"Config path exists: {os.path.exists(config_path)}")
    print(f"Preprocessing path exists: {os.path.exists(preprocessing_path)}")
    print(f"Data processing path exists: {os.path.exists(data_processing_path)}")
    print(f"Solar features path exists: {os.path.exists(solar_features_path)}")
    print(f"Weather features path exists: {os.path.exists(weather_features_path)}")

    # Load the config file
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            print(f"Successfully loaded config from {config_path}")
            print(f"Config contents: {json.dumps(config_dict, indent=2)[:500]}...")
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        print("Using default values")
        config_dict = {}
        sys.exit(1)  # Exit with error code

    # Create a dedicated working directory for our scripts
    working_dir = "/opt/ml/processing/working"
    os.makedirs(working_dir, exist_ok=True)
    print(f"Created working directory: {working_dir}")
   
    # Copy preprocessing.py and data_processing.py to the working directory
    try:
        shutil.copy2(preprocessing_path, os.path.join(working_dir, "preprocessing.py"))
        shutil.copy2(data_processing_path, os.path.join(working_dir, "data_processing.py"))
       
        # Copy new feature engineering modules
        if os.path.exists(solar_features_path):
            shutil.copy2(solar_features_path, os.path.join(working_dir, "solar_features.py"))
            print(f"Copied solar_features.py to {working_dir}")
        else:
            print(f"Warning: Solar features path {solar_features_path} not found")
           
        if os.path.exists(weather_features_path):
            shutil.copy2(weather_features_path, os.path.join(working_dir, "weather_features.py"))
            print(f"Copied weather_features.py to {working_dir}")
        else:
            print(f"Warning: Weather features path {weather_features_path} not found")
       
        print(f"Copied preprocessing and data_processing scripts to {working_dir}")
       
        # Verify file contents by printing first 10 lines
        with open(os.path.join(working_dir, "preprocessing.py"), 'r') as f:
            preprocessing_content = f.readlines()[:10]
            print(f"First 10 lines of preprocessing.py: {preprocessing_content}")
           
        with open(os.path.join(working_dir, "data_processing.py"), 'r') as f:
            data_processing_content = f.readlines()[:10]
            print(f"First 10 lines of data_processing.py: {data_processing_content}")
    except Exception as e:
        print(f"Error copying scripts: {e}")
        sys.exit(1)
   
    # Create config.py in the working directory
    with open(os.path.join(working_dir, "config.py"), "w") as f:
        f.write("# Generated config file\n\n")
        for key, value in config_dict.items():
            if isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            else:
                f.write(f"{key} = {value}\n")
    print(f"Created config.py in {working_dir}")

    with open(os.path.join(working_dir, "config.py"), 'r') as f:
        config_content = f.readlines()
        print(f"Content of config.py: {config_content}")
   
    # Create configs directory and __init__.py
    configs_dir = os.path.join(working_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)
   
    with open(os.path.join(configs_dir, "__init__.py"), "w") as f:
        f.write("# Package initialization\n")
    print(f"Created configs/__init__.py in {working_dir}/configs")

    # Copy the config.py to configs directory
    with open(os.path.join(configs_dir, "config.py"), "w") as f:
        f.write("# Generated config file\n\n")
        for key, value in config_dict.items():
            if isinstance(value, str):
                f.write(f"{key} = {safe_repr(value)}\n")
            else:
                f.write(f"{key} = {repr(value)}\n")
    print(f"Created configs/config.py in {working_dir}/configs")
   
    # Change to the working directory
    os.chdir(working_dir)
    print(f"Changed working directory to: {os.getcwd()}")
   
    # Add the working directory to Python path
    sys.path.insert(0, working_dir)
    print(f"Python path now includes: {sys.path[:3]}")
   
    # List directory contents
    print(f"Working directory contents: {os.listdir(working_dir)}")
   
    # Now run the preprocessing script with all original arguments
    preprocessing_script = os.path.join(working_dir, "preprocessing.py")
   
    # Build the command with parsed arguments
    cmd_args = [sys.executable, preprocessing_script]
   
    # Process the remaining arguments and parse template expressions
    i = 0
    while i < len(remaining):
        current_arg = remaining[i]
       
        # Skip our custom path arguments which shouldn't be passed to preprocessing.py
        if current_arg in ["--preprocessing-path", "--data-processing-path",
                         "--solar-features-path", "--weather-features-path"]:
            i += 2  # Skip this arg and its value
            continue
       
        # Add the argument name to command args
        cmd_args.append(current_arg)
       
        # If this is a parameter with a value (not a flag without value)
        if i + 1 < len(remaining) and not remaining[i+1].startswith("--"):
            # Parse the parameter value if it's a template expression
            param_value = parse_parameter_value(remaining[i+1])
            cmd_args.append(param_value)
            i += 2  # Move past both the arg and its value
        else:
            # This is a flag without a value
            i += 1  # Just move to the next arg
   
    print(f"Running preprocessing with command: {cmd_args}")
   
    try:
        # Run preprocessing script
        subprocess.check_call(cmd_args)
        print("Preprocessing completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running preprocessing script: {str(e)}")
        print(f"Return code: {e.returncode}")
        if hasattr(e, 'output'):
            print(f"Output: {e.output}")
        if hasattr(e, 'stderr'):
            print(f"Stderr: {e.stderr}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
   
    # Log debug information at the end
    print("\n=== FINAL STATE ===")
    print(f"Working directory contents after execution: {os.listdir(working_dir)}")
    print(f"Output directories:")
    for dir_name in ['train', 'validation', 'test']:
        output_dir = f"/opt/ml/processing/output/{dir_name}"
        if os.path.exists(output_dir):
            print(f"  {dir_name}: {os.listdir(output_dir)}")
        else:
            print(f"  {dir_name}: directory does not exist")
    print("=== END FINAL STATE ===\n")


# def install_packages():
#     """Install required packages."""
#     packages = [
#         "pyathena", "pandas", "numpy", "boto3", "s3fs", "holidays",
#         "requests", "openmeteo-requests", "requests-cache", "retry-requests"
#     ]

#     try:
#         print(f"Installing packages: {', '.join(packages)}")
#         subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
#         print("Package installation successful")
#     except Exception as e:
#         print(f"Error installing packages: {str(e)}")
#         if "openmeteo-requests" in str(e):
#             print("ERROR: Failed to install openmeteo-requests, which is required")
#             print("This is likely due to Python version incompatibility")
#             sys.exit(1)


def install_packages():
    """Install required packages from requirements.txt."""
    try:
        # Define the specific path where requirements.txt should be
        requirements_path = "/opt/ml/processing/input/requirements/requirements.txt"
       
        if os.path.exists(requirements_path):
            print(f"Found requirements.txt at {requirements_path}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            print("Package installation from requirements.txt successful")
        else:
            # Look in alternative locations as fallback
            fallback_paths = [
                "/opt/ml/processing/input/code/requirements.txt",
                "/opt/ml/processing/input/code/preprocessing/requirements.txt",
                "requirements.txt",
                "../requirements.txt"
            ]
           
            for path in fallback_paths:
                if os.path.exists(path):
                    print(f"Found requirements.txt at fallback location: {path}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", path])
                    print("Package installation from fallback requirements.txt successful")
                    break
            else:
                # If no requirements.txt is found, install minimum packages
                print("requirements.txt not found, installing minimum required packages")
                fallback_packages = [
                    "pandas==2.0.3",
                    "numpy==1.24.4",
                    "boto3==1.28.38",
                    "s3fs==2023.6.0",
                    "pyathena==2.25.2",
                    "holidays==0.29",
                    "requests==2.31.0",
                    "openmeteo-requests==1.4.0",
                    "requests-cache==1.1.1",
                    "retry-requests==2.0.0",
                    "urllib3==1.26.15"
                ]
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + fallback_packages)
                print("Fallback package installation successful")
    except Exception as e:
        print(f"Error installing packages: {str(e)}")
        # Continue anyway - they might already be installed


if __name__ == "__main__":
    main()
