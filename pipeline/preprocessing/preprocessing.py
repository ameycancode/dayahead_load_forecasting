#!/usr/bin/env python
# preprocessing.py
import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd

# Install required packages if not present
try:
    import holidays
    import pyathena
except ImportError:
    print("Installing required packages...")
    logging.info("Installing required packages before importing")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "pyathena",
            "pandas",
            "numpy",
            "boto3",
            "s3fs",
            "holidays",
        ]
    )
    import holidays
    import pyathena


# Import config
try:
    logging.info("importing configs from project root folder")
    from configs import config
except ImportError:
    # If running in container, config might be in different location
    # Try to look for config.py in current directory
    logging.info("importing config inside container using config.py")
    if os.path.exists("config.py"):
        sys.path.insert(0, os.path.abspath("."))
        from config import *
    else:
        # Create a minimal config class with defaults
        logging.info(
            "config.py was not found, hence creating configurations on the fly"
        )

        class Config:
            """Default configuration if config module is not available."""

            PREPROCESSING_LOG_FOLDER = "/opt/ml/processing/logs"
            PREPROCESSING_OUTPUT_TRAIN_FOLDER = "/opt/ml/processing/output/train"
            PREPROCESSING_OUTPUT_VAL_FOLDER = "/opt/ml/processing/output/validation"
            PREPROCESSING_OUTPUT_TEST_FOLDER = "/opt/ml/processing/output/test"
            USE_CSV_CACHE = True
            DEFAULT_METER_THRESHOLD = 1000
            DEFAULT_LOAD_PROFILE = "RES"
            DEFAULT_RATE_GROUP_FILTER = "NEM%"
            INITIAL_SUBMISSION_DELAY = 14
            FINAL_SUBMISSION_DELAY = 48
            DEFAULT_LAG_DAYS = [14, 21]
            EXTENDED_LAG_DAYS = [14, 21, 28, 35]
            TEST_DAYS = 30
            VALIDATION_DAYS = 60
            USE_WEATHER_FEATURES = True
            USE_SOLAR_FEATURES = True
            WEATHER_CACHE = True

        config = Config()

# Import data processing functions
try:
    logging.info("importing preprocessing modules from project code base")
    from pipeline.preprocessing.data_processing import (
        aggregate_timeseries,
        analyze_duck_curve,
        create_features,
        create_lags,
        find_meter_threshold,
        handle_missing,
        handle_outliers,
        preprocess_raw,
        query_data,
    )

    # Import feature engineering modules
    from pipeline.preprocessing.solar_features import (
        add_solar_features,
        add_solar_ratios,
    )
    from pipeline.preprocessing.weather_features import (
        add_weather_features,
        create_weather_solar_interactions,
    )

    logging.info("Imports worked as expected.")
except ImportError:
    # Try different import path
    logging.info("importing preprocessing modules inside container")
    try:
        import os
        import sys

        #  Add current directory to Python path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            logging.info(f"Added current directory to Python path: {current_dir}")

        #  Check if files exist
        for file in ["data_processing.py", "solar_features.py", "weather_features.py"]:
            file_path = os.path.join(current_dir, file)
            if os.path.exists(file_path):
                logging.info(f"Found file: {file_path}")
            else:
                logging.error(f"File not found: {file_path}")

        # If in the container with flattened structure
        logging.info("Attempting to import modules from current directory")
        from data_processing import (
            aggregate_timeseries,
            analyze_duck_curve,
            create_features,
            create_lags,
            find_meter_threshold,
            handle_missing,
            handle_outliers,
            preprocess_raw,
            query_data,
            process_data_for_forecasting, 
            mark_forecast_available_features
        )

        logging.info("Successfully imported data_processing module")

        # Import feature engineering modules
        from solar_features import (
            add_solar_features,
            add_solar_ratios,
        )

        logging.info("Successfully imported solar_features module")

        from weather_features import (
            add_weather_features,
            create_weather_solar_interactions,
        )

        logging.info("Successfully imported weather_features module")
    except ImportError:
        logging.error(f"Error importing modules: {e}")
        logging.warning("Data processing modules not found")
        logging.info("Data processing modules not found")


# Configure logging
def setup_logging(log_dir=config.PREPROCESSING_LOG_FOLDER):
    """Set up logging configuration.

    Args:
        log_dir (str): Directory for log files

    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # File handler
    log_file = os.path.join(
        log_dir, f'preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


# Initialize logger
logger = setup_logging()


def process_data(
    output_train,
    output_val,
    output_test,
    days_delay=14,
    use_reduced_features=True,
    meter_threshold=None,
    use_cache=None,
    query_limit=None,
    use_weather=None,
    use_solar=None,
    weather_cache=None,
):
    """
    Main data processing function.
    
    1. Fetch ALL available Final submission data 
    2. Find latest Final date (e.g., May 3, 2025)
    3. Fetch Initial data from latest Final + 1 day to current_date - days_delay (June 3, 2025)
    4. Ensure 3-year period ending at June 3, 2025 (back to June 3, 2022)
    5. Split using configurable TEST_DAYS and VALIDATION_DAYS

    Args:
        output_train_path: Path to save training data
        output_validation_path: Path to save validation data
        output_test_path: Path to save test data
        days_delay: Number of days delay in data availability
        use_reduced_features: Whether to use a reduced feature set
        meter_threshold: Minimum meter count threshold
        use_cache: Whether to use CSV cache (None to use config default)
        query_limit: Maximum number of rows to query (None for no limit)
        use_weather: Whether to use weather features (None to use config default)
        use_solar: Whether to use solar features (None to use config default)
        weather_cache: Whether to cache weather data (None to use config default)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    try:
        logger.info("Starting preprocessing pipeline")

        # Set defaults from config if not provided
        if use_weather is None:
            use_weather = getattr(config, "USE_WEATHER_FEATURES", True)
        if use_solar is None:
            use_solar = getattr(config, "USE_SOLAR_FEATURES", True)
        if weather_cache is None:
            weather_cache = getattr(config, "WEATHER_CACHE", True)

        # Track stats
        stats = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {
                "days_delay": days_delay,
                "use_reduced_features": use_reduced_features,
                "meter_threshold": meter_threshold,
                "use_cache": (
                    use_cache if use_cache is not None else config.USE_CSV_CACHE
                ),
                "query_limit": query_limit,
                "use_weather_features": use_weather,
                "use_solar_features": use_solar,
            },
        }

        # Get current date and calculate end date for 3-year period
        current_date = datetime.now()  # June 17, 2025
        three_year_end_date = current_date - timedelta(days=days_delay)  # June 3, 2025
        
        logger.info(f"Current date: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"3-year period end date: {three_year_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Data delay: {days_delay} days")

        # Step 1: Get data using new logic (3 years ending June 3, 2025)
        raw_df = query_data(
            current_date=current_date,
            load_profile=config.CUSTOMER_PROFILE,
            # rate_group_filter=config.DEFAULT_RATE_GROUP_FILTER,
            use_cache=config.USE_CSV_CACHE if use_cache is None else use_cache,
            query_limit=query_limit,
        )

        if raw_df.empty:
            logger.error("No data retrieved")
            # return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            # Create empty output files to satisfy the SageMaker pipeline output requirements
            save_empty_outputs(
                output_train, output_val, output_test
            )
            sys.exit(1)

        logger.info(f"Retrieved {len(raw_df)} rows of raw data")
        
        # Log date range of retrieved data
        if 'datetime' in raw_df.columns or ('tradedate' in raw_df.columns and 'tradetime' in raw_df.columns):
            if 'datetime' not in raw_df.columns:
                raw_df['datetime'] = pd.to_datetime(raw_df['tradedate'].astype(str) + ' ' + raw_df['tradetime'].astype(str))
            
            data_start = raw_df['datetime'].min()
            data_end = raw_df['datetime'].max()
            data_span_days = (data_end - data_start).days
            logger.info(f"Raw data date range: {data_start} to {data_end} ({data_span_days} days, ~{data_span_days/365:.1f} years)")

        # Step 2: Process the data for forecasting
        processed_df, stats = process_data_for_forecasting(
            raw_df, 
            meter_threshold, 
            stats, 
            load_profile=config.CUSTOMER_PROFILE
        )

        if processed_df.empty:
            logger.error("Processed data is empty after processing")
            save_empty_outputs(output_train, output_val, output_test)
            sys.exit(1)

        # Step 3: Ensure exactly 3-year period
        three_year_df = ensure_three_year_period(processed_df, three_year_end_date)
        
        if three_year_df.empty:
            logger.error("No data available for 3-year period")
            save_empty_outputs(output_train, output_val, output_test)
            sys.exit(1)

        # Step 4: Mark and filter features that are available at forecast time
        available_features = mark_forecast_available_features(three_year_df, forecast_delay_days=days_delay)

        # Step 5: Filter to include only available features plus the target
        forecast_ready_df = three_year_df[available_features + ['lossadjustedload']]

        # Step 6: Add weather features if enabled
        if use_weather:
            logger.info("Adding weather features")
            forecast_ready_df = add_weather_features(forecast_ready_df, weather_cache=weather_cache)
            forecast_ready_df = create_weather_solar_interactions(forecast_ready_df)
            stats["weather_features_added"] = True

        # Step 7: Add solar features if enabled
        if use_solar:
            logger.info("Adding solar features")
            forecast_ready_df = add_solar_features(forecast_ready_df)
            forecast_ready_df = add_solar_ratios(forecast_ready_df)
            stats["solar_features_added"] = True

        # Log feature info
        logger.info(f"Final dataset shape: {forecast_ready_df.shape}")
        logger.info(f"Feature count: {len(forecast_ready_df.columns)}")
        logger.info(f"Feature list: {forecast_ready_df.columns.tolist()}")

        # Step 8: Handle critical NaNs before splitting
        if "lossadjustedload" in forecast_ready_df.columns:
            nan_count = forecast_ready_df["lossadjustedload"].isna().sum()
            if nan_count > 0:
                logger.warning(f"Dropping {nan_count} rows with NaN in target")
                forecast_ready_df = forecast_ready_df.dropna(subset=["lossadjustedload"])

        # Step 9: Create train/validation/test splits using configurable days
        train, val, test = create_new_train_validation_test_splits(
            forecast_ready_df, 
            three_year_end_date,
            test_days=config.TEST_DAYS,
            validation_days=config.VALIDATION_DAYS
        )

        # Step 10: Log final split information
        logger.info("=== FINAL DATASET SPLITS ===")
        logger.info(f"Train dataset: {len(train)} rows, {len(train.columns)} features")
        logger.info(f"Validation dataset: {len(val)} rows, {len(val.columns)} features")
        logger.info(f"Test dataset: {len(test)} rows, {len(test.columns)} features")
        
        if not train.empty:
            logger.info(f"Train date range: {train['datetime'].min()} to {train['datetime'].max()}")
        if not val.empty:
            logger.info(f"Validation date range: {val['datetime'].min()} to {val['datetime'].max()}")
        if not test.empty:
            logger.info(f"Test date range: {test['datetime'].min()} to {test['datetime'].max()}")

        # Store updated stats
        stats["final_train_rows"] = len(train)
        stats["final_val_rows"] = len(val)
        stats["final_test_rows"] = len(test)
        stats["feature_count"] = len(forecast_ready_df.columns)
        stats["feature_list"] = forecast_ready_df.columns.tolist()
        stats["data_period_start"] = three_year_df['datetime'].min().strftime('%Y-%m-%d') if not three_year_df.empty else None
        stats["data_period_end"] = three_year_df['datetime'].max().strftime('%Y-%m-%d') if not three_year_df.empty else None

        # Step 11: Validate splits
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            logger.error("One or more splits are empty")
            logger.error(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            save_empty_outputs(output_train, output_val, output_test)
            sys.exit(1)

        # Step 12: Save splits and stats
        save_outputs(train, val, test, output_train, output_val, output_test, stats)

        logger.info("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
        return train, val, test

    except Exception as e:
        logger.error(f"Error in processing: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def ensure_three_year_period(df, end_date):
    """
    Ensure the dataset covers exactly 3 years ending at end_date
    
    Args:
        df: Combined dataframe
        end_date: End date for the 3-year period (e.g., June 3, 2025)
    
    Returns:
        DataFrame filtered to exactly 3 years
    """
    try:
        if df.empty:
            return df
            
        # Define 3-year start date
        start_date = end_date - timedelta(days=3*365)  # 3 years before end_date
        
        logger.info(f"Filtering data to 3-year period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            if 'tradedate' in df.columns and 'tradetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['tradedate'].astype(str) + ' ' + df['tradetime'].astype(str))
            else:
                logger.error("Cannot create datetime column - missing tradedate/tradetime")
                return df
        
        # Filter to 3-year period
        before_count = len(df)
        df_filtered = df[
            (df['datetime'] >= start_date) & 
            (df['datetime'] <= end_date)
        ].copy()
        
        after_count = len(df_filtered)
        logger.info(f"Filtered to 3-year period: {before_count} → {after_count} rows")
        
        # Verify date range
        if not df_filtered.empty:
            actual_start = df_filtered['datetime'].min()
            actual_end = df_filtered['datetime'].max()
            logger.info(f"Actual date range in data: {actual_start} to {actual_end}")
            
            # Calculate actual time span
            time_span_days = (actual_end - actual_start).days
            logger.info(f"Actual time span: {time_span_days} days (~{time_span_days/365:.1f} years)")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Error ensuring 3-year period: {str(e)}")
        return df


def create_new_train_validation_test_splits(df, end_date, test_days=30, validation_days=60):
    """
    Create train/validation/test splits using configurable days:
    - Test: Last test_days (default 30 days)
    - Validation: validation_days before test period (default 60 days) 
    - Train: All remaining data (rest of the 3-year period)
    
    Args:
        df: Processed dataframe covering 3 years
        end_date: End date of the 3-year period (e.g., June 3, 2025)
        test_days: Number of days for test set (from config.TEST_DAYS)
        validation_days: Number of days for validation set (from config.VALIDATION_DAYS)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    try:
        if df.empty:
            logger.warning("Empty dataframe provided for splitting")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            logger.error("No datetime column found for splitting")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Calculate split dates working backwards from end_date
        test_start = end_date - timedelta(days=test_days)  # Test period
        val_start = test_start - timedelta(days=validation_days)  # Validation period
        three_year_start = end_date - timedelta(days=3*365)  # Train period starts
        
        logger.info("=== TRAIN/VALIDATION/TEST SPLIT ===")
        logger.info(f"Using TEST_DAYS={test_days}, VALIDATION_DAYS={validation_days}")
        logger.info(f"Total period: {three_year_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Train period: {three_year_start.strftime('%Y-%m-%d')} to {val_start.strftime('%Y-%m-%d')} (~{(val_start - three_year_start).days} days)")
        logger.info(f"Validation period: {val_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')} ({validation_days} days)")
        logger.info(f"Test period: {test_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({test_days} days)")
        
        # Create splits
        train_df = df[
            (df['datetime'] >= three_year_start) & 
            (df['datetime'] < val_start)
        ].copy()
        
        val_df = df[
            (df['datetime'] >= val_start) & 
            (df['datetime'] < test_start)
        ].copy()
        
        test_df = df[
            (df['datetime'] >= test_start) & 
            (df['datetime'] <= end_date)
        ].copy()
        
        # Log split results
        logger.info(f"Split results:")
        if not train_df.empty:
            logger.info(f"  Train: {len(train_df)} rows ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
        else:
            logger.error("  Train: 0 rows - EMPTY!")
            
        if not val_df.empty:
            logger.info(f"  Validation: {len(val_df)} rows ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
        else:
            logger.error("  Validation: 0 rows - EMPTY!")
            
        if not test_df.empty:
            logger.info(f"  Test: {len(test_df)} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
        else:
            logger.error("  Test: 0 rows - EMPTY!")
        
        # Validate splits
        total_original = len(df)
        total_splits = len(train_df) + len(val_df) + len(test_df)
        logger.info(f"Total rows: Original={total_original}, Splits={total_splits}")
        
        if total_splits != total_original:
            logger.warning(f"Row count mismatch! Missing {total_original - total_splits} rows")
            
        # Check for empty splits
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            logger.error("One or more splits are empty!")
            logger.error(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
        # Verify no data leakage (no overlapping dates)
        if not train_df.empty and not val_df.empty:
            train_max = train_df['datetime'].max()
            val_min = val_df['datetime'].min()
            if train_max >= val_min:
                logger.warning(f"Potential data leakage: Train max ({train_max}) >= Val min ({val_min})")
                
        if not val_df.empty and not test_df.empty:
            val_max = val_df['datetime'].max()
            test_min = test_df['datetime'].min()
            if val_max >= test_min:
                logger.warning(f"Potential data leakage: Val max ({val_max}) >= Test min ({test_min})")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error creating train/validation/test splits: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def save_empty_outputs(train_path, val_path, test_path):
    """Save empty CSV files when no data is available to satisfy SageMaker output requirements.
    
    Args:
        train_path: Path to save empty training data
        val_path: Path to save empty validation data  
        test_path: Path to save empty test data
    """
    try:
        # Create output dirs
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        # Create empty DataFrame with minimal structure
        empty_df = pd.DataFrame(columns=['datetime', 'lossadjustedload'])
        
        # Save empty files
        empty_df.to_csv(os.path.join(train_path, "train.csv"), index=False)
        empty_df.to_csv(os.path.join(val_path, "validation.csv"), index=False)
        empty_df.to_csv(os.path.join(test_path, "test.csv"), index=False)
        
        logger.info("Created empty output files due to data retrieval failure")
        
    except Exception as e:
        logger.error(f"Error creating empty outputs: {e}")


def save_outputs(train, val, test, train_path, val_path, test_path, stats):
    """Save processed datasets and statistics.

    Args:
        train: Training dataframe
        val: Validation dataframe
        test: Test dataframe
        train_path: Path to save training data
        val_path: Path to save validation data
        test_path: Path to save test data
        stats: Dictionary of processing statistics
    """
    try:
        # Create output dirs
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Save data
        if not train.empty:
            train.to_csv(os.path.join(train_path, "train.csv"), index=False)
            logger.info(f"Train data saved to location {train_path} in file train.csv")
            s3 = boto3.client("s3")
            s3.upload_file(
                f"{train_path}/train.csv",
                config.S3_BUCKET,
                f"{config.PREPROCESSING_S3_BUCKET_TRAIN}/train.csv",
            )
            logger.info("Training data (train.csv) saved to S3")
        if not val.empty:
            val.to_csv(os.path.join(val_path, "validation.csv"), index=False)
            logger.info(
                f"Validation data saved to location {val_path} in file validation.csv"
            )
            s3 = boto3.client("s3")
            s3.upload_file(
                f"{val_path}/validation.csv",
                config.S3_BUCKET,
                f"{config.PREPROCESSING_S3_BUCKET_VAL}/validation.csv",
            )
            logger.info("Validation data (validation.csv) saved to S3")
        if not test.empty:
            test.to_csv(os.path.join(test_path, "test.csv"), index=False)
            logger.info(f"Test data saved to location {test_path} in file test.csv")

            s3 = boto3.client("s3")
            s3.upload_file(
                f"{test_path}/test.csv",
                config.S3_BUCKET,
                f"{config.PREPROCESSING_S3_BUCKET_TEST}/test.csv",
            )
            logger.info("Test data (test.csv) saved to S3")

            # Save latest data to S3
            try:
                with open("/tmp/latest_data.csv", "w") as f:
                    test.to_csv(f, index=False)

                s3 = boto3.client("s3")
                s3.upload_file(
                    "/tmp/latest_data.csv",
                    config.S3_BUCKET,
                    f"{config.PREPROCESSING_S3_BUCKET}/latest_data.csv",
                )
                logger.info("Latest data (that is similar to test data) saved to S3")
            except Exception as e:
                logger.warning(f"Error saving to S3: {e}")

        # Save stats
        with open(os.path.join(train_path, "data_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    except Exception as e:
        logger.error(f"Error saving outputs: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process energy load data")
    parser.add_argument(
        "--config-path",
        type=str,
        default="/opt/ml/processing/input/config/processing_config.json",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default=config.PREPROCESSING_OUTPUT_TRAIN_FOLDER,
    )
    parser.add_argument(
        "--output-validation",
        type=str,
        default=config.PREPROCESSING_OUTPUT_VAL_FOLDER,
    )
    parser.add_argument(
        "--output-test", type=str, default=config.PREPROCESSING_OUTPUT_TEST_FOLDER
    )
    parser.add_argument(
        "--days-delay", type=int, default=config.INITIAL_SUBMISSION_DELAY
    )
    parser.add_argument("--use-reduced-features", type=str, default="True")
    parser.add_argument(
        "--meter-threshold", type=int, default=config.DEFAULT_METER_THRESHOLD
    )
    parser.add_argument("--log-dir", type=str, default=config.PREPROCESSING_LOG_FOLDER)
    parser.add_argument(
        "--use-cache",
        type=str,
        default=str(config.USE_CSV_CACHE),
        help="Whether to use CSV cache (None to use config default)",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Maximum number of rows to query (None for no limit)",
    )
    parser.add_argument(
        "--use-weather",
        type=str,
        default="True",
        help="Whether to use weather features",
    )
    parser.add_argument(
        "--use-solar",
        type=str,
        default="True",
        help="Whether to use solar features",
    )
    parser.add_argument(
        "--weather-cache",
        type=str,
        default="True",
        help="Whether to cache weather data",
    )

    args = parser.parse_args()

    # Convert string booleans to actual booleans
    use_reduced_features = args.use_reduced_features.lower() == "true"
    use_cache = args.use_cache.lower() == "true"
    use_weather = args.use_weather.lower() == "true"
    use_solar = args.use_solar.lower() == "true"
    weather_cache = args.weather_cache.lower() == "true"

    # Set up logging with specified directory
    logger = setup_logging(args.log_dir)

    try:
        # Check if config path exists and try to load it
        if os.path.exists(args.config_path):
            logger.info(f"Loading configuration from {args.config_path}")
            try:
                with open(args.config_path, "r") as f:
                    config_dict = json.load(f)

                # Apply config values to the config module
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        logger.info(f"Set config.{key} = {value}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")

        # Process the data
        process_data(
            args.output_train,
            args.output_validation,
            args.output_test,
            args.days_delay,
            use_reduced_features,
            args.meter_threshold,
            use_cache,
            args.query_limit,
            use_weather,
            use_solar,
            weather_cache,
        )
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
