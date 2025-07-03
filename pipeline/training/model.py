"""
Core XGBoost model training for energy load forecasting.
This module handles model training, cross-validation, and time series handling.
"""

import logging
import os
import pickle
from functools import partial
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def create_time_series_splits(
    df: pd.DataFrame,
    date_column: str = "datetime",
    n_splits: int = 5,
    initial_train_months: int = 6,
    validation_months: int = 3,
    step_months: int = 3,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Create time series cross-validation splits with expanding training window.
    
    Args:
        df: DataFrame with time series data
        date_column: Name of the date/time column
        n_splits: Maximum number of cross-validation splits to create
        initial_train_months: Number of months for initial training set
        validation_months: Number of months for each validation set
        step_months: Number of months to step forward for each split
        
    Returns:
        List of (train_indices, test_indices) tuples for each split
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to create_time_series_splits")
        return []

    # Ensure the date column exists
    if date_column not in df.columns:
        logger.error(f"Date column '{date_column}' not found in DataFrame")
        return []
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        logger.info(f"Converting {date_column} to datetime")
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort DataFrame by date to ensure proper time ordering
    df = df.sort_values(by=date_column).copy()
    
    # Get min and max dates
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    
    total_months = ((max_date.year - min_date.year) * 12 + 
                    max_date.month - min_date.month)
    
    logger.info(f"Data spans {total_months} months from {min_date.date()} to {max_date.date()}")
    
    # Calculate how many splits we can make
    possible_splits = max(1, (total_months - initial_train_months - validation_months) // step_months + 1)
    actual_splits = min(n_splits, possible_splits)
    
    logger.info(f"Creating {actual_splits} time series splits with {initial_train_months} months initial "
                f"training, {validation_months} months validation, {step_months} months step")
    
    # Create splits
    splits = []
    
    for i in range(actual_splits):
        # Calculate cutoff dates for this split
        train_months = initial_train_months + (i * step_months)
        train_end_year = min_date.year + (min_date.month + train_months - 1) // 12
        train_end_month = (min_date.month + train_months - 1) % 12 + 1
        
        val_start_year = train_end_year
        val_start_month = train_end_month
        
        val_end_year = val_start_year + (val_start_month + validation_months - 1) // 12
        val_end_month = (val_start_month + validation_months - 1) % 12 + 1
        
        # Create cutoff dates
        try:
            train_end_date = pd.Timestamp(year=train_end_year, month=train_end_month, day=28)
            val_start_date = train_end_date + pd.Timedelta(days=1)
            val_end_date = pd.Timestamp(year=val_end_year, month=val_end_month, day=28)
        except ValueError as e:
            logger.error(f"Error creating dates for split {i+1}: {e}")
            continue
        
        # Get indices for train and validation sets
        train_mask = (df[date_column] <= train_end_date)
        val_mask = (df[date_column] > train_end_date) & (df[date_column] <= val_end_date)
        
        train_indices = df.index[train_mask]
        val_indices = df.index[val_mask]
        
        if len(train_indices) > 0 and len(val_indices) > 0:
            splits.append((train_indices, val_indices))
            logger.info(f"Split {i+1}: train={min_date.date()} to {train_end_date.date()}, "
                       f"val={val_start_date.date()} to {val_end_date.date()}, "
                       f"sizes: train={len(train_indices)}, val={len(val_indices)}")
        else:
            logger.warning(f"Skipping invalid split {i+1}: insufficient data")
    
    logger.info(f"Created {len(splits)} valid time series splits")
    return splits


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error - handles zero and near-zero values better.
   
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
       
    Returns:
        sMAPE value (lower is better)
    """
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error - better for zero-crossing data.
   
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
       
    Returns:
        WAPE value (lower is better)
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)

def mase(y_true, y_pred, y_train=None, seasonality=24):
    """
    Mean Absolute Scaled Error - scaled relative to naive forecast.
   
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        y_train: Training data to compute naive error scale (if None, uses y_true)
        seasonality: Seasonality period for naive forecast (default: 24 for hourly data)
       
    Returns:
        MASE value (lower is better)
    """
    if y_train is None:
        y_train = y_true
       
    # Create naive seasonal forecast (same hour, day before)
    if len(y_train) <= seasonality:
        # Not enough history, use mean as naive forecast
        naive_errors = np.abs(y_train - np.mean(y_train))
    else:
        # Use seasonal naive forecast
        naive_forecast = y_train[:-seasonality]
        naive_truth = y_train[seasonality:]
        naive_errors = np.abs(naive_truth - naive_forecast)
   
    # Calculate scale based on naive errors
    scale = np.mean(naive_errors) + 1e-8
   
    # Calculate model errors
    model_errors = np.abs(y_true - y_pred)
   
    # Return scaled errors
    return np.mean(model_errors / scale)

def transition_weighted_error(y_true, y_pred, threshold=20000):
    """
    Error metric that gives higher weight to transition periods (near zero).
   
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        threshold: Threshold to identify points near zero
       
    Returns:
        Weighted error value (lower is better)
    """
    # Identify transition points (near zero)
    near_zero_mask = np.abs(y_true) < threshold
   
    # Calculate RMSE for all points and transition points
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   
    # If there are transition points, calculate their RMSE
    if np.any(near_zero_mask):
        transition_rmse = np.sqrt(mean_squared_error(
            y_true[near_zero_mask], y_pred[near_zero_mask]))
    else:
        transition_rmse = 0
   
    # Weight overall and transition errors (60% overall, 40% transition)
    return 0.6 * overall_rmse + 0.4 * transition_rmse


def initialize_model(params: Optional[Dict] = None) -> XGBRegressor:
    """
    Initialize an XGBoost model with default or specified parameters.
    
    Args:
        params: Dictionary of XGBoost model parameters
        
    Returns:
        Initialized XGBRegressor model
    """
    # Default parameters optimized for energy load forecasting
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,  # Use all available cores
        "objective": "reg:squarederror",
    }
    
    # Update with user-provided parameters
    if params:
        default_params.update(params)
    
    logger.info(f"Initializing XGBoost model with parameters: {default_params}")
    return XGBRegressor(**default_params)


def train_model(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'lossadjustedload',
    model_params: Optional[Dict[str, Any]] = None,
    date_column: str = 'datetime',
    output_dir: str = None,
    run_id: str = None,
    create_plots: bool = True,
    customer_segment: str = 'RES_SOLAR'
) -> Tuple[Any, Dict[str, float]]:
    """
    Train an XGBoost model on the specified features and target.
   
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        model_params: Parameters for XGBoost model
        early_stopping_rounds: Number of rounds for early stopping
        eval_metric: Evaluation metric for early stopping
       
    Returns:
        Tuple of (trained model, training metrics)
    """
    if df.empty:
        logger.error("Empty DataFrame provided for training")
        return None, {}
   
    # Validate features and target exist in DataFrame
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in training data: {missing_cols}")
        return None, {}
   
    # Initialize model
    model = initialize_model(model_params)
    logger.info(f"Initializing XGBoost model with parameters: {model_params}")
   
    # Drop rows with NaN in features or target
    valid_rows = df[features + [target]].dropna()
   
    if len(valid_rows) < len(df):
        logger.warning(f"Dropped {len(df) - len(valid_rows)} rows with NaN values")
   
    if valid_rows.empty:
        logger.error("No valid rows for training after dropping NaN values")
        return None, {}
   
    # Prepare training data
    X = valid_rows[features]
    y = valid_rows[target]
   
    logger.info(f"Training XGBoost model on {len(X)} samples with {len(features)} features for segment {customer_segment}")
   
    # Training with validation split
    validation_size = min(0.2, 1000 / len(X)) if len(X) > 1000 else 0.2
    validation_count = int(len(X) * validation_size)
   
    # Sort by datetime if available to respect temporal ordering in validation split
    logger.info(f"Features in X: {X.columns}")
    if "datetime" in X.columns:
        logger.info("Sorting by datetime to respect temporal ordering in validation split")
        sorted_indices = X["datetime"].sort_values().index
        X = X.loc[sorted_indices]
        y = y.loc[sorted_indices]
       
    # Split into train and validation sets
    X_train = X.iloc[:-validation_count]
    y_train = y.iloc[:-validation_count]
    X_val = X.iloc[-validation_count:]
    y_val = y.iloc[-validation_count:]
   
    logger.info(f"Split into {len(X_train)} training samples and {len(X_val)} validation samples")
   
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
   
    # Calculate metrics on validation set
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
   
    # Calculate MAPE with handling for zeros
    y_true_safe = y_val.copy()
    y_true_safe = np.where(y_true_safe == 0, 1e-10, y_true_safe)
    mape = np.mean(np.abs((y_val - y_pred) / y_true_safe)) * 100

    # Calculate alternative metrics for zero-crossing data
    smape_val = smape(y_val, y_pred)
    wape_val = wape(y_val, y_pred)
    mase_val = mase(y_val, y_pred, y_train)
    trans_error = transition_weighted_error(y_val, y_pred)
   
    metrics = {
        "validation_rmse": rmse,
        "validation_mae": mae,
        "validation_r2": r2,
        "validation_mape": mape,
        "validation_smape": smape_val,
        "validation_wape": wape_val,
        "validation_mase": mase_val,
        "validation_trans_weighted_error": trans_error,
        "n_features": len(features),
        "n_samples": len(X),
        "customer_segment": customer_segment
    }
   
    logger.info(f"Model trained successfully for {customer_segment}. Validation RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, "
               f"sMAPE: {smape_val:.2f}%, WAPE: {wape_val:.2f}%, MASE: {mase_val:.4f}, R²: {r2:.4f}")
   
    # Add segment-specific period metrics if datetime column is available
    if "hour" in X_val.columns:
        # Create a copy of validation data with predictions
        val_df = X_val.copy()
        val_df[target] = y_val
        val_df['prediction'] = y_pred
       
        # Get segment-specific periods
        from configs.config import get_segment_config
        segment_config = get_segment_config(customer_segment)
        periods = segment_config.get("evaluation_periods", {})
       
        logger.info(f"\nSegment-specific validation metrics for {customer_segment}:")
        for period_name, (start_hour, end_hour) in periods.items():
            # Handle periods that cross midnight
            if start_hour <= end_hour:
                period_mask = (val_df['hour'] >= start_hour) & (val_df['hour'] < end_hour)
            else:
                period_mask = (val_df['hour'] >= start_hour) | (val_df['hour'] < end_hour)
           
            if period_mask.sum() > 0:
                period_rmse = np.sqrt(mean_squared_error(val_df[target][period_mask], val_df['prediction'][period_mask]))
               
                # Handle MAPE with zeros
                period_target_safe = val_df[target][period_mask].copy()
                period_target_safe = np.where(period_target_safe == 0, 1e-10, period_target_safe)
                period_mape = np.mean(np.abs((val_df[target][period_mask] - val_df['prediction'][period_mask]) / period_target_safe)) * 100

                period_smape = smape(val_df[target][period_mask], val_df['prediction'][period_mask])
                period_wape = wape(val_df[target][period_mask], val_df['prediction'][period_mask])
                period_mase = mase(val_df[target][period_mask], val_df['prediction'][period_mask])
               
                metrics[f'{period_name}_rmse'] = period_rmse
                metrics[f'{period_name}_mape'] = period_mape
                metrics[f'{period_name}_smape'] = period_smape
                metrics[f'{period_name}_wape'] = period_wape
                metrics[f'{period_name}_mase'] = period_mase
               
                logger.info(f"  - {period_name.replace('_', ' ').title()}: RMSE = {period_rmse:.2f}, "
                      f"MAPE = {period_mape:.2f}%, sMAPE = {period_smape:.2f}%, "
                      f"WAPE = {period_wape:.2f}%")
        
        # Add segment-specific critical metrics
        if customer_segment == "RES_NONSOLAR":
            # Special focus on evening super peak
            evening_peak_mask = (val_df['hour'] >= 17) & (val_df['hour'] < 21)
            if evening_peak_mask.sum() > 0:
                evening_actual = val_df[target][evening_peak_mask]
                evening_pred = val_df['prediction'][evening_peak_mask]
                
                # Peak magnitude accuracy
                actual_max = evening_actual.max()
                actual_max_idx = evening_actual.idxmax()
                pred_at_max_time = evening_pred.loc[actual_max_idx] if actual_max_idx in evening_pred.index else evening_pred.mean()
                
                peak_magnitude_error = abs(actual_max - pred_at_max_time) / (actual_max + 1e-8) * 100
                metrics['evening_peak_magnitude_error_pct'] = peak_magnitude_error
                
                logger.info(f"  - Evening Peak Magnitude Error: {peak_magnitude_error:.2f}%")
        
        elif customer_segment.endswith("_SOLAR"):  # Fixed: Use endswith instead of "in"
            # Special focus on duck curve transition
            duck_curve_mask = (val_df['hour'] >= 14) & (val_df['hour'] < 18)
            if duck_curve_mask.sum() > 0:
                duck_rmse = np.sqrt(mean_squared_error(val_df[target][duck_curve_mask], val_df['prediction'][duck_curve_mask]))
                metrics['duck_curve_transition_rmse'] = duck_rmse
                logger.info(f"  - Duck Curve Transition RMSE: {duck_rmse:.2f}")
       
        # Calculate metrics by load level with segment awareness
        logger.info("\nLoad level validation metrics:")
        # Create 5 load level bins
        val_df['load_level'] = pd.qcut(val_df[target], 5, labels=False)
       
        # Get bin edges for labeling
        bin_edges = pd.qcut(val_df[target], 5, retbins=True)[1]
       
        for level in range(5):
            level_df = val_df[val_df['load_level'] == level]
           
            if len(level_df) > 0:
                level_rmse = np.sqrt(mean_squared_error(level_df[target], level_df['prediction']))
               
                # Handle MAPE with zeros
                level_target_safe = level_df[target].copy()
                level_target_safe = np.where(level_target_safe == 0, 1e-10, level_target_safe)
                level_mape = np.mean(np.abs((level_df[target] - level_df['prediction']) / level_target_safe)) * 100

                level_smape = smape(level_df[target], level_df['prediction'])
                level_wape = wape(level_df[target], level_df['prediction'])
                level_mase = mase(level_df[target], level_df['prediction'])
               
                level_min = bin_edges[level]
                level_max = bin_edges[level + 1]
               
                metrics[f'load_level_{level}_rmse'] = level_rmse
                metrics[f'load_level_{level}_mape'] = level_mape
                metrics[f'load_level_{level}_smape'] = level_smape
                metrics[f'load_level_{level}_wape'] = level_wape
                metrics[f'load_level_{level}_mase'] = level_mase
               
                logger.info(f"  - Load level {level} ({level_min:.2f}-{level_max:.2f}): RMSE = {level_rmse:.2f}, "
                  f"MAPE = {level_mape:.2f}%, sMAPE = {level_smape:.2f}%, WAPE = {level_wape:.2f}%")
   
    # Log feature importance
    logger.info("\nTop 10 feature importance:")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
   
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        logger.info(f"  {i+1}. {row['Feature']} = {row['Importance']:.4f}")

    # Create visualizations if requested
    if create_plots and output_dir and run_id:
        try:
            from pipeline.training.visualization import visualize_training_results
            visualize_training_results(
                df=df,
                features=features,
                target=target,
                model=model,
                model_metrics=metrics,
                date_column=date_column,
                output_dir=output_dir,
                run_id=run_id,
                # customer_segment=customer_segment
            )
        except Exception as e:
            logger.error(f"Error creating training visualizations: {str(e)}")
   
    return model, metrics


def cross_validate_model(
    df: pd.DataFrame,
    features: List[str],
    target: str = "lossadjustedload",
    date_column: str = "datetime",
    model_params: Optional[Dict] = None,
    n_splits: int = 5,
    test_size_days: int = 14,
    gap_days: int = 7,
    custom_periods: Optional[Dict[str, Tuple[int, int]]] = None,
    customer_segment: str = "RES_SOLAR"  # Add this parameter
) -> Tuple[XGBRegressor, Dict]:
    """
    Perform time series cross-validation and train a final model.
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        date_column: Name of the date/time column
        model_params: Parameters for XGBoost model
        n_splits: Number of cross-validation splits
        test_size_days: Size of test set in days for each split
        gap_days: Gap between train and test periods in days
        custom_periods: Dictionary of custom time periods to evaluate,
                        e.g. {"peak_hours": (17, 20)}
        
    Returns:
        Tuple of (trained model on full dataset, cross-validation metrics)
    """
    if df.empty:
        logger.error("Empty DataFrame provided for cross-validation")
        return None, {}
    
    # Ensure columns exist
    required_cols = features + [target, date_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns for cross-validation: {missing_cols}")
        return None, {}
    
    # Get segment-specific periods if not provided
    if custom_periods is None:
        # Define periods directly or import from config
        try:
            from configs.config import get_segment_config
            segment_config = get_segment_config(customer_segment)
            custom_periods = segment_config.get("evaluation_periods", {})
            logger.info(f"Using segment-specific periods for {customer_segment}: {list(custom_periods.keys())}")
        except ImportError:
            # Fallback to default periods if config import fails
            logger.warning("Could not import segment config, using default periods")
            custom_periods = {
                "morning_peak": (7, 9),
                "evening_peak": (17, 21),
                "solar_peak": (11, 14) if customer_segment.endswith("_SOLAR") else (11, 14)
            }
    
    # Create time series splits
    cv_splits = create_time_series_splits(
        df,
        date_column=date_column,
        n_splits=n_splits,
        test_size_days=test_size_days,
        gap_days=gap_days
    )
    
    if not cv_splits:
        logger.error("Failed to create valid cross-validation splits")
        return None, {}
    
    # Track metrics across all splits
    cv_metrics = {
        "splits": [],
        "rmse": [],
        "mae": [],
        "mape": [],
        "r2": [],
        "customer_segment": customer_segment,
        "custom_periods": {name: {"rmse": [], "mae": [], "mape": [], "r2": []}
                          for name in (custom_periods or {})}
    }
    
    # Perform cross-validation
    logger.info(f"Starting {len(cv_splits)}-fold time series cross-validation for {customer_segment}")
    
    for i, (train_idx, test_idx) in enumerate(cv_splits):
        logger.info(f"CV Fold {i+1}/{len(cv_splits)}")
        
        # Split data
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target]
        X_test = df.loc[test_idx, features]
        y_test = df.loc[test_idx, target]
        
        # Train model on this fold
        model = initialize_model(model_params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE with handling for zeros
        y_test_safe = y_test.copy()
        y_test_safe = np.where(y_test_safe == 0, 1e-10, y_test_safe)
        mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
        
        # Save metrics for this fold
        cv_metrics["splits"].append(i)
        cv_metrics["rmse"].append(rmse)
        cv_metrics["mae"].append(mae)
        cv_metrics["mape"].append(mape)
        cv_metrics["r2"].append(r2)
        
        logger.info(f"Fold {i+1} metrics - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        
        # Evaluate on custom periods if provided
        if custom_periods and date_column in df.columns:
            test_df = df.loc[test_idx].copy()
            test_df["prediction"] = y_pred
            test_df["hour"] = test_df[date_column].dt.hour
            
            for period_name, (start_hour, end_hour) in custom_periods.items():
                # Handle periods that cross midnight
                if start_hour <= end_hour:
                    period_mask = (test_df["hour"] >= start_hour) & (test_df["hour"] < end_hour)
                else:
                    period_mask = (test_df["hour"] >= start_hour) | (test_df["hour"] < end_hour)
                
                if period_mask.sum() > 0:
                    # Calculate metrics for this period
                    period_rmse = np.sqrt(mean_squared_error(test_df[target][period_mask], test_df["prediction"][period_mask]))
                    period_mae = mean_absolute_error(test_df[target][period_mask], test_df["prediction"][period_mask])
                    period_r2 = r2_score(test_df[target][period_mask], test_df["prediction"][period_mask])
                    
                    # Calculate MAPE for this period
                    period_y_true_safe = test_df[target][period_mask].copy()
                    period_y_true_safe = np.where(period_y_true_safe == 0, 1e-10, period_y_true_safe)
                    period_mape = np.mean(np.abs((test_df[target][period_mask] - test_df["prediction"][period_mask]) / period_y_true_safe)) * 100
                    
                    # Save period metrics
                    cv_metrics["custom_periods"][period_name]["rmse"].append(period_rmse)
                    cv_metrics["custom_periods"][period_name]["mae"].append(period_mae)
                    cv_metrics["custom_periods"][period_name]["mape"].append(period_mape)
                    cv_metrics["custom_periods"][period_name]["r2"].append(period_r2)
                    
                    logger.debug(f"Fold {i+1} {period_name} metrics - RMSE: {period_rmse:.4f}, MAPE: {period_mape:.2f}%")
    
    # Calculate average metrics across all folds
    cv_metrics["avg_rmse"] = np.mean(cv_metrics["rmse"])
    cv_metrics["avg_mae"] = np.mean(cv_metrics["mae"])
    cv_metrics["avg_mape"] = np.mean(cv_metrics["mape"])
    cv_metrics["avg_r2"] = np.mean(cv_metrics["r2"])
    cv_metrics["std_rmse"] = np.std(cv_metrics["rmse"])
    cv_metrics["std_mape"] = np.std(cv_metrics["mape"])
    
    # Calculate average metrics for each custom period
    for period_name in (custom_periods or {}):
        period_metrics = cv_metrics["custom_periods"][period_name]
        if period_metrics["rmse"]:
            period_metrics["avg_rmse"] = np.mean(period_metrics["rmse"])
            period_metrics["avg_mae"] = np.mean(period_metrics["mae"])
            period_metrics["avg_mape"] = np.mean(period_metrics["mape"])
            period_metrics["avg_r2"] = np.mean(period_metrics["r2"])
            period_metrics["std_rmse"] = np.std(period_metrics["rmse"])
            period_metrics["std_mape"] = np.std(period_metrics["mape"])
    
    logger.info(f"Cross-validation complete for {customer_segment}. Average RMSE: {cv_metrics['avg_rmse']:.4f} (±{cv_metrics['std_rmse']:.4f}), "
                f"Average MAPE: {cv_metrics['avg_mape']:.2f}% (±{cv_metrics['std_mape']:.2f}%), "
                f"Average R²: {cv_metrics['avg_r2']:.4f}")
    
    # Train final model on all data
    logger.info(f"Training final model on all data for {customer_segment}")
    final_model, final_metrics = train_model(
        df, 
        features, 
        target=target, 
        model_params=model_params,
        customer_segment=customer_segment  # Pass segment info
    )
    
    # Combine metrics
    all_metrics = {
        "cv_metrics": cv_metrics,
        "final_model_metrics": final_metrics
    }
    
    return final_model, all_metrics


def save_model(
    model: XGBRegressor,
    metrics: Dict,
    features: List[str],
    output_dir: str,
    model_name: str = "xgboost_model"
) -> Dict:
    """
    Save trained model, features list, and metrics to the output directory.
    
    Args:
        model: Trained XGBoost model
        metrics: Dictionary of model metrics
        features: List of feature names used for training
        output_dir: Directory to save model files
        model_name: Base name for model files
        
    Returns:
        Dictionary with paths to saved files
    """
    if model is None:
        logger.error("Cannot save None model")
        return {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model file
    model_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pkl")
    feature_path = os.path.join(output_dir, f"{model_name}_{timestamp}_features.pkl")
    metrics_path = os.path.join(output_dir, f"{model_name}_{timestamp}_metrics.json")
    
    try:
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save features list
        with open(feature_path, "wb") as f:
            pickle.dump(features, f)
        
        # Save metrics as JSON (convert numpy values to Python types)
        import json
        with open(metrics_path, "w") as f:
            # Convert numpy types to Python native types
            cleaned_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    cleaned_metrics[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            cleaned_metrics[key][k] = {}
                            for k2, v2 in v.items():
                                if isinstance(v2, list):
                                    cleaned_metrics[key][k][k2] = [float(x) if isinstance(x, np.number) else x for x in v2]
                                else:
                                    cleaned_metrics[key][k][k2] = float(v2) if isinstance(v2, np.number) else v2
                        elif isinstance(v, list):
                            cleaned_metrics[key][k] = [float(x) if isinstance(x, np.number) else x for x in v]
                        else:
                            cleaned_metrics[key][k] = float(v) if isinstance(v, np.number) else v
                elif isinstance(value, list):
                    cleaned_metrics[key] = [float(x) if isinstance(x, np.number) else x for x in value]
                else:
                    cleaned_metrics[key] = float(value) if isinstance(value, np.number) else value
            
            json.dump(cleaned_metrics, f, indent=2)
        
        logger.info(f"Model saved successfully to {model_path}")
        
        # Return paths to saved files
        return {
            "model_path": model_path,
            "feature_path": feature_path,
            "metrics_path": metrics_path,
        }
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return {}


def load_model(model_path: str, feature_path: Optional[str] = None) -> Tuple[XGBRegressor, Optional[List[str]]]:
    """
    Load a trained model and optionally its feature list from saved files.
    
    Args:
        model_path: Path to the saved model file
        feature_path: Optional path to the saved feature list
        
    Returns:
        Tuple of (loaded model, feature list or None)
    """
    model = None
    features = None
    
    try:
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load features if provided
        if feature_path and os.path.exists(feature_path):
            with open(feature_path, "rb") as f:
                features = pickle.load(f)
        
        logger.info(f"Model loaded successfully from {model_path}")
        if features:
            logger.info(f"Loaded {len(features)} features")
        
        return model, features
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None


def predict_with_model(
    model: XGBRegressor,
    df: pd.DataFrame,
    features: List[str]
) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained XGBoost model
        df: DataFrame with feature columns
        features: List of feature names to use
        
    Returns:
        NumPy array of predictions
    """
    if model is None:
        logger.error("Cannot predict with None model")
        return np.array([])
    
    if df.empty:
        logger.warning("Empty DataFrame provided for prediction")
        return np.array([])
    
    # Check if all required features are present
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns for prediction: {missing_cols}")
        return np.array([])
    
    try:
        logger.info(f"Making predictions on {len(df)} samples")
        predictions = model.predict(df[features])
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return np.array([])


def compute_weighted_score(
    metrics: Dict[str, float],
    weights: Dict[str, float] = None,
    customer_segment: str = "RES_SOLAR"
) -> float:
    """
    Compute a weighted score from multiple metrics.
    
    Args:
        metrics: Dictionary of metric name to value
        weights: Dictionary of metric name to weight
        
    Returns:
        Weighted score (lower is better)
    """
    if not metrics:
        return float('inf')
    
    # Use segment-specific weights if not provided
    if weights is None:
        from configs.config import get_segment_config
        segment_config = get_segment_config(customer_segment)
        weights = segment_config.get("metric_weights", {})
    
    if not weights:
        return float('inf')
    
    score = 0.0
    total_weight = 0.0
    
    for metric_name, weight in weights.items():
        if metric_name in metrics:
            # For R² higher is better, so we convert to 1-R² for consistency
            if metric_name.startswith('r2'):
                metric_value = 1 - metrics[metric_name]
            else:
                metric_value = metrics[metric_name]
            
            score += weight * metric_value
            total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        score /= total_weight
    
    return score


def filter_by_rate_group(
    df: pd.DataFrame,
    rate_group_mapping: Dict[str, List[str]],
    rate_group_col: str = "rategroup"
) -> Dict[str, pd.DataFrame]:
    """
    Filter DataFrame by rate groups according to mapping.
    
    Args:
        df: DataFrame with rate group column
        rate_group_mapping: Dictionary mapping group name to list of rate codes
        rate_group_col: Column name containing rate group codes
        
    Returns:
        Dictionary of group name to filtered DataFrame
    """
    if df.empty or rate_group_col not in df.columns:
        logger.error(f"Cannot filter: DataFrame empty or missing column {rate_group_col}")
        return {"all": df}
    
    result = {}
    
    # Add the complete dataset
    result["all"] = df
    
    # Filter for each group in the mapping
    for group_name, rate_codes in rate_group_mapping.items():
        group_mask = df[rate_group_col].isin(rate_codes)
        group_df = df[group_mask]
        
        if group_df.empty:
            logger.warning(f"No data for rate group '{group_name}' with codes {rate_codes}")
        else:
            logger.info(f"Filtered {len(group_df)} rows for rate group '{group_name}'")
            result[group_name] = group_df
    
    return result
