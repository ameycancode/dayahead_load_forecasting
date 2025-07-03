"""
Evaluation utilities for energy load forecasting models.

This module provides functions for evaluating model performance,
generating visualizations, and analyzing prediction errors for both
solar and non-solar customer segments.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

from configs.config import CUSTOMER_SEGMENTS


logger = logging.getLogger(__name__)

# Import boto3 for S3 uploads
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available, S3 uploads will be disabled")


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error - handles zero and near-zero values better.
   
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
       
    Returns:
        sMAPE value (lower is better)
    """
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error - better for zero-crossing data.
   
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
       
    Returns:
        WAPE value (lower is better)
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)

def mase(y_true, y_pred, y_train=None, seasonality=24):
    """
    Mean Absolute Scaled Error - scaled relative to naive forecast.
   
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
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
        y_true: Ground truth values
        y_pred: Predicted values
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


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate standard regression metrics for predictions.
   
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Optional prefix for metric names
       
    Returns:
        Dictionary of metrics
    """
    # Add prefix with underscore if provided
    prefix = f"{prefix}_" if prefix else ""
   
    metrics = {}
   
    # Handle zero division in MAPE by replacing zeros with small value
    y_true_safe = np.array(y_true).copy()
    zero_mask = (y_true_safe == 0)
   
    if np.any(zero_mask):
        logger.warning(f"Found {np.sum(zero_mask)} zero values in ground truth. Replacing with small value for MAPE calculation.")
        y_true_safe[zero_mask] = 1e-10
   
    # Calculate standard metrics
    metrics[f"{prefix}rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics[f"{prefix}mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics[f"{prefix}r2"] = float(r2_score(y_true, y_pred))
   
    # Calculate MAPE with safe values
    mape = np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100
    metrics[f"{prefix}mape"] = float(mape)

    metrics[f"{prefix}smape"] = float(smape(y_true, y_pred))
    metrics[f"{prefix}wape"] = float(wape(y_true, y_pred))
    metrics[f"{prefix}mase"] = float(mase(y_true, y_pred))
   
    # Calculate transition-weighted error
    metrics[f"{prefix}trans_weighted_error"] = float(transition_weighted_error(y_true, y_pred))
   
    # Calculate normalized RMSE (NRMSE) - RMSE divided by the range of true values
    y_range = np.max(y_true) - np.min(y_true)
    if y_range > 0:
        metrics[f"{prefix}nrmse"] = float(metrics[f"{prefix}rmse"] / y_range)
    else:
        metrics[f"{prefix}nrmse"] = float('nan')
   
    # Calculate RMSE as percentage of mean (CV-RMSE)
    y_mean = np.mean(y_true)
    if y_mean > 0:
        metrics[f"{prefix}cv_rmse"] = float(metrics[f"{prefix}rmse"] / y_mean * 100)
    else:
        metrics[f"{prefix}cv_rmse"] = float('nan')
   
    return metrics


def calculate_time_weighted_metrics(df, predictions, target, date_column, segment_config):
    """Calculate metrics with different weights for different time periods."""
   
    profile = segment_config["profile"]
    has_solar = segment_config["has_solar"]
    commercial_metrics = segment_config.get("commercial_metrics", False)
   
    metrics = {}
   
    if commercial_metrics:
        # Commercial customers - business hours are most important
        business_hours_weight = 0.7
        off_hours_weight = 0.2
        weekend_weight = 0.1
       
        # Create time masks
        is_business_hour = (df[date_column].dt.hour >= 8) & (df[date_column].dt.hour < 18)
        is_weekday = df[date_column].dt.weekday < 5
        is_business_time = is_business_hour & is_weekday
       
        # Calculate weighted metrics
        business_mask = is_business_time
        off_hours_mask = ~is_business_hour & is_weekday  # Weekday off hours
        weekend_mask = ~is_weekday  # All weekend hours
       
    else:  # Residential customers
        # Residential customers - more even distribution but peak hours matter more
        peak_hours_weight = 0.4
        regular_hours_weight = 0.4
        off_peak_weight = 0.2
       
        # Peak hours: 7-9 AM, 5-9 PM
        is_morning_peak = (df[date_column].dt.hour >= 7) & (df[date_column].dt.hour < 9)
        is_evening_peak = (df[date_column].dt.hour >= 17) & (df[date_column].dt.hour < 21)
        is_peak = is_morning_peak | is_evening_peak
       
        business_mask = is_peak
        off_hours_mask = (df[date_column].dt.hour >= 22) | (df[date_column].dt.hour < 6)
        weekend_mask = ~is_peak & ~off_hours_mask
       
        business_hours_weight = peak_hours_weight
        off_hours_weight = off_peak_weight
        weekend_weight = regular_hours_weight
   
    # Calculate metrics for each period
    if business_mask.sum() > 0:
        business_metrics = evaluate_predictions(
            df[target][business_mask],
            predictions[business_mask],
            prefix="business_critical"
        )
        metrics.update(business_metrics)
   
    if off_hours_mask.sum() > 0:
        off_hours_metrics = evaluate_predictions(
            df[target][off_hours_mask],
            predictions[off_hours_mask],
            prefix="off_hours"
        )
        metrics.update(off_hours_metrics)
   
    if weekend_mask.sum() > 0:
        weekend_metrics = evaluate_predictions(
            df[target][weekend_mask],
            predictions[weekend_mask],
            prefix="weekend"
        )
        metrics.update(weekend_metrics)
   
    # Calculate weighted overall score
    weighted_rmse = 0
    weighted_mape = 0
    total_weight = 0
   
    if "business_critical_rmse" in metrics:
        weighted_rmse += metrics["business_critical_rmse"] * business_hours_weight
        weighted_mape += metrics["business_critical_mape"] * business_hours_weight
        total_weight += business_hours_weight
   
    if "off_hours_rmse" in metrics:
        weighted_rmse += metrics["off_hours_rmse"] * off_hours_weight
        weighted_mape += metrics["off_hours_mape"] * off_hours_weight
        total_weight += off_hours_weight
   
    if "weekend_rmse" in metrics:
        weighted_rmse += metrics["weekend_rmse"] * weekend_weight
        weighted_mape += metrics["weekend_mape"] * weekend_weight
        total_weight += weekend_weight
   
    if total_weight > 0:
        metrics["weighted_overall_rmse"] = float(weighted_rmse / total_weight)
        metrics["weighted_overall_mape"] = float(weighted_mape / total_weight)
   
    return metrics


def calculate_commercial_specific_metrics(df, predictions, target, date_column):
    """Calculate metrics specific to commercial load patterns."""
   
    metrics = {}
   
    # 1. Business Hours Performance (Most Critical)
    business_mask = (df[date_column].dt.hour >= 8) & (df[date_column].dt.hour < 18) & (df[date_column].dt.weekday < 5)
    if business_mask.sum() > 0:
        business_actual = df[target][business_mask]
        business_pred = predictions[business_mask]
       
        business_metrics = evaluate_predictions(business_actual, business_pred, prefix="business_hours")
        business_metrics.update({
            "business_hours_samples": business_mask.sum(),
            "business_hours_avg_load": float(business_actual.mean()),
            "business_hours_peak_load": float(business_actual.max())
        })
        metrics.update(business_metrics)
   
    # 2. Peak Business Hours (10 AM - 4 PM)
    peak_business_mask = (df[date_column].dt.hour >= 10) & (df[date_column].dt.hour < 16) & (df[date_column].dt.weekday < 5)
    if peak_business_mask.sum() > 0:
        peak_actual = df[target][peak_business_mask]
        peak_pred = predictions[peak_business_mask]
       
        peak_metrics = evaluate_predictions(peak_actual, peak_pred, prefix="peak_business")
        metrics.update(peak_metrics)
   
    # 3. Transition Hours (Business start/end)
    start_transition = (df[date_column].dt.hour >= 7) & (df[date_column].dt.hour < 9) & (df[date_column].dt.weekday < 5)
    end_transition = (df[date_column].dt.hour >= 17) & (df[date_column].dt.hour < 19) & (df[date_column].dt.weekday < 5)
   
    if start_transition.sum() > 0:
        start_metrics = evaluate_predictions(
            df[target][start_transition],
            predictions[start_transition],
            prefix="morning_transition"
        )
        metrics.update(start_metrics)
   
    if end_transition.sum() > 0:
        end_metrics = evaluate_predictions(
            df[target][end_transition],
            predictions[end_transition],
            prefix="evening_transition"
        )
        metrics.update(end_metrics)
   
    # 4. Low-Load Performance (Nights/Weekends)
    low_load_mask = (~business_mask) & (df[target] < df[target].quantile(0.2))
    if low_load_mask.sum() > 0:
        # Use different metrics for low-load periods
        low_actual = df[target][low_load_mask]
        low_pred = predictions[low_load_mask]
       
        # Use absolute errors instead of percentage errors for low loads
        mae = float(np.mean(np.abs(low_actual - low_pred)))
        rmse = float(np.sqrt(np.mean((low_actual - low_pred) ** 2)))
       
        metrics.update({
            "low_load_mae": mae,
            "low_load_rmse": rmse,
            "low_load_max_absolute_error": float(np.max(np.abs(low_actual - low_pred))),
            "low_load_samples": low_load_mask.sum()
        })
   
    # 5. Load Ramp Analysis (Important for commercial)
    df_copy = df.copy()
    df_copy['predictions'] = predictions
    df_copy['hour'] = df_copy[date_column].dt.hour
   
    # Morning ramp (7-10 AM)
    morning_ramp_hours = [7, 8, 9, 10]
    evening_ramp_hours = [16, 17, 18, 19]
   
    for ramp_type, hours in [("morning_ramp", morning_ramp_hours), ("evening_ramp", evening_ramp_hours)]:
        ramp_errors = []
        for hour in hours:
            hour_mask = (df_copy['hour'] == hour) & (df_copy[date_column].dt.weekday < 5)
            if hour_mask.sum() > 0:
                hour_error = np.mean(np.abs(df_copy[target][hour_mask] - df_copy['predictions'][hour_mask]))
                ramp_errors.append(hour_error)
       
        if ramp_errors:
            metrics.update({
                f"{ramp_type}_avg_mae": float(np.mean(ramp_errors)),
                f"{ramp_type}_max_mae": float(np.max(ramp_errors)),
                f"{ramp_type}_consistency": float(np.std(ramp_errors))  # Lower is better
            })
   
    return metrics


def calculate_solar_specific_metrics(df, predictions, target, date_column):
    """Calculate solar-specific metrics (only for solar customers)."""
    solar_metrics = {}
   
    # Duck curve hours (typically 3-6 PM when solar production drops but demand rises)
    duck_curve_mask = (df[date_column].dt.hour >= 15) & (df[date_column].dt.hour < 18)
    if duck_curve_mask.sum() > 0:
        duck_actual = df[target][duck_curve_mask]
        duck_pred = predictions[duck_curve_mask]
        duck_metrics = evaluate_predictions(duck_actual, duck_pred, prefix="duck_curve")
        solar_metrics.update(duck_metrics)
   
    # High solar generation hours (11 AM - 2 PM)
    solar_peak_mask = (df[date_column].dt.hour >= 11) & (df[date_column].dt.hour < 14)
    if solar_peak_mask.sum() > 0:
        solar_actual = df[target][solar_peak_mask]
        solar_pred = predictions[solar_peak_mask]
        solar_metrics_period = evaluate_predictions(solar_actual, solar_pred, prefix="solar_peak")
        solar_metrics.update(solar_metrics_period)
   
    return solar_metrics


def analyze_duck_curve_performance(df, predictions, target, date_column):
    """Analyze model performance during duck curve periods."""
    duck_metrics = {}
   
    # Calculate hour-by-hour performance during duck curve transition
    for hour in range(14, 19):  # 2 PM to 7 PM
        hour_mask = df[date_column].dt.hour == hour
        if hour_mask.sum() > 0:
            hour_actual = df[target][hour_mask]
            hour_pred = predictions[hour_mask]
            hour_metrics = evaluate_predictions(hour_actual, hour_pred, prefix=f"hour_{hour}")
            duck_metrics.update(hour_metrics)
   
    return duck_metrics


def evaluate_by_period(
    df: pd.DataFrame,
    date_column: str,
    true_column: str,
    pred_column: str,
    periods: Dict[str, Tuple[int, int]],
    has_solar: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions for specific time periods.
   
    Args:
        df: DataFrame with timestamps, true values, and predictions
        date_column: Name of date/time column
        true_column: Name of column with true values
        pred_column: Name of column with predictions
        periods: Dictionary mapping period names to (start_hour, end_hour) tuples
        has_solar: Whether this is a solar customer segment
       
    Returns:
        Dictionary of period name to metrics dictionary
    """
    if date_column not in df.columns or true_column not in df.columns or pred_column not in df.columns:
        logger.error(f"Missing required columns: {date_column}, {true_column}, or {pred_column}")
        return {}
   
    # Ensure datetime column
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
   
    # Get hour of day
    df['hour'] = df[date_column].dt.hour
   
    # Dictionary to store period metrics
    period_metrics = {}
   
    # Evaluate overall
    period_metrics["overall"] = evaluate_predictions(
        df[true_column], df[pred_column], prefix="overall"
    )
   
    # Evaluate each period
    for period_name, period_def in periods.items():
        # Initialize period_mask to None
        period_mask = None
       
        # Handle special case for weekend
        if period_name == "weekend" and period_def == "weekend":
            # Handle weekend as day-of-week filter instead of hour range
            weekend_mask = df[date_column].dt.weekday >= 5
            period_mask = weekend_mask
        else:
            # Handle regular hour-based periods
            if isinstance(period_def, tuple) and len(period_def) == 2:
                start_hour, end_hour = period_def
               
                # Handle periods that cross midnight
                if start_hour <= end_hour:
                    period_mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
                else:
                    period_mask = (df['hour'] >= start_hour) | (df['hour'] < end_hour)
            else:
                logger.warning(f"Invalid period definition for '{period_name}': {period_def}")
                continue
       
        # Only proceed if we have a valid period_mask
        if period_mask is not None and period_mask.sum() > 0:
            # Calculate metrics for this period
            period_metrics[period_name] = evaluate_predictions(
                df[true_column][period_mask], df[pred_column][period_mask], prefix=period_name
            )
           
            # Add extra info
            period_metrics[period_name][f"{period_name}_sample_count"] = period_mask.sum()
            period_metrics[period_name][f"{period_name}_mean_load"] = float(df[true_column][period_mask].mean())
           
            if isinstance(period_def, tuple) and len(period_def) == 2:
                start_hour, end_hour = period_def
                logger.info(f"Period '{period_name}' ({start_hour}-{end_hour}h): "
                     f"RMSE = {period_metrics[period_name][f'{period_name}_rmse']:.2f}, "
                     f"MAPE = {period_metrics[period_name][f'{period_name}_mape']:.2f}%, "
                     f"SMAPE = {period_metrics[period_name][f'{period_name}_smape']:.2f}, "
                     f"WAPE = {period_metrics[period_name][f'{period_name}_wape']:.2f}")
            else:
                logger.info(f"Period '{period_name}': "
                     f"RMSE = {period_metrics[period_name][f'{period_name}_rmse']:.2f}, "
                     f"MAPE = {period_metrics[period_name][f'{period_name}_mape']:.2f}%, "
                     f"SMAPE = {period_metrics[period_name][f'{period_name}_smape']:.2f}, "
                     f"WAPE = {period_metrics[period_name][f'{period_name}_wape']:.2f}")
        else:
            if period_mask is None:
                logger.warning(f"Could not create period mask for '{period_name}' with definition: {period_def}")
            else:
                logger.warning(f"No data for period '{period_name}' - mask sum: {period_mask.sum()}")
   
    return period_metrics


def evaluate_by_day_type(
    df: pd.DataFrame,
    date_column: str,
    true_column: str,
    pred_column: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions by day type (weekday vs weekend).
   
    Args:
        df: DataFrame with timestamps, true values, and predictions
        date_column: Name of date/time column
        true_column: Name of column with true values
        pred_column: Name of column with predictions
       
    Returns:
        Dictionary of day type to metrics dictionary
    """
    if date_column not in df.columns or true_column not in df.columns or pred_column not in df.columns:
        logger.error(f"Missing required columns: {date_column}, {true_column}, or {pred_column}")
        return {}
   
    # Ensure datetime column
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
   
    # Add day of week
    df['dayofweek'] = df[date_column].dt.dayofweek
   
    # Mark weekdays and weekends
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int) # 5, 6 are Sat and Sun
   
    # Dictionary to store day type metrics
    day_type_metrics = {}
   
    # Evaluate weekdays
    weekday_df = df[df['is_weekend'] == 0]
    if len(weekday_df) > 0:
        day_type_metrics["weekday"] = evaluate_predictions(
            weekday_df[true_column], weekday_df[pred_column], prefix="weekday"
        )
        day_type_metrics["weekday"]["weekday_sample_count"] = len(weekday_df)
   
    # Evaluate weekends
    weekend_df = df[df['is_weekend'] == 1]
    if len(weekend_df) > 0:
        day_type_metrics["weekend"] = evaluate_predictions(
            weekend_df[true_column], weekend_df[pred_column], prefix="weekend"
        )
        day_type_metrics["weekend"]["weekend_sample_count"] = len(weekend_df)
   
    return day_type_metrics


def evaluate_by_load_level(
    df: pd.DataFrame,
    true_column: str,
    pred_column: str,
    n_bins: int = 5,
    add_transition_bin: bool = True # New parameter to add special transition bin
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions by load level (binned load ranges).
   
    Args:
        df: DataFrame with true values and predictions
        true_column: Name of column with true values
        pred_column: Name of column with predictions
        n_bins: Number of load level bins to create
        add_transition_bin: Whether to add a special bin for transition region
       
    Returns:
        Dictionary of load level to metrics dictionary
    """
    if true_column not in df.columns or pred_column not in df.columns:
        logger.error(f"Missing required columns: {true_column} or {pred_column}")
        return {}
   
    df = df.copy()
   
    # Dictionary to store load level metrics
    load_level_metrics = {}
   
    if add_transition_bin:
        # First, create a special bin just for the transition region (near zero)
        transition_threshold = 20000 # Threshold for transition region (±20,000)
       
        # Create transition mask
        transition_mask = np.abs(df[true_column]) < transition_threshold
       
        if transition_mask.sum() > 0:
            # Calculate metrics for transition region
            transition_df = df[transition_mask]
           
            transition_metrics = evaluate_predictions(
                transition_df[true_column], transition_df[pred_column], prefix="transition"
            )
           
            # Add extra info
            transition_metrics["transition_sample_count"] = len(transition_df)
            transition_metrics["transition_mean_load"] = float(transition_df[true_column].mean())
            transition_metrics["transition_min_load"] = float(transition_df[true_column].min())
            transition_metrics["transition_max_load"] = float(transition_df[true_column].max())
           
            # Store metrics
            load_level_metrics["transition_region"] = transition_metrics
           
            logger.info(f"Transition region (±{transition_threshold}): "
                       f"RMSE = {transition_metrics['transition_rmse']:.2f}, "
                       f"MAPE = {transition_metrics['transition_mape']:.2f}%, "
                       f"sMAPE = {transition_metrics['transition_smape']:.2f}%, "
                       f"WAPE = {transition_metrics['transition_wape']:.2f}%")
   
    # Continue with regular binning for the rest
    # Create load level bins
    df['load_level'] = pd.qcut(df[true_column], n_bins, labels=False)
   
    # Get bin edges for labeling
    bin_edges = pd.qcut(df[true_column], n_bins, retbins=True)[1]
   
    # Evaluate each load level
    for level in range(n_bins):
        level_df = df[df['load_level'] == level]
       
        if len(level_df) > 0:
            # Create level name with range
            level_min = bin_edges[level]
            level_max = bin_edges[level + 1]
            level_name = f"load_{level}_({level_min:.0f}-{level_max:.0f})"
           
            # Calculate metrics for this level
            load_level_metrics[level_name] = evaluate_predictions(
                level_df[true_column], level_df[pred_column], prefix=f"load_{level}"
            )
           
            # Add extra info
            load_level_metrics[level_name][f"load_{level}_sample_count"] = len(level_df)
            load_level_metrics[level_name][f"load_{level}_mean_load"] = float(level_df[true_column].mean())
           
            logger.info(f"Load level {level} ({level_min:.0f}-{level_max:.0f}): "
                       f"RMSE = {load_level_metrics[level_name][f'load_{level}_rmse']:.2f}, "
                       f"MAPE = {load_level_metrics[level_name][f'load_{level}_mape']:.2f}%, "
                       f"sMAPE = {load_level_metrics[level_name][f'load_{level}_smape']:.2f}%, "
                       f"WAPE = {load_level_metrics[level_name][f'load_{level}_wape']:.2f}%")
   
    return load_level_metrics


def create_segment_specific_plots(df, predictions, target, date_column, segment_config, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create plots specific to the customer segment."""
    has_solar = segment_config["has_solar"]
    periods = segment_config["evaluation_periods"]
    commercial_metrics = segment_config.get("commercial_metrics", False)
   
    # Common plots for all segments
    create_time_series_plot(df, predictions, target, date_column, output_dir, prefix, run_id, upload_to_s3)
    create_scatter_plot(df[target], predictions, output_dir, prefix, run_id, upload_to_s3)
    create_residual_plots(df[target], predictions, output_dir, prefix, run_id, upload_to_s3)
   
    # Create daily profile plot with segment-specific highlighting
    create_daily_profile_plot(df, predictions, target, date_column, periods, output_dir, prefix, run_id, upload_to_s3)
   
    # Solar-specific plots
    if has_solar:
        create_duck_curve_plot(df, predictions, target, date_column, output_dir, prefix, run_id, upload_to_s3)
        create_solar_impact_plot(df, predictions, target, date_column, output_dir, prefix, run_id, upload_to_s3)
   
    # Commercial-specific plots
    if commercial_metrics:
        create_business_hours_plot(df, predictions, target, date_column, output_dir, prefix, run_id, upload_to_s3)
        create_weekday_weekend_comparison(df, predictions, target, date_column, output_dir, prefix, run_id, upload_to_s3)
    else:
        # Non-commercial (residential) specific plots
        create_load_pattern_plot(df, predictions, target, date_column, output_dir, prefix, run_id, upload_to_s3)


def create_time_series_plot(df, predictions, target, date_column, output_dir, prefix, run_id=None, upload_to_s3=True, n_days=7):
    """Create time series plot with last n days."""
    try:
        # Get the most recent n days of data
        df_copy = df.copy()
        df_copy['predictions'] = predictions
       
        latest_date = df_copy[date_column].max()
        start_date = latest_date - timedelta(days=n_days)
        recent_df = df_copy[df_copy[date_column] >= start_date]
       
        if len(recent_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
           
            # Plot true values
            ax.plot(recent_df[date_column], recent_df[target],
                    label='Actual', linewidth=2, color='#2077B4')
           
            # Plot predictions
            ax.plot(recent_df[date_column], recent_df['predictions'],
                    label='Predicted', linewidth=2, color='#FF7F0E', linestyle='--')
           
            # Shade weekend periods
            for i in range((n_days // 7) + 2):
                for day_offset in range(7):
                    test_date = start_date + timedelta(days=i*7 + day_offset)
                    if test_date.weekday() == 5:  # Saturday
                        weekend_start = test_date
                        weekend_end = test_date + timedelta(days=2)  # End of Sunday
                        ax.axvspan(weekend_start, weekend_end, alpha=0.1, color='gray',
                                   label='_Weekend' if i > 0 else 'Weekend')
           
            # Formatting
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Load (kW)', fontsize=12)
            ax.set_title(f'Energy Load Forecast vs Actual (Last {n_days} Days)', fontsize=14)
           
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
           
            # Save the plot
            ts_path = os.path.join(output_dir, f"{prefix}_time_series.png")
            plt.savefig(ts_path, dpi=300, bbox_inches='tight')
            plt.close()
           
            # Upload to S3 if requested
            if upload_to_s3 and run_id:
                upload_plot_to_s3(ts_path, "time_series", run_id)
           
            logger.info(f"Created time series plot at {ts_path}")
   
    except Exception as e:
        logger.error(f"Error creating time series plot: {str(e)}")


def create_scatter_plot(y_true, y_pred, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create scatter plot with true vs predicted values."""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
       
        # Plot scatter
        ax.scatter(y_true, y_pred, alpha=0.5, color='#2077B4')
       
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
       
        # Calculate R²
        r2 = r2_score(y_true, y_pred)
       
        # Formatting
        ax.set_xlabel('Actual Load (kW)', fontsize=12)
        ax.set_ylabel('Predicted Load (kW)', fontsize=12)
        ax.set_title(f'Predicted vs Actual Load (R² = {r2:.4f})', fontsize=14)
       
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
       
        # Save the plot
        scatter_path = os.path.join(output_dir, f"{prefix}_scatter.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(scatter_path, "scatter", run_id)
       
        logger.info(f"Created scatter plot at {scatter_path}")
   
    except Exception as e:
        logger.error(f"Error creating scatter plot: {str(e)}")


def create_residual_plots(y_true, y_pred, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create residual plot (error vs true value)."""
    try:
        # Calculate residuals
        residuals = y_true - y_pred
       
        fig, ax = plt.subplots(figsize=(12, 8))
       
        # Plot residuals
        ax.scatter(y_true, residuals, alpha=0.5, color='#2077B4')
       
        # Add zero line
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
       
        # Formatting
        ax.set_xlabel('Actual Load (kW)', fontsize=12)
        ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14)
       
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
       
        # Save the plot
        residual_path = os.path.join(output_dir, f"{prefix}_residual.png")
        plt.savefig(residual_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(residual_path, "residual", run_id)
       
        logger.info(f"Created residual plot at {residual_path}")
   
    except Exception as e:
        logger.error(f"Error creating residual plot: {str(e)}")


def create_daily_profile_plot(df, predictions, target, date_column, periods, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create daily load profile plot with period highlighting."""
    try:
        # Calculate hourly averages
        df_copy = df.copy()
        df_copy['predictions'] = predictions
        df_copy['hour'] = df_copy[date_column].dt.hour
       
        hourly_actual = df_copy.groupby('hour')[target].mean()
        hourly_pred = df_copy.groupby('hour')['predictions'].mean()
       
        fig, ax = plt.subplots(figsize=(12, 8))
       
        # Plot lines
        ax.plot(hourly_actual.index, hourly_actual.values, 'b-', label='Actual', linewidth=2)
        ax.plot(hourly_pred.index, hourly_pred.values, 'r--', label='Predicted', linewidth=2)
       
        # Highlight important periods
        colors = ['yellow', 'orange', 'lightcoral', 'lightblue', 'lightgreen', 'lightpink']
        for i, (period_name, (start_hour, end_hour)) in enumerate(periods.items()):
            color = colors[i % len(colors)]
           
            if start_hour <= end_hour:
                ax.axvspan(start_hour, end_hour, alpha=0.3, color=color,
                           label=period_name.replace('_', ' ').title())
            else:
                # Handle periods crossing midnight
                ax.axvspan(start_hour, 24, alpha=0.3, color=color,
                           label=f"{period_name.replace('_', ' ').title()} (part 1)")
                ax.axvspan(0, end_hour, alpha=0.3, color=color,
                           label=f"{period_name.replace('_', ' ').title()} (part 2)")
       
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Load (kW)')
        ax.set_title(f'Daily Load Profile - {prefix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
       
        # Save the plot
        profile_path = os.path.join(output_dir, f'{prefix}_daily_profile.png')
        plt.savefig(profile_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(profile_path, "daily_profile", run_id)
       
        logger.info(f"Created daily profile plot at {profile_path}")
   
    except Exception as e:
        logger.error(f"Error creating daily profile plot: {str(e)}")


def create_duck_curve_plot(df, predictions, target, date_column, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create duck curve specific plot for solar customers."""
    try:
        df_copy = df.copy()
        df_copy['predictions'] = predictions
        df_copy['hour'] = df_copy[date_column].dt.hour
       
        # Focus on duck curve hours (2 PM - 7 PM)
        duck_curve_hours = range(14, 20)
        duck_curve_df = df_copy[df_copy['hour'].isin(duck_curve_hours)]
       
        if len(duck_curve_df) > 0:
            # Calculate hourly averages for duck curve period
            hourly_actual = duck_curve_df.groupby('hour')[target].mean()
            hourly_pred = duck_curve_df.groupby('hour')['predictions'].mean()
           
            fig, ax = plt.subplots(figsize=(12, 8))
           
            # Plot lines
            ax.plot(hourly_actual.index, hourly_actual.values, 'b-o', label='Actual', linewidth=3, markersize=8)
            ax.plot(hourly_pred.index, hourly_pred.values, 'r--s', label='Predicted', linewidth=3, markersize=8)
           
            # Highlight critical duck curve transition
            ax.axvspan(15, 18, alpha=0.3, color='orange', label='Critical Duck Curve Period')
           
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Load (kW)')
            ax.set_title('Duck Curve Performance (Solar Impact Period)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(duck_curve_hours)
            plt.tight_layout()
           
            # Save the plot
            duck_path = os.path.join(output_dir, f'{prefix}_duck_curve.png')
            plt.savefig(duck_path, dpi=300, bbox_inches='tight')
            plt.close()
           
            # Upload to S3 if requested
            if upload_to_s3 and run_id:
                upload_plot_to_s3(duck_path, "duck_curve", run_id)
           
            logger.info(f"Created duck curve plot at {duck_path}")
   
    except Exception as e:
        logger.error(f"Error creating duck curve plot: {str(e)}")


def create_solar_impact_plot(df, predictions, target, date_column, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create solar impact analysis plot."""
    try:
        df_copy = df.copy()
        df_copy['predictions'] = predictions
        df_copy['hour'] = df_copy[date_column].dt.hour
        df_copy['error'] = df_copy[target] - df_copy['predictions']
        df_copy['abs_error'] = np.abs(df_copy['error'])
       
        # Calculate hourly error patterns
        hourly_errors = df_copy.groupby('hour')['abs_error'].mean()
       
        fig, ax = plt.subplots(figsize=(12, 8))
       
        # Plot hourly errors
        ax.plot(hourly_errors.index, hourly_errors.values, 'g-o', linewidth=2, markersize=6)
       
        # Highlight solar-specific periods
        ax.axvspan(11, 14, alpha=0.2, color='yellow', label='Solar Peak')
        ax.axvspan(15, 18, alpha=0.2, color='orange', label='Duck Curve')
        ax.axvspan(6, 9, alpha=0.2, color='lightblue', label='Morning Rise')
       
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Absolute Error (kW)')
        ax.set_title('Model Error Pattern Throughout Day (Solar Customer)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        plt.tight_layout()
       
        # Save the plot
        solar_path = os.path.join(output_dir, f'{prefix}_solar_impact.png')
        plt.savefig(solar_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(solar_path, "solar_impact", run_id)
       
        logger.info(f"Created solar impact plot at {solar_path}")
   
    except Exception as e:
        logger.error(f"Error creating solar impact plot: {str(e)}")


def create_business_hours_plot(df, predictions, target, date_column, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create business hours specific plot for commercial customers."""
    try:
        df_copy = df.copy()
        df_copy['predictions'] = predictions
        df_copy['hour'] = df_copy[date_column].dt.hour
        df_copy['is_weekday'] = df_copy[date_column].dt.weekday < 5
        df_copy['is_business_hour'] = (df_copy['hour'] >= 8) & (df_copy['hour'] < 18)
       
        # Create business vs non-business comparison
        business_df = df_copy[df_copy['is_business_hour'] & df_copy['is_weekday']]
        non_business_df = df_copy[~(df_copy['is_business_hour'] & df_copy['is_weekday'])]
       
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
       
        # Business hours performance
        if len(business_df) > 0:
            hourly_business_actual = business_df.groupby('hour')[target].mean()
            hourly_business_pred = business_df.groupby('hour')['predictions'].mean()
           
            ax1.plot(hourly_business_actual.index, hourly_business_actual.values, 'b-o', label='Actual', linewidth=2)
            ax1.plot(hourly_business_pred.index, hourly_business_pred.values, 'r--s', label='Predicted', linewidth=2)
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Load (kW)')
            ax1.set_title('Business Hours Performance (Weekdays 8-18)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(8, 19, 2))
       
        # Non-business hours performance
        if len(non_business_df) > 0:
            non_business_df['time_category'] = 'Other'
            non_business_df.loc[non_business_df[date_column].dt.weekday >= 5, 'time_category'] = 'Weekend'
            non_business_df.loc[(non_business_df[date_column].dt.weekday < 5) &
                               (~non_business_df['is_business_hour']), 'time_category'] = 'Off Hours'
           
            category_actual = non_business_df.groupby('time_category')[target].mean()
            category_pred = non_business_df.groupby('time_category')['predictions'].mean()
           
            x_pos = np.arange(len(category_actual))
            width = 0.35
           
            ax2.bar(x_pos - width/2, category_actual.values, width, label='Actual', alpha=0.8)
            ax2.bar(x_pos + width/2, category_pred.values, width, label='Predicted', alpha=0.8)
            ax2.set_xlabel('Time Category')
            ax2.set_ylabel('Average Load (kW)')
            ax2.set_title('Non-Business Hours Performance')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(category_actual.index)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
       
        plt.tight_layout()
       
        # Save the plot
        business_path = os.path.join(output_dir, f'{prefix}_business_hours.png')
        plt.savefig(business_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(business_path, "business_hours", run_id)
       
        logger.info(f"Created business hours plot at {business_path}")
   
    except Exception as e:
        logger.error(f"Error creating business hours plot: {str(e)}")


def create_weekday_weekend_comparison(df, predictions, target, date_column, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create weekday vs weekend comparison plot."""
    try:
        df_copy = df.copy()
        df_copy['predictions'] = predictions
        df_copy['hour'] = df_copy[date_column].dt.hour
        df_copy['is_weekend'] = df_copy[date_column].dt.weekday >= 5
       
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
       
        # Weekday pattern
        weekday_df = df_copy[~df_copy['is_weekend']]
        if len(weekday_df) > 0:
            hourly_weekday_actual = weekday_df.groupby('hour')[target].mean()
            hourly_weekday_pred = weekday_df.groupby('hour')['predictions'].mean()
           
            ax1.plot(hourly_weekday_actual.index, hourly_weekday_actual.values, 'b-', label='Actual', linewidth=2)
            ax1.plot(hourly_weekday_pred.index, hourly_weekday_pred.values, 'r--', label='Predicted', linewidth=2)
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Load (kW)')
            ax1.set_title('Weekday Load Pattern')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(0, 24, 4))
       
        # Weekend pattern
        weekend_df = df_copy[df_copy['is_weekend']]
        if len(weekend_df) > 0:
            hourly_weekend_actual = weekend_df.groupby('hour')[target].mean()
            hourly_weekend_pred = weekend_df.groupby('hour')['predictions'].mean()
           
            ax2.plot(hourly_weekend_actual.index, hourly_weekend_actual.values, 'b-', label='Actual', linewidth=2)
            ax2.plot(hourly_weekend_pred.index, hourly_weekend_pred.values, 'r--', label='Predicted', linewidth=2)
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Load (kW)')
            ax2.set_title('Weekend Load Pattern')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(0, 24, 4))
       
        plt.tight_layout()
       
        # Save the plot
        comparison_path = os.path.join(output_dir, f'{prefix}_weekday_weekend.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(comparison_path, "weekday_weekend", run_id)
       
        logger.info(f"Created weekday/weekend comparison plot at {comparison_path}")
   
    except Exception as e:
        logger.error(f"Error creating weekday/weekend comparison plot: {str(e)}")


def create_load_pattern_plot(df, predictions, target, date_column, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create load pattern plot for residential customers."""
    try:
        df_copy = df.copy()
        df_copy['predictions'] = predictions
        df_copy['hour'] = df_copy[date_column].dt.hour
        df_copy['error'] = df_copy[target] - df_copy['predictions']
        df_copy['abs_error'] = np.abs(df_copy['error'])
       
        # Calculate hourly patterns
        hourly_actual = df_copy.groupby('hour')[target].mean()
        hourly_pred = df_copy.groupby('hour')['predictions'].mean()
        hourly_error = df_copy.groupby('hour')['abs_error'].mean()
       
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
       
        # Load pattern
        ax1.plot(hourly_actual.index, hourly_actual.values, 'b-', label='Actual', linewidth=2)
        ax1.plot(hourly_pred.index, hourly_pred.values, 'r--', label='Predicted', linewidth=2)
       
        # Highlight residential peak periods
        ax1.axvspan(7, 9, alpha=0.2, color='yellow', label='Morning Peak')
        ax1.axvspan(17, 21, alpha=0.2, color='orange', label='Evening Peak')
       
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Load (kW)')
        ax1.set_title('Residential Load Pattern')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
       
        # Error pattern
        ax2.plot(hourly_error.index, hourly_error.values, 'g-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Mean Absolute Error (kW)')
        ax2.set_title('Model Error Pattern Throughout Day')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
       
        plt.tight_layout()
       
        # Save the plot
        pattern_path = os.path.join(output_dir, f'{prefix}_load_pattern.png')
        plt.savefig(pattern_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(pattern_path, "load_pattern", run_id)
       
        logger.info(f"Created load pattern plot at {pattern_path}")
   
    except Exception as e:
        logger.error(f"Error creating load pattern plot: {str(e)}")


def upload_plot_to_s3(local_path, plot_type, run_id):
    """Upload plot to S3 bucket."""
    if not BOTO3_AVAILABLE:
        return local_path
   
    try:
        s3_client = boto3.client('s3')
        bucket = os.environ.get('SM_HP_S3_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-data')
        prefix = os.environ.get('SM_HP_S3_PREFIX', 'res-load-forecasting')
        s3_key = f"{prefix}/plots/run_{run_id}/test_{plot_type}.png"
       
        s3_client.upload_file(local_path, bucket, s3_key)
        logger.info(f"Uploaded {plot_type} plot to s3://{bucket}/{s3_key}")
        return s3_key
    except Exception as e:
        logger.error(f"Error uploading {plot_type} plot to S3: {str(e)}")
        return local_path


def create_prediction_plots(
    df: pd.DataFrame,
    date_column: str,
    true_column: str,
    pred_column: str,
    output_dir: str,
    prefix: str = "prediction",
    periods: Optional[Dict[str, Tuple[int, int]]] = None,
    n_days: int = 7,
    run_id: str = None,
    upload_to_s3: bool = True,
    customer_segment: str = "RES_SOLAR"
) -> Dict[str, str]:
    """
    Create prediction plots and save to output directory.
   
    Args:
        df: DataFrame with timestamps, true values, and predictions
        date_column: Name of date/time column
        true_column: Name of column with true values
        pred_column: Name of column with predictions
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        periods: Dictionary mapping period names to (start_hour, end_hour) tuples
        n_days: Number of days to show in time series plot
        run_id: Run identifier for S3 uploads
        upload_to_s3: Whether to upload plots to S3
        customer_segment: Customer segment for segment-specific plots
       
    Returns:
        Dictionary mapping plot types to file paths
    """
    if date_column not in df.columns or true_column not in df.columns or pred_column not in df.columns:
        logger.error(f"Missing required columns: {date_column}, {true_column}, or {pred_column}")
        return {}
   
    # Get segment configuration
    segment_config = CUSTOMER_SEGMENTS.get(customer_segment, None)
    if segment_config is None:
        logger.warning(f"Unknown customer segment: {customer_segment}. Using RES_SOLAR default.")
        segment_config = CUSTOMER_SEGMENTS["RES_SOLAR"]
   
    # Ensure datetime column
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
   
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
   
    # Dictionary to store plot paths
    plot_paths = {}
   
    # Set larger figure size and improved styling
    plt.rcParams.update({'figure.figsize': (12, 8)})
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid') # Fallback for older matplotlib
   
    # Create segment-specific plots
    predictions_array = df[pred_column].values
    create_segment_specific_plots(df, predictions_array, true_column, date_column,
                                 segment_config, output_dir, prefix, run_id, upload_to_s3)
   
    # Create additional standard plots
    create_error_distribution_plot(df, true_column, pred_column, output_dir, prefix, run_id, upload_to_s3)
    create_error_heatmap(df, date_column, true_column, pred_column, output_dir, prefix, run_id, upload_to_s3)
   
    # Create period-specific performance plot if periods provided
    if periods:
        create_period_performance_plot(df, date_column, true_column, pred_column, periods, output_dir, prefix, run_id, upload_to_s3)
   
    return plot_paths


def create_error_distribution_plot(df, true_column, pred_column, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create distribution of errors plot."""
    try:
        # Calculate absolute percentage errors
        df_copy = df.copy()
        df_copy['ape'] = np.abs((df_copy[true_column] - df_copy[pred_column]) / df_copy[true_column].replace(0, 1e-10)) * 100
       
        fig, ax = plt.subplots(figsize=(12, 8))
       
        # Create histogram
        ax.hist(df_copy['ape'].clip(upper=100), bins=50, color='#2077B4', alpha=0.7)
       
        # Add vertical line at mean APE
        mean_ape = df_copy['ape'].mean()
        ax.axvline(x=mean_ape, color='r', linestyle='--', linewidth=2,
                   label=f'Mean APE: {mean_ape:.2f}%')
       
        # Formatting
        ax.set_xlabel('Absolute Percentage Error (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Prediction Errors', fontsize=14)
       
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
       
        # Save the plot
        error_dist_path = os.path.join(output_dir, f"{prefix}_error_distribution.png")
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(error_dist_path, "error_distribution", run_id)
       
        logger.info(f"Created error distribution plot at {error_dist_path}")
   
    except Exception as e:
        logger.error(f"Error creating error distribution plot: {str(e)}")


def create_error_heatmap(df, date_column, true_column, pred_column, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create heatmap of errors by hour and day of week."""
    try:
        df_copy = df.copy()
        df_copy['dayofweek'] = df_copy[date_column].dt.dayofweek
        df_copy['hour'] = df_copy[date_column].dt.hour
       
        # Create percentage error (not absolute)
        df_copy['pe'] = (df_copy[true_column] - df_copy[pred_column]) / df_copy[true_column].replace(0, 1e-10) * 100
       
        # Create pivot table
        error_pivot = df_copy.pivot_table(
            values='pe',
            index='dayofweek',
            columns='hour',
            aggfunc='mean'
        )
       
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
       
        # Create heatmap
        im = ax.imshow(error_pivot, cmap='RdBu_r', vmin=-50, vmax=50)
       
        # Set colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Percentage Error (%)', rotation=-90, va="bottom")
       
        # Set axis labels
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ax.set_yticks(np.arange(len(day_names)))
        ax.set_yticklabels(day_names)
       
        # Set x-axis ticks (hours)
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
       
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
       
        # Add title
        ax.set_title("Error Heatmap by Day of Week and Hour", fontsize=14)
       
        # Add grid
        for i in range(len(day_names)):
            for j in range(24):
                if j < error_pivot.shape[1] and i < error_pivot.shape[0]:
                    text = ax.text(j, i, f"{error_pivot.iloc[i, j]:.1f}",
                                   ha="center", va="center", color="black", fontsize=7)
       
        # Adjust layout
        fig.tight_layout()
       
        # Save the plot
        heatmap_path = os.path.join(output_dir, f"{prefix}_error_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(heatmap_path, "error_heatmap", run_id)
       
        logger.info(f"Created error heatmap at {heatmap_path}")
   
    except Exception as e:
        logger.error(f"Error creating error heatmap: {str(e)}")


def create_period_performance_plot(df, date_column, true_column, pred_column, periods, output_dir, prefix, run_id=None, upload_to_s3=True):
    """Create period-specific performance plot."""
    try:
        df_copy = df.copy()
        df_copy['hour'] = df_copy[date_column].dt.hour
       
        # Create dictionary to store metrics by period
        period_metrics = {}
       
        # Calculate metrics for each period
        for period_name, (start_hour, end_hour) in periods.items():
            # Filter data for this period
            if start_hour <= end_hour:
                period_mask = (df_copy['hour'] >= start_hour) & (df_copy['hour'] < end_hour)
            else:
                period_mask = (df_copy['hour'] >= start_hour) | (df_copy['hour'] < end_hour)
           
            period_df = df_copy[period_mask]
           
            if len(period_df) > 0:
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(period_df[true_column], period_df[pred_column]))
                y_true_safe = period_df[true_column].replace(0, 1e-10)
                mape = np.mean(np.abs((period_df[true_column] - period_df[pred_column]) / y_true_safe)) * 100
               
                # Store metrics
                period_metrics[period_name] = {
                    'rmse': rmse,
                    'mape': mape,
                    'count': len(period_df)
                }
       
        # Create figure for RMSE and MAPE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
       
        # Plot RMSE by period
        periods_list = list(period_metrics.keys())
        rmse_values = [period_metrics[p]['rmse'] for p in periods_list]
        mape_values = [period_metrics[p]['mape'] for p in periods_list]
       
        # RMSE plot
        ax1.bar(range(len(periods_list)), rmse_values, color='skyblue')
        ax1.set_xticks(range(len(periods_list)))
        ax1.set_xticklabels([p.replace('_', ' ').title() for p in periods_list], rotation=45)
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE by Time Period')
        ax1.grid(True, alpha=0.3, axis='y')
       
        # Add value labels
        for i, v in enumerate(rmse_values):
            ax1.text(i, v + max(rmse_values) * 0.02, f"{v:.1f}", ha='center')
       
        # MAPE plot
        ax2.bar(range(len(periods_list)), mape_values, color='lightgreen')
        ax2.set_xticks(range(len(periods_list)))
        ax2.set_xticklabels([p.replace('_', ' ').title() for p in periods_list], rotation=45)
        ax2.set_ylabel('MAPE (%)')
        ax2.set_title('MAPE by Time Period')
        ax2.grid(True, alpha=0.3, axis='y')
       
        # Add value labels
        for i, v in enumerate(mape_values):
            ax2.text(i, v + max(mape_values) * 0.02, f"{v:.1f}%", ha='center')
       
        # Adjust layout
        plt.tight_layout()
       
        # Save the plot
        period_perf_path = os.path.join(output_dir, f"{prefix}_period_performance.png")
        plt.savefig(period_perf_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        # Upload to S3 if requested
        if upload_to_s3 and run_id:
            upload_plot_to_s3(period_perf_path, "period_performance", run_id)
       
        logger.info(f"Created period performance plot at {period_perf_path}")
   
    except Exception as e:
        logger.error(f"Error creating period-specific plots: {str(e)}")


def patch_xgboost_model(model):
    # Patch the XGBoost model to ensure compatibility between different versions.
    if not hasattr(model, 'gpu_id'):
        model.gpu_id = None

    return model


def evaluate_model(
    model,
    df: pd.DataFrame,
    features: List[str],
    target: str = 'lossadjustedload',
    date_column: str = 'datetime',
    output_dir: str = None,
    periods: Optional[Dict[str, Tuple[int, int]]] = None,
    create_plots: bool = True,
    prefix: str = "evaluation",
    run_id: str = None,
    customer_segment: str = "RES_SOLAR"
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with metrics and optional plots.
   
    Args:
        model: Trained model with predict method
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        date_column: Name of date/time column
        output_dir: Directory to save results and plots
        periods: Dictionary mapping period names to (start_hour, end_hour) tuples (optional, uses segment defaults)
        create_plots: Whether to create and save plots
        prefix: Prefix for output files
        run_id: Run identifier for S3 uploads
        customer_segment: Customer segment (e.g., "RES_SOLAR", "MEDCI_NONSOLAR")
       
    Returns:
        Dictionary with evaluation results
    """
    if df.empty or target not in df.columns:
        logger.error(f"Empty DataFrame or missing target column '{target}'")
        return {}
   
    try:
        logger.info(f"Starting comprehensive model evaluation on {len(df)} samples for segment {customer_segment}")

        # Get segment configuration
        segment_config = CUSTOMER_SEGMENTS.get(customer_segment, CUSTOMER_SEGMENTS["RES_SOLAR"])
       
        # Use provided periods or segment defaults
        if periods is None:
            periods = segment_config["evaluation_periods"]
       
        # Create predictions
        df = df.copy()
        try:
            predictions = model.predict(df[features])
            df['prediction'] = predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {'error': str(e)}
       
        # Dictionary for all evaluation metrics
        evaluation_results = {
            'dataset_info': {
                'n_samples': len(df),
                'date_range': (df[date_column].min().strftime('%Y-%m-%d'),
                               df[date_column].max().strftime('%Y-%m-%d')),
                'target_mean': float(df[target].mean()),
                'target_std': float(df[target].std()),
                'target_min': float(df[target].min()),
                'target_max': float(df[target].max()),
                'customer_segment': customer_segment,
                'segment_config': segment_config
            },
            'metrics': {},
            'metrics_by_period': {},
            'metrics_by_day_type': {},
            'metrics_by_load_level': {},
            'plot_paths': {}
        }
       
        # Overall metrics
        evaluation_results['metrics'] = evaluate_predictions(df[target], df['prediction'])

        # Time-weighted metrics
        time_weighted_metrics = calculate_time_weighted_metrics(df, predictions, target, date_column, segment_config)
        evaluation_results['time_weighted_metrics'] = time_weighted_metrics

        # Metrics by time period
        evaluation_results['metrics_by_period'] = evaluate_by_period(
            df, date_column, target, 'prediction', periods, segment_config["has_solar"]
        )
       
        # Metrics by day type (weekday vs weekend)
        evaluation_results['metrics_by_day_type'] = evaluate_by_day_type(
            df, date_column, target, 'prediction'
        )
       
        # Metrics by load level
        evaluation_results['metrics_by_load_level'] = evaluate_by_load_level(
            df, target, 'prediction', n_bins=5, add_transition_bin=True
        )
       
        # Segment-specific analysis
        if customer_segment == "RES_NONSOLAR":
            # Special analysis for non-solar residential
            res_nonsolar_metrics = calculate_residential_nonsolar_metrics(df, predictions, target, date_column)
            evaluation_results['residential_nonsolar_metrics'] = res_nonsolar_metrics
       
        elif segment_config["solar_specific_metrics"] and segment_config["has_solar"]:
            # Solar-specific analysis
            solar_analysis = calculate_solar_specific_metrics(df, predictions, target, date_column)
            evaluation_results['solar_analysis'] = solar_analysis
       
        # Duck curve analysis (only if applicable)
        if segment_config.get("duck_curve_analysis", False) and segment_config["has_solar"]:
            duck_curve_metrics = analyze_duck_curve_performance(df, predictions, target, date_column)
            evaluation_results['duck_curve_analysis'] = duck_curve_metrics
       
        # Commercial-specific metrics (if applicable)
        if segment_config.get("commercial_metrics", False):
            commercial_metrics = calculate_commercial_specific_metrics(df, predictions, target, date_column)
            evaluation_results['commercial_metrics'] = commercial_metrics
       
        # Create plots if requested
        if create_plots and output_dir:
            evaluation_results['plot_paths'] = create_prediction_plots(
                df, date_column, target, 'prediction', output_dir, prefix, periods,
                run_id=run_id, upload_to_s3=True, customer_segment=customer_segment
            )
       
        # Save evaluation results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
           
            results_path = os.path.join(output_dir, f"{prefix}_evaluation_results.json")
           
            def clean_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: clean_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                else:
                    return obj
           
            clean_results = clean_numpy_types(evaluation_results)
           
            with open(results_path, 'w') as f:
                json.dump(clean_results, f, indent=2)
           
            logger.info(f"Evaluation results saved to {results_path}")
       
        logger.info(f"Evaluation complete for {customer_segment}. Overall RMSE: {evaluation_results['metrics'].get('rmse', 0):.4f}, "
                    f"MAPE: {evaluation_results['metrics'].get('mape', 0):.2f}%, "
                    f"sMAPE: {evaluation_results['metrics'].get('smape', 0):.2f}%, "
                    f"WAPE: {evaluation_results['metrics'].get('wape', 0):.2f}%")
       
        return evaluation_results
   
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def compute_combined_score(
    metrics: Dict[str, float],
    metric_weights: Dict[str, float]
) -> float:
    """
    Compute a weighted combined score from multiple metrics.
   
    Args:
        metrics: Dictionary of metrics
        metric_weights: Dictionary mapping metric names to weights
       
    Returns:
        Combined weighted score (lower is better)
    """
    if not metrics or not metric_weights:
        return float('inf')
   
    score = 0.0
    total_weight = 0.0
   
    for metric_name, weight in metric_weights.items():
        if metric_name in metrics:
            # Handle R² differently (higher is better, unlike other metrics)
            if metric_name.startswith('r2'):
                # Convert to 1-R² so lower is better, consistent with other metrics
                metric_value = 1.0 - metrics[metric_name]
            else:
                metric_value = metrics[metric_name]
           
            score += weight * metric_value
            total_weight += weight
           
            logger.debug(f"Metric {metric_name}: {metrics[metric_name]} * weight {weight}")
   
    # Return average weighted score if valid weights found
    if total_weight > 0:
        return score / total_weight
    else:
        return float('inf')


def load_evaluation_results(input_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
   
    Args:
        input_path: Path to the evaluation results file
       
    Returns:
        Dictionary with evaluation results
    """
    try:
        with open(input_path, 'r') as f:
            results = json.load(f)
       
        logger.info(f"Loaded evaluation results from {input_path}")
        return results
   
    except Exception as e:
        logger.error(f"Error loading evaluation results: {str(e)}")
        return {}


def calculate_residential_nonsolar_metrics(df, predictions, target, date_column):
    """Calculate metrics specific to non-solar residential patterns."""
   
    metrics = {}

    # Ensure predictions is a pandas Series with the same index as df
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=df.index)
   
    # 1. Evening Super Peak Analysis (Most Critical for RES_NONSOLAR)
    evening_peak_mask = (df[date_column].dt.hour >= 17) & (df[date_column].dt.hour < 21)
    if evening_peak_mask.sum() > 0:
        peak_actual = df[target][evening_peak_mask]
        peak_pred = predictions[evening_peak_mask]
       
        # Standard metrics for evening peak
        peak_metrics = evaluate_predictions(peak_actual, peak_pred, prefix="evening_super_peak")
        metrics.update(peak_metrics)
       
        # Peak magnitude accuracy
        actual_max = peak_actual.max()
        actual_max_idx = peak_actual.idxmax()
        pred_at_actual_max_time = predictions[actual_max_idx]
       
        peak_magnitude_error = abs(actual_max - pred_at_actual_max_time) / (actual_max + 1e-8) * 100
        metrics["evening_peak_magnitude_error_pct"] = float(peak_magnitude_error)
       
        # Peak timing accuracy
        actual_peak_hour = df.loc[actual_max_idx, date_column].hour
        pred_max_idx = peak_pred.idxmax()
        pred_peak_hour = df.loc[pred_max_idx, date_column].hour if pred_max_idx in df.index else actual_peak_hour
       
        peak_timing_error = abs(actual_peak_hour - pred_peak_hour)
        metrics["evening_peak_timing_error_hours"] = float(peak_timing_error)
       
        # Evening ramp rate accuracy (critical for grid management)
        afternoon_hours = range(14, 21)  # 2 PM to 9 PM
        ramp_errors = []
       
        for hour in afternoon_hours:
            hour_mask = df[date_column].dt.hour == hour
            if hour_mask.sum() > 0:
                hour_actual_mean = df[target][hour_mask].mean()
                hour_pred_mean = predictions[hour_mask].mean()
                hour_error = abs(hour_actual_mean - hour_pred_mean)
                ramp_errors.append(hour_error)
       
        if ramp_errors:
            metrics["evening_ramp_avg_error"] = float(np.mean(ramp_errors))
            metrics["evening_ramp_max_error"] = float(np.max(ramp_errors))
            metrics["evening_ramp_consistency"] = float(np.std(ramp_errors))
   
    # 2. Midday Stability Analysis (should be stable for non-solar)
    midday_mask = (df[date_column].dt.hour >= 10) & (df[date_column].dt.hour < 14)
    if midday_mask.sum() > 0:
        midday_actual = df[target][midday_mask]
        midday_pred = predictions[midday_mask]
       
        # Standard metrics
        midday_metrics = evaluate_predictions(midday_actual, midday_pred, prefix="midday_stability")
        metrics.update(midday_metrics)
       
        # Stability metrics (should be relatively flat)
        actual_midday_std = float(midday_actual.std())
        pred_midday_std = float(midday_pred.std())
        stability_difference = abs(actual_midday_std - pred_midday_std)
       
        metrics["midday_stability_difference"] = stability_difference
        metrics["midday_actual_std"] = actual_midday_std
        metrics["midday_pred_std"] = pred_midday_std
   
    return metrics
