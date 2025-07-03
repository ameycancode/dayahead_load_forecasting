"""
Visualization utilities for energy load forecasting.

This module provides functions for creating visualizations of model performance,
hyperparameter optimization results, and predictions.
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

# Import boto3 for S3 uploads
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available, S3 uploads will be disabled")

def upload_to_s3(local_path, s3_key):
    """Upload a file to S3."""
    if not BOTO3_AVAILABLE:
        logger.warning("boto3 not available, skipping S3 upload")
        return local_path
    
    bucket = os.environ.get('SM_HP_S3_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-data')
    prefix = os.environ.get('SM_HP_S3_PREFIX', 'res-load-forecasting')
    
    try:
        s3_client = boto3.client('s3')
        full_s3_key = f"{prefix}/{s3_key}"
        logger.info(f"S3 location to be used in bucket {bucket}: {full_s3_key}")
        s3_client.upload_file(local_path, bucket, full_s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{full_s3_key}")
        return f"s3://{bucket}/{full_s3_key}"
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return local_path

def visualize_hpo_results(optimization_results, output_dir, run_id):
    """
    Create visualizations of HPO process and results.
    
    Args:
        optimization_results: Dictionary with HPO results
        output_dir: Directory to save plots
        run_id: Unique run identifier
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import os
    
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract trial data
    trials = optimization_results.get('trials', [])
    if not trials:
        logger.warning("No trial data available for visualization")
        return
    
    # Convert trials to DataFrame for easier plotting
    trial_data = []
    for trial in trials:
        trial_dict = {
            'trial': trial['number'],
            'score': trial['value']
        }
        # Add parameters
        for param_name, param_value in trial['params'].items():
            trial_dict[f'param_{param_name}'] = param_value
        
        # Add metrics if available
        metrics = trial.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            trial_dict[metric_name] = metric_value
        
        trial_data.append(trial_dict)
    
    trials_df = pd.DataFrame(trial_data)
    
    # 1. Trial improvement plot
    plt.figure(figsize=(12, 6))
    plt.plot(trials_df['trial'], trials_df['score'], 'o-', color='blue', alpha=0.6)
    plt.plot(trials_df['trial'], trials_df['score'].cummin(), 'r-', linewidth=2)
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Score (lower is better)')
    plt.title('HPO Progress: Objective Score Improvement Across Trials')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save locally and upload to S3
    local_path = os.path.join(output_dir, f'hpo_trial_improvement.png')
    plt.savefig(local_path, dpi=300, bbox_inches='tight')
    s3_key = f"plots/run_{run_id}/hpo_trial_improvement.png"
    upload_to_s3(local_path, s3_key)
    plt.close()
    
    # 2. Parameter parallel coordinates plot (helps visualize best parameters)
    if len(trials_df) > 5:  # Only create if we have enough trials
        plt.figure(figsize=(14, 8))
        # Select top 20% of trials by score
        top_trials = trials_df.nsmallest(max(1, int(len(trials_df) * 0.2)), 'score')
        
        # Get parameter columns
        param_cols = [col for col in trials_df.columns if col.startswith('param_')]
        
        # Normalize parameters for parallel coordinates
        param_data = trials_df[param_cols].copy()
        for col in param_cols:
            param_data[col] = (param_data[col] - param_data[col].min()) / (param_data[col].max() - param_data[col].min() + 1e-10)
        
        # Add score and color column
        param_data['score'] = trials_df['score']
        param_data['color'] = 'blue'
        param_data.loc[top_trials.index, 'color'] = 'red'
        
        # Plot all trials in blue (transparent)
        for idx, row in param_data.iterrows():
            if row['color'] == 'blue':
                coords = [param_cols.index(col) for col in param_cols]
                values = [row[col] for col in param_cols]
                plt.plot(coords, values, 'o-', color='blue', alpha=0.1, linewidth=1)
        
        # Plot top trials in red
        for idx, row in param_data.iterrows():
            if row['color'] == 'red':
                coords = [param_cols.index(col) for col in param_cols]
                values = [row[col] for col in param_cols]
                plt.plot(coords, values, 'o-', color='red', alpha=0.7, linewidth=2)
        
        # Set x-ticks to parameter names
        plt.xticks(range(len(param_cols)), [col.replace('param_', '') for col in param_cols], rotation=45)
        plt.ylabel('Normalized Parameter Value')
        plt.title('Parameter Values for Best vs All Trials')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save locally and upload to S3
        local_path = os.path.join(output_dir, f'hpo_parallel_coords.png')
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        s3_key = f"plots/run_{run_id}/hpo_parallel_coords.png"
        upload_to_s3(local_path, s3_key)
        plt.close()
    
    # 3. CV Split Performance
    # Extract CV metrics if available
    cv_metrics = optimization_results.get('best_metrics', {})
    if cv_metrics:
        metrics_to_plot = [m for m in cv_metrics.keys() if 'rmse' in m or 'mape' in m]
        
        # Plot RMSE metrics
        rmse_metrics = [m for m in metrics_to_plot if 'rmse' in m]
        if rmse_metrics:
            plt.figure(figsize=(12, 6))
            metric_values = [cv_metrics[m] for m in rmse_metrics]
            plt.bar(range(len(rmse_metrics)), metric_values, color='skyblue')
            plt.xticks(range(len(rmse_metrics)), [m.replace('_rmse', '').replace('_', ' ').title() for m in rmse_metrics], rotation=45)
            plt.ylabel('RMSE')
            plt.title('RMSE by Time Period in Best Trial')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save locally and upload to S3
            local_path = os.path.join(output_dir, f'hpo_rmse_by_period.png')
            plt.savefig(local_path, dpi=300, bbox_inches='tight')
            s3_key = f"plots/run_{run_id}/hpo_rmse_by_period.png"
            upload_to_s3(local_path, s3_key)
            plt.close()
        
        # Plot MAPE metrics
        mape_metrics = [m for m in metrics_to_plot if 'mape' in m]
        if mape_metrics:
            plt.figure(figsize=(12, 6))
            metric_values = [cv_metrics[m] for m in mape_metrics]
            plt.bar(range(len(mape_metrics)), metric_values, color='lightgreen')
            plt.xticks(range(len(mape_metrics)), [m.replace('_mape', '').replace('_', ' ').title() for m in mape_metrics], rotation=45)
            plt.ylabel('MAPE (%)')
            plt.title('MAPE by Time Period in Best Trial')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save locally and upload to S3
            local_path = os.path.join(output_dir, f'hpo_mape_by_period.png')
            plt.savefig(local_path, dpi=300, bbox_inches='tight')
            s3_key = f"plots/run_{run_id}/hpo_mape_by_period.png"
            upload_to_s3(local_path, s3_key)
            plt.close()
    
    # 4. Parameter importance barplot
    param_importance = optimization_results.get('parameter_importance', {})
    if param_importance:
        plt.figure(figsize=(10, 6))
        params = list(param_importance.keys())
        importance = list(param_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        params = [params[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        plt.barh(params, importance, color='orange')
        plt.xlabel('Importance')
        plt.title('Parameter Importance')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save locally and upload to S3
        local_path = os.path.join(output_dir, f'hpo_parameter_importance.png')
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        s3_key = f"plots/run_{run_id}/hpo_parameter_importance.png"
        upload_to_s3(local_path, s3_key)
        plt.close()
        
    logger.info(f"Created and uploaded HPO visualization plots to S3")


def visualize_training_results(df, features, target, model, model_metrics, date_column, output_dir, run_id):
    """
    Create visualizations of final model performance on training/validation data.
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        model: Trained model
        model_metrics: Dictionary with model metrics
        date_column: Date column name
        output_dir: Directory to save plots
        run_id: Unique run identifier
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import os
    import boto3
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data for visualization (use same split as in train_model)
    train_ratio = 0.8
    train_size = int(len(df) * train_ratio)
    
    # Sort by date if available to maintain time series structure
    if date_column in df.columns:
        df = df.sort_values(by=date_column)
    
    # Split into train and validation sets
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Make predictions
    train_predictions = model.predict(train_df[features])
    val_predictions = model.predict(val_df[features])
    
    # Add predictions to DataFrames
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['prediction'] = train_predictions
    val_df['prediction'] = val_predictions
    
    # 1. Actual vs predicted time series plot
    if date_column in val_df.columns:
        plt.figure(figsize=(15, 6))
        
        # Get range of dates to plot (last 2 weeks of validation data)
        plot_days = 14
        if len(val_df) > 24 * plot_days:  # assuming hourly data
            plot_df = val_df.iloc[-24*plot_days:]
        else:
            plot_df = val_df
        
        plt.plot(plot_df[date_column], plot_df[target], 'b-', label='Actual', linewidth=2)
        plt.plot(plot_df[date_column], plot_df['prediction'], 'r--', label='Predicted', linewidth=2)
        
        # Highlight weekends with background shading
        min_date = plot_df[date_column].min()
        max_date = plot_df[date_column].max()
        
        # Get Saturdays and Sundays
        dates = pd.date_range(start=min_date, end=max_date, freq='D')
        weekends = [d for d in dates if d.weekday() >= 5]  # 5=Saturday, 6=Sunday
        
        for weekend in weekends:
            plt.axvspan(weekend, weekend + pd.Timedelta(days=1), 
                       alpha=0.1, color='gray', label='_' if weekend != weekends[0] else 'Weekend')
        
        plt.xlabel('Date')
        plt.ylabel('Load')
        plt.title('Validation Data: Actual vs Predicted Load')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save locally and upload to S3
        local_path = os.path.join(output_dir, f'training_time_series.png')
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        s3_key = f"plots/run_{run_id}/training_time_series.png"
        upload_to_s3(local_path, s3_key)
        plt.close()
    
    # 2. Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 10))
    
    # Plot identity line
    min_val = min(val_df[target].min(), val_df['prediction'].min())
    max_val = max(val_df[target].max(), val_df['prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Plot scatter
    plt.scatter(val_df[target], val_df['prediction'], alpha=0.5)
    
    plt.xlabel('Actual Load')
    plt.ylabel('Predicted Load')
    plt.title(f'Validation Data: Predicted vs Actual (R² = {model_metrics.get("validation_r2", 0):.4f})')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save locally and upload to S3
    local_path = os.path.join(output_dir, f'training_scatter.png')
    plt.savefig(local_path, dpi=300, bbox_inches='tight')
    s3_key = f"plots/run_{run_id}/training_scatter.png"
    upload_to_s3(local_path, s3_key)
    plt.close()
    
    # 3. Residual plot
    plt.figure(figsize=(10, 6))
    
    # Calculate residuals
    val_df['residual'] = val_df[target] - val_df['prediction']
    
    plt.scatter(val_df[target], val_df['residual'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Actual Load')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Validation Data: Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save locally and upload to S3
    local_path = os.path.join(output_dir, f'training_residuals.png')
    plt.savefig(local_path, dpi=300, bbox_inches='tight')
    s3_key = f"plots/run_{run_id}/training_residuals.png"
    upload_to_s3(local_path, s3_key)
    plt.close()
    
    # 4. Residual histogram
    plt.figure(figsize=(10, 6))
    
    plt.hist(val_df['residual'], bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Validation Data: Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save locally and upload to S3
    local_path = os.path.join(output_dir, f'training_residual_hist.png')
    plt.savefig(local_path, dpi=300, bbox_inches='tight')
    s3_key = f"plots/run_{run_id}/training_residual_hist.png"
    upload_to_s3(local_path, s3_key)
    plt.close()
    
    # 5. Daily profile plot
    if date_column in val_df.columns:
        # Add hour to DataFrame
        val_df['hour'] = val_df[date_column].dt.hour
        
        # Group by hour
        hourly_avg = val_df.groupby('hour')[[target, 'prediction']].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(hourly_avg['hour'], hourly_avg[target], 'b-', label='Actual', linewidth=2)
        plt.plot(hourly_avg['hour'], hourly_avg['prediction'], 'r--', label='Predicted', linewidth=2)
        
        # Highlight key periods
        periods = {
            "Morning Rise": (6, 9),
            "Solar Peak": (11, 14),
            "Evening Ramp": (16, 20),
            "Peak Demand": (17, 22)
        }
        colors = ['#FFDDDD', '#DDFFDD', '#DDDDFF', '#FFFFDD']
        
        for i, (period_name, (start_hour, end_hour)) in enumerate(periods.items()):
            plt.axvspan(start_hour, end_hour, alpha=0.2, color=colors[i % len(colors)], 
                       label=period_name)
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Load')
        plt.title('Validation Data: Average Daily Load Profile')
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save locally and upload to S3
        local_path = os.path.join(output_dir, f'training_daily_profile.png')
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        s3_key = f"plots/run_{run_id}/training_daily_profile.png"
        upload_to_s3(local_path, s3_key)
        plt.close()
    
    # 6. Feature importance plot
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        
        # Get feature importance
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Plot top 20 features or all if less than 20
        top_n = min(20, len(features))
        top_indices = indices[:top_n]
        
        plt.bar(range(top_n), importance[top_indices], align='center')
        plt.xticks(range(top_n), [features[i] for i in top_indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save locally and upload to S3
        local_path = os.path.join(output_dir, f'training_feature_importance.png')
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        s3_key = f"plots/run_{run_id}/training_feature_importance.png"
        upload_to_s3(local_path, s3_key)
        plt.close()
    
    # 7. Period metrics comparison
    period_metrics = {k: v for k, v in model_metrics.items() if ('solar_peak' in k or 'evening_ramp' in k or 
                                                               'peak_demand' in k or 'morning_rise' in k)}
    
    if period_metrics:
        # Group metrics by period
        period_groups = {}
        for metric, value in period_metrics.items():
            for period in ['solar_peak', 'evening_ramp', 'peak_demand', 'morning_rise']:
                if period in metric:
                    metric_type = metric.replace(f'{period}_', '')
                    if period not in period_groups:
                        period_groups[period] = {}
                    period_groups[period][metric_type] = value
        
        # Plot RMSE by period
        if all('rmse' in period_groups[p] for p in period_groups):
            plt.figure(figsize=(10, 6))
            
            periods = list(period_groups.keys())
            rmse_values = [period_groups[p]['rmse'] for p in periods]
            
            plt.bar(range(len(periods)), rmse_values, color='skyblue')
            plt.xticks(range(len(periods)), [p.replace('_', ' ').title() for p in periods], rotation=45)
            plt.ylabel('RMSE')
            plt.title('RMSE by Time Period')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save locally and upload to S3
            local_path = os.path.join(output_dir, f'training_rmse_by_period.png')
            plt.savefig(local_path, dpi=300, bbox_inches='tight')
            s3_key = f"plots/run_{run_id}/training_rmse_by_period.png"
            upload_to_s3(local_path, s3_key)
            plt.close()
        
        # Plot MAPE by period
        if all('mape' in period_groups[p] for p in period_groups):
            plt.figure(figsize=(10, 6))
            
            periods = list(period_groups.keys())
            mape_values = [period_groups[p]['mape'] for p in periods]
            
            plt.bar(range(len(periods)), mape_values, color='lightgreen')
            plt.xticks(range(len(periods)), [p.replace('_', ' ').title() for p in periods], rotation=45)
            plt.ylabel('MAPE (%)')
            plt.title('MAPE by Time Period')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save locally and upload to S3
            local_path = os.path.join(output_dir, f'training_mape_by_period.png')
            plt.savefig(local_path, dpi=300, bbox_inches='tight')
            s3_key = f"plots/run_{run_id}/training_mape_by_period.png"
            upload_to_s3(local_path, s3_key)
            plt.close()
    
    logger.info(f"Created and uploaded training visualization plots to S3")
