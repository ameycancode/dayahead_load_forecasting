"""
Hyperparameter optimization using Optuna.
"""

import logging
import os
import pickle
import json
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Import local modules
from pipeline.training.model import (
  create_time_series_splits,
  initialize_model
)

logger = logging.getLogger(__name__)

# Import Optuna
try:
  import optuna
  OPTUNA_AVAILABLE = True
except ImportError:
  OPTUNA_AVAILABLE = False
  logger.warning("Optuna library not available. Installing it...")
  try:
    import subprocess
    subprocess.check_call(["pip", "install", "optuna"])
    import optuna
    OPTUNA_AVAILABLE = True
    logger.info("Optuna successfully installed")
  except Exception as e:
    logger.error(f"Failed to install Optuna: {str(e)}")


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error"""
    return 100 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


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


def compute_weighted_score(
  metrics: Dict[str, float],
  weights: Dict[str, float]
) -> float:
  """
  Compute a weighted combined score from multiple metrics.
  
  Args:
    metrics: Dictionary of metrics
    weights: Dictionary mapping metric names to weights
    
  Returns:
    Combined weighted score (lower is better)
  """
  if not metrics or not weights:
    return float('inf')
  
  score = 0.0
  total_weight = 0.0
  
  logger.info("Computing weighted score from metrics:")
  for metric_name, weight in weights.items():
    if metric_name in metrics:
      # Handle R² differently (higher is better, unlike other metrics)
      if metric_name.startswith('r2'):
        # Convert to 1-R² so lower is better, consistent with other metrics
        metric_value = 1.0 - metrics[metric_name]
      else:
        metric_value = metrics[metric_name]
      
      score += weight * metric_value
      total_weight += weight
      
      logger.info(f"  - {metric_name}: {metrics[metric_name]:.4f} * weight {weight:.2f} = {weight * metric_value:.4f}")
  
  # Add special weighting for transition periods
  if all(key in metrics for key in ['rmse_overall', 'validation_smape']):
    # Create a synthetic transition score combining RMSE and sMAPE
    transition_score = 0.5 * metrics['rmse_overall'] + 0.5 * metrics['validation_smape']
    # Add with high weight to prioritize transition performance
    score += 0.25 * transition_score
    total_weight += 0.25
    logger.info(f"  - transition_score: {transition_score:.4f} * weight 0.25 = {0.25 * transition_score:.4f}")
  
  # Return average weighted score if valid weights found
  if total_weight > 0:
    final_score = score / total_weight
    logger.info(f"  Final weighted score (lower is better): {final_score:.4f}")
    return final_score
  else:
    logger.warning("No valid weights found, returning infinity score")
    return float('inf')


class OptunaHPO:
  """Hyperparameter optimization with Optuna."""
  
  def __init__(
    self,
    df: pd.DataFrame,
    features: List[str],
    target: str = "lossadjustedload",
    date_column: str = "datetime",
    n_splits: int = 4,
    initial_train_months: int = 9,
    validation_months: int = 6,
    step_months: int = 6,
    n_trials: int = 50,
    custom_periods: Optional[Dict[str, Tuple[int, int]]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
  ):
    """
    Initialize the HPO class.
    
    Args:
      df: DataFrame with features and target
      features: List of feature column names
      target: Target column name
      date_column: Name of date column
      n_splits: Maximum number of cross-validation splits
      initial_train_months: Number of months for initial training set
      validation_months: Number of months for each validation set
      step_months: Number of months to step forward for each split
      n_trials: Number of optimization trials
      custom_periods: Dictionary of period names to (start_hour, end_hour)
      metric_weights: Dictionary of metric names to weights
    """
    self.df = df
    self.features = features
    self.target = target
    self.date_column = date_column
    self.n_splits = n_splits
    self.initial_train_months = initial_train_months
    self.validation_months = validation_months
    self.step_months = step_months
    self.n_trials = n_trials
    self.custom_periods = custom_periods or {}
    self.metric_weights = metric_weights or {
      "rmse_overall": 1.0 # Default to just RMSE if no weights provided
    }
    
    # Create time series splits for cross-validation
    self.cv_splits = create_time_series_splits(
      df,
      date_column=date_column,
      n_splits=n_splits,
      initial_train_months=initial_train_months,
      validation_months=validation_months,
      step_months=step_months
    )
    
    logger.info(f"Initialized Optuna HPO with {len(self.cv_splits)} CV splits")
    logger.info(f"Using {len(features)} features")
    
    # Log each split's date range and sample count
    for i, (train_idx, val_idx) in enumerate(self.cv_splits):
        train_df = self.df.iloc[train_idx]
        val_df = self.df.iloc[val_idx]
        
        train_start = train_df[date_column].min().strftime('%Y-%m-%d')
        train_end = train_df[date_column].max().strftime('%Y-%m-%d')
        val_start = val_df[date_column].min().strftime('%Y-%m-%d')
        val_end = val_df[date_column].max().strftime('%Y-%m-%d')
        
        logger.info(f"Split {i+1}: train={train_start} to {train_end} ({len(train_df)} samples), "
                   f"val={val_start} to {val_end} ({len(val_df)} samples)")
    
    # Store study and best parameters
    self.study = None
    self.best_params = None
    self.best_score = None
    self.results = {}

  def create_objective(self) -> Callable:
    """Create objective function for Optuna."""
    def objective(trial):
      # Define the parameter space
      params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1
      }
      
      logger.info(f"\n{'='*80}\nTrial {trial.number}: Evaluating parameters:")
      for param_name, param_value in params.items():
          logger.info(f"  - {param_name}: {param_value}")
      
      # Track metrics across CV splits
      cv_metrics = []
      
      # Perform cross-validation
      for split_idx, (train_idx, test_idx) in enumerate(self.cv_splits):
        logger.info(f"\nCV split {split_idx+1}/{len(self.cv_splits)}")
        
        # Get train and test data for this split
        X_train = self.df.loc[train_idx, self.features]
        y_train = self.df.loc[train_idx, self.target]
        X_test = self.df.loc[test_idx, self.features]
        y_test = self.df.loc[test_idx, self.target]
        
        logger.info(f"Training with {len(X_train)} samples, validating with {len(X_test)} samples")
        
        # Initialize and train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        split_metrics = {}
        
        # Overall metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Handle MAPE with zeros
        y_test_safe = np.array(y_test).copy()
        y_test_safe[y_test_safe == 0] = 1e-10
        mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
        
        # Add overall metrics
        split_metrics["rmse_overall"] = rmse
        split_metrics["mae_overall"] = mae
        split_metrics["mape_overall"] = mape
        split_metrics["r2_overall"] = r2

        # Calculate alternative metrics for zero-crossing
        smape_overall = smape(y_test, y_pred)
        wape_overall = wape(y_test, y_pred)
        split_metrics["smape_overall"] = smape_overall
        split_metrics["wape_overall"] = wape_overall
        
        # Add transition-specific metrics
        transition_rmse = 0
        transition_smape = 0
        transition_wape = 0
        near_zero_mask = np.abs(y_test) < 20000  # Threshold for transition region
        if sum(near_zero_mask) > 0:
          transition_rmse = np.sqrt(mean_squared_error(y_test[near_zero_mask], y_pred[near_zero_mask]))
          transition_smape = smape(y_test[near_zero_mask], y_pred[near_zero_mask])
          transition_wape = wape(y_test[near_zero_mask], y_pred[near_zero_mask])

          split_metrics["transition_rmse"] = transition_rmse
          split_metrics["transition_smape"] = transition_smape
          split_metrics["transition_wape"] = transition_wape
        else:
          split_metrics["transition_rmse"] = 0
          split_metrics["transition_smape"] = 0
          split_metrics["transition_wape"] = 0
            
        # Add weighted transition score
        trans_weighted_err = transition_weighted_error(y_test, y_pred)
        split_metrics["trans_weighted_error"] = trans_weighted_err
        
        logger.info(f"Split {split_idx+1} overall metrics: RMSE = {rmse:.2f}, MAPE = {mape:.2f}%, R² = {r2:.4f}, SMAPE = {smape_overall:.2f}%, WAPE = {wape_overall:.2f}%")

        logger.info(f"Split {split_idx+1} transition metrics: RMSE = {transition_rmse:.2f}, SMAPE = {transition_smape:.2f}%, WAPE = {transition_wape:.2f}%, Weighted Error = {trans_weighted_err:.2f}")
        
        
        # Add custom period metrics if available
        if self.custom_periods and self.date_column in self.df.columns:
          test_df = self.df.loc[test_idx].copy()
          test_df["prediction"] = y_pred
          
          for period_name, (start_hour, end_hour) in self.custom_periods.items():
            # Filter by hours in this period
            period_mask = (
              (test_df[self.date_column].dt.hour >= start_hour) & 
              (test_df[self.date_column].dt.hour <= end_hour)
            )
            period_df = test_df[period_mask]
            
            if not period_df.empty:
              # Calculate period metrics
              period_rmse = np.sqrt(mean_squared_error(
                period_df[self.target], period_df["prediction"]
              ))
              period_mae = mean_absolute_error(
                period_df[self.target], period_df["prediction"]
              )
              period_r2 = r2_score(
                period_df[self.target], period_df["prediction"]
              )
                
              # Calculate period MAPE
              period_y_true_safe = period_df[self.target].copy()
              period_y_true_safe = np.where(period_y_true_safe == 0, 1e-10, period_y_true_safe)
              period_mape = np.mean(np.abs(
                (period_df[self.target] - period_df["prediction"]) / period_y_true_safe
              )) * 100
                
              # Add period metrics
              split_metrics[f"rmse_{period_name}"] = period_rmse
              split_metrics[f"mae_{period_name}"] = period_mae
              split_metrics[f"mape_{period_name}"] = period_mape
              split_metrics[f"r2_{period_name}"] = period_r2

              period_smape = smape(period_df[self.target], period_df["prediction"])
              period_wape = wape(period_df[self.target], period_df["prediction"])
                
              split_metrics[f"smape_{period_name}"] = period_smape
              split_metrics[f"wape_{period_name}"] = period_wape
              
              logger.info(f"  - {period_name.replace('_', ' ').title()}: RMSE = {period_rmse:.2f}, MAPE = {period_mape:.2f}%, R² = {period_r2:.2f}, SMAPE = {period_smape:.2f}%, WAPE = {period_wape:.2f}%")
        
        cv_metrics.append(split_metrics)
      
      # Calculate average metrics across splits
      avg_metrics = {}
      for metric in cv_metrics[0].keys():
        values = [metrics.get(metric, float('inf')) for metrics in cv_metrics]
        avg_metrics[metric] = np.mean(values)
      
      # Log average metrics across splits
      logger.info("\nAverage metrics across all splits:")
      for key, value in sorted(avg_metrics.items()):
        logger.info(f"  - {key}: {value:.4f}")
      
      # Calculate weighted score
      custom_weights = self.metric_weights.copy()
    
      # Normalize weights to ensure sum is 1.0
      total_weight = sum(custom_weights.values())
      if total_weight != 1.0:
        custom_weights = {k: v/total_weight for k, v in custom_weights.items()}
    
      # Use these adjusted weights
      score = compute_weighted_score(avg_metrics, custom_weights)
          
      # Store split metrics for reference
      trial.set_user_attr("cv_metrics", cv_metrics)
      trial.set_user_attr("avg_metrics", avg_metrics)
      
      return score
    
    return objective
  
  def optimize(self, timeout=None) -> Dict[str, any]:
    """
    Run the optimization process.
    
    Args:
      timeout: Optional timeout in seconds
      
    Returns:
      Dictionary with best parameters and results
    """
    logger.info(f"Starting Optuna hyperparameter optimization with {self.n_trials} trials")
    
    if not OPTUNA_AVAILABLE:
      logger.error("Optuna is not available, falling back to default parameters")
      self.best_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1
      }
      self.best_score = float('inf')
      self.results = {"error": "Optuna not available"}
      return {
        "best_params": self.best_params,
        "best_score": self.best_score,
        "results": self.results
      }
    
    try:
      # Create study
      self.study = optuna.create_study(direction="minimize")
      
      # Create objective function
      objective = self.create_objective()
      
      # Run optimization with optional timeout
      if timeout:
        self.study.optimize(objective, n_trials=self.n_trials, timeout=timeout)
      else:
        self.study.optimize(objective, n_trials=self.n_trials)
      
      # Log all trials summary
      logger.info("\nAll trials summary (sorted by performance):")
      logger.info(f"{'Trial':>6} {'Score':>10} {'n_est':>6} {'depth':>6} {'lr':>10} {'subsamp':>8} {'colsamp':>8}")
      
      for trial in sorted(self.study.trials, key=lambda t: t.value):
          if trial.value is None:
              continue
          params = trial.params
          logger.info(f"{trial.number:6d} {trial.value:10.4f} {params.get('n_estimators', 0):6d} "
                    f"{params.get('max_depth', 0):6d} {params.get('learning_rate', 0):.6f} "
                    f"{params.get('subsample', 0):.4f} {params.get('colsample_bytree', 0):.4f}")
      
      # Get best parameters
      self.best_params = self.study.best_params
      self.best_score = self.study.best_value
      
      # Add fixed parameters
      self.best_params.update({
        "random_state": 42,
        "n_jobs": -1
      })
      
      # Get best trial metrics
      best_trial = self.study.best_trial
      avg_metrics = best_trial.user_attrs.get("avg_metrics", {})
      cv_metrics = best_trial.user_attrs.get("cv_metrics", [])
      
      # Store results
      self.results = {
        "best_params": self.best_params,
        "best_score": self.best_score,
        "best_metrics": avg_metrics,
        "cv_metrics": cv_metrics,
        "n_trials": self.n_trials,
        "completed_trials": len(self.study.trials),
        "datetime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
      }
      
      # Create parameter importance if enough trials
      if len(self.study.trials) >= 10:
        try:
          importance = optuna.importance.get_param_importances(self.study)
          self.results["parameter_importance"] = {k: float(v) for k, v in importance.items()}
          
          # Log parameter importance
          logger.info("\nParameter importance:")
          for param, importance in sorted(importance.items(), key=lambda x: x[1], reverse=True):
              logger.info(f"  - {param}: {importance:.4f}")
        except Exception as e:
          logger.warning(f"Could not compute parameter importance: {str(e)}")
      
      logger.info(f"Optimization complete: {self.n_trials} trials")
      logger.info(f"Best score: {self.best_score:.4f}")
      logger.info(f"Best parameters: {self.best_params}")
      
      if avg_metrics:
        logger.info("Best metrics:")
        for metric_name, value in sorted(avg_metrics.items()):
          logger.info(f"- {metric_name}: {value:.4f}")
      
      return {
        "best_params": self.best_params,
        "best_score": self.best_score,
        "results": self.results
      }
      
    except Exception as e:
      logger.error(f"Error in Optuna optimization: {str(e)}")
      self.best_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1
      }
      self.best_score = float('inf')
      self.results = {"error": str(e)}
      return {
        "best_params": self.best_params,
        "best_score": self.best_score,
        "results": self.results
      }
  
  def save_results(self, output_path: str) -> bool:
    """
    Save optimization results to a file.
    
    Args:
      output_path: Path to save results
      
    Returns:
      True if successful, False otherwise
    """
    try:
      # Create directory if it doesn't exist
      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      
      # Convert numpy types to Python types for JSON serialization
      def clean_types(obj):
        if isinstance(obj, dict):
          return {k: clean_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
          return [clean_types(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
          return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
          return float(obj)
        elif isinstance(obj, (np.ndarray,)):
          return obj.tolist()
        else:
          return obj
      
      # Save results to JSON file
      with open(output_path, 'w') as f:
        json.dump(clean_types(self.results), f, indent=2)
      
      logger.info(f"Optimization results saved to {output_path}")
      return True
    
    except Exception as e:
      logger.error(f"Error saving optimization results: {str(e)}")
      return False


def optimize_hyperparameters(
    df: pd.DataFrame,
    features: List[str],
    target: str = "lossadjustedload",
    date_column: str = "datetime",
    n_trials: int = 50,
    n_splits: int = 5,
    initial_train_months: int = 6,
    validation_months: int = 6,
    step_months: int = 3,
    custom_periods: Optional[Dict[str, Tuple[int, int]]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    output_dir: str = None,
    run_id: str = None,
) -> Tuple[Dict[str, any], Dict[str, any]]:
  """
  Optimize hyperparameters for XGBoost model using Optuna.
  
  Args:
    df: DataFrame with features and target
    features: List of feature column names
    target: Target column name
    date_column: Name of date column
    n_trials: Number of optimization trials
    n_splits: Number of cross-validation splits
    initial_train_months: Initial training period in months
    validation_months: Validation period in months
    step_months: Step size in months
    custom_periods: Dictionary of period names to (start_hour, end_hour)
    metric_weights: Dictionary of metric names to weights
    
  Returns:
    Tuple of (best parameters, optimization results)
  """
  logger.info(f"Starting hyperparameter optimization with {n_trials} trials and {n_splits} CV splits")
  logger.info(f"Data spans from {df[date_column].min().date()} to {df[date_column].max().date()}")
  logger.info(f"Target column: {target}")
  logger.info(f"Using {len(features)} features")
  
  if custom_periods:
      logger.info("Evaluating specific time periods:")
      for period_name, (start_hour, end_hour) in custom_periods.items():
          logger.info(f"  - {period_name}: {start_hour}:00 to {end_hour}:00")
  
  if metric_weights:
      logger.info("Using weighted metrics for optimization:")
      for metric, weight in metric_weights.items():
          logger.info(f"  - {metric}: weight {weight}")
  
  # Initialize optimizer
  optimizer = OptunaHPO(
    df=df,
    features=features,
    target=target,
    date_column=date_column,
    n_splits=n_splits,
    initial_train_months=initial_train_months,
    validation_months=validation_months,
    step_months=step_months,
    n_trials=n_trials,
    custom_periods=custom_periods,
    metric_weights=metric_weights,
  )
  
  # Run optimization
  optimization_results = optimizer.optimize()
  
  logger.info("\nHyperparameter optimization complete!")
  logger.info(f"Evaluated {n_trials} different parameter combinations across {n_splits} CV splits")
  logger.info(f"Best parameter set achieved score: {optimization_results['best_score']:.4f}")

  if output_dir and run_id:
    try:
      from pipeline.training.visualization import visualize_hpo_results
      visualize_hpo_results(
        optimization_results=optimization_results,
        output_dir=output_dir,
        run_id=run_id
      )
    except Exception as e:
      logger.error(f"Error creating HPO visualizations: {str(e)}")
  
  # Return results
  return optimization_results["best_params"], optimization_results["results"]


# API function for backward compatibility with existing code
def cross_validated_optimization(
  df: pd.DataFrame,
  features: List[str],
  target: str = 'lossadjustedload',
  date_column: str = 'datetime',
  hpo_method: str = 'bayesian',
  n_splits: int = 3,
  test_size_days: int = 14,
  gap_days: int = 7,
  n_iter: int = 50,
  param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
  custom_periods: Optional[Dict[str, Tuple[int, int]]] = None,
  metric_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, any], Dict[str, any]]:
  """
  Backward compatibility function for existing code.
  
  Args:
    df: DataFrame with features and target
    features: List of feature column names
    target: Target column name
    date_column: Name of date column
    hpo_method: Hyperparameter optimization method (ignored, always uses Optuna)
    n_splits: Number of cross-validation splits
    test_size_days: Size of test set in days
    gap_days: Gap between train and test in days
    n_iter: Number of optimization iterations/trials
    param_bounds: Parameter bounds (ignored, uses Optuna's built-in bounds)
    custom_periods: Dictionary of period names to (start_hour, end_hour)
    metric_weights: Dictionary of metric names to weights
    
  Returns:
    Tuple of (best parameters, optimization results)
  """
  logger.info("Using Optuna for hyperparameter optimization with expanding window CV")
  
  # Calculate approximate settings based on old parameters
  initial_train_months = 6 # Start with 6 months training
  validation_months = max(1, int(test_size_days / 30)) # Convert days to months, minimum 1
  step_months = max(1, int(test_size_days / 60)) # Step size is half the validation period, minimum 1
  
  # Use default metric weights if none provided
  if not metric_weights:
    metric_weights = {
      "rmse_overall": 0.3,
      "mape_overall": 0.2,
      "rmse_evening_ramp": 0.2 if "evening_ramp" in (custom_periods or {}) else 0,
      "mape_evening_ramp": 0.1 if "evening_ramp" in (custom_periods or {}) else 0,
      "rmse_peak_demand": 0.1 if "peak_demand" in (custom_periods or {}) else 0,
      "mape_peak_demand": 0.05 if "peak_demand" in (custom_periods or {}) else 0,
      "r2_overall": 0.05
    }
    # Remove any weights for periods that don't exist
    metric_weights = {k: v for k, v in metric_weights.items() if v > 0}
  
  # Run Optuna optimization
  return optimize_hyperparameters(
    df=df,
    features=features,
    target=target,
    date_column=date_column,
    n_trials=n_iter,
    n_splits=n_splits,
    initial_train_months=initial_train_months,
    validation_months=validation_months,
    step_months=step_months,
    custom_periods=custom_periods,
    metric_weights=metric_weights,
  )
