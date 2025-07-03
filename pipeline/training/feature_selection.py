"""
Feature selection techniques for energy load forecasting.

This module provides methods for selecting the most relevant features
using various approaches such as feature importance, correlation analysis,
and SHAP values.
"""

import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available. SHAP-based feature selection will be disabled.")


def consensus_feature_selection(
    method_results: Dict[str, List[str]],
    threshold: float = 0.5,
    top_n: Optional[int] = None
) -> List[str]:
    """
    Select features that appear in multiple selection methods.
    
    Args:
        method_results: Dictionary mapping method names to their selected features
        threshold: Minimum fraction of methods a feature must appear in (0.0-1.0)
        top_n: Maximum number of features to return
        
    Returns:
        List of selected feature names that meet the consensus criteria
    """
    if not method_results:
        logger.warning("No method results provided for consensus selection")
        return []
    
    # Count occurrences of each feature across methods
    feature_counts = {}
    for method, features in method_results.items():
        for feature in features:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1
    
    # Calculate minimum count based on threshold
    method_count = len(method_results)
    min_count = max(1, int(threshold * method_count))
    
    # Select features that appear in at least min_count methods
    consensus_features = [
        feature for feature, count in feature_counts.items() 
        if count >= min_count
    ]
    
    # Sort by occurrence count (descending)
    consensus_features.sort(
        key=lambda f: feature_counts[f], 
        reverse=True
    )
    
    # Apply top_n limit if specified
    if top_n is not None and len(consensus_features) > top_n:
        consensus_features = consensus_features[:top_n]
    
    logger.info(f"Selected {len(consensus_features)} features by consensus "
                f"(threshold={threshold}, methods={list(method_results.keys())})")
    
    return consensus_features


def select_features_by_importance(
    model: XGBRegressor,
    feature_names: List[str],
    top_n: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[str]:
    """
    Select top features based on XGBoost feature importance scores.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to select (if None, use threshold)
        threshold: Minimum importance score to keep (if None, use top_n)
        
    Returns:
        List of selected feature names
    """
    if model is None or not feature_names:
        logger.error("Invalid model or empty feature list")
        return []
    
    try:
        # Get feature importance scores
        importance = model.feature_importances_
        
        # Create a dataframe of features and importance scores
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 40 features by importance: {feature_importance['feature'].head(40).tolist()}")
        
        # Select features based on criteria
        if top_n is not None:
            # Select top N features
            top_n = min(top_n, len(feature_names))
            selected_features = feature_importance.head(top_n)['feature'].tolist()
            logger.info(f"Selected {len(selected_features)} features by top-n={top_n}")
        elif threshold is not None:
            # Select features above threshold
            selected_features = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()
            logger.info(f"Selected {len(selected_features)} features by threshold={threshold}")
        else:
            # Default: select all features
            selected_features = feature_names
            logger.info(f"No criteria specified, keeping all {len(selected_features)} features")
        
        return selected_features
    
    except Exception as e:
        logger.error(f"Error in feature selection by importance: {str(e)}")
        return feature_names


def select_features_by_correlation(
    df: pd.DataFrame,
    target: str = 'lossadjustedload',
    correlation_method: str = 'pearson',
    correlation_threshold: float = 0.85,
    target_correlation_threshold: float = 0.05,
) -> List[str]:
    """
    Select features by removing those that are highly correlated with each other,
    while ensuring features have a minimum correlation with the target.
    
    Args:
        df: DataFrame with features and target
        target: Target column name
        correlation_method: Method for correlation calculation ('pearson', 'spearman', 'kendall')
        correlation_threshold: Threshold for identifying highly correlated features
        target_correlation_threshold: Minimum absolute correlation with target
        
    Returns:
        List of selected feature names
    """
    if df.empty or target not in df.columns:
        logger.error(f"Empty DataFrame or target '{target}' not found")
        return list(df.columns)
    
    try:
        # Start with all columns except the target
        features = [col for col in df.columns if col != target]
        
        # Calculate correlation matrix
        corr_matrix = df[features + [target]].corr(method=correlation_method)
        
        # Absolute correlation with target
        target_corr = corr_matrix[target].abs()
        
        # Filter features by correlation with target
        features_corr_target = [col for col in features if target_corr[col] >= target_correlation_threshold]
        logger.info(f"Kept {len(features_corr_target)}/{len(features)} features with correlation >= {target_correlation_threshold} with target")
        
        if not features_corr_target:
            logger.warning(f"No features met the minimum correlation threshold of {target_correlation_threshold}")
            return features
        
        # Get just the feature correlation matrix
        feature_corr = corr_matrix.loc[features_corr_target, features_corr_target]
        
        # Track features to remove
        to_drop = set()
        
        # Find highly correlated feature pairs
        for i in range(len(features_corr_target)):
            if features_corr_target[i] in to_drop:
                continue
                
            for j in range(i+1, len(features_corr_target)):
                if features_corr_target[j] in to_drop:
                    continue
                    
                if abs(feature_corr.iloc[i, j]) > correlation_threshold:
                    # If two features are highly correlated, drop the one with lower correlation to target
                    if target_corr[features_corr_target[i]] < target_corr[features_corr_target[j]]:
                        to_drop.add(features_corr_target[i])
                        break
                    else:
                        to_drop.add(features_corr_target[j])
        
        # Get the final list of features
        selected_features = [f for f in features_corr_target if f not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} features due to high correlation with other features")
        logger.info(f"Final selection: {len(selected_features)} features")
        
        return selected_features
    
    except Exception as e:
        logger.error(f"Error in feature selection by correlation: {str(e)}")
        return features


# def select_features_by_shap(
#     model: XGBRegressor,
#     X: pd.DataFrame,
#     feature_names: List[str],
#     top_n: Optional[int] = None,
#     threshold: Optional[float] = None,
#     max_shap_samples: int = 500,
# ) -> List[str]:
#     """
#     Select features based on SHAP values.
    
#     Args:
#         model: Trained XGBoost model
#         X: DataFrame with feature data
#         feature_names: List of feature names
#         top_n: Number of top features to select (if None, use threshold)
#         threshold: Minimum absolute SHAP value to keep (if None, use top_n)
#         max_shap_samples: Maximum number of samples to use for SHAP calculation
        
#     Returns:
#         List of selected feature names
#     """
#     if not SHAP_AVAILABLE:
#         logger.warning("SHAP not available, falling back to feature importance")
#         return select_features_by_importance(model, feature_names, top_n, threshold)
    
#     if model is None or X.empty or not feature_names:
#         logger.error("Invalid model, empty data, or empty feature list")
#         return []
    
#     try:
#         # Sample data if needed to limit computation
#         if len(X) > max_shap_samples:
#             logger.info(f"Sampling {max_shap_samples} rows for SHAP calculation")
#             X_sample = X.sample(max_shap_samples, random_state=42)
#         else:
#             X_sample = X
        
#         # Calculate SHAP values
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X_sample)
        
#         # Calculate mean absolute SHAP values for each feature
#         shap_importance = pd.DataFrame({
#             'feature': feature_names,
#             'importance': np.mean(np.abs(shap_values), axis=0)
#         }).sort_values('importance', ascending=False)
        
#         logger.info(f"Top 40 features by SHAP: {shap_importance['feature'].head(40).tolist()}")
        
#         # Select features based on criteria
#         if top_n is not None:
#             # Select top N features
#             top_n = min(top_n, len(feature_names))
#             selected_features = shap_importance.head(top_n)['feature'].tolist()
#             logger.info(f"Selected {len(selected_features)} features by SHAP top-n={top_n}")
#         elif threshold is not None:
#             # Select features above threshold
#             selected_features = shap_importance[shap_importance['importance'] > threshold]['feature'].tolist()
#             logger.info(f"Selected {len(selected_features)} features by SHAP threshold={threshold}")
#         else:
#             # Default: select all features
#             selected_features = feature_names
#             logger.info(f"No criteria specified, keeping all {len(selected_features)} features")
        
#         return selected_features
    
#     except Exception as e:
#         logger.error(f"Error in feature selection by SHAP: {str(e)}")
#         return feature_names

def create_energy_representative_sample(df, target_sample_size=2500, date_column='datetime'):
    """
    Create a representative sample for energy time series data.
    Ensures coverage of all important patterns while keeping sample size manageable.
    
    Args:
        df: DataFrame with time series data
        target_sample_size: Target number of rows to sample
        date_column: Name of datetime column
        
    Returns:
        DataFrame with representative sample
    """
    logger.info(f"Creating representative sample from {len(df)} total rows")
    
    df_sample = df.copy()
    
    # Ensure datetime column exists and is properly formatted
    if date_column not in df_sample.columns:
        logger.warning(f"Date column '{date_column}' not found, using row-based sampling")
        return df_sample.sample(n=min(target_sample_size, len(df_sample)), random_state=42)
    
    df_sample[date_column] = pd.to_datetime(df_sample[date_column])
    
    # Create stratification variables
    df_sample['month'] = df_sample[date_column].dt.month
    df_sample['hour'] = df_sample[date_column].dt.hour
    df_sample['day_type'] = df_sample[date_column].dt.weekday.apply(lambda x: 'weekend' if x >= 5 else 'weekday')
    df_sample['season'] = df_sample['month'].apply(lambda x: 
        'winter' if x in [12, 1, 2] else
        'spring' if x in [3, 4, 5] else
        'summer' if x in [6, 7, 8] else 'fall'
    )
    
    # Strategy: Ensure representation across key dimensions
    # 4 seasons × 2 day types × 24 hours = 192 combinations
    total_combinations = len(df_sample.groupby(['season', 'day_type', 'hour']))
    samples_per_stratum = max(1, target_sample_size // total_combinations)
    
    logger.info(f"Found {total_combinations} unique time pattern combinations")
    logger.info(f"Target samples per combination: {samples_per_stratum}")
    
    sampled_dfs = []
    
    for (season, day_type, hour), group in df_sample.groupby(['season', 'day_type', 'hour']):
        if len(group) > 0:
            # Calculate samples for this stratum
            # Minimum 1 sample, but take more if stratum is large
            n_samples = min(
                max(samples_per_stratum, max(1, len(group) // 50)),  # At least 2% of each stratum
                len(group)
            )
            sampled_dfs.append(group.sample(n=n_samples, random_state=42))
    
    # Combine all samples
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we have too many samples, randomly select target amount
    if len(result) > target_sample_size:
        result = result.sample(n=target_sample_size, random_state=42)
    
    # Clean up temporary columns
    result = result.drop(['month', 'hour', 'day_type', 'season'], axis=1, errors='ignore')
    
    logger.info(f"Created representative sample of {len(result)} rows from {len(df)} total rows")
    
    # Verify we have good coverage
    if len(result) < target_sample_size * 0.5:
        logger.warning(f"Sample size {len(result)} is much smaller than target {target_sample_size}")
        logger.warning("Consider using a smaller target_sample_size or check data distribution")
    
    return result


def select_features_by_shap(
    model: XGBRegressor,
    X: pd.DataFrame,
    feature_names: List[str],
    top_n: Optional[int] = None,
    threshold: Optional[float] = None,
    max_shap_samples: int = 2500,
    use_energy_sampling: bool = True,
) -> List[str]:
    """
    Select features based on SHAP values with optimized sampling for energy time series data.
    
    Args:
        model: Trained XGBoost model
        X: DataFrame with feature data
        feature_names: List of feature names
        top_n: Number of top features to select (if None, use threshold)
        threshold: Minimum absolute SHAP value to keep (if None, use top_n)
        max_shap_samples: Maximum number of samples to use for SHAP calculation
        use_energy_sampling: Whether to use energy-specific stratified sampling
        
    Returns:
        List of selected feature names
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, falling back to feature importance")
        return select_features_by_importance(model, feature_names, top_n, threshold)
    
    if model is None or X.empty or not feature_names:
        logger.error("Invalid model, empty data, or empty feature list")
        return []
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Starting SHAP feature selection with {len(X)} total rows")
        
        # Prepare sample for SHAP calculation
        if len(X) > max_shap_samples:
            if use_energy_sampling and 'datetime' in X.columns:
                logger.info("Using energy-specific stratified sampling for SHAP")
                X_sample = create_energy_representative_sample(
                    X, 
                    target_sample_size=max_shap_samples,
                    date_column='datetime'
                )
            else:
                logger.info("Using random sampling for SHAP")
                X_sample = X.sample(max_shap_samples, random_state=42)
        else:
            X_sample = X.copy()
        
        logger.info(f"Using {len(X_sample)} rows for SHAP calculation")
        
        # Ensure we only use the features that the model expects
        available_features = [f for f in feature_names if f in X_sample.columns]
        if len(available_features) != len(feature_names):
            missing_features = [f for f in feature_names if f not in X_sample.columns]
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
        
        # Prepare data for SHAP
        X_shap = X_sample[available_features].copy()
        
        # Handle missing values
        if X_shap.isnull().any().any():
            logger.info("Handling missing values in SHAP data")
            X_shap = X_shap.fillna(X_shap.median())
        
        # For very large samples, use a smaller subset for actual SHAP calculation
        if len(X_shap) > 2000:
            logger.info(f"Sample size {len(X_shap)} is large, using subset of 2000 for SHAP calculation")
            X_shap_final = X_shap.sample(n=2000, random_state=42)
        else:
            X_shap_final = X_shap
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap_final)
        
        # Calculate mean absolute SHAP values for each feature
        shap_importance = pd.DataFrame({
            'feature': available_features,
            'importance': np.mean(np.abs(shap_values), axis=0)
        }).sort_values('importance', ascending=False)
        
        computation_time = time.time() - start_time
        logger.info(f"SHAP calculation completed in {computation_time:.1f} seconds")
        logger.info(f"Top 10 features by SHAP importance: {shap_importance['feature'].head(10).tolist()}")
        
        # Select features based on criteria
        if top_n is not None:
            # Select top N features
            top_n = min(top_n, len(available_features))
            selected_features = shap_importance.head(top_n)['feature'].tolist()
            logger.info(f"Selected {len(selected_features)} features by SHAP top-n={top_n}")
        elif threshold is not None:
            # Select features above threshold
            selected_features = shap_importance[shap_importance['importance'] > threshold]['feature'].tolist()
            logger.info(f"Selected {len(selected_features)} features by SHAP threshold={threshold}")
        else:
            # Default: select all features (sorted by importance)
            selected_features = available_features
            logger.info(f"No criteria specified, returning all {len(selected_features)} features sorted by SHAP importance")
        
        # Log feature importance statistics
        if len(shap_importance) > 0:
            logger.info(f"SHAP importance stats - Min: {shap_importance['importance'].min():.6f}, "
                       f"Max: {shap_importance['importance'].max():.6f}, "
                       f"Mean: {shap_importance['importance'].mean():.6f}")
        
        return selected_features
    
    except Exception as e:
        logger.error(f"Error in SHAP feature selection: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # Fallback to original feature list
        logger.warning("Falling back to original feature list due to SHAP error")
        return feature_names


def select_features_by_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    top_n: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[str]:
    """
    Select features based on mutual information with target.
    
    Args:
        X: DataFrame with feature data
        y: Series with target values
        feature_names: List of feature names
        top_n: Number of top features to select (if None, use threshold)
        threshold: Minimum mutual information score to keep (if None, use top_n)
        
    Returns:
        List of selected feature names
    """
    if X.empty or y.empty or not feature_names:
        logger.error("Empty data or feature list")
        return []
    
    try:
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y)
        
        # Create a dataframe of features and scores
        mi_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 40 features by mutual information: {mi_importance['feature'].head(40).tolist()}")
        
        # Select features based on criteria
        if top_n is not None:
            # Select top N features
            top_n = min(top_n, len(feature_names))
            selected_features = mi_importance.head(top_n)['feature'].tolist()
            logger.info(f"Selected {len(selected_features)} features by mutual information top-n={top_n}")
        elif threshold is not None:
            # Select features above threshold
            selected_features = mi_importance[mi_importance['importance'] > threshold]['feature'].tolist()
            logger.info(f"Selected {len(selected_features)} features by mutual information threshold={threshold}")
        else:
            # Default: select all features
            selected_features = feature_names
            logger.info(f"No criteria specified, keeping all {len(selected_features)} features")
        
        return selected_features
    
    except Exception as e:
        logger.error(f"Error in feature selection by mutual information: {str(e)}")
        return feature_names


def select_features(
  df: pd.DataFrame,
  target: str = 'lossadjustedload',
  method: str = 'importance',
  n_features: Optional[int] = None,
  importance_threshold: Optional[float] = None,
  correlation_threshold: float = 0.85,
  target_correlation_threshold: float = 0.05,
  model: Optional[XGBRegressor] = None,
  consensus_threshold: float = 0.5,
) -> Tuple[List[str], Dict]:
  """
  Select features using the specified method.
  
  Args:
    df: DataFrame with features and target
    target: Target column name
    method: Feature selection method ('importance', 'correlation', 'shap', 'mutual_info', 'consensus')
    n_features: Number of features to select
    importance_threshold: Minimum importance score
    correlation_threshold: Maximum correlation between features
    target_correlation_threshold: Minimum correlation with target
    model: Optional pre-trained model for importance and SHAP methods
    consensus_threshold: Threshold for consensus feature selection
    
  Returns:
    Tuple of (selected feature names, selection metadata)
  """
  if df.empty or target not in df.columns:
    logger.error(f"Empty DataFrame or target '{target}' not found")
    return [], {}
  
  # Get all feature columns (excluding target)
  all_features = [col for col in df.columns if col != target]
  
  if not all_features:
    logger.error("No features found in DataFrame")
    return [], {}
  
  # Dictionary to store metadata about the selection process
  metadata = {
    "method": method,
    "total_features": len(all_features),
    "n_features_requested": n_features,
    "importance_threshold": importance_threshold,
  }
  
  try:
    # Create a copy of the DataFrame for preprocessing
    df_processed = df.copy()
    
    # Preprocess DataFrame to handle categorical and datetime features
    numeric_features = []
    for col in all_features:
      # Skip columns with all NaN values
      if df_processed[col].isna().all():
        continue
        
      # Handle datetime columns
      if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
        # Skip datetime columns for feature selection
        continue
        
      # Handle categorical columns by encoding them
      if pd.api.types.is_object_dtype(df_processed[col]) or pd.api.types.is_categorical_dtype(df_processed[col]):
        # For correlation and mutual info, we use one-hot encoding
        if method in ['correlation', 'mutual_info']:
          # Get unique values (limiting to top 10 categories to avoid explosion)
          top_categories = df_processed[col].value_counts().nlargest(10).index
          
          # One-hot encode top categories
          for cat in top_categories:
            new_col = f"{col}_{cat}"
            df_processed[new_col] = (df_processed[col] == cat).astype(int)
            numeric_features.append(new_col)
        else:
          # For tree-based methods, we can use label encoding
          from sklearn.preprocessing import LabelEncoder
          le = LabelEncoder()
          df_processed[col] = le.fit_transform(df_processed[col].astype(str))
          numeric_features.append(col)
      else:
        # For numeric columns, handle NaN values
        if df_processed[col].isna().any():
          df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        numeric_features.append(col)
    
    # Make sure the target column is numeric
    if pd.api.types.is_object_dtype(df_processed[target]) or pd.api.types.is_categorical_dtype(df_processed[target]):
      logger.error(f"Target column '{target}' must be numeric")
      return all_features, {"error": "Target column must be numeric"}
    
    # Handle NaN values in target
    if df_processed[target].isna().any():
      df_processed[target] = df_processed[target].fillna(df_processed[target].median())
    
    # Update metadata
    metadata["processed_features_count"] = len(numeric_features)
    
    # Handle the consensus method differently - run multiple methods and combine
    if method == 'consensus':
        # Run multiple feature selection methods
        method_results = {}
        
        # Initialize model if needed for tree-based methods
        if model is None:
            logger.info("Initializing XGBoost model for consensus feature selection")
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            # Fit model on preprocessed data
            model.fit(df_processed[numeric_features], df_processed[target])
            metadata["model_trained"] = True
        
        # Run importance-based selection
        importance_features = select_features_by_importance(
            model, numeric_features, top_n=n_features, threshold=importance_threshold
        )
        method_results['importance'] = importance_features
        
        # Run correlation-based selection
        correlation_features = select_features_by_correlation(
            df_processed[numeric_features + [target]], target, 'pearson', 
            correlation_threshold, target_correlation_threshold
        )
        method_results['correlation'] = correlation_features
        
        # Run mutual information selection
        mi_features = select_features_by_mutual_information(
            df_processed[numeric_features], df_processed[target], numeric_features, 
            top_n=n_features, threshold=importance_threshold
        )
        method_results['mutual_info'] = mi_features
        
        # Try SHAP selection if available
        if SHAP_AVAILABLE:
            try:
                shap_features = select_features_by_shap(
                    model, df_processed[numeric_features], numeric_features, 
                    top_n=n_features, threshold=importance_threshold
                )
                method_results['shap'] = shap_features
            except Exception as e:
                logger.warning(f"SHAP selection failed: {str(e)}")
        
        # Apply consensus selection
        selected_features = consensus_feature_selection(
            method_results, threshold=consensus_threshold, top_n=n_features
        )
        
        # Update metadata
        metadata["consensus_methods"] = list(method_results.keys())
        metadata["consensus_threshold"] = consensus_threshold
        metadata["method_feature_counts"] = {k: len(v) for k, v in method_results.items()}
    else:
        # Initialize model if needed and not provided
        if method in ['importance', 'shap'] and model is None:
          logger.info("Initializing XGBoost model for feature selection")
          model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
          )
          # Fit model on preprocessed data
          model.fit(df_processed[numeric_features], df_processed[target])
          metadata["model_trained"] = True
        
        # Apply selected method
        if method == 'importance':
          logger.info("Using feature importance for selection")
          selected_features = select_features_by_importance(
            model, numeric_features, n_features, importance_threshold
          )
        elif method == 'correlation':
          logger.info("Using correlation analysis for selection")
          # Create correlation matrix using numeric features only
          selected_features = select_features_by_correlation(
            df_processed[numeric_features + [target]], target, 'pearson', correlation_threshold, target_correlation_threshold
          )
        elif method == 'shap':
          logger.info("Using SHAP values for selection")
          selected_features = select_features_by_shap(
            model, df_processed[numeric_features], numeric_features, n_features, importance_threshold
          )
        elif method == 'mutual_info':
          logger.info("Using mutual information for selection")
          selected_features = select_features_by_mutual_information(
            df_processed[numeric_features], df_processed[target], numeric_features, n_features, importance_threshold
          )
        else:
          logger.warning(f"Unknown method '{method}', using all features")
          selected_features = numeric_features
    
    # Update metadata
    metadata["selected_features_count"] = len(selected_features)
    metadata["top_features"] = selected_features[:10] if len(selected_features) > 10 else selected_features
    
    return selected_features, metadata
  
  except Exception as e:
    logger.error(f"Error in feature selection: {str(e)}")
    logger.error(traceback.format_exc())
    return all_features, {"error": str(e), "method": method, "total_features": len(all_features)}


def save_feature_selection(selected_features: List[str], metadata: Dict, output_path: str) -> bool:
    """
    Save feature selection results to a file.
    
    Args:
        selected_features: List of selected feature names
        metadata: Dictionary with selection metadata
        output_path: Path to save the selection
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Create a dictionary with all data
        selection_data = {
            "selected_features": selected_features,
            "metadata": metadata,
        }
        
        # Save to file
        with open(output_path, 'wb') as f:
            pickle.dump(selection_data, f)
        
        logger.info(f"Feature selection saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving feature selection: {str(e)}")
        return False


def load_feature_selection(input_path: str) -> Tuple[List[str], Dict]:
    """
    Load feature selection results from a file.
    
    Args:
        input_path: Path to load the selection from
        
    Returns:
        Tuple of (selected feature names, metadata)
    """
    try:
        # Load from file
        with open(input_path, 'rb') as f:
            selection_data = pickle.load(f)
        
        selected_features = selection_data.get("selected_features", [])
        metadata = selection_data.get("metadata", {})
        
        logger.info(f"Loaded feature selection from {input_path}: {len(selected_features)} features")
        return selected_features, metadata
    
    except Exception as e:
        logger.error(f"Error loading feature selection: {str(e)}")
        return [], {}
