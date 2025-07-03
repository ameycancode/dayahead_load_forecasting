"""
Feature engineering module for energy load forecasting.
Handles creation of time features, lag features, and other derived features.
"""
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import holidays

# Set up logging
logger = logging.getLogger(__name__)


def add_weather_features(df, weather_df):
    """
    Add weather features to dataframe using Open-Meteo data.
    From your weather_features.py.
    
    Args:
        df: DataFrame with datetime column
        weather_df: DataFrame with weather forecast data
        
    Returns:
        DataFrame with additional weather features
    """
    if df.empty:
        logger.warning("Empty dataframe provided to add_weather_features")
        return df
    
    if 'datetime' not in df.columns:
        logger.error("DataFrame missing datetime column")
        return df
    
    try:
        logger.info("Adding weather features to dataset")

        # logger.info(f"incoming forecast columns: {df.columns}")
        # logger.info(f"incoming weather columns: {weather_df.columns}")
        
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Rename weather time column to match datetime for merging
        weather_df = weather_df.rename(columns={'time': 'datetime'})
        
        # Round datetimes to nearest hour for joining
        df['datetime_hour'] = df['datetime'].dt.floor('h')
        weather_df['datetime_hour'] = weather_df['datetime'].dt.floor('h')
        # logger.info(f"updated weather columns: {weather_df.columns}")
        
        # Merge weather data with main dataframe
        result_df = pd.merge(df, weather_df, on='datetime_hour', how='left')
        result_df = result_df.rename(columns={'datetime_x': 'datetime'})
        
        # Clean up temporary column
        result_df = result_df.drop('datetime_hour', axis=1)
        
        # Create additional derived features
        if all(col in result_df.columns for col in ['temperature_2m', 'apparent_temperature']):
            # Temperature difference (feels like vs actual)
            result_df['temperature_difference'] = result_df['apparent_temperature'] - result_df['temperature_2m']
        
        # Solar radiation features
        if all(col in result_df.columns for col in ['direct_radiation', 'diffuse_radiation']):
            # Total radiation
            result_df['total_radiation'] = result_df['direct_radiation'] + result_df['diffuse_radiation']
            
            # Radiation ratio (direct to diffuse)
            result_df['radiation_ratio'] = np.where(
                result_df['diffuse_radiation'] > 0,
                result_df['direct_radiation'] / result_df['diffuse_radiation'],
                0
            )
        
        # Create heating and cooling degree hours
        if 'temperature_2m' in result_df.columns:
            result_df['heating_degree_hours'] = np.maximum(18.3 - result_df['temperature_2m'], 0)  # Below 65°F
            result_df['cooling_degree_hours'] = np.maximum(result_df['temperature_2m'] - 23.9, 0)  # Above 75°F
        
        # Time-of-day specific temperatures
        if 'temperature_2m' in result_df.columns:
            # Create hour of day if not present
            if 'hour' not in result_df.columns:
                result_df['hour'] = result_df['datetime'].dt.hour
                
            # Morning temperature
            morning_mask = (result_df['hour'] >= 6) & (result_df['hour'] <= 9)
            # Evening temperature
            evening_mask = (result_df['hour'] >= 17) & (result_df['hour'] <= 22)
            
            result_df['morning_temp'] = np.where(morning_mask, result_df['temperature_2m'], 0)
            result_df['evening_temp'] = np.where(evening_mask, result_df['temperature_2m'], 0)

        # Add daily temperature range (daily min/max)
        if "temperature_2m" in result_df.columns:
            # Group by day to get daily min/max
            if "date" not in result_df.columns:
                result_df["date"] = result_df["datetime"].dt.date

            daily_temp = (
                result_df.groupby("date")["temperature_2m"].agg(["min", "max"]).reset_index()
            )
            daily_temp.columns = ["date", "daily_min_temp", "daily_max_temp"]

            # Merge back to main dataframe
            result_df = pd.merge(result_df, daily_temp, on="date", how="left")

            # Add daily temperature range
            result_df["daily_temp_range"] = result_df["daily_max_temp"] - result_df["daily_min_temp"]
        
        # Cloud impact on solar generation
        if all(col in result_df.columns for col in ['cloudcover', 'is_solar_window']):
            # Cloud attenuation during solar window
            result_df['solar_window_cloudcover'] = result_df['cloudcover'] * result_df['is_solar_window'] / 100
        
        # Clean up any NaN values in weather columns
        weather_cols = [col for col in result_df.columns if col not in df.columns or col in ['temperature_2m', 'cloudcover']]
        for col in weather_cols:
            if col in result_df.columns and result_df[col].isna().any():
                # Use appropriate fill method
                if col.endswith('_change_3h') or col.endswith('_change_24h'):
                    # For change columns, fill with 0
                    result_df[col] = result_df[col].fillna(0)
                else:
                    # For others, use forward/backward fill
                    result_df[col] = result_df[col].ffill().bfill()
        
        logger.info(f"Added weather features to dataset")
        logger.info(f"result_df columns: {result_df.columns}")

        return result_df
    
    except Exception as e:
        logger.error(f"Error adding weather features: {e}")
        return df


def add_solar_features(df):
    """
    Add solar features to dataframe using weather data.
    From your solar_features.py.
    
    Args:
        df: DataFrame with datetime column and weather features
        
    Returns:
        DataFrame with additional solar features
    """
    if df.empty:
        logger.warning("Empty dataframe provided to add_solar_features")
        return df
    
    if 'datetime' not in df.columns:
        logger.error("DataFrame missing datetime column")
        return df
    
    try:
        logger.info("Adding solar features to dataset")
        
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Add hour column if not present for temporal features
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        
        # Create solar window features (9am-4pm typically peak solar)
        df['is_solar_window'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        
        # Morning rise, solar peak, and evening ramp periods
        df['is_morning_rise'] = ((df['hour'] >= 6) & (df['hour'] < 9)).astype(int)
        df['is_solar_peak'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        df['is_evening_ramp'] = ((df['hour'] >= 16) & (df['hour'] <= 20)).astype(int)

        # Make sure all required radiation data is available
        if all(
            col in df.columns
            for col in ["direct_radiation", "diffuse_radiation", "shortwave_radiation"]
        ):
            # Direct to total ratio (indicates clear vs cloudy conditions)
            df["direct_radiation_ratio"] = np.where(
                df["shortwave_radiation"] > 0,
                df["direct_radiation"] / df["shortwave_radiation"],
                0,
            )
        
        # Check if weather data with solar information is available
        has_weather_data = all(
            col in df.columns
            for col in ['direct_radiation', 'diffuse_radiation', 'is_day']
        )
        
        if has_weather_data:
            logger.info("Using Open-Meteo API solar data")
            
            # If total radiation not provided but components are, calculate it
            if 'shortwave_radiation' not in df.columns and all(
                col in df.columns for col in ['direct_radiation', 'diffuse_radiation']
            ):
                df['total_radiation'] = df['direct_radiation'] + df['diffuse_radiation']
            elif 'shortwave_radiation' in df.columns:
                df['total_radiation'] = df['shortwave_radiation']
            
            # Solar energy potential calculation - normalize radiation values for the day
            if 'total_radiation' in df.columns:
                # Group by date to normalize within each day
                if 'date' not in df.columns:
                    df['date'] = df['datetime'].dt.date
                
                # Calculate daily max radiation for normalization
                daily_max = df.groupby('date')['total_radiation'].transform('max')
                # Avoid division by zero
                df['solar_potential'] = np.where(
                    daily_max > 0, df['total_radiation'] / daily_max, 0
                )
            
            # Use is_day from API directly
            if 'is_day' in df.columns:
                df['solar_daylight'] = df['is_day']
            
            # Calculate direct:diffuse ratio - important indicator for solar effectiveness
            if all(
                col in df.columns for col in ['direct_radiation', 'diffuse_radiation']
            ):
                # Avoid division by zero
                df['direct_diffuse_ratio'] = np.where(
                    df['diffuse_radiation'] > 0,
                    df['direct_radiation'] / df['diffuse_radiation'],
                    0
                )
                
                # Cap extreme values
                if len(df) > 1:
                    upper_limit = df['direct_diffuse_ratio'].quantile(0.99)
                    df['direct_diffuse_ratio'] = df['direct_diffuse_ratio'].clip(
                        0, upper_limit
                    )
        
        # Clean up any NaN values
        solar_cols = [
            col
            for col in df.columns
            if 'solar_' in col or 'is_' in col or 'radiation' in col or col == 'direct_diffuse_ratio'
        ]
        for col in solar_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        logger.info(f"Added solar features to dataset")
        return df
    
    except Exception as e:
        logger.error(f"Error adding solar features: {e}")
        return df


def add_solar_ratios(df, forecast_delay_days):
    """
    Add solar generation ratio features.
    From your solar_features.py.
    
    Args:
        df: DataFrame with load and generation columns
        
    Returns:
        DataFrame with additional solar ratio features
    """
    if df.empty:
        return df
    
    try:
        # Make copy to avoid warnings
        df = df.copy()

        lag_hours = forecast_delay_days * 24
        loadlal_lag_feat = f'loadlal_lag_{lag_hours}h'
        genlal_lag_feat = f'genlal_lag_{lag_hours}h'
        lossadjustedload_lag_feat = f'lossadjustedload_lag_{lag_hours}h'
        
        # Generation to load ratio - indicates solar penetration
        if all(col in df.columns for col in [loadlal_lag_feat, genlal_lag_feat]):
            # Avoid division by zero
            df['gen_load_ratio'] = df[genlal_lag_feat].abs() / df[loadlal_lag_feat].replace(0, np.nan)
            df['gen_load_ratio'] = df['gen_load_ratio'].fillna(0)
            
            # Cap extreme values
            if len(df) > 1:
                upper_limit = df['gen_load_ratio'].quantile(0.99)
                df['gen_load_ratio'] = df['gen_load_ratio'].clip(upper=upper_limit)
            
            # Add features about maximum generation periods
            if len(df) > 3:
                df['is_high_gen_period'] = (df['gen_load_ratio'] > df['gen_load_ratio'].quantile(0.75)).astype(int)
        
        # Add duck curve metrics if we have load data
        if 'datetime' in df.columns and lossadjustedload_lag_feat in df.columns:
            # Group by day to find daily load patterns
            if 'date' not in df.columns:
                df['date'] = df['datetime'].dt.date
            
            # Calculate duck curve metrics
            daily_metrics = []
            
            for date, day_df in df.groupby('date'):
                if len(day_df) < 24:  # Skip incomplete days
                    continue
                    
                morning_peak = day_df[day_df['hour'].between(6, 9)][lossadjustedload_lag_feat].max()
                solar_dip = day_df[day_df['hour'].between(11, 15)][lossadjustedload_lag_feat].min()
                evening_peak = day_df[day_df['hour'].between(17, 21)][lossadjustedload_lag_feat].max()
                
                # Calculate duck curve metrics
                if solar_dip > 0:
                    evening_to_dip = evening_peak / solar_dip if solar_dip else 1
                else:
                    evening_to_dip = 1

                daily_metrics.append(
                    {
                        "date": date,
                        "duck_curve_ratio": evening_to_dip,  # Main duck curve indicator
                    }
                )
            
            if daily_metrics:
                metrics_df = pd.DataFrame(daily_metrics)
                
                # Join back to main dataframe
                df = df.merge(metrics_df, on='date', how='left')
                
                # Fill missing values for days with incomplete data
                for col in metrics_df.columns:
                    if col != 'date' and col in df.columns:
                        df[col] = df[col].ffill().bfill()

        # Consumption to generation ratio
        if loadlal_lag_feat in df.columns and genlal_lag_feat in df.columns:
            df['cons_to_gen_ratio'] = df[loadlal_lag_feat] / df[genlal_lag_feat].replace(0, np.nan)
            df['cons_to_gen_ratio'] = df['cons_to_gen_ratio'].fillna(0)
            
            # Cap extreme ratios
            ratio_cap = pd.Series(df['cons_to_gen_ratio']).quantile(0.99)
            df['cons_to_gen_ratio'] = df['cons_to_gen_ratio'].clip(upper=ratio_cap)

        # Add generation ratio
        df['generation_ratio'] = np.abs(df[genlal_lag_feat] / df[loadlal_lag_feat].replace(0, np.nan))
        df['generation_ratio'] = df['generation_ratio'].fillna(0)
        
        return df
    
    except Exception as e:
        logger.error(f"Error adding solar ratio features: {e}")
        return df


def create_weather_solar_interactions(df):
    """
    Create interaction features between weather and solar data.
    From your weather_features.py.
    
    Args:
        df: DataFrame with weather and solar features
        
    Returns:
        DataFrame with additional interaction features
    """
    if df.empty:
        return df
    
    try:
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Wind cooling effect on buildings
        if all(col in df.columns for col in ['windspeed_10m', 'temperature_2m']):
            # Wind chill effect (increases heating load)
            if 'heating_degree_hours' in df.columns:
                df['wind_heating_effect'] = np.where(
                    df['temperature_2m'] < 18.3,  # Below heating threshold (65°F)
                    df['windspeed_10m'] * df['heating_degree_hours'] / 10,
                    0
                )
            
            # Wind cooling effect (reduces cooling load)
            if 'cooling_degree_hours' in df.columns:
                df['wind_cooling_effect'] = np.where(
                    df['temperature_2m'] > 23.9,  # Above cooling threshold (75°F)
                    df['windspeed_10m'] * df['cooling_degree_hours'] / 10,
                    0
                )
        
        # Humidity impact on cooling needs
        if all(col in df.columns for col in ['relativehumidity_2m', 'cooling_degree_hours']):
            df['humidity_cooling_impact'] = np.where(
                df['cooling_degree_hours'] > 0,
                df['relativehumidity_2m'] * df['cooling_degree_hours'] / 100,
                0
            )
        
        # Evening humidity discomfort
        if all(col in df.columns for col in ['relativehumidity_2m', 'evening_temp']):
            evening_hot_mask = df['evening_temp'] > 23.9  # Hot evenings
            df['evening_humidity_discomfort'] = np.where(
                evening_hot_mask,
                df['relativehumidity_2m'] * (df['evening_temp'] - 23.9) / 100,
                0
            )
        
        # Clean up NaN values in interaction features
        interaction_cols = [col for col in df.columns 
                            if 'effect' in col or 'impact' in col or 'discomfort' in col]
        
        for col in interaction_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating weather-solar interactions: {e}")
        return df
