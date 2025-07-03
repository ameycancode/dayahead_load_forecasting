"""
Weather service module for energy load forecasting.
Handles fetching weather data from Open-Meteo API.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import pytz
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Set up logging
logger = logging.getLogger(__name__)


def get_openmeteo_client(cache_dir='/tmp/weather_cache', expire_after=3600):
    """
    Set up an Open-Meteo API client with caching.
    
    Args:
        cache_dir: Directory for the cache
        expire_after: Cache expiration time in seconds
        
    Returns:
        OpenMeteo client with retry capabilities
    """
    # Create cache directory if specified
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Setup the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession(
        cache_dir, expire_after=expire_after
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_weather_data(
    target_date: datetime,
    config=None,
    days: int = 5
) -> Optional[pd.DataFrame]:
    """
    Fetch weather data for any date (historical or forecast).
    
    Args:
        target_date: Target date for which to fetch weather data
        days: Number of days to fetch (for forecast only)
        config (EnvironmentConfig): Centralized configuration object
        
    Returns:
        DataFrame with hourly weather data or None if error
    """
    try:
        import pytz

        if config is None:
            raise ValueError("Config object is required")
        
        # Use configuration values
        latitude = config.get('DEFAULT_LATITUDE', 32.7157)
        longitude = config.get('DEFAULT_LONGITUDE', -117.1611)
        timezone = config.get('DEFAULT_TIMEZONE', 'America/Los_Angeles')
        weather_variables = config.get('WEATHER_VARIABLES', [])
        
        # Configure Open-Meteo client
        openmeteo = get_openmeteo_client()
        
        # Get current date in the specified timezone
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        
        # Ensure target_date is a datetime (not just a date)
        if not isinstance(target_date, datetime) and not hasattr(target_date, 'hour'):
            target_date = datetime.combine(target_date, datetime.min.time())
        
        # Add timezone info if missing
        if getattr(target_date, 'tzinfo', None) is None:
            try:
                target_date = tz.localize(target_date)
            except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError):
                # Handle DST transitions - default to standard time
                target_date = tz.localize(target_date, is_dst=False)
        
        # Determine if we need historical or forecast data
        is_historical = target_date.date() < now.date()
        
        logger.info(f"Fetching {'historical' if is_historical else 'forecast'} weather data for {target_date.strftime('%Y-%m-%d')}")
        
        if is_historical:
            # Use historical/archive API
            url = "https://archive-api.open-meteo.com/v1/archive"
            
            # Format dates for the API
            start_date = target_date.strftime("%Y-%m-%d")
            end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": weather_variables,
                "timezone": timezone
            }
        else:
            # Use forecast API
            url = "https://api.open-meteo.com/v1/forecast"
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": weather_variables,
                "forecast_days": min(days, 16),  # API limit is 16 days
                "timezone": timezone
            }
        
        # Make the API request
        responses = openmeteo.weather_api(url, params=params)
        
        # Process the first location
        if not responses or len(responses) == 0:
            logger.error("No response from Open-Meteo API")
            return None
            
        response = responses[0]
        
        # Process hourly data
        hourly = response.Hourly()
        hourly_data = {}
        
        # Create date range
        time_index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        
        # Convert to Series to enable .dt accessor
        hourly_data["time"] = pd.Series(time_index)
        
        # If timezone is provided, convert to local time
        if timezone:
            hourly_data["time"] = hourly_data["time"].dt.tz_convert(timezone)
            
        # Convert to timezone-naive datetime for easier processing
        hourly_data["time"] = hourly_data["time"].dt.tz_localize(None)
        
        # IMPORTANT: Ensure time format is consistent (HH:00:00)
        hourly_data["time"] = hourly_data["time"].dt.floor('H')
        
        # Extract all variables
        for i, variable in enumerate(weather_variables):
            if i < hourly.VariablesLength():
                hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()
                
        # Create DataFrame
        df = pd.DataFrame(data=hourly_data)
        
        # Filter to just the target date
        target_date_start = pd.Timestamp(target_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None))
        target_date_end = target_date_start + pd.Timedelta(days=1)
        df = df[(df['time'] >= target_date_start) & (df['time'] < target_date_end)]
        
        # Add hour column for consistency
        df['hour'] = df['time'].dt.hour
        
        # Check if we got all 24 hours
        if len(df) != 24:
            logger.warning(f"Expected 24 hours of weather data, got {len(df)}")
            
            # If we're missing hours, create a full day template and fill with available data
            full_day = pd.DataFrame({
                'time': pd.date_range(start=target_date_start, periods=24, freq='H'),
                'hour': range(24)
            })
            
            # Merge with available data
            df = pd.merge(full_day, df, on=['time', 'hour'], how='left')
        
        # Log to confirm time format
        logger.info(f"Weather data time format: {df['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S') if not df.empty else 'No data'}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
