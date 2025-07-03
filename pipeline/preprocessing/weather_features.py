"""
Weather feature integration for energy load forecasting.
This module handles fetching and processing weather data from Open-Meteo API.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from configs import config

# Configure logging
logger = logging.getLogger(__name__)

# Default location coordinates for San Diego, CA
DEFAULT_LATITUDE = 32.7157
DEFAULT_LONGITUDE = -117.1611
DEFAULT_TIMEZONE = "America/Los_Angeles"

# Open-Meteo cache directory
OPEN_METEO_CACHE_DIR = "/tmp/weather_cache"

# Essential weather variables for energy forecasting
WEATHER_VARIABLES = [
    "temperature_2m",  # Main driver of heating/cooling load
    "apparent_temperature",  # Human-perceived temperature (comfort)
    "cloudcover",  # Overall cloud cover - impacts both lighting and solar
    "direct_radiation",  # Direct solar radiation at surface
    "diffuse_radiation",  # Diffuse solar radiation at surface
    "shortwave_radiation",  # Global solar radiation
    "windspeed_10m",  # Wind impacts building thermal envelope
    "is_day",  # Boolean indicator if sun is up
    "relativehumidity_2m",  # Humidity impacts cooling loads
]


def get_openmeteo_client(cache_dir=None, expire_after=3600):
    """
    Set up an Open-Meteo API client with caching.

    Args:
        cache_dir: Directory for the cache (None for default)
        expire_after: Cache expiration time in seconds

    Returns:
        OpenMeteo client with retry capabilities
    """
    # Create cache directory if specified
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Setup the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession(
        cache_dir or ".cache", expire_after=expire_after
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_historical_weather(
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    timezone: str = DEFAULT_TIMEZONE,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical weather data from Open-Meteo API.

    Args:
        start_date: Start date for weather data
        end_date: End date for weather data
        latitude: Location latitude
        longitude: Location longitude
        timezone: Timezone identifier
        use_cache: Whether to use cached data if available

    Returns:
        DataFrame with hourly weather data or None if error
    """
    try:
        # Convert dates to strings if needed
        if not isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        if not isinstance(end_date, str):
            end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")

        # Create cache directory if doesn't exist
        if use_cache and not os.path.exists(OPEN_METEO_CACHE_DIR):
            os.makedirs(OPEN_METEO_CACHE_DIR, exist_ok=True)

        # Create cache filename
        cache_file = None
        if use_cache:
            cache_file = os.path.join(
                OPEN_METEO_CACHE_DIR,
                f"weather_{start_date}_{end_date}_{latitude}_{longitude}.csv",
            )

            # Return cached data if exists
            if os.path.exists(cache_file):
                logger.info(f"Loading cached weather data from {cache_file}")
                df = pd.read_csv(cache_file)

                # Convert time column to datetime
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])

                return df

        # Set up OpenMeteo client
        logger.info(
            f"Fetching weather data from Open-Meteo: {start_date} to {end_date}"
        )

        # Setup the Open-Meteo client with caching
        openmeteo = get_openmeteo_client(
            cache_dir=OPEN_METEO_CACHE_DIR if use_cache else None
        )

        # Configure API request parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": WEATHER_VARIABLES,
            "timezone": timezone,
        }

        # Make the API request
        responses = openmeteo.weather_api(url, params=params)

        # Process the first location
        if not responses or len(responses) == 0:
            logger.error("No response from Open-Meteo API")
            return None

        response = responses[0]

        # Log response details
        logger.debug(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        logger.debug(f"Elevation: {response.Elevation()} m asl")
        logger.debug(
            f"Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}"
        )

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

        # Extract all variables
        for i, variable in enumerate(WEATHER_VARIABLES):
            if i < hourly.VariablesLength():
                hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

        # Create DataFrame
        df = pd.DataFrame(data=hourly_data)

        # Save to cache if enabled
        if use_cache and cache_file:
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached weather data to {cache_file}")

        return df

    except Exception as e:
        logger.error(f"Error fetching historical weather: {str(e)}")
        return None


def fetch_forecast_weather(
    start_date: Union[str, datetime, pd.Timestamp],
    days: int = 16,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    timezone: str = DEFAULT_TIMEZONE,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch weather forecast data from Open-Meteo API.

    Args:
        start_date: Start date for forecast data (typically today)
        days: Number of forecast days to fetch (max 16)
        latitude: Location latitude
        longitude: Location longitude
        timezone: Timezone identifier
        use_cache: Whether to use cached data if available

    Returns:
        DataFrame with hourly forecast data or None if error
    """
    try:
        # Convert date to string if needed
        if not isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        # Create cache directory if doesn't exist
        if use_cache and not os.path.exists(OPEN_METEO_CACHE_DIR):
            os.makedirs(OPEN_METEO_CACHE_DIR, exist_ok=True)

        # Create cache filename
        cache_file = None
        if use_cache:
            cache_file = os.path.join(
                OPEN_METEO_CACHE_DIR,
                f"forecast_{start_date}_{days}days_{latitude}_{longitude}.csv",
            )

            # Check cache age - only use if created today
            if os.path.exists(cache_file):
                file_time = os.path.getmtime(cache_file)
                file_date = datetime.fromtimestamp(file_time).date()

                if file_date == datetime.now().date():
                    logger.info(f"Loading cached forecast data from {cache_file}")
                    df = pd.read_csv(cache_file)

                    # Convert time column to datetime
                    if "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"])

                    return df
                else:
                    logger.info("Cached forecast is outdated, fetching fresh data")

        # Set up OpenMeteo client
        logger.info(
            f"Fetching forecast data from Open-Meteo: {start_date} for {days} days"
        )

        # Setup the Open-Meteo client with caching
        openmeteo = get_openmeteo_client(
            cache_dir=OPEN_METEO_CACHE_DIR if use_cache else None
        )

        # Configure API request parameters
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": WEATHER_VARIABLES,
            "forecast_days": min(days, 16),  # API limit is 16 days
            "timezone": timezone,
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

        # Create date range time Series
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

        # Extract all variables
        for i, variable in enumerate(WEATHER_VARIABLES):
            if i < hourly.VariablesLength():
                hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

        # Create DataFrame
        df = pd.DataFrame(data=hourly_data)

        # Save to cache if enabled
        if use_cache and cache_file:
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached forecast data to {cache_file}")

        return df

    except Exception as e:
        logger.error(f"Error fetching forecast weather: {str(e)}")
        return None


def add_weather_features(
    df: pd.DataFrame, use_forecast: bool = True, weather_cache: bool = True
) -> pd.DataFrame:
    """
    Add weather features to dataframe.

    Args:
        df: DataFrame with datetime column
        use_forecast: Whether to use forecast data for future dates
        weather_cache: Whether to use cached weather data

    Returns:
        DataFrame with additional weather features
    """
    if df.empty:
        logger.warning("Empty dataframe provided to add_weather_features")
        return df

    if "datetime" not in df.columns:
        logger.error("DataFrame missing datetime column")
        return df

    try:
        logger.info("Adding weather features to dataset")

        # Make a copy to avoid warnings
        df = df.copy()

        # Get date range for historical data
        min_date = df["datetime"].min().date()
        max_date = df["datetime"].max().date()

        # Fetch historical weather data
        weather_df = fetch_historical_weather(
            min_date, max_date, use_cache=weather_cache
        )

        # Handle missing historical data
        if weather_df is None or weather_df.empty:
            logger.warning("Failed to fetch historical weather data")
            return df

        # If future dates exist and forecast enabled, add forecast data
        if use_forecast:
            today = datetime.now().date()

            # Check if dataframe has future dates
            if max_date > today:
                days_ahead = (max_date - today).days + 1

                forecast_df = fetch_forecast_weather(
                    today, days=days_ahead, use_cache=weather_cache
                )

                if forecast_df is not None and not forecast_df.empty:
                    # Combine historical and forecast data
                    weather_df = pd.concat(
                        [
                            # Only keep historical data up to yesterday
                            weather_df[
                                pd.to_datetime(weather_df["time"]).dt.date < today
                            ],
                            # Add forecast data from today onwards
                            forecast_df[
                                pd.to_datetime(forecast_df["time"]).dt.date >= today
                            ],
                        ]
                    )

        # Rename time column to match datetime
        weather_df = weather_df.rename(columns={"time": "datetime"})

        # Round datetimes to nearest hour to ensure proper joining
        df["datetime_hour"] = df["datetime"].dt.floor("h")
        weather_df["datetime_hour"] = weather_df["datetime"].dt.floor("h")

        # Merge weather data with main dataframe
        df = pd.merge(
            df, weather_df, on="datetime_hour", how="left", suffixes=("", "_weather")
        )

        # Clean up temporary column
        df = df.drop("datetime_hour", axis=1)

        # Create additional derived features
        if all(col in df.columns for col in ["temperature_2m", "apparent_temperature"]):
            # Temperature difference (feels like vs actual)
            df["temperature_difference"] = (
                df["apparent_temperature"] - df["temperature_2m"]
            )

        # Create heating and cooling degree hours
        if "temperature_2m" in df.columns:
            df["heating_degree_hours"] = np.maximum(
                18.3 - df["temperature_2m"], 0
            )  # Below 65°F (18.3°C)
            df["cooling_degree_hours"] = np.maximum(
                df["temperature_2m"] - 23.9, 0
            )  # Above 75°F (23.9°C)

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

        # Time-of-day specific temperatures (critical for load patterns)
        if "temperature_2m" in df.columns:
            # Create hour of day if not present
            if "hour" not in df.columns:
                df["hour"] = df["datetime"].dt.hour

            # Morning temperature (when people wake up/get ready)
            morning_mask = (df["hour"] >= 6) & (df["hour"] <= 9)
            # Evening temperature (when people return home)
            evening_mask = (df["hour"] >= 17) & (df["hour"] <= 22)

            # These are critical periods for residential load
            df["morning_temp"] = np.where(morning_mask, df["temperature_2m"], 0)
            df["evening_temp"] = np.where(evening_mask, df["temperature_2m"], 0)

        # Cloud impact on solar generation
        if all(col in df.columns for col in ["cloudcover", "is_solar_window"]):
            # Only care about clouds during solar generation window
            df["solar_window_cloudcover"] = df["cloudcover"] * df.get(
                "is_solar_window", 0
            )

        # Add daily temperature range (daily min/max)
        if "temperature_2m" in df.columns:
            # Group by day to get daily min/max
            if "date" not in df.columns:
                df["date"] = df["datetime"].dt.date

            daily_temp = (
                df.groupby("date")["temperature_2m"].agg(["min", "max"]).reset_index()
            )
            daily_temp.columns = ["date", "daily_min_temp", "daily_max_temp"]

            # Merge back to main dataframe
            df = pd.merge(df, daily_temp, on="date", how="left")

            # Add daily temperature range
            df["daily_temp_range"] = df["daily_max_temp"] - df["daily_min_temp"]

        # Clean up any NaN values in weather columns
        weather_cols = [
            col
            for col in df.columns
            if col not in df.columns[: df.columns.get_loc("datetime") + 1]
        ]
        for col in weather_cols:
            if col in df.columns and df[col].isna().any():
                # Use appropriate fill method based on column type
                if col.endswith("_change_3h") or col.endswith("_change_24h"):
                    # For change columns, fill with 0
                    df[col] = df[col].fillna(0)
                else:
                    # For others, use forward/backward fill
                    df[col] = df[col].ffill().bfill()

        logger.info(f"Added weather features to dataset")
        return df

    except Exception as e:
        import traceback
        logger.error(f"Error adding weather features: {e}")
        logger.error(traceback.format_exc())
        return df


def create_weather_solar_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between weather and solar data.

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

        # We only need a few targeted interactions that directly impact energy load

        # 1. Solar production efficiency based on temperature
        #    (solar panels are less efficient at higher temperatures)
        if all(col in df.columns for col in ["temperature_2m", "is_solar_window"]):
            df["solar_temp_impact"] = np.where(
                df["is_solar_window"] == 1,
                # Simplified temperature impact - panels lose efficiency above 25°C
                np.maximum(0, (df["temperature_2m"] - 25) / 100),
                0,
            )

        # 2. Wind cooling effect on buildings
        #    (wind increases heat loss in cold weather, beneficial cooling in hot weather)
        if all(col in df.columns for col in ["windspeed_10m", "temperature_2m"]):
            # Wind chill effect (increases heating load)
            df["wind_heating_effect"] = np.where(
                df["temperature_2m"] < 18.3,  # Below heating threshold (65°F)
                df["windspeed_10m"] * df["heating_degree_hours"] / 10,
                0,
            )

            # Wind cooling effect (reduces cooling load)
            df["wind_cooling_effect"] = np.where(
                df["temperature_2m"] > 23.9,  # Above cooling threshold (75°F)
                df["windspeed_10m"] * df["cooling_degree_hours"] / 10,
                0,
            )

        # 3. Humidity impact on cooling needs
        #    (high humidity makes cooling less effective)
        if all(
            col in df.columns for col in ["relativehumidity_2m", "cooling_degree_hours"]
        ):
            df["humidity_cooling_impact"] = np.where(
                df["cooling_degree_hours"] > 0,
                df["relativehumidity_2m"] * df["cooling_degree_hours"] / 100,
                0,
            )

        # 4. Evening humidity discomfort
        #    (high evening humidity with high temperature influences cooling loads)
        if all(col in df.columns for col in ["relativehumidity_2m", "evening_temp"]):
            evening_hot_mask = df["evening_temp"] > 23.9  # Hot evenings
            df["evening_humidity_discomfort"] = np.where(
                evening_hot_mask,
                df["relativehumidity_2m"] * (df["evening_temp"] - 23.9) / 100,
                0,
            )

        # Clean up NaN values in interaction features
        interaction_cols = [
            "solar_temp_impact",
            "wind_heating_effect",
            "wind_cooling_effect",
            "humidity_cooling_impact",
            "evening_humidity_discomfort",
        ]

        for col in interaction_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(0)

        return df

    except Exception as e:
        logger.error(f"Error creating weather-solar interactions: {e}")
        return df
