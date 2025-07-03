"""
Solar-specific feature engineering for energy load forecasting.
This module provides functions to generate solar position and production features
using Open-Meteo API data where possible.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Get module logger
logger = logging.getLogger(__name__)

# Default location coordinates for San Diego, CA
DEFAULT_LATITUDE = 32.7157
DEFAULT_LONGITUDE = -117.1611
DEFAULT_TIMEZONE = -8  # UTC-8 for Pacific Time


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add solar position and clear sky features to dataframe.
    Uses weather API data if available, otherwise calculates them.

    Args:
        df: DataFrame with datetime column and weather features from Open-Meteo

    Returns:
        DataFrame with additional solar features
    """
    if df.empty:
        logger.warning("Empty dataframe provided to add_solar_features")
        return df

    if "datetime" not in df.columns:
        logger.error("DataFrame missing datetime column")
        return df

    try:
        logger.info("Adding solar features to dataset")

        # Make a copy to avoid warnings
        df = df.copy()

        # Add hour column if not present for temporal features
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour

        # Create solar window features (9am-4pm typically peak solar)
        df["is_solar_window"] = ((df["hour"] >= 9) & (df["hour"] <= 16)).astype(int)

        # Morning rise, solar peak, and evening ramp periods
        df["is_morning_rise"] = ((df["hour"] >= 6) & (df["hour"] < 9)).astype(int)
        df["is_solar_peak"] = ((df["hour"] >= 11) & (df["hour"] <= 14)).astype(int)
        df["is_evening_ramp"] = ((df["hour"] >= 16) & (df["hour"] <= 20)).astype(int)

        # Check if weather data with solar information is available
        has_weather_data = all(
            col in df.columns
            for col in ["direct_radiation", "diffuse_radiation", "is_day"]
        )

        if has_weather_data:
            logger.info("Using Open-Meteo API solar data")

            # If total radiation not provided but components are, calculate it
            if "shortwave_radiation" not in df.columns and all(
                col in df.columns for col in ["direct_radiation", "diffuse_radiation"]
            ):
                df["total_radiation"] = df["direct_radiation"] + df["diffuse_radiation"]
            elif "shortwave_radiation" in df.columns:
                df["total_radiation"] = df["shortwave_radiation"]

            # Solar energy potential calculation - normalize radiation values for the day
            if "total_radiation" in df.columns:
                # Group by date to normalize within each day
                df["date"] = df["datetime"].dt.date

                # Calculate daily max radiation for normalization
                daily_max = df.groupby("date")["total_radiation"].transform("max")
                # Avoid division by zero
                df["solar_potential"] = np.where(
                    daily_max > 0, df["total_radiation"] / daily_max, 0
                )

            # Use is_day from API directly
            if "is_day" in df.columns:
                df["solar_daylight"] = df["is_day"]

            # Calculate direct:diffuse ratio - important indicator for solar effectiveness
            if all(
                col in df.columns for col in ["direct_radiation", "diffuse_radiation"]
            ):
                # Avoid division by zero
                df["direct_diffuse_ratio"] = np.where(
                    df["diffuse_radiation"] > 0,
                    df["direct_radiation"] / df["diffuse_radiation"],
                    0,
                )

                # Cap extreme values
                upper_limit = df["direct_diffuse_ratio"].quantile(0.99)
                df["direct_diffuse_ratio"] = df["direct_diffuse_ratio"].clip(
                    0, upper_limit
                )
        else:
            logger.info(
                "Weather data not available, calculating solar features manually"
            )
            # Calculate solar position for each datetime
            solar_features = df["datetime"].apply(calculate_solar_position)
            solar_df = pd.json_normalize(solar_features)

            for col in solar_df.columns:
                df[f"solar_{col}"] = solar_df[col]

            # Add clear sky radiation
            df["clear_sky_radiation"] = df["datetime"].apply(
                calculate_clear_sky_radiation
            )

            # Add solar potential (normalized clear sky radiation)
            df["date"] = df["datetime"].dt.date
            daily_max = df.groupby("date")["clear_sky_radiation"].transform("max")
            # Avoid division by zero
            df["solar_potential"] = np.where(
                daily_max > 0, df["clear_sky_radiation"] / daily_max, 0
            )

        # Clean up any NaN values
        solar_cols = [
            col
            for col in df.columns
            if "solar_" in col
            or "is_" in col
            or "radiation" in col
            or col == "direct_diffuse_ratio"
        ]
        for col in solar_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(0)

        logger.info(f"Added solar features to dataset")
        return df

    except Exception as e:
        logger.error(f"Error adding solar features: {e}")
        return df


def add_solar_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add solar generation ratio features.

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

        # Generation to load ratio - indicates solar penetration
        if all(col in df.columns for col in ["loadlal", "genlal"]):
            # Avoid division by zero
            df["gen_load_ratio"] = df["genlal"].abs() / df["loadlal"].replace(0, np.nan)
            df["gen_load_ratio"] = df["gen_load_ratio"].fillna(0)

            # Cap extreme values
            upper_limit = df["gen_load_ratio"].quantile(0.99)
            df["gen_load_ratio"] = df["gen_load_ratio"].clip(upper=upper_limit)

        # Add duck curve metrics if we have load data
        if "datetime" in df.columns and "lossadjustedload" in df.columns:
            # Group by day to find daily load patterns
            if "date" not in df.columns:
                df["date"] = df["datetime"].dt.date

            # Calculate duck curve metrics
            daily_metrics = []

            for date, day_df in df.groupby("date"):
                if len(day_df) < 24:  # Skip incomplete days
                    continue

                morning_peak = day_df[day_df["hour"].between(6, 9)][
                    "lossadjustedload"
                ].max()
                solar_dip = day_df[day_df["hour"].between(11, 15)][
                    "lossadjustedload"
                ].min()
                evening_peak = day_df[day_df["hour"].between(17, 21)][
                    "lossadjustedload"
                ].max()

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
                df = df.merge(metrics_df, on="date", how="left")

                # Fill missing values for days with incomplete data
                for col in metrics_df.columns:
                    if col != "date" and col in df.columns:
                        df[col] = df[col].ffill().bfill()

        return df

    except Exception as e:
        logger.error(f"Error adding solar ratio features: {e}")
        return df


# These functions are kept for fallback when weather API data isn't available


def calculate_solar_position(
    dt: Union[datetime, pd.Timestamp],
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> Dict[str, float]:
    """
    Calculate solar position (zenith angle, azimuth, elevation) for a given time and location.
    Used as fallback when API data isn't available.

    Args:
        dt: Datetime for which to calculate solar position
        latitude: Location latitude in degrees (default: San Diego)
        longitude: Location longitude in degrees (default: San Diego)

    Returns:
        Dictionary with solar position parameters
    """
    try:
        # Ensure datetime is in UTC
        if hasattr(dt, "tz") and dt.tz is not None:
            dt_utc = dt.tz_convert("UTC")
        else:
            # Assume local time and convert to UTC
            dt_utc = pd.Timestamp(dt).tz_localize("US/Pacific").tz_convert("UTC")

        # Calculate day of year
        doy = dt_utc.timetuple().tm_yday

        # Calculate hour of day (decimal)
        hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

        # Convert lat/long to radians
        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)

        # Time equation and declination
        b = (2 * math.pi * (doy - 81)) / 365
        equation_of_time = (
            9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
        )
        declination = math.radians(
            23.45 * math.sin(math.radians((360 / 365) * (doy - 81)))
        )

        # True solar time
        solar_time = hour + equation_of_time / 60 + (longitude / 15)

        # Hour angle
        hour_angle = math.radians(15 * (solar_time - 12))

        # Calculate zenith angle (angle from vertical)
        cos_zenith = math.sin(lat_rad) * math.sin(declination) + math.cos(
            lat_rad
        ) * math.cos(declination) * math.cos(hour_angle)
        cos_zenith = min(max(cos_zenith, -1), 1)  # Clamp to [-1, 1]
        zenith = math.acos(cos_zenith)

        # Calculate azimuth
        cos_azimuth = (math.sin(declination) - math.sin(lat_rad) * math.cos(zenith)) / (
            math.cos(lat_rad) * math.sin(zenith)
        )
        cos_azimuth = min(max(cos_azimuth, -1), 1)  # Clamp to [-1, 1]
        azimuth = math.acos(cos_azimuth)

        # Adjust azimuth for afternoon
        if hour_angle > 0:
            azimuth = 2 * math.pi - azimuth

        # Convert to degrees
        zenith_deg = math.degrees(zenith)
        azimuth_deg = math.degrees(azimuth)

        # Elevation is 90 - zenith
        elevation_deg = 90 - zenith_deg

        # Adjustments for edge cases
        if elevation_deg < -90:
            elevation_deg = -90
        elif elevation_deg > 90:
            elevation_deg = 90

        return {
            "zenith": zenith_deg,
            "azimuth": azimuth_deg,
            "elevation": elevation_deg,
            "daylight": 1 if elevation_deg > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Error calculating solar position: {e}")
        # Return default values to avoid breaking pipeline
        return {"zenith": 90.0, "azimuth": 0.0, "elevation": 0.0, "daylight": 0}


def calculate_clear_sky_radiation(
    dt: Union[datetime, pd.Timestamp],
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> float:
    """
    Calculate theoretical clear-sky radiation for a given time and location.
    Used as fallback when API data isn't available.

    Args:
        dt: Datetime for which to calculate clear sky radiation
        latitude: Location latitude in degrees
        longitude: Location longitude in degrees

    Returns:
        Clear sky radiation value in W/m²
    """
    try:
        # Get solar position
        solar_pos = calculate_solar_position(dt, latitude, longitude)

        # If sun is below horizon, radiation is 0
        if solar_pos["elevation"] <= 0:
            return 0.0

        # Solar constant (W/m²)
        solar_constant = 1361.0

        # Calculate day of year
        if isinstance(dt, pd.Timestamp):
            doy = dt.dayofyear
        else:
            doy = dt.timetuple().tm_yday

        # Eccentricity correction factor
        theta = 2 * math.pi * doy / 365.0
        eccentricity = (
            1.00011
            + 0.034221 * math.cos(theta)
            + 0.00128 * math.sin(theta)
            + 0.000719 * math.cos(2 * theta)
            + 0.000077 * math.sin(2 * theta)
        )

        # Atmospheric transmittance - simplified model
        # Higher values at elevation, lower values at sunrise/sunset
        elevation_rad = math.radians(solar_pos["elevation"])
        air_mass = 1 / (
            math.sin(elevation_rad)
            + 0.50572 * (math.degrees(elevation_rad) + 6.07995) ** -1.6364
        )
        transmittance = 0.7 ** (air_mass**0.678)

        # Calculate clear sky radiation
        clear_sky = (
            solar_constant * eccentricity * transmittance * math.sin(elevation_rad)
        )

        # Ensure non-negative
        return max(0.0, clear_sky)
    except Exception as e:
        logger.error(f"Error calculating clear sky radiation: {e}")
        return 0.0
