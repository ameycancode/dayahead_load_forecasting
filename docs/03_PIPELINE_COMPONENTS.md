# Pipeline Components - Technical Reference

## Table of Contents
- [Overview](#overview)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Training Pipeline](#training-pipeline)
- [Orchestration](#orchestration)
- [Data Flow Summary](#data-flow-summary)

---

## Overview

The pipeline is divided into three main stages:

1. **Preprocessing**: Data retrieval, feature engineering, train/val/test splits
2. **Training**: Feature selection, hyperparameter optimization, model training
3. **Orchestration**: SageMaker pipeline creation and management

Each stage is implemented as modular Python code that can run locally or on SageMaker.

---

## Preprocessing Pipeline

Location: `pipeline/preprocessing/`

### Module: preprocessing.py - Main Orchestration

**Purpose**: Central entry point that orchestrates the complete preprocessing workflow.

#### Main Function: `process_data()`

**Signature**:
```python
def process_data(
    output_train: str,
    output_val: str,
    output_test: str,
    days_delay: int = 14,
    use_reduced_features: bool = False,
    meter_threshold: int = 10,
    use_cache: bool = True,
    use_weather: bool = True,
    use_solar: bool = True,
    weather_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

**Parameters**:
- `output_train/val/test`: Output directory paths for CSV files
- `days_delay`: Data availability delay (default 14 days)
  - Historical data must be at least 14 days old
  - Prevents using data not available at forecast time
- `use_reduced_features`: If True, use only top features (faster)
- `meter_threshold`: Minimum meter count for filtering
- `use_cache`: Enable CSV caching (speeds up dev iterations)
- `use_weather`: Add weather features from Open-Meteo API
- `use_solar`: Add solar position and radiation features
- `weather_cache`: Cache weather API responses

**Execution Flow**:
```python
# 1. Query raw data from Redshift/Athena
df = query_data(config, current_date, load_profile, rate_group_filter)
# Returns: 3-year time series with columns:
#   datetime, loadlal, genlal, metercount, loadmetercount,
#   genmetercount, loadprofile, rategroup, submission

# 2. Ensure exactly 3-year period
df = ensure_three_year_period(df, current_date, days_delay)
# Filters to: (current_date - days_delay - 3 years) to (current_date - days_delay)

# 3. Mark forecast-available features
df = mark_forecast_available_features(df)
# Adds column: is_forecast_available (boolean)
# True for: time features, weather, solar position
# False for: historical lags, past actuals

# 4. Add weather features (if enabled)
if use_weather:
    weather_df = fetch_historical_weather(start_date, end_date, config)
    df = add_weather_features(df, weather_df)
# Adds 17 weather variables from Open-Meteo API

# 5. Add solar features (if enabled)
if use_solar:
    df = add_solar_features(df)
# Adds: solar position, radiation, duck curve indicators

# 6. Create train/validation/test splits
train_df, val_df, test_df = create_new_train_validation_test_splits(
    df,
    end_date=current_date - timedelta(days=days_delay),
    test_days=config.TEST_DAYS,
    validation_days=config.VALIDATION_DAYS
)

# 7. Save outputs
save_outputs(train_df, val_df, test_df, output_train, output_val, output_test)
# Saves to local directories AND uploads to S3

# 8. Return dataframes
return train_df, val_df, test_df
```

**Output Statistics**:
```json
{
    "train_rows": 24500,
    "val_rows": 1440,
    "test_rows": 720,
    "total_features": 95,
    "forecast_available_features": 42,
    "historical_features": 53,
    "start_date": "2023-01-14",
    "end_date": "2025-12-30",
    "total_days": 1082,
    "missing_data_pct": 0.5
}
```

#### Helper Function: `ensure_three_year_period()`

**Purpose**: Enforce exactly 3-year training window

**Implementation**:
```python
def ensure_three_year_period(df, current_date, days_delay):
    end_date = current_date - timedelta(days=days_delay)
    start_date = end_date - timedelta(days=3*365)  # 3 years

    df = df[
        (df['datetime'] >= start_date) &
        (df['datetime'] <= end_date)
    ]

    actual_days = (df['datetime'].max() - df['datetime'].min()).days
    expected_days = 3 * 365

    if actual_days < expected_days - 7:  # Allow 1 week tolerance
        logger.warning(f"Less than 3 years of data: {actual_days} days")

    return df
```

#### Helper Function: `create_new_train_validation_test_splits()`

**Purpose**: Split data into train/validation/test with temporal ordering

**Implementation**:
```python
def create_new_train_validation_test_splits(
    df: pd.DataFrame,
    end_date: datetime,
    test_days: int = 30,
    validation_days: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Sort by datetime (critical for time series)
    df = df.sort_values('datetime')

    # Calculate split dates
    test_start = end_date - timedelta(days=test_days)
    val_start = test_start - timedelta(days=validation_days)

    # Split data
    test_df = df[df['datetime'] >= test_start]
    val_df = df[(df['datetime'] >= val_start) & (df['datetime'] < test_start)]
    train_df = df[df['datetime'] < val_start]

    # Validate splits
    assert len(test_df) > 0, "Test set is empty"
    assert len(val_df) > 0, "Validation set is empty"
    assert len(train_df) > 0, "Train set is empty"

    # Log split info
    logger.info(f"Train: {len(train_df)} rows ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    logger.info(f"Val: {len(val_df)} rows ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    logger.info(f"Test: {len(test_df)} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

    return train_df, val_df, test_df
```

**Time Series Split Diagram**:
```
|←────────────── 3 years (1,095 days) ──────────────→|
|                                                      |
|←──────── Train (1,005 days) ────────→|←Val→|←Test→|
|                                       |60d  |30d   |
|                                       |     |      |
Start                                 Val    Test   End
2023-01-14                         2025-11-01  2025-12-01  2025-12-30
```

---

### Module: data_processing.py - Data Retrieval

**Purpose**: Query data from Redshift/Athena and perform basic transformations.

#### Main Function: `query_data()`

**Signature**:
```python
def query_data(
    config: Dict,
    current_date: datetime,
    load_profile: str,
    rate_group_filter: Optional[str] = None,
    query_limit: Optional[int] = None
) -> pd.DataFrame
```

**Execution**:
```python
if config['database_type'] == 'redshift':
    df = query_data_redshift(config, current_date, load_profile, rate_group_filter, query_limit)
elif config['database_type'] == 'athena':
    df = query_data_athena(config, current_date, load_profile, rate_group_filter, query_limit)
else:
    raise ValueError(f"Unsupported database type: {config['database_type']}")

return df
```

#### Function: `query_data_redshift()`

**Purpose**: Query Redshift using Data API with optimized date ranges

**Key Steps**:

1. **Calculate Optimal Date Ranges**:
```python
start_date, end_date, final_cutoff_date = calculate_data_date_ranges(
    current_date, config, data_delay_days=14
)

# Example output:
# start_date: 2025-11-04 (70 days back for lag features)
# end_date: 2025-12-30 (current_date - 14 days)
# final_cutoff_date: 2025-12-27 (latest Final submission available)
```

2. **Query Final Submissions** (verified data):
```sql
SELECT
    tradedatelocal AS tradedate,
    tradehourstartlocal AS tradetime,
    loadprofile,
    rategroup,
    baseload,
    lossadjustedload,
    metercount,
    loadbl,
    loadlal,
    loadmetercount,
    genbl,
    genlal,
    genmetercount,
    submission,
    createddate AS created
FROM {schema}.{table}
WHERE loadprofile = '{load_profile}'
  AND submission = 'Final'
  AND tradedatelocal >= '{start_date}'
  AND tradedatelocal <= '{final_cutoff_date}'
  {AND rate_group_filter if provided}
ORDER BY tradedatelocal, tradehourstartlocal
```

3. **Query Initial Submissions** (recent preliminary data):
```sql
SELECT ... [same columns] ...
FROM {schema}.{table}
WHERE loadprofile = '{load_profile}'
  AND submission = 'Initial'
  AND tradedatelocal >= '{final_cutoff_date + 1 day}'
  AND tradedatelocal <= '{end_date}'
  {AND rate_group_filter if provided}
ORDER BY tradedatelocal, tradehourstartlocal
```

4. **Combine Results**:
```python
df_combined = pd.concat([final_df, initial_df], ignore_index=True)
df_combined = df_combined.sort_values(['tradedate', 'tradetime'])
```

#### Function: `execute_redshift_query_via_data_api()`

**Purpose**: Execute query using Redshift Data API (no direct connection needed)

**Implementation**:
```python
def execute_redshift_query_via_data_api(
    query: str,
    database: str,
    cluster_identifier: str,
    db_user: str,
    region: str
) -> pd.DataFrame:

    # 1. Submit query
    redshift_data = boto3.client('redshift-data', region_name=region)
    response = redshift_data.execute_statement(
        ClusterIdentifier=cluster_identifier,
        Database=database,
        DbUser=db_user,
        Sql=query
    )
    query_id = response['Id']

    # 2. Poll for completion
    max_wait = 1800  # 30 minutes
    poll_interval = 5
    elapsed = 0

    while elapsed < max_wait:
        status_response = redshift_data.describe_statement(Id=query_id)
        status = status_response['Status']

        if status == 'FINISHED':
            break
        elif status in ['FAILED', 'ABORTED']:
            error = status_response.get('Error', 'Unknown error')
            raise Exception(f"Query {status}: {error}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    if elapsed >= max_wait:
        raise TimeoutError(f"Query timeout after {max_wait} seconds")

    # 3. Get paginated results
    df = get_paginated_results(redshift_data, query_id)

    return df
```

#### Function: `get_paginated_results()`

**Purpose**: Retrieve all query results handling pagination (critical for large datasets)

**Implementation**:
```python
def get_paginated_results(redshift_data_client, query_id):
    all_records = []
    next_token = None
    page_num = 0

    while True:
        page_num += 1

        if next_token:
            result = redshift_data_client.get_statement_result(
                Id=query_id,
                NextToken=next_token
            )
        else:
            result = redshift_data_client.get_statement_result(Id=query_id)

        # Extract column metadata (first page only)
        if page_num == 1:
            columns = [col['name'] for col in result['ColumnMetadata']]

        # Extract records
        records = result.get('Records', [])
        all_records.extend(records)

        logger.info(f"Page {page_num}: Retrieved {len(records)} records (total: {len(all_records)})")

        # Check for more pages
        next_token = result.get('NextToken')
        if not next_token:
            break

    # Convert to DataFrame
    df = records_to_dataframe(all_records, columns)
    return df
```

**Why Pagination is Critical**:
- Redshift Data API returns max 100,000 records per page
- 3-year hourly data ≈ 26,000 rows (fits in 1 page)
- But if multiple rate groups or profiles queried together, can exceed 100K
- Missing pagination caused data loss bug in early versions

#### Function: `convert_column_types()`

**Purpose**: Convert DataFrame columns to appropriate types

**Implementation**:
```python
def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric columns
    numeric_cols = [
        'baseload', 'lossadjustedload',
        'loadbl', 'loadlal', 'genbl', 'genlal'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Integer columns
    int_cols = ['metercount', 'loadmetercount', 'genmetercount']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # DateTime construction
    # Redshift: tradedate (date) + tradehourstartlocal (hour number 0-23)
    if 'tradedatelocal' in df.columns and 'tradehourstartlocal' in df.columns:
        df['datetime'] = (
            pd.to_datetime(df['tradedatelocal']) +
            pd.to_timedelta(df['tradehourstartlocal'], unit='h')
        )

    # Date features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['date'] = df['datetime'].dt.date

    return df
```

---

### Module: solar_features.py - Solar Feature Engineering

**Purpose**: Calculate solar position, radiation, and duck curve metrics.

#### Function: `add_solar_features()`

**Signature**:
```python
def add_solar_features(
    df: pd.DataFrame,
    latitude: float = 32.7157,
    longitude: float = -117.1611
) -> pd.DataFrame
```

**Features Created**:

**1. Time-Based Solar Windows**:
```python
# Morning rise (6 AM - 9 AM)
df['is_morning_rise'] = ((df['hour'] >= 6) & (df['hour'] < 9)).astype(int)

# Solar window (9 AM - 4 PM) - peak generation period
df['is_solar_window'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)

# Solar peak (11 AM - 2 PM) - maximum efficiency
df['is_solar_peak'] = ((df['hour'] >= 11) & (df['hour'] < 14)).astype(int)

# Evening ramp (4 PM - 8 PM) - generation decreases
df['is_evening_ramp'] = ((df['hour'] >= 16) & (df['hour'] < 20)).astype(int)
```

**2. Radiation Features** (if weather data available):
```python
if 'direct_radiation' in df.columns and 'diffuse_radiation' in df.columns:
    # Total radiation
    df['total_radiation'] = df['direct_radiation'] + df['diffuse_radiation']

    # Solar potential (normalized to daily max)
    df['solar_potential'] = df.groupby('date')['total_radiation'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )

    # Direct/Diffuse ratio (indicates cloud cover)
    df['direct_diffuse_ratio'] = np.where(
        df['diffuse_radiation'] > 0,
        df['direct_radiation'] / df['diffuse_radiation'],
        0
    )

    # Cap extreme values at 99th percentile
    cap = df['direct_diffuse_ratio'].quantile(0.99)
    df['direct_diffuse_ratio'] = df['direct_diffuse_ratio'].clip(upper=cap)
```

**3. Solar Position** (if weather not available):
```python
# Calculate for each timestamp
df['solar_position'] = df.apply(
    lambda row: calculate_solar_position(
        row['datetime'], latitude, longitude
    ),
    axis=1
)

# Extract zenith and azimuth
df['solar_zenith'] = df['solar_position'].apply(lambda x: x['zenith'])
df['solar_azimuth'] = df['solar_position'].apply(lambda x: x['azimuth'])
df['solar_elevation'] = df['solar_position'].apply(lambda x: x['elevation'])

# Calculate clear-sky radiation from position
df['clearsky_radiation'] = df.apply(
    lambda row: calculate_clear_sky_radiation(
        row['datetime'], latitude, longitude
    ),
    axis=1
)
```

#### Function: `calculate_solar_position()`

**Purpose**: Calculate sun position using astronomical formulas

**Implementation** (Spencer Equation):
```python
def calculate_solar_position(dt: datetime, lat: float, lon: float) -> dict:
    # Convert to UTC
    dt_utc = dt.astimezone(pytz.UTC)

    # Day of year (1-365)
    doy = dt_utc.timetuple().tm_yday

    # Fractional year in radians
    gamma = 2 * np.pi * (doy - 1) / 365

    # Equation of time (minutes)
    eot = 229.18 * (
        0.000075 +
        0.001868 * np.cos(gamma) -
        0.032077 * np.sin(gamma) -
        0.014615 * np.cos(2*gamma) -
        0.040849 * np.sin(2*gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918 -
        0.399912 * np.cos(gamma) +
        0.070257 * np.sin(gamma) -
        0.006758 * np.cos(2*gamma) +
        0.000907 * np.sin(2*gamma) -
        0.002697 * np.cos(3*gamma) +
        0.00148 * np.sin(3*gamma)
    )

    # Time offset (minutes)
    time_offset = eot + 4 * lon

    # True solar time (minutes)
    tst = dt_utc.hour * 60 + dt_utc.minute + time_offset

    # Hour angle (radians)
    hour_angle = np.radians((tst / 4) - 180)

    # Zenith angle (radians)
    lat_rad = np.radians(lat)
    zenith = np.arccos(
        np.sin(lat_rad) * np.sin(decl) +
        np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    )

    # Elevation angle
    elevation = np.pi/2 - zenith

    # Azimuth angle (from north, clockwise)
    azimuth = np.arccos(
        (np.sin(decl) * np.cos(lat_rad) -
         np.cos(decl) * np.sin(lat_rad) * np.cos(hour_angle)) /
        np.sin(zenith)
    )

    # Adjust azimuth for afternoon (hour_angle > 0)
    if hour_angle > 0:
        azimuth = 2 * np.pi - azimuth

    return {
        'zenith': np.degrees(zenith),
        'azimuth': np.degrees(azimuth),
        'elevation': np.degrees(elevation),
        'daylight': elevation > 0
    }
```

#### Function: `add_solar_ratios()`

**Purpose**: Calculate generation/load ratios and duck curve metrics

**Implementation**:
```python
def add_solar_ratios(df: pd.DataFrame, forecast_delay_days: int = 14) -> pd.DataFrame:
    # Require lag features to exist (historical data)
    lag_hours = forecast_delay_days * 24

    required_cols = [
        f'genlal_lag_{lag_hours}h',
        f'loadlal_lag_{lag_hours}h'
    ]

    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing lag features, skipping solar ratios")
        return df

    # 1. Generation-to-Load Ratio
    df['gen_load_ratio'] = np.where(
        df[f'loadlal_lag_{lag_hours}h'] > 0,
        np.abs(df[f'genlal_lag_{lag_hours}h']) / df[f'loadlal_lag_{lag_hours}h'],
        0
    )

    # Cap at 99th percentile
    cap = df['gen_load_ratio'].quantile(0.99)
    df['gen_load_ratio'] = df['gen_load_ratio'].clip(upper=cap)

    # 2. High Generation Period Indicator
    threshold = df['gen_load_ratio'].quantile(0.75)
    df['is_high_gen_period'] = (df['gen_load_ratio'] > threshold).astype(int)

    # 3. Duck Curve Ratio (daily metric)
    def calculate_duck_curve_ratio(group):
        # Morning peak (6-9 AM)
        morning_peak = group[
            (group['hour'] >= 6) & (group['hour'] < 9)
        ]['lossadjustedload'].max()

        # Solar dip (11 AM - 3 PM) - minimum load during solar
        solar_dip = group[
            (group['hour'] >= 11) & (group['hour'] < 15)
        ]['lossadjustedload'].min()

        # Evening peak (5-9 PM)
        evening_peak = group[
            (group['hour'] >= 17) & (group['hour'] < 21)
        ]['lossadjustedload'].max()

        # Duck curve ratio: evening peak / solar dip
        # Higher ratio = more severe duck curve
        ratio = evening_peak / solar_dip if solar_dip > 0 else 0

        return pd.Series({
            'morning_peak': morning_peak,
            'solar_dip': solar_dip,
            'evening_peak': evening_peak,
            'duck_curve_ratio': ratio
        })

    # Calculate daily
    daily_metrics = df.groupby('date').apply(calculate_duck_curve_ratio)

    # Merge back to hourly data
    df = df.merge(daily_metrics, on='date', how='left')

    return df
```

**Duck Curve Visualization**:
```
Load (MW)
    ↑
    │     Morning Peak
    │        ↗↘
    │       ↗  ↘            Evening Peak
    │      ↗    ↘             ↗↗↗
    │     ↗      ↘           ↗
    │    ↗        ↘  Solar  ↗
    │   ↗          ↘  Dip  ↗
    │  ↗            ↘___↙ ← (Minimum load during solar)
    │ ↗
    └──────────────────────────────→ Hour
      6am    12pm    6pm    12am

Duck Curve Ratio = Evening Peak / Solar Dip
Higher ratio = More severe ramp rate = Grid stress
```

---

### Module: weather_features.py - Weather Integration

**Purpose**: Fetch and integrate weather data from Open-Meteo API.

#### Function: `fetch_historical_weather()`

**Signature**:
```python
def fetch_historical_weather(
    start_date: Union[str, date],
    end_date: Union[str, date],
    latitude: float = 32.7157,
    longitude: float = -117.1611,
    timezone: str = 'America/Los_Angeles',
    use_cache: bool = True
) -> pd.DataFrame
```

**API Call**:
```python
# Open-Meteo Archive API (free, no key)
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": [
        "temperature_2m",
        "apparent_temperature",
        "cloudcover",
        "direct_radiation",
        "diffuse_radiation",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "windspeed_10m",
        "winddirection_10m",
        "windgusts_10m",
        "relativehumidity_2m",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "is_day"
    ],
    "timezone": timezone
}

client = get_openmeteo_client(cache_dir='/tmp/weather_cache')
responses = client.weather_api(url, params=params)
```

**Response Processing**:
```python
# Extract hourly data
response = responses[0]
hourly = response.Hourly()

# Create time series
hourly_data = {
    "time": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
}

# Convert to local timezone
hourly_data["time"] = hourly_data["time"].dt.tz_convert(timezone)
hourly_data["time"] = hourly_data["time"].dt.tz_localize(None)

# Extract variables
for i, variable in enumerate(params["hourly"]):
    hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

# Create DataFrame
weather_df = pd.DataFrame(data=hourly_data)

return weather_df
```

#### Function: `add_weather_features()`

**Purpose**: Merge weather data and create derived features

**Implementation**:
```python
def add_weather_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:

    # 1. Merge on hourly timestamp
    df['datetime_hour'] = df['datetime'].dt.floor('h')
    weather_df['datetime_hour'] = weather_df['time'].dt.floor('h')

    df = pd.merge(df, weather_df, on='datetime_hour', how='left')

    # 2. Temperature difference (comfort indicator)
    df['temperature_difference'] = (
        df['apparent_temperature'] - df['temperature_2m']
    )

    # 3. Heating degree hours (below 65°F = 18.3°C)
    df['heating_degree_hours'] = np.maximum(18.3 - df['temperature_2m'], 0)

    # 4. Cooling degree hours (above 75°F = 23.9°C)
    df['cooling_degree_hours'] = np.maximum(df['temperature_2m'] - 23.9, 0)

    # 5. Direct radiation ratio
    df['direct_radiation_ratio'] = np.where(
        df['shortwave_radiation'] > 0,
        df['direct_radiation'] / df['shortwave_radiation'],
        0
    )

    # 6. Period-specific temperatures
    df['morning_temp'] = np.where(
        (df['hour'] >= 6) & (df['hour'] < 9),
        df['temperature_2m'],
        0
    )

    df['evening_temp'] = np.where(
        (df['hour'] >= 17) & (df['hour'] < 22),
        df['temperature_2m'],
        0
    )

    # 7. Solar window cloud cover
    df['solar_window_cloudcover'] = (
        df['cloudcover'] * df['is_solar_window']
    )

    # 8. Daily temperature features
    daily_temp = df.groupby('date')['temperature_2m'].agg(['min', 'max'])
    daily_temp['daily_temp_range'] = daily_temp['max'] - daily_temp['min']

    df = df.merge(
        daily_temp.rename(columns={
            'min': 'daily_min_temp',
            'max': 'daily_max_temp'
        }),
        on='date',
        how='left'
    )

    # 9. Handle missing values
    weather_cols = [
        'temperature_2m', 'apparent_temperature', 'cloudcover',
        'direct_radiation', 'diffuse_radiation', 'shortwave_radiation',
        'windspeed_10m', 'relativehumidity_2m'
    ]

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    return df
```

#### Function: `create_weather_solar_interactions()`

**Purpose**: Create interaction features between weather and solar

**Implementation**:
```python
def create_weather_solar_interactions(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Wind heating effect (cold + windy = increased heating)
    if 'windspeed_10m' in df.columns and 'heating_degree_hours' in df.columns:
        df['wind_heating_effect'] = (
            df['windspeed_10m'] * df['heating_degree_hours'] / 10
        )

    # 2. Wind cooling effect (hot + windy = reduced cooling need)
    if 'windspeed_10m' in df.columns and 'cooling_degree_hours' in df.columns:
        df['wind_cooling_effect'] = (
            df['windspeed_10m'] * df['cooling_degree_hours'] / 10
        )

    # 3. Humidity cooling impact (high humidity reduces AC efficiency)
    if 'relativehumidity_2m' in df.columns and 'cooling_degree_hours' in df.columns:
        df['humidity_cooling_impact'] = np.where(
            df['cooling_degree_hours'] > 0,
            df['relativehumidity_2m'] * df['cooling_degree_hours'] / 100,
            0
        )

    # 4. Evening humidity discomfort (hot + humid evenings)
    if 'relativehumidity_2m' in df.columns and 'evening_temp' in df.columns:
        df['evening_humidity_discomfort'] = np.where(
            df['evening_temp'] > 23.9,  # Above 75°F
            df['relativehumidity_2m'] * (df['evening_temp'] - 23.9) / 100,
            0
        )

    return df
```

---

## Training Pipeline

Location: `pipeline/training/`

### Module: model.py - XGBoost Training

**Purpose**: Core model training, cross-validation, and prediction.

#### Function: `initialize_model()`

**Purpose**: Create XGBoost model with default parameters

**Implementation**:
```python
def initialize_model(params: Optional[dict] = None) -> xgb.XGBRegressor:
    default_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,      # L1 regularization
        'reg_lambda': 1.0,     # L2 regularization
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }

    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    return model
```

#### Function: `train_model()`

**Purpose**: Train XGBoost model with validation monitoring

**Signature**:
```python
def train_model(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'lossadjustedload',
    test_size: float = 0.2,
    params: Optional[dict] = None,
    custom_periods: Optional[dict] = None
) -> Tuple[xgb.XGBRegressor, dict]
```

**Execution**:
```python
# 1. Validate inputs
assert target in df.columns, f"Target {target} not in DataFrame"
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    logger.warning(f"Missing features: {missing_features}")
    features = [f for f in features if f in df.columns]

# 2. Prepare data
df = df.dropna(subset=features + [target])
df = df.sort_values('datetime')  # Maintain temporal order

X = df[features]
y = df[target]

# 3. Train/validation split (temporal)
split_idx = int(len(df) * (1 - test_size))
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

# 4. Initialize and train model
model = initialize_model(params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# 5. Predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

# 6. Calculate metrics
metrics = {}

# Standard metrics
metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_pred_val))
metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
metrics['val_mae'] = mean_absolute_error(y_val, y_pred_val)
metrics['train_r2'] = r2_score(y_train, y_pred_train)
metrics['val_r2'] = r2_score(y_val, y_pred_val)

# Custom metrics
metrics['train_mape'] = calculate_mape(y_train, y_pred_train)
metrics['val_mape'] = calculate_mape(y_val, y_pred_val)
metrics['train_smape'] = smape(y_train, y_pred_train)
metrics['val_smape'] = smape(y_val, y_pred_val)
metrics['train_wape'] = wape(y_train, y_pred_train)
metrics['val_wape'] = wape(y_val, y_pred_val)
metrics['train_mase'] = mase(y_train, y_pred_train)
metrics['val_mase'] = mase(y_val, y_pred_val)

# 7. Period-specific metrics (if custom_periods provided)
if custom_periods:
    val_df_with_pred = df.iloc[split_idx:].copy()
    val_df_with_pred['prediction'] = y_pred_val

    for period_name, (start_hour, end_hour) in custom_periods.items():
        period_mask = (
            (val_df_with_pred['hour'] >= start_hour) &
            (val_df_with_pred['hour'] < end_hour)
        )

        if period_mask.sum() > 0:
            y_period = val_df_with_pred.loc[period_mask, target]
            y_pred_period = val_df_with_pred.loc[period_mask, 'prediction']

            metrics[f'val_{period_name}_rmse'] = np.sqrt(
                mean_squared_error(y_period, y_pred_period)
            )
            metrics[f'val_{period_name}_mape'] = calculate_mape(
                y_period, y_pred_period
            )

return model, metrics
```

#### Custom Metrics

**1. SMAPE (Symmetric Mean Absolute Percentage Error)**:
```python
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Handles zeros better than MAPE.
    Range: 0-200%
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Avoid division by zero
    mask = denominator > 0
    smape_values = np.zeros_like(numerator)
    smape_values[mask] = numerator[mask] / denominator[mask]

    return 100 * np.mean(smape_values)
```

**2. WAPE (Weighted Absolute Percentage Error)**:
```python
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weights errors by actual load magnitude.
    Better for aggregated forecasting.
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
```

**3. MASE (Mean Absolute Scaled Error)**:
```python
def mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 24) -> float:
    """
    Compares to seasonal naive forecast.
    MASE < 1: Better than naive
    MASE > 1: Worse than naive
    """
    # Model error
    mae = mean_absolute_error(y_true, y_pred)

    # Naive forecast error (yesterday same hour)
    naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    mae_naive = np.mean(naive_errors)

    return mae / mae_naive if mae_naive > 0 else np.inf
```

**4. Transition Weighted Error**:
```python
def transition_weighted_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 20000
) -> float:
    """
    Emphasizes accuracy during load transitions (near-zero periods).
    Critical for duck curve modeling.
    """
    # Overall RMSE
    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))

    # Transition points (low load)
    transition_mask = np.abs(y_true) < threshold

    if transition_mask.sum() > 0:
        rmse_transition = np.sqrt(
            mean_squared_error(y_true[transition_mask], y_pred[transition_mask])
        )
    else:
        rmse_transition = rmse_all

    # Weighted combination
    weighted_error = 0.6 * rmse_all + 0.4 * rmse_transition

    return weighted_error
```

#### Function: `create_time_series_splits()`

**Purpose**: Create expanding window cross-validation splits

**Implementation**:
```python
def create_time_series_splits(
    df: pd.DataFrame,
    initial_train_months: int = 6,
    validation_months: int = 3,
    step_months: int = 3,
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates time series CV splits with expanding training window.

    Example with 24 months of data:
    Split 1: [0-6mo] train, [6-9mo] test
    Split 2: [0-9mo] train, [9-12mo] test
    Split 3: [0-12mo] train, [12-15mo] test
    Split 4: [0-15mo] train, [15-18mo] test
    Split 5: [0-18mo] train, [18-21mo] test
    """
    df = df.sort_values('datetime')

    total_months = (df['datetime'].max() - df['datetime'].min()).days / 30

    if total_months < initial_train_months + validation_months:
        logger.warning(f"Insufficient data for CV: {total_months} months")
        return []

    splits = []

    for i in range(n_splits):
        train_end_months = initial_train_months + (i * step_months)
        test_start_months = train_end_months
        test_end_months = test_start_months + validation_months

        if test_end_months > total_months:
            break

        # Calculate date thresholds
        min_date = df['datetime'].min()
        train_end = min_date + pd.DateOffset(months=train_end_months)
        test_start = train_end
        test_end = min_date + pd.DateOffset(months=test_end_months)

        # Create indices
        train_idx = df[df['datetime'] < train_end].index.values
        test_idx = df[
            (df['datetime'] >= test_start) &
            (df['datetime'] < test_end)
        ].index.values

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits
```

**CV Split Visualization**:
```
Total data: 24 months
│←─────────────────────────────────────────────→│

Split 1:
│←─ 6mo train ─→│←3mo test→│
                 ▲

Split 2:
│←──── 9mo train ────→│←3mo test→│
                       ▲

Split 3:
│←────── 12mo train ──────→│←3mo test→│
                             ▲

Split 4:
│←──────── 15mo train ────────→│←3mo test→│
                                 ▲

Split 5:
│←────────── 18mo train ──────────→│←3mo test→│
                                     ▲
```

---

### Module: hyperparameter_optimization.py - Bayesian Optimization

**Purpose**: Optimize XGBoost hyperparameters using Optuna.

#### Class: `OptunaHPO`

**Initialization**:
```python
class OptunaHPO:
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        date_column: str = 'datetime',
        n_splits: int = 5,
        n_trials: int = 50,
        custom_periods: Optional[dict] = None,
        metric_weights: Optional[dict] = None
    ):
        self.df = df
        self.features = features
        self.target = target
        self.date_column = date_column
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.custom_periods = custom_periods or {}
        self.metric_weights = metric_weights or {
            'rmse': 0.4,
            'mape': 0.3,
            'r2': 0.2,
            'smape': 0.1
        }

        # Create CV splits
        self.cv_splits = create_time_series_splits(df, n_splits=n_splits)
```

#### Method: `optimize()`

**Purpose**: Run Optuna optimization study

**Implementation**:
```python
def optimize(self) -> Tuple[dict, optuna.Study]:
    """
    Run Bayesian optimization to find best hyperparameters.

    Returns:
        best_params: Best hyperparameter configuration
        study: Optuna study object with trial history
    """
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run optimization
    study.optimize(
        self.objective,
        n_trials=self.n_trials,
        show_progress_bar=True
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params, study
```

#### Method: `objective()`

**Purpose**: Objective function for Optuna trials

**Implementation**:
```python
def objective(self, trial: optuna.Trial) -> float:
    """
    Objective function evaluated for each trial.

    Returns:
        score: Weighted combination of metrics (lower is better)
    """
    # Sample hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10, log=True),
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }

    # Evaluate on CV splits
    cv_scores = []

    for train_idx, test_idx in self.cv_splits:
        X_train = self.df.iloc[train_idx][self.features]
        y_train = self.df.iloc[train_idx][self.target]
        X_test = self.df.iloc[test_idx][self.features]
        y_test = self.df.iloc[test_idx][self.target]

        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        smape_score = smape(y_test, y_pred)

        # Weighted score
        score = self.compute_weighted_score({
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'smape': smape_score
        })

        cv_scores.append(score)

    # Return mean CV score
    return np.mean(cv_scores)
```

#### Method: `compute_weighted_score()`

**Purpose**: Combine multiple metrics with weights

**Implementation**:
```python
def compute_weighted_score(self, metrics: dict) -> float:
    """
    Compute weighted combination of metrics.
    Lower is better (R² converted to 1-R²).
    """
    weights = self.metric_weights

    # Convert R² (higher is better) to loss (lower is better)
    r2_loss = 1 - metrics.get('r2', 0)

    score = (
        weights.get('rmse', 0) * metrics.get('rmse', 0) +
        weights.get('mape', 0) * metrics.get('mape', 0) +
        weights.get('r2', 0) * r2_loss +
        weights.get('smape', 0) * metrics.get('smape', 0)
    )

    # Add transition score if available
    if 'transition_score' in metrics:
        score += 0.25 * metrics['transition_score']

    return score
```

**Hyperparameter Search Space**:
```
n_estimators: 50-500 (integer)
  - Number of boosting rounds
  - More trees = better fit but longer training

max_depth: 3-10 (integer)
  - Maximum tree depth
  - Deeper = more complex interactions

learning_rate: 0.01-0.2 (log scale)
  - Step size shrinkage
  - Lower = more conservative, needs more trees

subsample: 0.5-1.0 (float)
  - Row sampling ratio
  - < 1.0 helps prevent overfitting

colsample_bytree: 0.5-1.0 (float)
  - Feature sampling ratio per tree
  - < 1.0 adds randomness

reg_alpha: 1e-6 to 10 (log scale)
  - L1 regularization
  - Promotes feature sparsity

reg_lambda: 1e-6 to 10 (log scale)
  - L2 regularization
  - Smooth weights, reduces overfitting
```

---

### Module: feature_selection.py - Feature Selection

**Purpose**: Select most relevant features using multiple methods.

#### Function: `consensus_feature_selection()`

**Purpose**: Select features that appear across multiple methods

**Implementation**:
```python
def consensus_feature_selection(
    feature_sets: List[List[str]],
    threshold: float = 0.5,
    top_n: Optional[int] = None
) -> List[str]:
    """
    Select features based on consensus across methods.

    Args:
        feature_sets: List of feature lists from different methods
        threshold: Minimum fraction of methods (0.0-1.0)
        top_n: Maximum features to return

    Returns:
        Selected features sorted by occurrence count
    """
    from collections import Counter

    # Count feature occurrences
    all_features = [f for feature_set in feature_sets for f in feature_set]
    feature_counts = Counter(all_features)

    # Filter by threshold
    min_count = int(threshold * len(feature_sets))
    selected = [
        feature for feature, count in feature_counts.items()
        if count >= min_count
    ]

    # Sort by count (descending)
    selected = sorted(selected, key=lambda f: feature_counts[f], reverse=True)

    # Limit to top_n
    if top_n:
        selected = selected[:top_n]

    logger.info(f"Consensus selection: {len(selected)} features from {len(feature_counts)} total")

    return selected
```

#### Function: `select_features_by_importance()`

**Purpose**: Select features based on XGBoost feature importance

**Implementation**:
```python
def select_features_by_importance(
    model: xgb.XGBRegressor,
    features: List[str],
    top_n: Optional[int] = None,
    threshold: Optional[float] = None
) -> List[str]:
    """
    Select features based on model's feature_importances_.

    Args:
        model: Trained XGBoost model
        features: Feature names
        top_n: Select top N features
        threshold: Select features with importance > threshold

    Returns:
        Selected feature names
    """
    # Get importances
    importances = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Apply selection criteria
    if top_n:
        selected = importance_df.head(top_n)['feature'].tolist()
    elif threshold:
        selected = importance_df[
            importance_df['importance'] > threshold
        ]['feature'].tolist()
    else:
        selected = importance_df['feature'].tolist()

    logger.info(f"Importance-based selection: {len(selected)} features")

    return selected
```

#### Function: `select_features_by_correlation()`

**Purpose**: Select features based on target correlation and remove collinear features

**Implementation**:
```python
def select_features_by_correlation(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    target_corr_threshold: float = 0.05,
    feature_corr_threshold: float = 0.85
) -> List[str]:
    """
    Two-stage selection:
    1. Keep features correlated with target
    2. Remove highly correlated feature pairs

    Args:
        df: DataFrame with features and target
        features: Feature names
        target: Target column name
        target_corr_threshold: Min |correlation| with target
        feature_corr_threshold: Max |correlation| between features

    Returns:
        Selected features
    """
    # Stage 1: Target correlation
    target_corrs = df[features + [target]].corr()[target].abs()
    target_corrs = target_corrs[target_corrs.index != target]

    selected = target_corrs[
        target_corrs >= target_corr_threshold
    ].index.tolist()

    logger.info(f"After target correlation: {len(selected)} features")

    # Stage 2: Remove multicollinearity
    feature_corr_matrix = df[selected].corr().abs()

    # Find pairs with high correlation
    to_remove = set()
    for i in range(len(feature_corr_matrix.columns)):
        for j in range(i+1, len(feature_corr_matrix.columns)):
            if feature_corr_matrix.iloc[i, j] > feature_corr_threshold:
                # Remove feature with lower target correlation
                feat_i = feature_corr_matrix.columns[i]
                feat_j = feature_corr_matrix.columns[j]

                if target_corrs[feat_i] < target_corrs[feat_j]:
                    to_remove.add(feat_i)
                else:
                    to_remove.add(feat_j)

    selected = [f for f in selected if f not in to_remove]

    logger.info(f"After multicollinearity removal: {len(selected)} features")

    return selected
```

**Example Usage**:
```python
# Train initial model with all features
model, _ = train_model(train_df, all_features, target)

# Method 1: Importance-based
features_importance = select_features_by_importance(model, all_features, top_n=50)

# Method 2: Correlation-based
features_correlation = select_features_by_correlation(
    train_df, all_features, target,
    target_corr_threshold=0.05,
    feature_corr_threshold=0.85
)

# Method 3: Mutual information (if implemented)
# features_mi = select_features_by_mutual_info(train_df, all_features, target)

# Consensus
final_features = consensus_feature_selection(
    [features_importance, features_correlation],
    threshold=0.5,  # Must appear in 50%+ of methods
    top_n=40
)

# Retrain with selected features
final_model, final_metrics = train_model(train_df, final_features, target)
```

---

### Module: evaluation.py - Model Evaluation

**Purpose**: Calculate comprehensive metrics and segment-specific evaluations.

#### Function: `evaluate_predictions()`

**Purpose**: Calculate all standard metrics

**Implementation**:
```python
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ''
) -> dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., 'test_')

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # RMSE
    metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)

    # R²
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)

    # MAPE
    metrics[f'{prefix}mape'] = calculate_mape(y_true, y_pred)

    # sMAPE
    metrics[f'{prefix}smape'] = smape(y_true, y_pred)

    # WAPE
    metrics[f'{prefix}wape'] = wape(y_true, y_pred)

    # MASE
    metrics[f'{prefix}mase'] = mase(y_true, y_pred)

    # Normalized RMSE
    value_range = y_true.max() - y_true.min()
    metrics[f'{prefix}nrmse'] = metrics[f'{prefix}rmse'] / value_range

    # CV(RMSE) - Coefficient of Variation
    metrics[f'{prefix}cv_rmse'] = (metrics[f'{prefix}rmse'] / y_true.mean()) * 100

    # Transition weighted error
    metrics[f'{prefix}transition_weighted'] = transition_weighted_error(y_true, y_pred)

    return metrics
```

#### Function: `calculate_time_weighted_metrics()`

**Purpose**: Calculate period-specific metrics with weights

**Implementation**:
```python
def calculate_time_weighted_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    customer_segment: str
) -> dict:
    """
    Calculate metrics for specific time periods with weights.

    Segment-specific periods:
    - RES_SOLAR: Duck curve (14-18h, 35%), evening peak (17-21h, 25%)
    - RES_NONSOLAR: Evening super peak (17-21h, 60%)
    - MEDCI/SMLCOM: Business hours (8-18h, 70%)
    """
    metrics = {}

    # Define periods by segment
    if customer_segment == 'RES_SOLAR':
        periods = {
            'duck_curve_critical': (14, 18, 0.35),
            'evening_peak': (17, 21, 0.25),
            'midday_low': (10, 14, 0.15),
            'overnight': (22, 6, 0.10),
            'morning': (6, 10, 0.15)
        }
    elif customer_segment == 'RES_NONSOLAR':
        periods = {
            'evening_super_peak': (17, 21, 0.60),
            'afternoon_build': (14, 17, 0.20),
            'morning': (6, 10, 0.10),
            'overnight': (22, 6, 0.10)
        }
    elif customer_segment in ['MEDCI_SOLAR', 'MEDCI_NONSOLAR']:
        periods = {
            'business_hours': (8, 18, 0.70),
            'off_hours_evening': (18, 22, 0.15),
            'off_hours_night': (22, 8, 0.10),
            'weekend': (0, 24, 0.05)  # Special handling
        }
    elif customer_segment in ['SMLCOM_SOLAR', 'SMLCOM_NONSOLAR']:
        periods = {
            'business_hours': (8, 18, 0.60),
            'extended_hours': (6, 20, 0.30),
            'overnight': (20, 6, 0.10)
        }
    else:
        # Default periods
        periods = {
            'all_day': (0, 24, 1.0)
        }

    # Calculate metrics for each period
    for period_name, (start_hour, end_hour, weight) in periods.items():
        if period_name == 'weekend':
            mask = df['dayofweek'].isin([5, 6])  # Saturday, Sunday
        elif end_hour < start_hour:  # Overnight period
            mask = (df['hour'] >= start_hour) | (df['hour'] < end_hour)
        else:
            mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)

        if mask.sum() == 0:
            continue

        y_true_period = df.loc[mask, y_true_col]
        y_pred_period = df.loc[mask, y_pred_col]

        period_metrics = evaluate_predictions(
            y_true_period.values,
            y_pred_period.values,
            prefix=f'{period_name}_'
        )

        # Add weight
        period_metrics[f'{period_name}_weight'] = weight
        period_metrics[f'{period_name}_hours'] = mask.sum() / 24  # Days

        metrics.update(period_metrics)

    # Calculate weighted overall score
    weighted_rmse = sum(
        metrics.get(f'{period}_rmse', 0) * weight
        for period, (_, _, weight) in periods.items()
        if f'{period}_rmse' in metrics
    )
    metrics['weighted_overall_rmse'] = weighted_rmse

    return metrics
```

---

### Module: inference.py - SageMaker Interface

**Purpose**: SageMaker-compatible inference functions for endpoint deployment.

#### Function: `model_fn()`

**Purpose**: Load model from SageMaker artifact directory

**Implementation**:
```python
def model_fn(model_dir: str) -> dict:
    """
    Load model and features from SageMaker model directory.

    Args:
        model_dir: Path to model artifacts (e.g., /opt/ml/model)

    Returns:
        Dict with 'model' and 'features' keys
    """
    import pickle
    import xgboost as xgb

    # Load model
    model_path = os.path.join(model_dir, 'xgboost-model')
    model = pickle.load(open(model_path, 'rb'))

    # Load features
    features_path = os.path.join(model_dir, 'features.pkl')
    if os.path.exists(features_path):
        features = pickle.load(open(features_path, 'rb'))
    else:
        features = None

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Features: {len(features) if features else 'None'}")

    return {'model': model, 'features': features}
```

#### Function: `input_fn()`

**Purpose**: Parse input request

**Implementation**:
```python
def input_fn(request_body: str, content_type: str = 'application/json') -> pd.DataFrame:
    """
    Parse JSON request body into DataFrame.

    Expected format:
    {
        "instances": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...}
        ]
    }

    Or single instance:
    {"feature1": value1, "feature2": value2, ...}
    """
    import json

    if content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)

    # Handle both formats
    if isinstance(data, dict):
        if 'instances' in data:
            instances = data['instances']
        else:
            instances = [data]
    elif isinstance(data, list):
        instances = data
    else:
        raise ValueError("Invalid input format")

    # Convert to DataFrame
    df = pd.DataFrame(instances)

    # Parse datetime if present
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_year'] = df['datetime'].dt.dayofyear

    return df
```

#### Function: `predict_fn()`

**Purpose**: Generate predictions

**Implementation**:
```python
def predict_fn(input_data: pd.DataFrame, model_dict: dict) -> np.ndarray:
    """
    Generate predictions using loaded model.

    Args:
        input_data: Parsed input DataFrame
        model_dict: Dict with 'model' and 'features' from model_fn

    Returns:
        Numpy array of predictions
    """
    model = model_dict['model']
    features = model_dict.get('features')

    # Filter to available features
    if features:
        available_features = [f for f in features if f in input_data.columns]
        missing_features = [f for f in features if f not in input_data.columns]

        if missing_features:
            logger.warning(f"Missing features: {missing_features}")

        X = input_data[available_features]
    else:
        X = input_data

    # Generate predictions
    predictions = model.predict(X)

    return predictions
```

#### Function: `output_fn()`

**Purpose**: Format response

**Implementation**:
```python
def output_fn(predictions: np.ndarray, accept: str = 'application/json') -> str:
    """
    Format predictions as JSON response.

    Args:
        predictions: Numpy array of predictions
        accept: Accept header (default application/json)

    Returns:
        JSON string
    """
    import json

    if accept != 'application/json':
        raise ValueError(f"Unsupported accept type: {accept}")

    response = {
        'predictions': predictions.tolist()
    }

    return json.dumps(response)
```

**Complete Inference Flow**:
```
1. SageMaker receives POST request with JSON body

2. input_fn() parses JSON to DataFrame
   Input: {"instances": [{"hour": 10, "temp": 20, ...}]}
   Output: DataFrame with 1 row

3. model_fn() loads model (cached after first call)
   Output: {'model': XGBRegressor, 'features': [...]}

4. predict_fn() generates predictions
   Output: np.array([12345.67])

5. output_fn() formats response
   Output: {"predictions": [12345.67]}

6. SageMaker returns JSON response to client
```

---

## Orchestration

Location: `pipeline/orchestration/`

### Module: pipeline.py - SageMaker Pipeline

**Purpose**: Create and manage SageMaker ML pipelines.

#### Function: `create_preprocessing_pipeline()`

**Purpose**: Create preprocessing-only pipeline

**Signature**:
```python
def create_preprocessing_pipeline(
    role: str,
    pipeline_name: str,
    bucket: str,
    prefix: str,
    region: str = 'us-west-2'
) -> Pipeline
```

**Implementation**:
```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# 1. Define SKLearn processor
sklearn_processor = SKLearnProcessor(
    framework_version='1.0-1',
    instance_type='ml.m5.large',
    instance_count=1,
    role=role,
    base_job_name=f'{pipeline_name}-preprocessing'
)

# 2. Define processing step
preprocessing_step = ProcessingStep(
    name='Preprocessing',
    processor=sklearn_processor,
    code=f's3://{bucket}/{prefix}/scripts/processing_wrapper.py',
    inputs=[
        ProcessingInput(
            source=f's3://{bucket}/{prefix}/scripts/processing_config.json',
            destination='/opt/ml/processing/input/config'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='training',
            source='/opt/ml/processing/output/train',
            destination=f's3://{bucket}/{prefix}/processed/training'
        ),
        ProcessingOutput(
            output_name='validation',
            source='/opt/ml/processing/output/val',
            destination=f's3://{bucket}/{prefix}/processed/validation'
        ),
        ProcessingOutput(
            output_name='test',
            source='/opt/ml/processing/output/test',
            destination=f's3://{bucket}/{prefix}/processed/test'
        )
    ]
)

# 3. Create pipeline
pipeline = Pipeline(
    name=pipeline_name,
    steps=[preprocessing_step],
    sagemaker_session=sagemaker.Session()
)

return pipeline
```

#### Function: `create_training_pipeline()`

**Purpose**: Create complete preprocessing + training pipeline

**Implementation**:
```python
def create_training_pipeline(
    role: str,
    pipeline_name: str,
    bucket: str,
    prefix: str,
    xgboost_image_uri: str,
    region: str = 'us-west-2'
) -> Pipeline:

    # 1. Preprocessing step (as above)
    preprocessing_step = ...

    # 2. Training step
    from sagemaker.estimator import Estimator
    from sagemaker.workflow.steps import TrainingStep
    from sagemaker.inputs import TrainingInput

    xgboost_estimator = Estimator(
        image_uri=xgboost_image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=f's3://{bucket}/{prefix}/models',
        base_job_name=f'{pipeline_name}-training',
        hyperparameters={
            'objective': 'reg:squarederror',
            'num_round': '200',
            'max_depth': '6',
            'eta': '0.05',
            'subsample': '0.8',
            'colsample_bytree': '0.8'
        }
    )

    training_step = TrainingStep(
        name='Training',
        estimator=xgboost_estimator,
        inputs={
            'train': TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs['training'].S3Output.S3Uri,
                content_type='text/csv'
            ),
            'validation': TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )

    # 3. Create pipeline with dependencies
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[preprocessing_step, training_step],
        sagemaker_session=sagemaker.Session()
    )

    return pipeline
```

**Pipeline Execution**:
```python
# Create pipeline
pipeline = create_training_pipeline(
    role=SAGEMAKER_ROLE_ARN,
    pipeline_name='energy-forecasting-complete',
    bucket=S3_BUCKET,
    prefix='RES-SOLAR',
    xgboost_image_uri='<xgboost-image>'
)

# Upsert (create or update)
pipeline.upsert(role_arn=SAGEMAKER_ROLE_ARN)

# Start execution
execution = pipeline.start(
    parameters={
        'DaysDelay': 14,
        'UseWeather': 'true',
        'UseSolar': 'true'
    }
)

# Monitor
execution.wait(delay=60, max_attempts=120)

# Get status
status = execution.describe()['PipelineExecutionStatus']
print(f"Pipeline status: {status}")
```

---

## Data Flow Summary

### Complete Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA RETRIEVAL (data_processing.py)                         │
│    • Query Redshift: 3 years hourly data                       │
│    • Final submissions (verified, 48+ days old)                │
│    • Initial submissions (preliminary, 14-48 days old)         │
│    Output: Raw time series DataFrame                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. WEATHER INTEGRATION (weather_features.py)                   │
│    • Fetch from Open-Meteo API (3 years)                       │
│    • 17 weather variables (temp, radiation, cloud, wind)       │
│    • Merge on hourly timestamp                                 │
│    Output: DataFrame with weather columns                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. SOLAR FEATURES (solar_features.py)                          │
│    • Calculate solar position (zenith, azimuth)                │
│    • Solar windows (morning, peak, evening)                    │
│    • Direct/diffuse radiation ratios                           │
│    • Duck curve metrics (daily)                                │
│    Output: DataFrame with 80-120 features                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. TRAIN/VAL/TEST SPLIT (preprocessing.py)                     │
│    • Sort by datetime (temporal order)                         │
│    • Train: 2,305+ days (~85%)                                 │
│    • Validation: 60 days (~5%)                                 │
│    • Test: 30 days (~2.5%)                                     │
│    • Save CSVs to S3                                            │
│    Output: 3 CSV files                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. FEATURE SELECTION (feature_selection.py)                    │
│    • Method 1: Importance-based (XGBoost)                      │
│    • Method 2: Correlation-based (remove collinear)            │
│    • Method 3: Mutual information (optional)                   │
│    • Consensus: Select from 50%+ of methods                    │
│    Output: ~40 selected features                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. HYPERPARAMETER OPTIMIZATION (hyperparameter_optimization.py)│
│    • Optuna Bayesian optimization                              │
│    • 50 trials (configurable)                                  │
│    • Time series CV (5 folds)                                  │
│    • Weighted objective (RMSE, MAPE, R², sMAPE)                │
│    Output: Best hyperparameters                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. FINAL TRAINING (model.py)                                   │
│    • Train XGBoost with best params                            │
│    • Use train + validation data                               │
│    • Evaluate on held-out test set                             │
│    Output: Trained model, metrics, feature list                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. EVALUATION (evaluation.py)                                  │
│    • Standard metrics (RMSE, MAE, R², MAPE, etc.)              │
│    • Segment-specific metrics (duck curve, business hours)     │
│    • Period-weighted scores                                    │
│    • Visualizations (plots, feature importance)                │
│    Output: Comprehensive metrics JSON, plots                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. DEPLOYMENT (inference.py + SageMaker)                       │
│    • Package model + features + inference code                 │
│    • Deploy to SageMaker endpoint                              │
│    • DELETE endpoint (cost optimization)                       │
│    • Store config in S3                                         │
│    Output: Endpoint config JSON in S3                          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Data Transformations

**Initial Data Shape**:
```
Rows: ~26,000 (3 years × 365 days × 24 hours)
Columns: 12 (raw Redshift columns)
```

**After Weather Integration**:
```
Rows: ~26,000
Columns: 29 (12 raw + 17 weather)
```

**After Solar Features**:
```
Rows: ~26,000
Columns: 95 (29 + 66 solar/derived)
```

**After Feature Selection**:
```
Rows: ~26,000
Columns: 40 (selected features only)
```

**Train/Val/Test Split**:
```
Train: 24,500 rows × 40 features
Val: 1,440 rows × 40 features
Test: 720 rows × 40 features
```

**Model Output**:
```
Predictions: 720 values (test set)
Metrics: 15+ evaluation metrics
Artifacts: Model file (~5 MB), feature list, metadata
```

---

## Performance Benchmarks

| Stage | Duration | Notes |
|-------|----------|-------|
| Data Retrieval | 5-15 min | Depends on Redshift cluster performance |
| Weather API | 30-60 sec | Cached after first call |
| Feature Engineering | 2-5 min | Vectorized operations |
| Feature Selection | 10-15 min | 3 methods + consensus |
| HPO (50 trials) | 2-4 hours | Parallelizable |
| Final Training | 5-10 min | Single model on full data |
| Evaluation | 2-5 min | Metrics + plots |
| **Total** | **3-5 hours** | For complete pipeline |

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Status**: Production Ready
