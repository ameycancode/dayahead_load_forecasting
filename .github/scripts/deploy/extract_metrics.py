#!/usr/bin/env python3
"""
Model Metrics Extraction Script for Energy Load Forecasting Pipeline
Extracts metrics from model_metrics.json and formats them for shell consumption
"""
import json
import sys
import os

def extract_metrics():
    """Extract metrics from model_metrics.json file"""
   
    try:
        # Check if the metrics file exists
        if not os.path.exists('model_metrics.json'):
            print('ERROR_EXTRACTING_METRICS=model_metrics.json file not found')
            return False
       
        # Load the metrics
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
       
        # Extract metrics with fallback values
        rmse = metrics.get('rmse', 'N/A')
        mape = metrics.get('mape', 'N/A')
        r2 = metrics.get('r2', 'N/A')
        wape = metrics.get('wape', 'N/A')
        smape = metrics.get('smape', 'N/A')
       
        # Additional metrics that might be present
        mae = metrics.get('mae', 'N/A')
        mse = metrics.get('mse', 'N/A')
        cv_rmse = metrics.get('cv_rmse', 'N/A')
        cv_mape = metrics.get('cv_mape', 'N/A')
       
        # Handle NaN values
        if str(cv_rmse).lower() == 'nan':
            cv_rmse = 'N/A'
        if str(cv_mape).lower() == 'nan':
            cv_mape = 'N/A'
       
        # Print in format suitable for shell variables (only the variable assignments)
        print(f'RMSE_VALUE={rmse}')
        print(f'MAPE_VALUE={mape}')
        print(f'R2_VALUE={r2}')
        print(f'WAPE_VALUE={wape}')
        print(f'SMAPE_VALUE={smape}')
        print(f'MAE_VALUE={mae}')
        print(f'MSE_VALUE={mse}')
        print(f'CV_RMSE_VALUE={cv_rmse}')
        print(f'CV_MAPE_VALUE={cv_mape}')
       
        return True
       
    except json.JSONDecodeError as e:
        print(f'ERROR_EXTRACTING_METRICS=Invalid JSON format in model_metrics.json: {str(e)}')
        return False
    except Exception as e:
        print(f'ERROR_EXTRACTING_METRICS={str(e)}')
        return False

if __name__ == '__main__':
    success = extract_metrics()
    sys.exit(0 if success else 1)
