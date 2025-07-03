"""
SageMaker Pipeline orchestration script for energy load forecasting.
"""

import os
import tempfile
import shutil
import sys
import json
import logging
import boto3
import sagemaker
from datetime import datetime
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.functions import Join
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep

from configs import config

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%m/%d/%y %H:%M:%S",
)

def create_preprocessing_pipeline(
    role,
    pipeline_name,
    bucket=None,
    prefix=None,
    preprocessing_s3_uri=None,
    days_delay=7,
    use_reduced_features=True,
    meter_threshold=1000,
    use_cache=True,
    use_weather=True,
    use_solar=True,
    weather_cache=True,
):
    """Create a SageMaker Pipeline for preprocessing.
    
    Args:
        role (str): AWS IAM role ARN for SageMaker
        pipeline_name (str): Name of the pipeline
        bucket (str, optional): S3 bucket name. Defaults to config value.
        prefix (str, optional): S3 prefix for all pipeline files. Defaults to config value.
        preprocessing_s3_uri (str, optional): S3 URI for preprocessing script.
        days_delay (int, optional): Days delay in data availability. Defaults to 7.
        use_reduced_features (bool, optional): Whether to use reduced features. Defaults to True.
        meter_threshold (int, optional): Meter count threshold. Defaults to 1000.
        use_cache (bool, optional): Whether to use CSV cache. Defaults to True.
        use_weather (bool, optional): Whether to use weather features. Defaults to True.
        use_solar (bool, optional): Whether to use solar features. Defaults to True.
        weather_cache (bool, optional): Whether to cache weather data. Defaults to True.
    
    Returns:
        Pipeline: SageMaker pipeline instance
    """
    logger.info(f"Creating preprocessing pipeline: {pipeline_name}")

    # Use config defaults if not provided
    bucket = bucket or config.S3_BUCKET
    prefix = prefix or config.S3_PREFIX

    # Create a SageMaker session
    session = sagemaker.Session()
    region = boto3.session.Session().region_name

    # Set up default locations if not provided
    if not preprocessing_s3_uri:
        preprocessing_s3_uri = f"s3://{bucket}/{prefix}/scripts/preprocessing.py"

    # Define pipeline parameters
    days_delay_param = ParameterInteger(name="DaysDelay", default_value=days_delay)
    use_reduced_features_param = ParameterBoolean(
        name="UseReducedFeatures", default_value=use_reduced_features
    )
    meter_threshold_param = ParameterInteger(
        name="MeterThreshold", default_value=meter_threshold
    )
    use_cache_param = ParameterBoolean(name="UseCache", default_value=use_cache)
    query_limit_param = ParameterInteger(name="QueryLimit", default_value=-1)
    use_weather_param = ParameterBoolean(name="UseWeather", default_value=use_weather)
    use_solar_param = ParameterBoolean(name="UseSolar", default_value=use_solar)
    weather_cache_param = ParameterBoolean(
        name="WeatherCache", default_value=weather_cache
    )

    # Configure processor
    sklearn_processor = SKLearnProcessor(
        framework_version=config.PREPROCESSING_FRAMEWORK_VERSION,
        role=role,
        instance_type=config.PREPROCESSING_INSTANCE_TYPE,
        instance_count=config.PREPROCESSING_INSTANCE_COUNT,
        base_job_name=config.PREPROCESSING_BASE_JOB_NAME,
    )

    # Define preprocessing step
    preprocessing_step = ProcessingStep(
        name="PreprocessRESDeva",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/processing_config.json",
                destination="/opt/ml/processing/input/config",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/preprocessing.py",
                destination="/opt/ml/processing/input/code/preprocessing",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/data_processing.py",
                destination="/opt/ml/processing/input/code/data_processing",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/solar_features.py",
                destination="/opt/ml/processing/input/code/solar_features",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/weather_features.py",
                destination="/opt/ml/processing/input/code/weather_features",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/processing_wrapper.py",
                destination="/opt/ml/processing/input/code/wrapper",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/requirements.txt",
                destination="/opt/ml/processing/input/requirements",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="training",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/{prefix}/processed/training",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{bucket}/{prefix}/processed/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/{prefix}/processed/test",
            ),
        ],
        code=f"s3://{bucket}/{prefix}/scripts/processing_wrapper.py",
        job_arguments=[
            "--days-delay",
            days_delay_param.to_string(),
            "--use-reduced-features",
            use_reduced_features_param.to_string(),
            "--meter-threshold",
            meter_threshold_param.to_string(),
            "--use-cache",
            use_cache_param.to_string(),
            "--query-limit",
            query_limit_param.to_string(),
            "--use-weather",
            use_weather_param.to_string(),
            "--use-solar",
            use_solar_param.to_string(),
            "--weather-cache",
            weather_cache_param.to_string(),
            "--config-path",
            "/opt/ml/processing/input/config/processing_config.json",
            "--preprocessing-path",
            "/opt/ml/processing/input/code/preprocessing/preprocessing.py",
            "--data-processing-path",
            "/opt/ml/processing/input/code/data_processing/data_processing.py",
            "--solar-features-path",
            "/opt/ml/processing/input/code/solar_features/solar_features.py",
            "--weather-features-path",
            "/opt/ml/processing/input/code/weather_features/weather_features.py",
        ],
    )

    # Create the pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            days_delay_param,
            use_reduced_features_param,
            meter_threshold_param,
            use_cache_param,
            query_limit_param,
            use_weather_param,
            use_solar_param,
            weather_cache_param,
        ],
        steps=[preprocessing_step],
        sagemaker_session=session,
    )

    # Create or update pipeline in SageMaker
    pipeline_arn = pipeline.create(role_arn=role)
    logger.info(f"Preprocessing pipeline created with ARN: {pipeline_arn}")

    return pipeline


def prepare_training_code(bucket, prefix):
    # Create a temporary directory for our code
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
   
    # Create source directory inside temp_dir
    source_dir = os.path.join(temp_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
   
    # Download required files from S3
    s3_client = boto3.client('s3')
    files_to_download = [
        "training_wrapper.py",
        "model.py",
        "feature_selection.py",
        "hyperparameter_optimization.py",
        "evaluation.py",
        "visualization.py",        
    ]
   
    for file in files_to_download:
        local_path = os.path.join(source_dir, file)
        s3_key = f"{prefix}/scripts/{file}"
        try:
            s3_client.download_file(bucket, s3_key, local_path)
            print(f"Downloaded {s3_key} to {local_path}")
        except Exception as e:
            print(f"Error downloading {file}: {str(e)}")
   
    # Create pipeline directory structure in source_dir
    pipeline_dir = os.path.join(source_dir, "pipeline")
    training_dir = os.path.join(pipeline_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
   
    # Create __init__.py files
    with open(os.path.join(source_dir, "__init__.py"), "w") as f:
        f.write("# Package initialization\n")
        
    with open(os.path.join(pipeline_dir, "__init__.py"), "w") as f:
        f.write("# Package initialization\n")
   
    with open(os.path.join(training_dir, "__init__.py"), "w") as f:
        f.write("# Package initialization\n")
   
    # Copy Python module files to the training dir
    for file in ["model.py", "feature_selection.py", "hyperparameter_optimization.py", "evaluation.py", "visualization.py"]:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(training_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file} to {dst_path}")
   
    return source_dir


def create_training_pipeline(
    role,
    pipeline_name,
    bucket=None,
    prefix=None,
    preprocessing_pipeline_name=None,
    customer_segment="RES_SOLAR",
    feature_selection_method="importance",
    feature_count=40,
    correlation_threshold=0.85,
    hpo_method="bayesian",
    hpo_max_evals=50,
    cv_folds=5,
    cv_gap_days=7,
    enable_multi_model=False,
    rate_groups=None,
    deploy_model=False,
    endpoint_name=None,
):
    """Create a SageMaker Pipeline for energy load forecasting training.

    Args:
        role (str): AWS IAM role ARN for SageMaker
        pipeline_name (str): Name of the pipeline
        bucket (str, optional): S3 bucket name. Defaults to config value.
        prefix (str, optional): S3 prefix for all pipeline files. Defaults to config value.
        preprocessing_pipeline_name (str, optional): Name of preprocessing pipeline to depend on
        feature_selection_method (str): Feature selection method
        feature_count (int): Target number of features to select
        correlation_threshold (float): Correlation threshold for feature selection
        hpo_method (str): Hyperparameter optimization method
        hpo_max_evals (int): Maximum number of hyperparameter evaluations
        cv_folds (int): Number of cross-validation folds
        cv_gap_days (int): Gap days for time series cross-validation
        enable_multi_model (bool): Whether to enable multi-model training
        rate_groups (str, optional): Comma-separated list of rate groups for multi-model training
        deploy_model (bool): Whether to deploy the model to an endpoint
        endpoint_name (str, optional): Name for the endpoint

    Returns:
        Pipeline: SageMaker pipeline instance
    """
    logger.info(f"Creating training pipeline: {pipeline_name}")

    # Use config defaults if not provided
    bucket = bucket or config.S3_BUCKET
    prefix = prefix or config.S3_PREFIX

    # Create a SageMaker session
    session = sagemaker.Session()
    region = boto3.session.Session().region_name
    
    # Define pipeline parameters
    customer_segment_param = ParameterString(
        name="CustomerSegment", default_value=customer_segment
    )
    
    feature_selection_method_param = ParameterString(
        name="FeatureSelectionMethod", default_value=feature_selection_method
    )
    feature_count_param = ParameterInteger(
        name="FeatureCount", default_value=feature_count
    )
    correlation_threshold_param = ParameterInteger(
        name="CorrelationThreshold", default_value=int(correlation_threshold * 100)
    )
    hpo_method_param = ParameterString(
        name="HPOMethod", default_value=hpo_method
    )
    hpo_max_evals_param = ParameterInteger(
        name="HPOMaxEvals", default_value=hpo_max_evals
    )
    cv_folds_param = ParameterInteger(
        name="CVFolds", default_value=cv_folds
    )
    cv_gap_days_param = ParameterInteger(
        name="CVGapDays", default_value=cv_gap_days
    )
    enable_multi_model_param = ParameterBoolean(
        name="EnableMultiModel", default_value=enable_multi_model
    )
    rate_groups_param = ParameterString(
        name="RateGroups", default_value=rate_groups or ""
    )
    model_name_param = ParameterString(
        name="ModelName", default_value=f"energy-forecasting-res-{datetime.now().strftime('%Y%m%d')}"
    )

    def prepare_training_code():
        # Create a temporary directory for our code
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # Download required files from S3
        s3_client = boto3.client('s3')
        files_to_download = [
            "training_wrapper.py",
            "model.py",
            "feature_selection.py", 
            "hyperparameter_optimization.py",
            "evaluation.py",
            "visualization.py",
            "requirements.txt",
            "config.py"
        ]
        
        for file in files_to_download:
            local_path = os.path.join(temp_dir, file)
            s3_key = f"{prefix}/scripts/{file}"
            try:
                s3_client.download_file(bucket, s3_key, local_path)
                print(f"Downloaded {s3_key} to {local_path}")
            except Exception as e:
                print(f"Error downloading {file}: {str(e)}")
        
        # Create a simple setup.py file
        setup_py = """
from setuptools import setup, find_packages

setup(
    name="energy_forecasting",
    version="0.1",
    packages=find_packages(),
)
"""
        with open(os.path.join(temp_dir, "setup.py"), "w") as f:
            f.write(setup_py)
        
        # Create __init__.py file
        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            f.write("# Package initialization\n")
        
        return temp_dir   
    
    # Create local code directory
    code_dir = prepare_training_code()
    
    # Define XGBoost estimator for training
    xgboost_estimator = XGBoost(
        entry_point="training_wrapper.py",
        source_dir=code_dir,
        dependencies=[],
        model_server_workers=1,
        role=role,
        instance_type='ml.m5.xlarge',
        instance_count=1,
        framework_version='1.5-1',
        # py_version='py38',
        output_path=f"s3://{bucket}/{prefix}/models",
        code_location=f"s3://{bucket}/{prefix}/code",
        hyperparameters={
            'customer-segment': customer_segment_param,
            'feature_selection_method': feature_selection_method_param,
            'feature_count': feature_count_param,
            'correlation_threshold': correlation_threshold_param,
            'hpo_method': hpo_method_param,
            'hpo_max_evals': hpo_max_evals_param,
            'cv_folds': cv_folds_param,
            'cv_gap_days': cv_gap_days_param,
            'enable_multi_model': enable_multi_model_param,
            'rate_groups': rate_groups_param,
            'model_name': model_name_param
        },
        sagemaker_session=session
    )
    
    # Define training step
    training_step = TrainingStep(
        name="TrainRESModel",
        estimator=xgboost_estimator,
        inputs={
            "train": f"s3://{bucket}/{prefix}/processed/training",
            "validation": f"s3://{bucket}/{prefix}/processed/validation",
            "test": f"s3://{bucket}/{prefix}/processed/test"
        }
    )

    # Initialize pipeline steps
    steps = [training_step]
    
    # Add deployment step if requested
    if deploy_model:
        # Set up endpoint name if not provided
        if not endpoint_name:
            endpoint_name = f"energy-forecast-{datetime.now().strftime('%Y%m%d')}"
            
        endpoint_name_param = ParameterString(
            name="EndpointName", 
            default_value=endpoint_name
        )
        
        # Create model
        model = Model(
            image_uri=f"763104351884.dkr.ecr.{region}.amazonaws.com/xgboost-inference:1.3-1",
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            role=role,
            name=model_name_param,
            env={
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM": "inference.py"
            }
        )
        
        # Create model inputs
        inputs = CreateModelInput(
            instance_type="ml.m5.large",
            accelerator_type=None
        )
        
        # Create model step
        create_model_step = CreateModelStep(
            name="CreateEnergyForecastModel",
            model=model,
            inputs=inputs
        )
        
        # Create endpoint config
        endpoint_config_name = Join(on="-", values=[model_name_param, "config"])
        endpoint_config_step = CreateEndpointConfigStep(
            name="CreateEnergyForecastEndpointConfig",
            endpoint_config_name=endpoint_config_name,
            model_name=create_model_step.properties.ModelName,
            initial_instance_count=1,
            instance_type="ml.m5.large"
        )
        
        # Create endpoint
        endpoint_step = CreateEndpointStep(
            name="CreateEnergyForecastEndpoint",
            endpoint_name=endpoint_name_param,
            endpoint_config_name=endpoint_config_step.properties.EndpointConfigName
        )
        
        # Add deployment steps
        steps.extend([create_model_step, endpoint_config_step, endpoint_step])
    
    # Create the pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            feature_selection_method_param,
            feature_count_param,
            correlation_threshold_param,
            hpo_method_param,
            hpo_max_evals_param,
            cv_folds_param,
            cv_gap_days_param,
            enable_multi_model_param,
            rate_groups_param,
            model_name_param,
            customer_segment_param,
        ],
        steps=steps,
        sagemaker_session=session
    )
    
    # Create or update pipeline in SageMaker
    pipeline_arn = pipeline.create(role_arn=role)
    logger.info(f"Training pipeline created with ARN: {pipeline_arn}")
    
    return pipeline

def create_complete_pipeline(
    role,
    pipeline_name,
    pipeline_type="complete",
    bucket=None,
    prefix=None,
    days_delay=7,
    use_reduced_features=True,
    meter_threshold=1000,
    use_cache=True,
    use_weather=True,
    use_solar=True,
    weather_cache=True,
    customer_segment="RES_SOLAR",
    customer_profile="RES",
    feature_selection_method="importance",
    feature_count=40,
    correlation_threshold=0.85,
    hpo_method="optuna",
    hpo_max_evals=50,
    cv_folds=5,
    cv_gap_days=7,
    enable_multi_model=False,
    rate_groups=None,
    deploy_model=False,
    endpoint_name=None,
    instance_type=None,
    instance_count=None,
):
    """Create a complete pipeline with preprocessing, training, and deployment.
    
    Args:
        role (str): AWS IAM role ARN for SageMaker
        pipeline_name (str): Name of the pipeline
        bucket (str, optional): S3 bucket name
        prefix (str, optional): S3 prefix
        days_delay (int, optional): Days of delay in data availability
        use_reduced_features (bool, optional): Whether to use reduced features
        meter_threshold (int, optional): Meter count threshold
        use_cache (bool, optional): Whether to use CSV cache
        use_weather (bool, optional): Whether to use weather features
        use_solar (bool, optional): Whether to use solar features
        weather_cache (bool, optional): Whether to cache weather data
        feature_selection_method (str, optional): Feature selection method
        feature_count (int, optional): Target number of features
        correlation_threshold (float, optional): Correlation threshold
        hpo_method (str, optional): Hyperparameter optimization method
        hpo_max_evals (int, optional): Max hyperparameter evaluations
        cv_folds (int, optional): Cross-validation folds
        cv_gap_days (int, optional): Gap days for CV
        enable_multi_model (bool, optional): Enable multi-model training
        rate_groups (str, optional): Rate groups for multi-model
        deploy_model (bool, optional): Whether to deploy the model
        endpoint_name (str, optional): Name for the endpoint
        instance_type (str, optional): Endpoint instance type
        instance_count (int, optional): Endpoint instance count
    
    Returns:
        Pipeline: SageMaker pipeline instance
    """
    # Use config defaults if not provided
    bucket = bucket or config.S3_BUCKET
    prefix = prefix or config.S3_PREFIX
    instance_type = instance_type or config.INFERENCE_INSTANCE_TYPE
    instance_count = instance_count or config.INFERENCE_INSTANCE_COUNT
    
    # Create a SageMaker session
    session = sagemaker.Session()
    region = boto3.session.Session().region_name
    
    # Generate endpoint name if not provided
    if endpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        endpoint_name = f"energy-forecasting-endpoint-{timestamp}"
    
    # Define preprocessing parameters
    customer_segment_param = ParameterString(
        name="CustomerSegment", default_value=customer_segment
    )
    
    days_delay_param = ParameterInteger(
        name="DaysDelay", default_value=days_delay
    )
    use_reduced_features_param = ParameterBoolean(
        name="UseReducedFeatures", default_value=use_reduced_features
    )
    meter_threshold_param = ParameterInteger(
        name="MeterThreshold", default_value=meter_threshold
    )
    use_cache_param = ParameterBoolean(
        name="UseCache", default_value=use_cache
    )
    query_limit_param = ParameterInteger(
        name="QueryLimit", default_value=-1
    )
    use_weather_param = ParameterBoolean(
        name="UseWeather", default_value=use_weather
    )
    use_solar_param = ParameterBoolean(
        name="UseSolar", default_value=use_solar
    )
    weather_cache_param = ParameterBoolean(
        name="WeatherCache", default_value=weather_cache
    )
    
    # Define training parameters
    feature_selection_method_param = ParameterString(
        name="FeatureSelectionMethod", default_value=feature_selection_method
    )
    feature_count_param = ParameterInteger(
        name="FeatureCount", default_value=feature_count
    )
    correlation_threshold_param = ParameterInteger(
        name="CorrelationThreshold", default_value=int(correlation_threshold * 100)
    )
    hpo_method_param = ParameterString(
        name="HPOMethod", default_value=hpo_method
    )
    hpo_max_evals_param = ParameterInteger(
        name="HPOMaxEvals", default_value=hpo_max_evals
    )
    cv_folds_param = ParameterInteger(
        name="CVFolds", default_value=cv_folds
    )
    cv_gap_days_param = ParameterInteger(
        name="CVGapDays", default_value=cv_gap_days
    )
    enable_multi_model_param = ParameterBoolean(
        name="EnableMultiModel", default_value=enable_multi_model
    )
    rate_groups_param = ParameterString(
        name="RateGroups", default_value=rate_groups or ""
    )
    model_name_param = ParameterString(
        name="ModelName", default_value=f"energy-forecasting-res-{datetime.now().strftime('%Y%m%d')}"
    )
    endpoint_name_param = ParameterString(
        name="EndpointName", default_value=endpoint_name
    )
    
    # Configure preprocessing processor
    preprocessing_processor = SKLearnProcessor(
        framework_version=config.PREPROCESSING_FRAMEWORK_VERSION,
        role=role,
        instance_type=config.PREPROCESSING_INSTANCE_TYPE,
        instance_count=config.PREPROCESSING_INSTANCE_COUNT,
        base_job_name=config.PREPROCESSING_BASE_JOB_NAME,
    )
    
    # Define preprocessing step
    preprocessing_step = ProcessingStep(
        name="PreprocessRESDeva",
        processor=preprocessing_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/processing_config.json",
                destination="/opt/ml/processing/input/config",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/preprocessing.py",
                destination="/opt/ml/processing/input/code/preprocessing",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/data_processing.py",
                destination="/opt/ml/processing/input/code/data_processing",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/solar_features.py",
                destination="/opt/ml/processing/input/code/solar_features",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/weather_features.py",
                destination="/opt/ml/processing/input/code/weather_features",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/processing_wrapper.py",
                destination="/opt/ml/processing/input/code/wrapper",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/scripts/requirements.txt",
                destination="/opt/ml/processing/input/requirements",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="training",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/{prefix}/processed/training",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{bucket}/{prefix}/processed/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/{prefix}/processed/test",
            ),
        ],
        code=f"s3://{bucket}/{prefix}/scripts/processing_wrapper.py",
        job_arguments=[
            "--days-delay",
            days_delay_param.to_string(),
            "--use-reduced-features",
            use_reduced_features_param.to_string(),
            "--meter-threshold",
            meter_threshold_param.to_string(),
            "--use-cache",
            use_cache_param.to_string(),
            "--query-limit",
            query_limit_param.to_string(),
            "--use-weather",
            use_weather_param.to_string(),
            "--use-solar",
            use_solar_param.to_string(),
            "--weather-cache",
            weather_cache_param.to_string(),
            "--config-path",
            "/opt/ml/processing/input/config/processing_config.json",
            "--preprocessing-path",
            "/opt/ml/processing/input/code/preprocessing/preprocessing.py",
            "--data-processing-path",
            "/opt/ml/processing/input/code/data_processing/data_processing.py",
            "--solar-features-path",
            "/opt/ml/processing/input/code/solar_features/solar_features.py",
            "--weather-features-path",
            "/opt/ml/processing/input/code/weather_features/weather_features.py",
        ],
    )
    
    def prepare_training_code():
        # Create a temporary directory for our code
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # Download required files from S3
        s3_client = boto3.client('s3')
        files_to_download = [
            "training_wrapper.py",
            "model.py",
            "feature_selection.py", 
            "hyperparameter_optimization.py",
            "evaluation.py",
            "visualization.py",
            "requirements.txt",
            "config.py"
        ]
        
        for file in files_to_download:
            local_path = os.path.join(temp_dir, file)
            s3_key = f"{prefix}/scripts/{file}"
            try:
                s3_client.download_file(bucket, s3_key, local_path)
                print(f"Downloaded {s3_key} to {local_path}")
            except Exception as e:
                print(f"Error downloading {file}: {str(e)}")
        
        # Create a simple setup.py file
        setup_py = """
from setuptools import setup, find_packages

setup(
  name="energy_forecasting",
  version="0.1",
  packages=find_packages(),
)
"""
        with open(os.path.join(temp_dir, "setup.py"), "w") as f:
            f.write(setup_py)
        
        # Create __init__.py file
        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            f.write("# Package initialization\n")
        
        return temp_dir  
    
    # Create local code directory
    code_dir = prepare_training_code()
    
    # Define XGBoost estimator for training
    xgboost_estimator = XGBoost(
        entry_point="training_wrapper.py",
        source_dir=code_dir,
        dependencies=[],
        model_server_workers=1,
        role=role,
        instance_type='ml.m5.xlarge',
        instance_count=1,
        framework_version='1.5-1',
        # py_version='py38',
        output_path=f"s3://{bucket}/{prefix}/models",
        code_location=f"s3://{bucket}/{prefix}/code",
        hyperparameters={
            'customer_segment': customer_segment_param,
            'feature_selection_method': feature_selection_method_param,
            'feature_count': feature_count_param,
            'correlation_threshold': correlation_threshold_param,
            'hpo_method': hpo_method_param,
            'hpo_max_evals': hpo_max_evals_param,
            'cv_folds': cv_folds_param,
            'cv_gap_days': cv_gap_days_param,
            'enable_multi_model': enable_multi_model_param,
            'rate_groups': rate_groups_param,
            'model_name': model_name_param,
            's3_bucket': bucket,
            's3_prefix': prefix
        },
        sagemaker_session=session
    )
    
    # Define training step
    training_step = TrainingStep(
        name="TrainRESModel",
        estimator=xgboost_estimator,
        inputs={
            "train": preprocessing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
            "validation": preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            "test": preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
        }
    )
    
    # Initialize steps list with preprocessing, training, and evaluation
    steps = [preprocessing_step, training_step]
    
    # Create the pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            # Preprocessing parameters
            days_delay_param,
            use_reduced_features_param,
            meter_threshold_param,
            use_cache_param,
            query_limit_param,
            use_weather_param,
            use_solar_param,
            weather_cache_param,
            # Training parameters
            feature_selection_method_param,
            feature_count_param,
            correlation_threshold_param,
            hpo_method_param,
            hpo_max_evals_param,
            cv_folds_param,
            cv_gap_days_param,
            enable_multi_model_param,
            rate_groups_param,
            model_name_param,
            customer_segment_param,
            # Deployment parameter (if applicable)
            endpoint_name_param if deploy_model else None
        ],
        steps=steps,
        sagemaker_session=session
    )
    
    # Remove None parameters
    pipeline.parameters = [p for p in pipeline.parameters if p is not None]
    
    # Create or update pipeline in SageMaker
    pipeline_arn = pipeline.create(role_arn=role)
    logger.info(f"Complete pipeline created with ARN: {pipeline_arn}")
    
    return pipeline


def execute_preprocessing_pipeline(pipeline_name, parameters=None):
    """Execute the preprocessing pipeline.

    Args:
        pipeline_name (str): Name of the pipeline to execute
        parameters (dict, optional): Parameter values to pass to the pipeline

    Returns:
        str: Pipeline execution ARN
    """
    logger.info(f"Executing preprocessing pipeline: {pipeline_name}")

    # Get the pipeline definition to check valid parameters
    client = boto3.client("sagemaker")
    try:
        # Get pipeline description
        pipeline_desc = client.describe_pipeline(PipelineName=pipeline_name)

        # Pipeline definition is in JSON format as a string
        pipeline_def = json.loads(pipeline_desc["PipelineDefinition"])

        # Extract parameter names
        valid_params = [param["Name"] for param in pipeline_def.get("Parameters", [])]
        logger.info(f"Valid pipeline parameters: {valid_params}")
        logger.info(f"Passed pipeline parameters: {parameters}")

        # Filter out invalid parameters
        filtered_params = {}
        if parameters:
            for key, value in parameters.items():
                if key in valid_params:
                    filtered_params[key] = value
                else:
                    logger.warning(
                        f"Parameter '{key}' is not defined in the pipeline and will be ignored."
                    )

        logger.info(f"Filtered pipeline parameters: {filtered_params}")

        # Convert parameters to the format expected by start_pipeline_execution
        pipeline_parameters = []
        for key, value in filtered_params.items():
            pipeline_parameters.append({"Name": key, "Value": str(value)})

        logger.info(f"Final pipeline parameters: {pipeline_parameters}")

        # Start the pipeline execution
        response = client.start_pipeline_execution(
            PipelineName=pipeline_name, PipelineParameters=pipeline_parameters
        )

        execution_arn = response["PipelineExecutionArn"]
        logger.info(f"Pipeline execution started with ARN: {execution_arn}")

        return execution_arn

    except Exception as e:
        logger.error(f"Error executing pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def execute_training_pipeline(pipeline_name, parameters=None):
    """Execute the training pipeline.

    Args:
        pipeline_name (str): Name of the pipeline to execute
        parameters (dict, optional): Parameter values to pass to the pipeline

    Returns:
        str: Pipeline execution ARN
    """
    logger.info(f"Executing training pipeline: {pipeline_name}")

    # Get the pipeline definition to check valid parameters
    client = boto3.client("sagemaker")
    try:
        # Get pipeline description
        pipeline_desc = client.describe_pipeline(PipelineName=pipeline_name)

        # Pipeline definition is in JSON format as a string
        pipeline_def = json.loads(pipeline_desc["PipelineDefinition"])

        # Extract parameter names
        valid_params = [param["Name"] for param in pipeline_def.get("Parameters", [])]
        logger.info(f"Valid pipeline parameters: {valid_params}")
        logger.info(f"Passed pipeline parameters: {parameters}")

        # Filter out invalid parameters
        filtered_params = {}
        if parameters:
            for key, value in parameters.items():
                if key in valid_params:
                    filtered_params[key] = value
                else:
                    logger.warning(
                        f"Parameter '{key}' is not defined in the pipeline and will be ignored."
                    )

        logger.info(f"Filtered pipeline parameters: {filtered_params}")

        # Convert parameters to the format expected by start_pipeline_execution
        pipeline_parameters = []
        for key, value in filtered_params.items():
            pipeline_parameters.append({"Name": key, "Value": str(value)})

        logger.info(f"Final pipeline parameters: {pipeline_parameters}")

        # Start the pipeline execution
        response = client.start_pipeline_execution(
            PipelineName=pipeline_name, PipelineParameters=pipeline_parameters
        )

        execution_arn = response["PipelineExecutionArn"]
        logger.info(f"Pipeline execution started with ARN: {execution_arn}")

        return execution_arn

    except Exception as e:
        logger.error(f"Error executing pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def execute_complete_pipeline(pipeline_name, parameters=None):
    """Execute the complete pipeline with preprocessing and training.

    Args:
        pipeline_name (str): Name of the pipeline to execute
        parameters (dict, optional): Parameter values to pass to the pipeline

    Returns:
        str: Pipeline execution ARN
    """
    logger.info(f"Executing complete pipeline: {pipeline_name}")

    # Get the pipeline definition to check valid parameters
    client = boto3.client("sagemaker")
    try:
        # Get pipeline description
        pipeline_desc = client.describe_pipeline(PipelineName=pipeline_name)

        # Pipeline definition is in JSON format as a string
        pipeline_def = json.loads(pipeline_desc["PipelineDefinition"])

        # Extract parameter names
        valid_params = [param["Name"] for param in pipeline_def.get("Parameters", [])]
        logger.info(f"Valid pipeline parameters: {valid_params}")
        logger.info(f"Passed pipeline parameters: {parameters}")

        # Filter out invalid parameters
        filtered_params = {}
        if parameters:
            for key, value in parameters.items():
                if key in valid_params:
                    filtered_params[key] = value
                else:
                    logger.warning(
                        f"Parameter '{key}' is not defined in the pipeline and will be ignored."
                    )

        logger.info(f"Filtered pipeline parameters: {filtered_params}")

        # Convert parameters to the format expected by start_pipeline_execution
        pipeline_parameters = []
        for key, value in filtered_params.items():
            pipeline_parameters.append({"Name": key, "Value": str(value)})

        logger.info(f"Final pipeline parameters: {pipeline_parameters}")

        # Start the pipeline execution
        response = client.start_pipeline_execution(
            PipelineName=pipeline_name, PipelineParameters=pipeline_parameters
        )

        execution_arn = response["PipelineExecutionArn"]
        logger.info(f"Pipeline execution started with ARN: {execution_arn}")

        return execution_arn

    except Exception as e:
        logger.error(f"Error executing pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
