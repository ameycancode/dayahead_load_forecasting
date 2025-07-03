#!/bin/bash
# package_lambda.sh

set -e  # Exit on any error

echo "=== Starting Lambda packaging process ==="

# Initialize variables
CONDA_ACTIVATED=false

# Check if conda is available
if command -v conda >/dev/null 2>&1; then
    echo "Conda available, attempting to create environment..."
    
    # Check if py39 environment exists
    if conda env list | grep -q "py39"; then
        echo "py39 environment already exists"
    else
        echo "Creating py39 conda environment..."
        conda create -n py39 python=3.9 -y
    fi
    
    # Try to activate conda environment
    echo "Attempting to activate conda environment..."
    if conda activate py39 >/dev/null 2>&1; then
        echo " Successfully activated conda environment"
        CONDA_ACTIVATED=true
    else
        echo " Could not activate conda environment, using system Python"
        CONDA_ACTIVATED=false
    fi
else
    echo " Conda not available, using system Python"
    CONDA_ACTIVATED=false
fi

# Create output directory
echo "Creating lambda_package directory..."
mkdir -p lambda_package
rm -rf lambda_package/*

if [ -f "lambda_forecast.zip" ]; then
    rm -f lambda_forecast.zip
    echo "Removed existing lambda_forecast.zip file..."
fi

# Copy Lambda code
echo "Copying Lambda code files..."
if [ -f "lambda_function.py" ]; then
    cp lambda_function.py lambda_package/
    echo " Copied lambda_function.py"
else
    echo " lambda_function.py not found"
    exit 1
fi

if [ -d "forecast" ]; then
    cp -r forecast lambda_package/
    echo " Copied forecast directory"
else
    echo " forecast directory not found"
    exit 1
fi

# Create virtual environment as fallback if conda failed
if [ "$CONDA_ACTIVATED" = "false" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
    echo "Activating virtual environment..."
    # Use dot instead of source for better compatibility
    . venv/bin/activate
    echo " Virtual environment activated"
fi

# Install requirements
echo "Installing Python packages..."
pip install pandas==1.4.3 numpy==1.22.4 boto3==1.28.38 pyathena==2.25.2 pytz==2023.3 \
    holidays==0.29 openmeteo-requests==1.1.0 requests-cache==1.1.1 retry-requests==2.0.0 -t lambda_package/

if [ $? -eq 0 ]; then
    echo " Successfully installed all packages"
else
    echo " Failed to install packages"
    exit 1
fi

# Check initial size before cleanup
echo "=== Package size before cleanup ==="
INITIAL_SIZE=$(du -sh lambda_package | cut -f1)
echo "Initial package size: $INITIAL_SIZE"
INITIAL_COUNT=$(find lambda_package -type f | wc -l)
echo "File count before cleanup: $INITIAL_COUNT"

# Clean up unnecessary files
echo "Cleaning up unnecessary files..."

# Count items before removal for verification
DIST_INFO_COUNT=$(find lambda_package -type d -name "__pycache__" | wc -l)
TESTS_COUNT=$(find lambda_package -type d -name "tests" | wc -l)
PYC_COUNT=$(find lambda_package -type f -name "*.pyc" | wc -l)
PYO_COUNT=$(find lambda_package -type f -name "*.pyo" | wc -l)

echo "Found $PYCACHE_COUNT __pycache__ directories to remove"
echo "Found $TESTS_COUNT test directories to remove"
echo "Found $PYC_COUNT .pyc files to remove"
echo "Found $PYO_COUNT .pyo files to remove"

# Remove files and directories
find lambda_package -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null || true
# Remove test directories
echo "Removing test directories..."
find lambda_package -type d -name "tests" -exec rm -rf {} \; 2>/dev/null || true
# Remove .pyc files
echo "Removing .pyc files..."
find lambda_package -type f -name "*.pyc" -delete 2>/dev/null || true
# Remove .pyo files
echo "Removing .pyo files..."
find lambda_package -type f -name "*.pyo" -delete 2>/dev/null || true
# Remove other unnecessary files
echo "Removing other unnecessary files..."
find lambda_package -type f -name "*.py~" -delete 2>/dev/null || true
find lambda_package -type f -name ".DS_Store" -delete 2>/dev/null || true
find lambda_package -type d -name ".git" -exec rm -rf {} \; 2>/dev/null || true
find lambda_package -type d -name ".pytest_cache" -exec rm -rf {} \; 2>/dev/null || true

echo " Cleanup completed"

# Verify cleanup worked
PYCACHE_REMAINING=$(find lambda_package -type d -name "__pycache__" | wc -l)
TESTS_REMAINING=$(find lambda_package -type d -name "tests" | wc -l) 
PYC_REMAINING=$(find lambda_package -type f -name "*.pyc" | wc -l)
PYO_REMAINING=$(find lambda_package -type f -name "*.pyo" | wc -l)

echo "Cleanup verification:"
echo "  __pycache__ directories remaining: $PYCACHE_REMAINING"
echo "  test directories remaining: $TESTS_REMAINING"
echo "  .pyc files remaining: $PYC_REMAINING"
echo "  .pyo files remaining: $PYO_REMAINING"

# Check size after cleanup
echo "=== Package size after cleanup ==="
FINAL_SIZE=$(du -sh lambda_package | cut -f1)
echo "Final package size: $FINAL_SIZE"
FINAL_COUNT=$(find lambda_package -type f | wc -l)
echo "File count after cleanup: $FINAL_COUNT"

FILES_REMOVED=$((INITIAL_COUNT - FINAL_COUNT))
echo "Files removed: $FILES_REMOVED"

# Show top largest items
echo "=== Top 10 largest items in package ==="
if [ -d "lambda_package" ]; then
    du -sh lambda_package/* 2>/dev/null | sort -hr | head -10 || echo "Could not analyze package contents"
fi

# Create ZIP file
echo "Creating ZIP package..."
cd lambda_package
zip -r ../lambda_forecast.zip . 2>/dev/null || zip -r ../lambda_forecast.zip .
cd ..

# Check final ZIP size
if [ -f "lambda_forecast.zip" ]; then
    ZIP_SIZE=$(du -sh lambda_forecast.zip | cut -f1)
    echo " Created lambda_forecast.zip successfully"
    echo "ZIP file size: $ZIP_SIZE"
else
    echo " Failed to create ZIP file"
    exit 1
fi

# Clean up environments
echo "Cleaning up..."

# Deactivate virtual environment if we used it
if [ "$CONDA_ACTIVATED" = "false" ] && [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || echo " Could not deactivate virtual environment"
    echo " Virtual environment deactivated"
fi

# Deactivate conda environment if we used it
if [ "$CONDA_ACTIVATED" = "true" ]; then
    conda deactivate >/dev/null 2>&1 || echo " Could not deactivate conda environment (normal in CI)"
fi

echo "=== Lambda packaging completed successfully ==="
echo "Package location: $(pwd)/lambda_forecast.zip"
echo "Final ZIP size: $ZIP_SIZE"
