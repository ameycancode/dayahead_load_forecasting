```bash
cat > test-automation/README.md << 'EOF'
# Energy Testing Framework

Automated unit test generation for energy forecasting codebase using AWS Bedrock Claude 3.5 Sonnet v2.

## Quick Start

1. **Setup (one-time)**:
   ```bash
   cd test-automation
   python run_energy_testing.py --setup-only
   cd ..
   ```

2. **Demo Mode**:
   ```bash
   python test-automation/run_energy_testing.py --demo
   ```

3. **Process Files**:
   ```bash
   # Single file
   python test-automation/run_energy_testing.py --budget 25 --target configs/config.py
   
   # Directory
   python test-automation/run_energy_testing.py --budget 100 --target pipeline/ --recursive
   ```

4. **Run Tests**:
   ```bash
   python run_tests.py
   ```

## Cost Estimates

- Simple config files: $2-5
- Medium complexity: $8-15  
- Large complex files: $20-40

## Target Coverage: 85%+
EOF
```

## 🎯 **Recommended Processing Order**

With this structure, here's the optimal processing sequence:

### **Phase 1: Foundation ($25 budget)**
```bash
python test-automation/run_energy_testing.py --budget 25 --target configs/config.py
```

### **Phase 2: Core Pipeline ($50 budget)**
```bash
python test-automation/run_energy_testing.py --budget 50 --target pipeline/orchestration/ --recursive
```

### **Phase 3: Processing Layer ($75 budget)**
```bash
python test-automation/run_energy_testing.py --budget 75 --target pipeline/preprocessing/ --recursive
```

### **Phase 4: Training Components ($100 budget)**
```bash
python test-automation/run_energy_testing.py --budget 100 --target pipeline/training/ --recursive
```

## 🔄 **Workflow Integration**

### **Development Workflow**
1. **Generate tests**: Use framework to create tests for new/modified files
2. **Run tests**: `python run_tests.py` 
3. **Check coverage**: Open `reports/coverage/index.html`
4. **Commit**: Generated tests are committed with your code

### **CI/CD Integration**
The generated `.github/workflows/tests.yml` will automatically:
- Run tests on every push/PR
- Generate coverage reports
- Fail if coverage drops below 85%

This structure keeps your **testing framework separate** from your **generated tests**, makes it easy to **version control**, and provides a **clear workflow** for automated test generation.