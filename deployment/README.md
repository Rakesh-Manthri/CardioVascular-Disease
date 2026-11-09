# Phase 4: Production-Ready CAD Diagnostic & Staging System

## Overview
This directory contains the production-ready software artifacts, serialized pipelines, and interactive deployment notebooks for the Coronary Artery Disease (CAD) triage system developed on the UCI Cleveland clinical cohort.

## Architecture: Dual-Stage Diagnostic Cascade

```
[Raw Clinical Patient Inputs]
              │
              ▼
   [Zero-Leakage Preprocessor]
  (Median Imputation, Mode Imputation, OHE, MinMax Normalization)
              │
              ▼
    [Stage 1: Screening Engine]
  Lightweight Neural Network (MLP)
  Operating Threshold: tau* = 0.391 (Youden's J Optimized)
              │
      ┌───────┴───────┐
      ▼               ▼
[Clean Coronaries] [CAD Detected]
  (Tier 0: Healthy)   │
                      ▼
            [Stage 2: Care Staging]
          Random Forest Classifier
          (Cost-Sensitive Balanced Weights)
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
  [Tier 1: Medical]     [Tier 2: Surgical]
  Outpatient Statins    Urgent Angiography &
  & Antiplatelets       Revascularization (PCI/CABG)
```

## Directory Structure
```
deployment/
├── models/
│   ├── cad_preprocessor.joblib         # Scaler, imputers, and aligned feature schema
│   ├── cad_stage1_screening.joblib     # Calibrated MLP Neural Network binary model
│   ├── cad_stage2_staging.joblib       # Balanced Random Forest 3-tier staging model
│   └── pipeline_metadata.json          # Input schema, valid ranges, and operating thresholds
├── phase4_deployment.ipynb             # Interactive, fully executed production notebook
├── predict_cad.py                      # Standalone CLI and Python inference module
└── README.md                           # Documentation and deployment instructions
```

## Usage Instructions

### 1. Automated Self-Test
Run the built-in test suite to verify pipeline integrity across representative patient profiles:
```bash
python predict_cad.py --test
```

### 2. Single Patient CLI Assessment
Evaluate a single patient record directly from the command line:
```bash
python predict_cad.py --age 62 --sex 1 --cp 4 --trestbps 140 --chol 268 --fbs 0 --restecg 2 --thalach 160 --exang 0 --oldpeak 3.6 --slope 3 --ca 2 --thal 3
```

### 3. Batch Scoring via CSV
Process a cohort of patient records from a CSV file:
```bash
python predict_cad.py --batch input_patients.csv --output scored_patients.csv
```

### 4. Programmatic Python API Integration
```python
from deployment.predict_cad import CADInferenceEngine

engine = CADInferenceEngine()

patient = {
    'age': 58.0, 'sex': 1, 'cp': 4, 'trestbps': 136.0, 'chol': 250.0,
    'fbs': 0, 'restecg': 1, 'thalach': 140.0, 'exang': 1, 'oldpeak': 1.6,
    'slope': 2, 'ca': 1.0, 'thal': 7
}

result = engine.predict_patient(patient)
print(result['screening_status'])         # 'Positive (CAD Detected)'
print(result['cad_probability'])          # 1.0
print(result['recommended_care_tier'])     # 1
print(result['tier_description'])         # 'Tier 1: Mild / Moderate CAD (1-2 Vessel Disease)'
print(result['clinical_action_plan'])     # Guideline-directed medical therapy
```

## Clinical Safety Metrics (Holdout Test Cohort N=61)
- **Diagnostic Sensitivity**: **92.86%** (26 out of 28 CAD patients identified)
- **Surgical Capture**: **67.0%** of critical Tier 2 multi-vessel CAD cases captured
- **Clean Specificity**: **72.73% - 87.88%** of healthy patients cleared without unnecessary invasive angiography
