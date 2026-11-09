#!/usr/bin/env python3
"""
Coronary Artery Disease (CAD) Dual-Stage Diagnostic Inference Engine
===================================================================
Production-ready clinical inference script for CAD screening and triage.

Features:
- Stage 1: Calibrated Binary CAD Detection (MLP Neural Network, Youden's J Threshold)
- Stage 2: 3-Tier Clinical Care Staging (Random Forest Classifier)
- Automated schema validation, zero-leakage imputation, and MinMax normalization
- Multi-mode support: Single Patient Dictionary, Interactive CLI, and Batch CSV Scoring
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# Paths to models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'cad_preprocessor.joblib')
STAGE1_PATH = os.path.join(MODEL_DIR, 'cad_stage1_screening.joblib')
STAGE2_PATH = os.path.join(MODEL_DIR, 'cad_stage2_staging.joblib')
METADATA_PATH = os.path.join(MODEL_DIR, 'pipeline_metadata.json')

class CADInferenceEngine:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        prep_path = os.path.join(self.model_dir, 'cad_preprocessor.joblib')
        s1_path = os.path.join(self.model_dir, 'cad_stage1_screening.joblib')
        s2_path = os.path.join(self.model_dir, 'cad_stage2_staging.joblib')
        meta_path = os.path.join(self.model_dir, 'pipeline_metadata.json')

        if not all(os.path.exists(p) for p in [prep_path, s1_path, s2_path, meta_path]):
            raise FileNotFoundError(f"Missing one or more required model artifacts in {self.model_dir}")

        self.preprocessor = joblib.load(prep_path)
        self.model_stage1 = joblib.load(s1_path)
        self.model_stage2 = joblib.load(s2_path)

        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.threshold = float(self.metadata.get('decision_threshold', 0.40))
        self.expected_cols = self.preprocessor['expected_ohe_cols']
        self.cont_cols = self.preprocessor['cont_cols']
        self.nominal_cols = self.preprocessor['nominal_cols']
        self.ca_median = self.preprocessor['ca_median']
        self.thal_mode = self.preprocessor['thal_mode']
        self.scaler = self.preprocessor['scaler']

    def validate_input(self, data):
        """Validates that input dictionary contains all required biomarkers with clinically valid ranges."""
        schema = self.metadata['input_schema']
        validated = {}
        for key, rules in schema.items():
            if key not in data or data[key] is None or pd.isna(data[key]):
                if key == 'ca':
                    val = self.ca_median
                elif key == 'thal':
                    val = self.thal_mode
                else:
                    raise ValueError(f"Missing required clinical biomarker: '{key}' ({rules.get('desc')})")
            else:
                val = data[key]
            
            # Numeric conversion
            try:
                val = float(val) if rules['type'] == 'float' else int(val)
            except Exception:
                raise TypeError(f"Biomarker '{key}' must be numeric (received {val})")

            # Range / Value check
            if 'values' in rules and val not in rules['values']:
                raise ValueError(f"Invalid value for '{key}': {val}. Allowed choices: {rules['values']}")
            if 'min' in rules and val < rules['min']:
                warnings.warn(f"Biomarker '{key}' value {val} is below typical clinical minimum ({rules['min']})")
            if 'max' in rules and val > rules['max']:
                warnings.warn(f"Biomarker '{key}' value {val} is above typical clinical maximum ({rules['max']})")
            
            validated[key] = val
        return validated

    def preprocess(self, validated_dict):
        """Converts validated dictionary to normalized feature vector."""
        df = pd.DataFrame([validated_dict])
        df_imp = df.copy()
        df_imp['ca'] = df_imp['ca'].fillna(self.ca_median)
        df_imp['thal'] = df_imp['thal'].fillna(self.thal_mode)

        df_ohe = pd.get_dummies(df_imp, columns=self.nominal_cols, drop_first=False)
        df_aligned = df_ohe.reindex(columns=self.expected_cols, fill_value=0)

        df_scaled = df_aligned.copy()
        df_scaled[self.cont_cols] = self.scaler.transform(df_aligned[self.cont_cols])
        return df_scaled

    def predict_patient(self, patient_dict, custom_threshold=None):
        """
        Executes the dual-stage clinical diagnostic pipeline for a single patient record.
        """
        cutoff = custom_threshold if custom_threshold is not None else self.threshold
        clean_input = self.validate_input(patient_dict)
        X_proc = self.preprocess(clean_input)

        # Stage 1: Binary CAD Screening
        prob_stage1 = float(self.model_stage1.predict_proba(X_proc)[0, 1])
        is_cad_present = bool(prob_stage1 >= cutoff)

        # Stage 2: Clinical Care Staging
        probs_stage2 = self.model_stage2.predict_proba(X_proc)[0]
        tier_pred = int(np.argmax(probs_stage2))

        tier_labels = {
            0: "Tier 0: Healthy / No Significant CAD (<50% stenosis)",
            1: "Tier 1: Mild / Moderate CAD (1-2 Vessel Disease)",
            2: "Tier 2: Severe / Critical CAD (3-4 Vessel / Left Main Disease)"
        }

        recommendations = {
            0: "Low Risk. Recommend routine preventive lifestyle counseling, annual lipid panel, and primary care follow-up.",
            1: "Moderate Risk. Recommend outpatient guideline-directed medical therapy (high-intensity statin, ACEi, beta-blocker, aspirin) and cardiology consultation.",
            2: "CRITICAL RISK. URGENT: Recommend immediate invasive coronary angiography / catheterization and cardiac surgical consultation for revascularization (PCI / CABG)."
        }

        # Override consistency: if Stage 1 is negative (< cutoff), overall status is Tier 0
        if not is_cad_present:
            final_status = "Negative (Healthy)"
            final_tier = 0
            rec = recommendations[0]
        else:
            final_status = "Positive (CAD Detected)"
            final_tier = max(1, tier_pred) # At least Tier 1 if Stage 1 flagged CAD
            rec = recommendations[final_tier]

        return {
            'screening_status': final_status,
            'cad_probability': round(prob_stage1, 4),
            'decision_threshold_applied': round(cutoff, 4),
            'recommended_care_tier': final_tier,
            'tier_description': tier_labels[final_tier],
            'stage2_tier_probabilities': {
                'Tier 0 (No CAD)': round(float(probs_stage2[0]), 4),
                'Tier 1 (Medical)': round(float(probs_stage2[1]), 4),
                'Tier 2 (Surgical)': round(float(probs_stage2[2]), 4)
            },
            'clinical_action_plan': rec
        }

    def predict_batch(self, csv_path, output_csv_path=None):
        """Scores a batch CSV file of patient records."""
        df_in = pd.read_csv(csv_path)
        print(f"Loaded {len(df_in)} patient records from: {csv_path}")
        results = []
        for idx, row in df_in.iterrows():
            record = row.to_dict()
            try:
                res = self.predict_patient(record)
                results.append({
                    'patient_index': idx,
                    'screening_status': res['screening_status'],
                    'cad_probability': res['cad_probability'],
                    'recommended_tier': res['recommended_care_tier'],
                    'tier_name': res['tier_description'].split(':')[1].strip(),
                    'clinical_recommendation': res['clinical_action_plan']
                })
            except Exception as e:
                results.append({
                    'patient_index': idx,
                    'screening_status': 'ERROR',
                    'cad_probability': np.nan,
                    'recommended_tier': np.nan,
                    'tier_name': str(e),
                    'clinical_recommendation': 'Data Validation Failed'
                })

        df_out = pd.DataFrame(results)
        df_combined = pd.concat([df_in.reset_index(drop=True), df_out.drop(columns=['patient_index'])], axis=1)

        if output_csv_path:
            df_combined.to_csv(output_csv_path, index=False)
            print(f"[OK] Batch predictions successfully exported to: {output_csv_path}")
        return df_combined


def run_unit_tests():
    print("=======================================================")
    print("RUNNING AUTOMATED CLINICAL INFERENCE SELF-TEST SUITE")
    print("=======================================================")
    engine = CADInferenceEngine()

    test_cases = [
        {
            "name": "Case 1: Low-Risk Healthy Patient",
            "data": {
                'age': 45.0, 'sex': 0, 'cp': 2, 'trestbps': 118.0, 'chol': 180.0,
                'fbs': 0, 'restecg': 0, 'thalach': 175.0, 'exang': 0, 'oldpeak': 0.0,
                'slope': 1, 'ca': 0.0, 'thal': 3
            },
            "expected_tier": 0
        },
        {
            "name": "Case 2: Intermediate CAD (Medical Therapy Candidate)",
            "data": {
                'age': 58.0, 'sex': 1, 'cp': 4, 'trestbps': 136.0, 'chol': 250.0,
                'fbs': 0, 'restecg': 1, 'thalach': 140.0, 'exang': 1, 'oldpeak': 1.6,
                'slope': 2, 'ca': 1.0, 'thal': 7
            },
            "expected_tier": 1
        },
        {
            "name": "Case 3: Critical Severe CAD (Urgent Surgical Candidate)",
            "data": {
                'age': 67.0, 'sex': 1, 'cp': 4, 'trestbps': 160.0, 'chol': 286.0,
                'fbs': 1, 'restecg': 2, 'thalach': 108.0, 'exang': 1, 'oldpeak': 3.6,
                'slope': 3, 'ca': 3.0, 'thal': 7
            },
            "expected_tier": 2
        }
    ]

    all_passed = True
    for tc in test_cases:
        print(f"\nEvaluating {tc['name']}...")
        res = engine.predict_patient(tc['data'])
        print(f"  Screening Status:  {res['screening_status']}")
        print(f"  CAD Probability:   {res['cad_probability'] * 100:.1f}%")
        print(f"  Care Tier:         {res['tier_description']}")
        print(f"  Action Plan:       {res['clinical_action_plan']}")
        
        passed = (res['recommended_care_tier'] == tc['expected_tier'])
        status_str = "PASS" if passed else "FAIL"
        print(f"  Test Assertion:    [{status_str}] (Expected Tier {tc['expected_tier']}, Got {res['recommended_care_tier']})")
        if not passed:
            all_passed = False

    print("\n=======================================================")
    if all_passed:
        print("[SUCCESS] ALL CLINICAL UNIT TESTS PASSED WITH 100% ACCURACY!")
    else:
        print("[WARNING] One or more clinical test cases diverged from expected tiers.")
    print("=======================================================")
    return all_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coronary Artery Disease Dual-Stage Inference CLI")
    parser.add_argument('--test', action='store_true', help="Run automated diagnostic self-test suite")
    parser.add_argument('--batch', type=str, help="Path to input CSV file for batch processing")
    parser.add_argument('--output', type=str, default='batch_predictions.csv', help="Path to output CSV file for batch results")
    
    # Biomarker arguments for single patient CLI evaluation
    parser.add_argument('--age', type=float, help="Age in years (e.g. 58.0)")
    parser.add_argument('--sex', type=int, choices=[0, 1], help="1=Male, 0=Female")
    parser.add_argument('--cp', type=int, choices=[1, 2, 3, 4], help="Chest Pain: 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic")
    parser.add_argument('--trestbps', type=float, help="Resting BP in mm Hg (e.g. 130.0)")
    parser.add_argument('--chol', type=float, help="Serum cholesterol in mg/dl (e.g. 240.0)")
    parser.add_argument('--fbs', type=int, choices=[0, 1], help="Fasting blood sugar > 120 mg/dl (1=True, 0=False)")
    parser.add_argument('--restecg', type=int, choices=[0, 1, 2], help="Resting ECG: 0=Normal, 1=ST abnormality, 2=LV hypertrophy")
    parser.add_argument('--thalach', type=float, help="Max heart rate achieved (e.g. 150.0)")
    parser.add_argument('--exang', type=int, choices=[0, 1], help="Exercise-induced angina (1=Yes, 0=No)")
    parser.add_argument('--oldpeak', type=float, help="ST depression in mm (e.g. 1.5)")
    parser.add_argument('--slope', type=int, choices=[1, 2, 3], help="Peak ST slope: 1=Upsloping, 2=Flat, 3=Downsloping")
    parser.add_argument('--ca', type=float, default=0.0, help="Number of major vessels (0-3) colored by fluoroscopy")
    parser.add_argument('--thal', type=int, choices=[3, 6, 7], default=3, help="Thallium scintigraphy: 3=Normal, 6=Fixed, 7=Reversible")
    parser.add_argument('--threshold', type=float, default=None, help="Custom decision threshold (e.g. 0.40)")

    args = parser.parse_args()

    if args.test:
        run_unit_tests()
    elif args.batch:
        engine = CADInferenceEngine()
        engine.predict_batch(args.batch, args.output)
    elif args.age is not None:
        engine = CADInferenceEngine()
        patient_record = {
            'age': args.age, 'sex': args.sex, 'cp': args.cp, 'trestbps': args.trestbps,
            'chol': args.chol, 'fbs': args.fbs, 'restecg': args.restecg, 'thalach': args.thalach,
            'exang': args.exang, 'oldpeak': args.oldpeak, 'slope': args.slope,
            'ca': args.ca, 'thal': args.thal
        }
        res = engine.predict_patient(patient_record, args.threshold)
        print("\n=== CLINICAL DIAGNOSTIC REPORT ===")
        print(f"Screening Status:       {res['screening_status']}")
        print(f"CAD Probability:        {res['cad_probability'] * 100:.2f}% (Threshold: {res['decision_threshold_applied']})")
        print(f"Recommended Care Tier:  {res['tier_description']}")
        print(f"Action Recommendation:  {res['clinical_action_plan']}")
        print("Stage 2 Staging Probabilities:")
        for t_name, t_p in res['stage2_tier_probabilities'].items():
            print(f"  - {t_name}: {t_p * 100:.2f}%")
        print("==================================")
    else:
        # Default behavior: run unit test demo
        print("No clinical arguments provided. Executing automated test suite by default:\n")
        run_unit_tests()
