import os
import sys
import json
import pandas as pd
import numpy as np

# Windows asyncio ProactorEventLoop fix to suppress harmless WinError 10054 on tab disconnect
if sys.platform == 'win32':
    import asyncio
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
        def _silent_call_connection_lost(self, exc=None):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                pass
        _ProactorBasePipeTransport._call_connection_lost = _silent_call_connection_lost
    except Exception:
        pass

import streamlit as st

# Ensure deployment package is in path
DEPLOYMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deployment')
if DEPLOYMENT_DIR not in sys.path:
    sys.path.append(DEPLOYMENT_DIR)

from predict_cad import CADInferenceEngine

# Page configuration
st.set_page_config(
    page_title="CardioScan AI | CAD Triage & Staging System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .status-positive {
        background-color: #FFF5F5;
        border-left: 5px solid #E53E3E;
        padding: 14px;
        border-radius: 6px;
    }
    .status-negative {
        background-color: #F0FFF4;
        border-left: 5px solid #38A169;
        padding: 14px;
        border-radius: 6px;
    }
    .tier-surgical {
        background-color: #742A2A;
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .tier-medical {
        background-color: #C05621;
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .tier-healthy {
        background-color: #276749;
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    models_dir = os.path.join(DEPLOYMENT_DIR, 'models')
    return CADInferenceEngine(model_dir=models_dir)

engine = load_engine()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
    st.markdown("### **CardioScan AI**")
    st.markdown("**Dual-Stage CAD Diagnostic Engine**")
    st.caption("Cleveland Heart Disease Clinical Suite")
    st.divider()

    st.markdown("#### **Engine Calibration**")
    st.markdown(f"• **Decision Cutoff ($\\tau^*$):** `{engine.threshold:.3f}`")
    st.markdown("• **Screening Model:** Calibrated MLP")
    st.markdown("• **Staging Model:** Balanced Random Forest")
    st.markdown("• **Clinical Sensitivity:** `92.86%`")
    st.divider()

    st.markdown("#### **Operating Modes**")
    app_mode = st.radio("Select Workflow:", ["Single Patient Triage", "Cohort Batch Scoring", "Benchmark Architecture Matrix"])

    st.divider()
    st.caption("AIML Clinical Diagnostics Lab • Zero-Leakage Pipeline v1.0")


if app_mode == "Single Patient Triage":
    st.markdown('<p class="main-header">Coronary Artery Disease (CAD) Triage & Care Staging</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated screening and actionable clinical care staging for ischemic heart disease.</p>', unsafe_allow_html=True)

    # Preset selection
    col_preset, col_info = st.columns([2, 1])
    with col_preset:
        preset = st.selectbox(
            "📋 Quick-Fill Patient Profile:",
            [
                "Custom Patient (Manual Entry)",
                "Preset 1: Healthy Screening Patient (Clean Coronaries)",
                "Preset 2: Intermediate CAD Patient (Medical Therapy Candidate)",
                "Preset 3: Critical CAD Patient (Urgent Surgical Candidate)"
            ]
        )

    # Preset values dictionary
    if preset == "Preset 1: Healthy Screening Patient (Clean Coronaries)":
        d_age, d_sex, d_cp, d_trestbps, d_chol = 45, "Female", "Atypical Angina (2)", 118, 180
        d_fbs, d_restecg, d_thalach, d_exang = "No (<= 120 mg/dl)", "Normal (0)", 175, "No (0)"
        d_oldpeak, d_slope, d_ca, d_thal = 0.0, "Upsloping (1)", 0, "Normal (3)"
    elif preset == "Preset 2: Intermediate CAD Patient (Medical Therapy Candidate)":
        d_age, d_sex, d_cp, d_trestbps, d_chol = 58, "Male", "Asymptomatic (4)", 136, 250
        d_fbs, d_restecg, d_thalach, d_exang = "No (<= 120 mg/dl)", "ST-T Abnormality (1)", 140, "Yes (1)"
        d_oldpeak, d_slope, d_ca, d_thal = 1.6, "Flat (2)", 1, "Reversible Defect (7)"
    elif preset == "Preset 3: Critical CAD Patient (Urgent Surgical Candidate)":
        d_age, d_sex, d_cp, d_trestbps, d_chol = 67, "Male", "Asymptomatic (4)", 160, 286
        d_fbs, d_restecg, d_thalach, d_exang = "Yes (> 120 mg/dl)", "LV Hypertrophy (2)", 108, "Yes (1)"
        d_oldpeak, d_slope, d_ca, d_thal = 3.6, "Downsloping (3)", 3, "Reversible Defect (7)"
    else:
        d_age, d_sex, d_cp, d_trestbps, d_chol = 55, "Male", "Typical Angina (1)", 130, 240
        d_fbs, d_restecg, d_thalach, d_exang = "No (<= 120 mg/dl)", "Normal (0)", 150, "No (0)"
        d_oldpeak, d_slope, d_ca, d_thal = 1.0, "Flat (2)", 0, "Normal (3)"

    st.markdown("#### **Enter Patient Biomarkers**")
    
    with st.form("patient_entry_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### **1. Demographics & Vitals**")
            age = st.slider("Patient Age (years):", 20, 90, d_age)
            sex = st.selectbox("Biological Sex:", ["Female", "Male"], index=0 if d_sex == "Female" else 1)
            trestbps = st.slider("Resting Blood Pressure (mm Hg):", 80, 220, d_trestbps)
            chol = st.slider("Serum Cholesterol (mg/dl):", 100, 550, d_chol)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl:", ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"], index=0 if "No" in d_fbs else 1)

        with col2:
            st.markdown("##### **2. Symptoms & Exercise Stress**")
            cp_options = ["Typical Angina (1)", "Atypical Angina (2)", "Non-Anginal Pain (3)", "Asymptomatic (4)"]
            cp_idx = [i for i, c in enumerate(cp_options) if d_cp in c][0]
            cp = st.selectbox("Chest Pain Type:", cp_options, index=cp_idx)
            
            exang = st.selectbox("Exercise-Induced Angina:", ["No (0)", "Yes (1)"], index=0 if "No" in d_exang else 1)
            thalach = st.slider("Maximum Heart Rate Achieved (bpm):", 60, 220, d_thalach)
            
            slope_options = ["Upsloping (1)", "Flat (2)", "Downsloping (3)"]
            slope_idx = [i for i, s in enumerate(slope_options) if d_slope in s][0]
            slope = st.selectbox("Peak Exercise ST Slope:", slope_options, index=slope_idx)
            
            oldpeak = st.slider("ST Depression (mm relative to rest):", 0.0, 6.2, d_oldpeak, step=0.1)

        with col3:
            st.markdown("##### **3. Diagnostic Imaging & ECG**")
            restecg_options = ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"]
            restecg_idx = [i for i, r in enumerate(restecg_options) if d_restecg in r][0]
            restecg = st.selectbox("Resting ECG Results:", restecg_options, index=restecg_idx)
            
            ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca):", [0, 1, 2, 3], index=d_ca)
            
            thal_options = ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"]
            thal_idx = [i for i, t in enumerate(thal_options) if d_thal in t][0]
            thal = st.selectbox("Thallium Scintigraphy (thal):", thal_options, index=thal_idx)

        submit_btn = st.form_submit_button("⚡ Run Clinical Diagnostic Assessment", type="primary", width="stretch")

    if submit_btn:
        # Encode inputs
        sex_val = 1 if sex == "Male" else 0
        cp_val = int(cp.split('(')[1][0])
        fbs_val = 1 if "Yes" in fbs else 0
        restecg_val = int(restecg.split('(')[1][0])
        exang_val = 1 if "Yes" in exang else 0
        slope_val = int(slope.split('(')[1][0])
        ca_val = float(ca)
        thal_val = int(thal.split('(')[1][0])

        patient_dict = {
            'age': float(age), 'sex': sex_val, 'cp': cp_val, 'trestbps': float(trestbps),
            'chol': float(chol), 'fbs': fbs_val, 'restecg': restecg_val, 'thalach': float(thalach),
            'exang': exang_val, 'oldpeak': float(oldpeak), 'slope': slope_val,
            'ca': ca_val, 'thal': thal_val
        }

        # Inference
        result = engine.predict_patient(patient_dict)

        st.divider()
        st.markdown("### 📊 Diagnostic Evaluation Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### **Stage 1: Binary Screening**")
            p_val = result['cad_probability']
            cutoff = result['decision_threshold_applied']

            if "Positive" in result['screening_status']:
                st.markdown(f"""
                <div class="status-positive">
                    <h3 style="color:#C53030; margin:0;">🚨 POSITIVE: CAD DETECTED</h3>
                    <p style="margin:5px 0 0 0; color:#4A5568;">Significant Coronary Artery Disease (&ge; 50% stenosis in &ge; 1 major vessel).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-negative">
                    <h3 style="color:#22543D; margin:0;">✅ NEGATIVE: CLEAN CORONARIES</h3>
                    <p style="margin:5px 0 0 0; color:#4A5568;">No hemodynamically significant stenosis (&lt; 50% stenosis).</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Predicted CAD Probability:** `{p_val * 100:.1f}%` (Calibrated Operating Cutoff: `{cutoff:.3f}`)")
            st.progress(min(1.0, float(p_val)))

            st.caption("Engine: Lightweight Neural Network (MLP) with Youden's J Threshold Optimization")

        with res_col2:
            st.markdown("#### **Stage 2: Clinical Care Staging**")
            tier_num = result['recommended_care_tier']

            if tier_num == 0:
                badge_html = '<span class="tier-healthy">TIER 0: LOW RISK / ROUTINE PREVENTION</span>'
            elif tier_num == 1:
                badge_html = '<span class="tier-medical">TIER 1: MODERATE CAD / MEDICAL THERAPY</span>'
            else:
                badge_html = '<span class="tier-surgical">TIER 2: CRITICAL CAD / SURGICAL PCI-CABG</span>'

            st.markdown(f"**Recommended Pathway:** {badge_html}", unsafe_allow_html=True)
            st.write("")

            probs = result['stage2_tier_probabilities']
            df_probs = pd.DataFrame({
                'Care Staging Tier': list(probs.keys()),
                'Probability': list(probs.values())
            })
            st.bar_chart(df_probs.set_index('Care Staging Tier'), color="#2B6CB0")

        # Action plan box
        st.markdown("#### **Recommended Clinical Action Plan**")
        st.info(f"📋 **Action Guideline:** {result['clinical_action_plan']}")


elif app_mode == "Cohort Batch Scoring":
    st.markdown('<p class="main-header">Cohort Batch Scoring via CSV</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a CSV file containing patient records to process entire clinical cohorts automatically.</p>', unsafe_allow_html=True)

    sample_csv = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n" \
                 "45,0,2,118,180,0,0,175,0,0.0,1,0,3\n" \
                 "58,1,4,136,250,0,1,140,1,1.6,2,1,7\n" \
                 "67,1,4,160,286,1,2,108,1,3.6,3,3,7\n"

    st.download_button(
        label="📥 Download Sample Clinical Template CSV",
        data=sample_csv,
        file_name="sample_patient_cohort.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df_input)} patient records.")

            if st.button("🚀 Process Batch Cohort", type="primary"):
                with st.spinner("Processing patients through dual-stage diagnostic pipeline..."):
                    # Save temporary file
                    tmp_in = "temp_batch_in.csv"
                    tmp_out = "temp_batch_out.csv"
                    df_input.to_csv(tmp_in, index=False)
                    
                    df_scored = engine.predict_batch(tmp_in, tmp_out)
                    
                    st.markdown("### Scored Cohort Results")
                    st.dataframe(df_scored, width="stretch")

                    # Summary Metrics
                    total_pts = len(df_scored)
                    cad_pos = sum(df_scored['screening_status'] == 'Positive (CAD Detected)')
                    surgical_pts = sum(df_scored['recommended_tier'] == 2)

                    s_col1, s_col2, s_col3 = st.columns(3)
                    s_col1.metric("Total Patients", total_pts)
                    s_col2.metric("CAD Detected (Stage 1)", f"{cad_pos} ({cad_pos/total_pts*100:.1f}%)")
                    s_col3.metric("Urgent Surgical Tier 2", f"{surgical_pts} ({surgical_pts/total_pts*100:.1f}%)")

                    # Download button
                    csv_data = df_scored.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Cohort CSV",
                        data=csv_data,
                        file_name="cardioscan_scored_cohort.csv",
                        mime="text/csv"
                    )

                    # Clean up
                    for p in [tmp_in, tmp_out]:
                        if os.path.exists(p):
                            os.remove(p)

        except Exception as e:
            st.error(f"Error processing CSV: {e}")


else: # Benchmark Architecture Matrix
    st.markdown('<p class="main-header">Benchmark Architecture Comparison Matrix</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive empirical results across all 5 architectures and 4 clinical problem formulations.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Category 1: 2-Tier Binary",
        "Category 2: 3-Tier Staging",
        "Category 3: Authentic 5-Class",
        "Category 4: 5-Class SMOTE"
    ])

    with tab1:
        st.markdown("#### **Category 1: 2-Tier Binary CAD Detection (Healthy vs. CAD &ge; 50% Stenosis)**")
        data_2tier = {
            'Architecture': ['Neural Network (MLP) 🏆', 'Logistic Regression', 'Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine'],
            'Optimal Hyperparameters': ["hidden=(64, 32, 16), act='tanh'", "C=50.0, class_weight='balanced'", "criterion='entropy', depth=12", "k=11, metric='manhattan'", "C=5.0, kernel='rbf'"],
            'Test Accuracy': ['90.16%', '86.89%', '86.89%', '86.89%', '80.33%'],
            'Sensitivity': ['92.86%', '92.86%', '92.86%', '85.71%', '85.71%'],
            'Specificity': ['87.88%', '81.82%', '81.82%', '87.88%', '75.76%'],
            'F1-Score': ['0.8966', '0.8667', '0.8667', '0.8571', '0.8000'],
            'ROC-AUC': ['0.9448', '0.9491', '0.9459', '0.9286', '0.9286']
        }
        st.dataframe(pd.DataFrame(data_2tier), width="stretch")

    with tab2:
        st.markdown("#### **Category 2: 3-Tier Clinically Actionable Care Staging (None vs. Medical vs. Surgical)**")
        data_3tier = {
            'Architecture': ['Random Forest 🏆', 'Logistic Regression', 'Support Vector Machine', 'Neural Network (MLP)', 'K-Nearest Neighbors'],
            'Optimal Hyperparameters': ["criterion='entropy', depth=8, n_est=50", "C=10.0, solver='lbfgs'", "C=50.0, kernel='rbf'", "hidden=(64, 32, 16), act='tanh'", "k=21, metric='manhattan'"],
            'Test Accuracy': ['70.49%', '68.85%', '67.21%', '65.57%', '62.30%'],
            'Tier 0 Recall': ['90.9%', '84.8%', '84.8%', '78.8%', '84.8%'],
            'Tier 1 Recall': ['55.6%', '61.1%', '50.0%', '77.8%', '55.6%'],
            'Tier 2 Recall': ['30.0%', '30.0%', '40.0% 🏆', '0.0%', '0.0%'],
            'F1-Weighted': ['0.6914', '0.6800', '0.6659', '0.6117', '0.5830'],
            'Macro ROC-AUC': ['0.8346', '0.8168', '0.8308', '0.8227', '0.7866']
        }
        st.dataframe(pd.DataFrame(data_3tier), width="stretch")

    with tab3:
        st.markdown("#### **Category 3: Authentic 5-Class Disease Distribution (Without SMOTE)**")
        data_no_smote = {
            'Architecture': ['Neural Network (MLP) 🏆', 'Logistic Regression', 'Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine'],
            'Optimal Hyperparameters': ["hidden=(128, 64), act='tanh'", "C=10.0, solver='lbfgs'", "class_weight='balanced', depth=4", "k=15, metric='euclidean'", "C=10.0, kernel='linear'"],
            'Test Accuracy': ['62.30%', '57.38%', '55.74%', '55.74%', '54.10%'],
            'Precision (Weighted)': ['0.5597', '0.5328', '0.5954', '0.4538', '0.5967'],
            'Recall (Weighted)': ['0.6230', '0.5738', '0.5574', '0.5574', '0.5410'],
            'F1-Weighted': ['0.5825', '0.5515', '0.5735', '0.5001', '0.5645'],
            'Macro ROC-AUC': ['0.8477', '0.8128', '0.7516', '0.6764', '0.8246']
        }
        st.dataframe(pd.DataFrame(data_no_smote), width="stretch")

    with tab4:
        st.markdown("#### **Category 4: Multi-Class CAD Severity With SMOTE (Synthetic Oversampling)**")
        data_smote = {
            'Architecture': ['Logistic Regression 🏆', 'Random Forest', 'Neural Network (MLP)', 'Support Vector Machine', 'K-Nearest Neighbors'],
            'Optimal Hyperparameters': ["C=10.0, solver='newton-cg'", "depth=12, leaf=1, n_est=200", "hidden=(128, 64), act='relu'", "C=50.0, kernel='rbf'", "k=3, metric='manhattan'"],
            'Test Accuracy': ['57.38%', '54.10%', '54.10%', '50.82%', '47.54%'],
            'Precision (Weighted)': ['0.5541', '0.5103', '0.5562', '0.4956', '0.5138'],
            'Recall (Weighted)': ['0.5738', '0.5410', '0.5410', '0.5082', '0.4754'],
            'F1-Weighted': ['0.5622', '0.5228', '0.5457', '0.5004', '0.4855'],
            'Macro ROC-AUC': ['0.7724', '0.7658', '0.7650', '0.6717', '0.6038']
        }
        st.dataframe(pd.DataFrame(data_smote), width="stretch")
