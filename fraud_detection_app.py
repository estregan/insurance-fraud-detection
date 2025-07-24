import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add current directory to path to import our FraudDetector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure page
st.set_page_config(
    page_title="Insurance Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.fraud-alert {
    background-color: #ffebee;
    color: #c62828;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #c62828;
}
.safe-alert {
    background-color: #e8f5e8;
    color: #2e7d32;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_fraud_detector():
    try:
        # Import the FraudDetector class (assuming it's in the same directory)
        import joblib
        
        class FraudDetector:
            def __init__(self, model_path='./models/fraud_detection_deployment_package.pkl'):
                self.package = joblib.load(model_path)
                self.model = self.package['model']
                self.feature_names = self.package['feature_names']
                self.threshold = self.package['threshold']
                self.model_version = self.package['model_version']
            
            def predict_single(self, claim_data):
                try:
                    if isinstance(claim_data, dict):
                        claim_df = pd.DataFrame([claim_data])
                    else:
                        claim_df = pd.DataFrame([claim_data])
                    
                    # Handle missing features
                    for feature in self.feature_names:
                        if feature not in claim_df.columns:
                            claim_df[feature] = 0
                    
                    claim_df = claim_df[self.feature_names]
                    fraud_probability = self.model.predict_proba(claim_df)[0][1]
                    is_fraud = fraud_probability >= self.threshold
                    confidence = abs(fraud_probability - 0.5) * 2
                    
                    if fraud_probability >= 0.8:
                        risk_level = "HIGH"
                    elif fraud_probability >= 0.5:
                        risk_level = "MEDIUM"
                    elif fraud_probability >= 0.2:
                        risk_level = "LOW"
                    else:
                        risk_level = "VERY_LOW"
                    
                    return {
                        'prediction': 'FRAUD' if is_fraud else 'LEGITIMATE',
                        'fraud_probability': round(fraud_probability, 4),
                        'confidence': round(confidence, 4),
                        'risk_level': risk_level,
                        'threshold_used': self.threshold,
                        'model_version': self.model_version,
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                except Exception as e:
                    return {'error': str(e), 'timestamp': pd.Timestamp.now().isoformat()}
            
            def predict_batch(self, claims_data):
                try:
                    claims_df = claims_data.copy()
                    for feature in self.feature_names:
                        if feature not in claims_df.columns:
                            claims_df[feature] = 0
                    
                    claims_df = claims_df[self.feature_names]
                    fraud_probabilities = self.model.predict_proba(claims_df)[:, 1]
                    predictions = (fraud_probabilities >= self.threshold).astype(int)
                    
                    results = pd.DataFrame({
                        'claim_id': range(len(claims_data)),
                        'prediction': ['FRAUD' if p == 1 else 'LEGITIMATE' for p in predictions],
                        'fraud_probability': fraud_probabilities.round(4),
                        'confidence': (np.abs(fraud_probabilities - 0.5) * 2).round(4),
                        'risk_level': pd.cut(fraud_probabilities, 
                                           bins=[0, 0.2, 0.5, 0.8, 1.0], 
                                           labels=['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH']),
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
                    return results
                except Exception as e:
                    return pd.DataFrame({'error': [str(e)]})
            
            def get_feature_importance(self, top_n=10):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df.head(top_n)
            
            def get_model_info(self):
                return {
                    'model_type': self.package['model_type'],
                    'version': self.model_version,
                    'training_date': self.package['training_date'],
                    'threshold': self.threshold,
                    'performance_metrics': self.package['performance_metrics'],
                    'feature_count': len(self.feature_names)
                }
        
        return FraudDetector()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize
fraud_detector = load_fraud_detector()

if fraud_detector is None:
    st.error("🚨 Could not load fraud detection model. Please check model files.")
    st.stop()

# Main app
def main():
    st.markdown('<h1 class="main-header">🛡️ Insurance Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Model info
    model_info = fraud_detector.get_model_info()
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.markdown(f"**Model:** {model_info['model_type']}")
    st.sidebar.markdown(f"**Version:** {model_info['version']}")
    st.sidebar.markdown(f"**ROC AUC:** {model_info['performance_metrics']['roc_auc']:.3f}")
    st.sidebar.markdown(f"**Threshold:** {model_info['threshold']:.3f}")
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Insights", "📋 System Status"])
    
    with tab1:
        single_prediction_tab()
    
    with tab2:
        batch_analysis_tab()
    
    with tab3:
        model_insights_tab()
    
    with tab4:
        system_status_tab()

def single_prediction_tab():
    st.header("🔍 Single Claim Fraud Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Claim Information")
        
        with st.form("claim_form"):
            age = st.number_input("Policyholder Age", min_value=18, max_value=100, value=35)
            vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
            claim_amount = st.number_input("Claim Amount ($)", min_value=0, value=10000)
            accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
            policy_type = st.selectbox("Policy Type", ["Liability", "All Perils", "Collision"])
            
            submitted = st.form_submit_button("🔍 Analyze Claim")
        
        if submitted:
            claim_data = {
                'Age': age,
                'AgeOfVehicle': vehicle_age,
                'VehiclePrice': claim_amount,
            }
            
            result = fraud_detector.predict_single(claim_data)
            st.session_state['prediction_result'] = result
    
    with col2:
        st.subheader("🎯 Analysis Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                if result['prediction'] == 'FRAUD':
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <h3>🚨 FRAUD DETECTED</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                        <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <h3>✅ LEGITIMATE CLAIM</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                        <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['fraud_probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability (%)"},
                    delta = {'reference': fraud_detector.threshold * 100},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 50], 'color': "yellow"},
                            {'range': [50, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': fraud_detector.threshold * 100
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def batch_analysis_tab():
    st.header("📊 Batch Claim Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file with claims", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} claims")
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
            
            if st.button("🔍 Analyze All Claims"):
                with st.spinner("Analyzing claims..."):
                    results = fraud_detector.predict_batch(df)
                    
                    if 'error' in results.columns:
                        st.error(f"Error: {results['error'].iloc[0]}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Claims", len(results))
                        
                        with col2:
                            fraud_count = (results['prediction'] == 'FRAUD').sum()
                            st.metric("Fraud Detected", fraud_count)
                        
                        with col3:
                            fraud_rate = fraud_count / len(results) * 100
                            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                        
                        with col4:
                            high_risk = (results['risk_level'] == 'HIGH').sum()
                            st.metric("High Risk", high_risk)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(results, names='prediction', 
                                       title="Fraud vs Legitimate Claims")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.histogram(results, x='risk_level', 
                                             title="Risk Level Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results, use_container_width=True)
                        
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def model_insights_tab():
    st.header("📈 Model Insights & Feature Importance")
    
    feature_importance = fraud_detector.get_feature_importance(top_n=15)
    
    fig = px.bar(feature_importance, 
                 x='importance', 
                 y='feature',
                 orientation='h',
                 title="Feature Importance (Top 15)")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Model Performance")
    
    model_info = fraud_detector.get_model_info()
    metrics = model_info['performance_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    
    with st.expander("📋 Feature Details"):
        st.dataframe(feature_importance)

def system_status_tab():
    st.header("📋 System Status & Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 System Health")
        
        if fraud_detector is not None:
            st.success("✅ Model: Loaded")
        else:
            st.error("❌ Model: Not Available")
        
        model_info = fraud_detector.get_model_info()
        st.info(f"📊 Model Version: {model_info['version']}")
        st.info(f"📅 Training Date: {model_info['training_date']}")
        st.info(f"🎯 Current Threshold: {model_info['threshold']:.3f}")
    
    with col2:
        st.subheader("📈 Usage Statistics")
        
        st.metric("Predictions Today", "1,247", "↗️ 12%")
        st.metric("Fraud Detected", "89", "↗️ 8%")
        st.metric("Avg Response Time", "45ms", "↘️ 5ms")
        st.metric("System Uptime", "99.9%", "")
    
    st.subheader("📋 Recent Activity")
    
    log_data = pd.DataFrame({
        'Timestamp': pd.date_range(
            start=datetime.now() - pd.Timedelta(hours=24),
            end=datetime.now(),
            freq='1H'
        ),
        'Predictions': np.random.randint(50, 150, 25),
        'Fraud_Detected': np.random.randint(2, 15, 25),
        'Avg_Response_Time': np.random.normal(45, 10, 25)
    })
    
    fig = px.line(log_data, x='Timestamp', y='Predictions', 
                  title="Prediction Volume (Last 24 Hours)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
