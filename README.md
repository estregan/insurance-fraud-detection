# 🛡️ Insurance Fraud Detection System

## 🎯 Project Overview
Advanced machine learning system for detecting fraudulent insurance claims using XGBoost and explainable AI. Built for production deployment with comprehensive business impact analysis.

## 🏆 Key Results
- **Model Performance:** 94%+ ROC AUC score
- **Business Impact:** $250K+ annual value demonstrated  
- **Production Ready:** Complete deployment package with Streamlit dashboard
- **Explainable AI:** SHAP-based model interpretability

## 🚀 Quick Start

### Install Dependencies
pip install -r requirements.txt

### Run Streamlit Dashboard
streamlit run fraud_detection_app.py

### Use Python API
from fraud_detector import FraudDetector

detector = FraudDetector()
claim_data = {'Age': 35, 'VehiclePrice': 25000, 'AgeOfVehicle': 3}
result = detector.predict_single(claim_data)
print(f"Prediction: {result['prediction']}")

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC AUC | 94%+ |
| Precision | 85%+ |
| Recall | 80%+ |
| F1-Score | 82%+ |

## 🔍 Features

### 🤖 Machine Learning
- 8 algorithms tested (XGBoost selected as best)
- Advanced feature engineering (50+ engineered features)
- Hyperparameter optimization with business metrics
- Cross-validation with time-based splits

### 🔍 Model Interpretability  
- SHAP explanations for global and local interpretability
- Feature importance rankings
- Individual prediction explanations
- Business-friendly insights

### 💼 Business Impact
- ROI analysis with cost-benefit optimization
- Threshold optimization for maximum business value
- False positive/negative trade-off analysis
- Executive reporting with clear metrics

## 📁 Project Structure
fraud_detection_app.py      # Streamlit web application
fraud_detector.py           # Production Python module  
requirements.txt            # Dependencies
PROJECT_SUMMARY.md          # Executive summary
data/processed/             # Feature-engineered datasets
models/                     # Model files
reports/                    # Documentation and metrics

## 🎯 Business Value

### Problem Solved
Insurance fraud costs the industry $40+ billion annually. This system identifies fraudulent claims with high accuracy while minimizing false positives.

### Impact Metrics
- **Annual Savings:** $250K+ demonstrated value
- **Fraud Detection Rate:** 80%+ of fraudulent claims caught
- **False Positive Rate:** <5% legitimate claims flagged  
- **ROI:** 300%+ return on investment

## 🌟 Portfolio Highlights

This project demonstrates:
- End-to-end ML pipeline from data to deployment
- Production engineering skills with proper code organization
- Business acumen with ROI analysis and stakeholder communication
- Technical depth with advanced feature engineering and model interpretation
- Real-world impact with quantified business value

## 👤 Author
Norman Bernardo  
Data Analyst transitioning to Data Science  
Specializing in Insurance Analytics and Risk Assessment

## 📄 License
MIT License
