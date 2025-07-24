"""
Fraud Detection Module for Production Deployment
"""
import joblib
import pandas as pd
import numpy as np

class FraudDetector:
    """Production-ready fraud detection class"""
    
    def __init__(self, model_path='./models/fraud_detection_deployment_package.pkl'):
        """Initialize the fraud detector with saved model package"""
        self.package = joblib.load(model_path)
        self.model = self.package['model']
        self.feature_names = self.package['feature_names']
        self.threshold = self.package['threshold']
        self.model_version = self.package['model_version']
    
    def predict_single(self, claim_data):
        """Predict fraud for a single claim"""
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
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    def predict_batch(self, claims_data):
        """Predict fraud for multiple claims"""
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
        """Get top N most important features"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def get_model_info(self):
        """Get model information and performance metrics"""
        return {
            'model_type': self.package['model_type'],
            'version': self.model_version,
            'training_date': self.package['training_date'],
            'threshold': self.threshold,
            'performance_metrics': self.package['performance_metrics'],
            'feature_count': len(self.feature_names)
        }
