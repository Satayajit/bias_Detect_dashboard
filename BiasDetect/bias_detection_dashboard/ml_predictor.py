import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

class MLPredictor:
    def check_ml_readiness(self, df, sensitive_cols):
        score = 100.0
        issues = []

        # Check for missing values
        missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.1:
            score -= 20
            issues.append("High missing value ratio (>10%)")
        
        # Check data balance for sensitive columns
        for col in sensitive_cols:
            if df[col].nunique() > 1:
                value_counts = df[col].value_counts(normalize=True)
                if value_counts.min() < 0.1:
                    score -= 15
                    issues.append(f"Imbalanced data in {col} (min group < 10%)")
        
        # Check for sufficient data
        if df.shape[0] < 100:
            score -= 10
            issues.append("Dataset too small (<100 rows)")
        
        # Check for numerical stability
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if numeric_cols.any():
            if df[numeric_cols].var().min() < 1e-6:
                score -= 10
                issues.append("Low variance in some numerical columns")

        message = ", ".join(issues) if issues else "No issues detected."
        readiness = score >= 80
        return readiness, message, score

    def predict(self, df, target_col):
        try:
            df_copy = df.copy()
            for col in df_copy.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            
            X = df_copy.drop(columns=[target_col])
            y = df_copy[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            return {"Accuracy": accuracy, "Feature Importance": feature_importance}
        except Exception as e:
            st.error(f"Error during prediction: {e}", icon="❌")
            return None