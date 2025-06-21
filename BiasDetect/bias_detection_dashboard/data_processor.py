import pandas as pd
import streamlit as st

class DataProcessor:
    def load_data(self, file):
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}", icon="❌")
            return None

    def clean_data(self, df):
        issues = []
        original_df = df.copy()
        
        # Handle missing values
        if df.isna().sum().sum() > 0:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            issues.append("Missing values filled (numerical: mean, categorical: mode)")

        # Remove duplicates
        if df.duplicated().sum() > 0:
            df.drop_duplicates(inplace=True)
            issues.append(f"Removed {original_df.duplicated().sum()} duplicate rows")

        return issues

    def detect_sensitive_columns(self, df):
        sensitive_keywords = ['gender', 'age', 'race', 'ethnicity', 'religion', 'disability']
        sensitive_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in sensitive_keywords)]
        return sensitive_cols