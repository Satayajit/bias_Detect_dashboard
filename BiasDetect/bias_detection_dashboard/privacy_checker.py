import pandas as pd
import re

class PrivacyChecker:
    def detect_pii(self, df):
        pii_columns = []
        pii_patterns = {
            'email': r'@\S+\.\S+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'name': r'^(name|full_name|first_name|last_name)$'
        }
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if any(pattern in col.lower() for pattern in pii_patterns['name'].split('|')):
                    pii_columns.append(col)
                elif df[col].str.contains(pii_patterns['email'], na=False).any():
                    pii_columns.append(col)
                elif df[col].str.contains(pii_patterns['phone'], na=False).any():
                    pii_columns.append(col)
        
        return pii_columns

    def get_pii_recommendations(self, pii_columns):
        recommendations = []
        for col in pii_columns:
            if 'email' in col.lower():
                recommendations.append(f"Column '{col}' contains email addresses. Consider anonymizing or removing this column.")
            elif 'phone' in col.lower():
                recommendations.append(f"Column '{col}' contains phone numbers. Consider anonymizing or removing this column.")
            elif 'name' in col.lower():
                recommendations.append(f"Column '{col}' contains names. Consider anonymizing or removing this column.")
            else:
                recommendations.append(f"Column '{col}' may contain PII. Review and consider anonymizing or removing this column.")
        return recommendations