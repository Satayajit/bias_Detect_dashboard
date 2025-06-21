from fairlearn.metrics import MetricFrame, demographic_parity_ratio, equalized_odds_ratio
import pandas as pd
import numpy as np
import streamlit as st

class BiasAnalyzer:
    def is_binary(self, series):
        unique_values = series.dropna().unique()
        return len(unique_values) == 2 and set(unique_values).issubset({0, 1})

    def bin_continuous_column(self, series, bins=5):
        """Bin a continuous column into categorical groups."""
        if pd.api.types.is_numeric_dtype(series):
            # Check if the column has enough variation to bin
            if series.nunique() > 1:
                return pd.cut(series, bins=bins, include_lowest=True, labels=[f"Group_{i+1}" for i in range(bins)])
            else:
                raise ValueError(f"Column {series.name} has insufficient variation for binning.")
        return series

    def calculate_fairness_metrics(self, df, sensitive_col, target_col):
        try:
            if sensitive_col not in df.columns or target_col not in df.columns:
                raise ValueError(f"Columns {sensitive_col} or {target_col} not found in dataset.")
            
            # Bin the sensitive column if it's continuous
            sensitive_series = self.bin_continuous_column(df[sensitive_col])
            df = df.copy()
            df[sensitive_col] = sensitive_series

            # Validate inputs
            if df[sensitive_col].nunique() < 2:
                raise ValueError(f"Sensitive column {sensitive_col} has insufficient variation (needs at least 2 unique values).")
            if df[target_col].nunique() < 2:
                raise ValueError(f"Target column {target_col} has insufficient variation (needs at least 2 unique values).")
            if df[sensitive_col].isna().any() or df[target_col].isna().any():
                raise ValueError(f"Columns {sensitive_col} and/or {target_col} contain missing values.")
            if not self.is_binary(df[target_col]):
                raise ValueError(f"Target column {target_col} must be binary (0 or 1) for fairness metrics.")

            metrics = {}
            # Simplified MetricFrame to calculate selection rate only
            gm = MetricFrame(
                metrics={
                    'selection_rate': lambda y_true, y_pred: np.mean(y_pred)
                },
                y_true=df[target_col],
                y_pred=df[target_col],
                sensitive_features=df[sensitive_col]
            )

            # Calculate fairness metrics
            metrics['Disparate Impact'] = demographic_parity_ratio(
                y_true=df[target_col],
                y_pred=df[target_col],
                sensitive_features=df[sensitive_col]
            )
            metrics['Equalized Odds'] = equalized_odds_ratio(
                y_true=df[target_col],
                y_pred=df[target_col],
                sensitive_features=df[sensitive_col]
            )
            metrics['Selection Rate by Group'] = {
                'selection_rate': gm.by_group['selection_rate'].to_dict()
            }
            return metrics
        except Exception as e:
            st.error(f"Error calculating fairness metrics for {sensitive_col}: {str(e)}", icon="❌")
            return {}

    def get_recommendations(self, df, sensitive_cols):
        recommendations = []
        for col in sensitive_cols:
            if df[col].nunique() < 2:
                recommendations.append(f"Column {col} has insufficient variation. Collect more diverse data.")
            if df[col].isna().sum() / len(df) > 0.1:
                recommendations.append(f"Column {col} has >10% missing values. Consider imputing or removing.")
            if df[col].dtype == 'object' and df[col].str.contains(r'@\S+\.\S+', na=False).any():
                recommendations.append(f"Column {col} may contain emails. Remove for privacy.")
        return recommendations

    def mitigate_bias(self, df, sensitive_cols):
        try:
            cleaned_df = df.copy()
            for col in sensitive_cols:
                if col in cleaned_df.columns:
                    group_counts = cleaned_df[col].value_counts()
                    weights = 1 / group_counts
                    cleaned_df['weight'] = cleaned_df[col].map(weights)
                    cleaned_df['weight'] = cleaned_df['weight'] / cleaned_df['weight'].sum()
            return cleaned_df
        except Exception as e:
            st.error(f"Error mitigating bias: {e}", icon="❌")
            return df