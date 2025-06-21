import streamlit as st
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

class PDFGenerator:
    def __init__(self):
        self.df = st.session_state.get('cleaned_df')
        self.fairness_metrics = st.session_state.get('fairness_metrics', {})
        self.pii_columns = st.session_state.get('pii_columns', [])
        self.ml_score = st.session_state.get('ml_score', 0)
        self.model = None
        self.label_encoders = {}

    def preprocess_data(self, df):
        """Preprocess the data for ML model training or prediction."""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['EducationLevel', 'University']
        for col in categorical_cols:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Select features for the model
        feature_cols = ['YearsExperience', 'EducationLevel', 'University', 'GapYears']
        X = df_processed[feature_cols]
        return X

    def generate_initial_labels(self, df):
        """Generate initial shortlisting labels using the rule-based approach."""
        weights = {
            'YearsExperience': 0.4,
            'EducationLevel': 0.3,
            'University': 0.2,
            'GapYears': -0.1
        }

        # Normalize features
        df['ExperienceScore'] = df['YearsExperience'] / df['YearsExperience'].max()
        df['EducationScore'] = df['EducationLevel'].map({'Bachelor': 0.5, 'Master': 0.75, 'PhD': 1.0}).fillna(0.5)
        df['UniversityScore'] = df['University'].map({'MIT': 1.0, 'Stanford': 1.0, 'Harvard': 0.9, 'Yale': 0.9, 'Berkeley': 0.8}).fillna(0.8)
        df['GapPenalty'] = df['GapYears'].apply(lambda x: -0.1 * x if x > 0 else 0)

        # Calculate total score
        df['Score'] = (
            weights['YearsExperience'] * df['ExperienceScore'] +
            weights['EducationLevel'] * df['EducationScore'] +
            weights['University'] * df['UniversityScore'] +
            weights['GapYears'] * df['GapPenalty']
        )
        df['Score'] = df['Score'].clip(0, 1)

        # Shortlist candidates with scores above the median
        threshold = df['Score'].quantile(0.5)
        df['shortlisted'] = (df['Score'] >= threshold).astype(int)

        # Drop temporary columns
        df = df.drop(columns=['ExperienceScore', 'EducationScore', 'UniversityScore', 'GapPenalty', 'Score'])
        return df

    def train_model(self, X, y):
        """Train a logistic regression model."""
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, y)

    def shortlist_candidates(self):
        """Use an ML model to shortlist candidates."""
        df = self.df.copy()
        
        # If 'shortlisted' column already exists, validate and return
        if 'shortlisted' in df.columns:
            if not df['shortlisted'].isin([0, 1]).all():
                st.error("The 'shortlisted' column must contain only binary values (0 or 1).", icon="❌")
                return df
            return df

        # Step 1: Generate initial labels using the rule-based approach for training
        df_with_labels = self.generate_initial_labels(df.copy())
        y = df_with_labels['shortlisted']

        # Step 2: Preprocess the data
        X = self.preprocess_data(df)

        # Step 3: Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Train the model
        self.train_model(X_train, y_train)

        # Step 5: Predict shortlisting for all candidates
        predictions = self.model.predict(X)
        df['shortlisted'] = predictions

        return df

    def compute_disparate_impact_ratio(self, sensitive_col, target_col):
        if self.df.empty or target_col not in self.df.columns:
            return 0.0
        selection_rates = self.df.groupby(sensitive_col)[target_col].mean()
        if len(selection_rates) < 2:
            return 1.0
        min_rate = selection_rates.min()
        max_rate = selection_rates.max()
        return min_rate / max_rate if max_rate > 0 else 0.0

    def check_feature_correlation(self, sensitive_col):
        correlations = []
        for col in self.df.columns:
            if col == sensitive_col or col in ['shortlisted', 'name', 'email', 'phone']:
                continue
            if self.df[col].dtype == 'object' or self.df[sensitive_col].dtype == 'object':
                contingency_table = pd.crosstab(self.df[sensitive_col], self.df[col])
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value < 0.05:
                    correlations.append((col, p_value))
        return correlations

    def filter_biased_candidates(self):
        if self.df.empty or not self.fairness_metrics:
            return self.df

        shortlisted_df = self.df[self.df['shortlisted'] == 1].copy()
        if shortlisted_df.empty:
            return shortlisted_df

        df_filtered = shortlisted_df
        biased_groups = []

        for sensitive_col, metrics in self.fairness_metrics.items():
            if 'Demographic Parity Difference' in metrics:
                dpd = metrics['Demographic Parity Difference']
                if dpd > 0.1:
                    selection_rates = self.df.groupby(sensitive_col)['shortlisted'].mean()
                    overall_rate = self.df['shortlisted'].mean()
                    for group, rate in selection_rates.items():
                        if abs(rate - overall_rate) > 0.1:
                            biased_groups.append((sensitive_col, group))

        for sensitive_col, group in biased_groups:
            group_df = df_filtered[(df_filtered[sensitive_col] == group)]
            if not group_df.empty:
                remove_count = len(group_df) // 2
                if remove_count > 0:
                    remove_indices = group_df.sample(n=remove_count).index
                    df_filtered = df_filtered.drop(remove_indices)

        return df_filtered

    def generate_pdf(self):
        if not all([self.df is not None, self.pii_columns is not None, self.ml_score is not None]):
            st.error("Please complete all analysis steps before generating the report.", icon="❌")
            return None, None

        # Create shortlisted column using the ML model
        self.df = self.shortlist_candidates()

        # Compute fairness metrics
        fairness_metrics = {}
        for sensitive_col in ['Gender']:
            selection_rates = self.df.groupby(sensitive_col)['shortlisted'].mean()
            overall_rate = self.df['shortlisted'].mean()
            dpd = selection_rates.max() - selection_rates.min()
            fairness_metrics[sensitive_col] = {
                'Demographic Parity Difference': dpd,
                'Selection Rate by Group': selection_rates.to_dict()
            }
        self.fairness_metrics = fairness_metrics

        # Filter biased candidates
        accepted_candidates = self.filter_biased_candidates()

        # Compute Disparate Impact Ratio for Gender
        dir_gender = self.compute_disparate_impact_ratio('Gender', 'shortlisted')

        # Check for feature correlations with Gender
        correlated_features = self.check_feature_correlation('Gender')

        # Calculate gender distribution and shortlisting percentages
        gender_dist = self.df['Gender'].value_counts(normalize=True) * 100
        shortlisted_dist = self.df[self.df['shortlisted'] == 1]['Gender'].value_counts(normalize=True) * 100
        accepted_dist = accepted_candidates['Gender'].value_counts(normalize=True) * 100

        # Calculate percentage of shortlisted candidates within each gender group
        shortlisted_percentages = self.df.groupby('Gender')['shortlisted'].mean() * 100
        accepted_percentages = (accepted_candidates['Gender'].value_counts() / self.df['Gender'].value_counts()) * 100

        # Create a buffer for PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Progress bar (5 steps)
        progress_bar = st.progress(0)
        steps = 5
        step = 0

        # PDF: Title
        elements.append(Paragraph("Bias Detection Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Step 1: Dataset Overview
        step += 1
        progress_bar.progress(step / steps)
        elements.append(Paragraph("Dataset Overview", styles['Heading2']))
        elements.append(Paragraph(f"Dataset Shape: {self.df.shape}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Statistical Summary", styles['Heading3']))
        stats = self.df.describe().T
        stats_data = [[''] + list(stats.columns)]
        for index, row in stats.iterrows():
            stats_data.append([index] + [f"{val:.2f}" for val in row])
        table = Table(stats_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Step 2: Distribution Analysis
        step += 1
        progress_bar.progress(step / steps)
        elements.append(Paragraph("Distribution Analysis", styles['Heading2']))
        elements.append(Paragraph("Gender Distribution in Applicants:", styles['Heading3']))
        for gender, percentage in gender_dist.items():
            elements.append(Paragraph(f"Applicants ({gender}): {percentage:.1f}%", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Gender Distribution in Initially Shortlisted Candidates:", styles['Heading3']))
        for gender, percentage in shortlisted_dist.items():
            elements.append(Paragraph(f"Shortlisted ({gender}): {percentage:.1f}%", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Percentage of Candidates Shortlisted Within Each Gender Group (Before Bias Filtering):", styles['Heading3']))
        for gender, percentage in shortlisted_percentages.items():
            elements.append(Paragraph(f"{gender}: {percentage:.1f}% of {gender} candidates shortlisted", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Gender Distribution in Accepted Candidates (After Bias Filtering):", styles['Heading3']))
        for gender, percentage in accepted_dist.items():
            elements.append(Paragraph(f"Accepted ({gender}): {percentage:.1f}%", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Percentage of Candidates Accepted Within Each Gender Group (After Bias Filtering):", styles['Heading3']))
        for gender, percentage in accepted_percentages.items():
            elements.append(Paragraph(f"{gender}: {percentage:.1f}% of {gender} candidates accepted", styles['Normal']))
        elements.append(Spacer(1, 12))

        # Step 3: Fairness Metrics
        step += 1
        progress_bar.progress(step / steps)
        elements.append(Paragraph("Fairness Metrics", styles['Heading2']))
        elements.append(Paragraph(f"Disparate Impact Ratio (Gender): {dir_gender:.2f} (Threshold: 0.8)", styles['Normal']))
        elements.append(Paragraph("Note: DIR < 0.8 indicates significant bias (EEOC standard).", styles['Normal']))
        elements.append(Spacer(1, 12))

        for sensitive_col, metrics in list(self.fairness_metrics.items())[:2]:
            elements.append(Paragraph(f"Fairness Metrics for {sensitive_col}", styles['Heading3']))
            metric_data = [['Metric', 'Value']]
            for metric, value in metrics.items():
                if metric != 'Selection Rate by Group':
                    metric_data.append([metric, f"{value:.3f}"])
            table = Table(metric_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

        # Step 4: Feature Contribution Check
        step += 1
        progress_bar.progress(step / steps)
        elements.append(Paragraph("Feature Contribution Check", styles['Heading2']))
        if correlated_features:
            elements.append(Paragraph("Features potentially correlated with Gender (p-value < 0.05):", styles['Normal']))
            for feature, p_value in correlated_features:
                elements.append(Paragraph(f"- {feature} (p-value: {p_value:.3f})", styles['Normal']))
            elements.append(Paragraph("Recommendation: These features may act as proxies for gender, contributing to bias.", styles['Normal']))
        else:
            elements.append(Paragraph("No significant correlations with Gender detected.", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Privacy Check", styles['Heading2']))
        if self.pii_columns:
            for col in self.pii_columns:
                elements.append(Paragraph(f"- {col}: Consider removing or anonymizing.", styles['Normal']))
        else:
            elements.append(Paragraph("No PII columns detected.", styles['Normal']))
        elements.append(Spacer(1, 12))

        # Step 5: Accepted Candidates Summary
        step += 1
        progress_bar.progress(step / steps)
        elements.append(Paragraph("Accepted Candidates Summary", styles['Heading2']))
        elements.append(Paragraph(f"Total Shortlisted Candidates (Before Bias Filtering): {len(self.df[self.df['shortlisted'] == 1])}", styles['Normal']))
        elements.append(Paragraph(f"Total Accepted Candidates (After Bias Filtering): {len(accepted_candidates)}", styles['Normal']))
        elements.append(Paragraph("Candidates were filtered based on fairness metrics to reduce bias.", styles['Normal']))
        elements.append(Spacer(1, 12))

        if 'name' in accepted_candidates.columns:
            elements.append(Paragraph("Accepted Candidates:", styles['Heading3']))
            accepted_names = accepted_candidates['name'].head(50).tolist()
            if len(accepted_candidates) > 50:
                accepted_names.append("... (and more)")
            names_text = ", ".join(accepted_names)
            elements.append(Paragraph(names_text, styles['Normal']))
        else:
            elements.append(Paragraph("No 'name' column found in the dataset.", styles['Normal']))

        # Build PDF
        pdf_file = "bias_detection_report.pdf"
        try:
            doc.build(elements)
            pdf_data = buffer.getvalue()
            buffer.close()
            with open(pdf_file, "wb") as f:
                f.write(pdf_data)
            st.info(f"PDF report saved as '{pdf_file}' in the project directory.", icon="ℹ️")
            return pdf_data, accepted_candidates
        except Exception as e:
            st.error(f"Failed to generate PDF: {str(e)}", icon="❌")
            return None, None

def generate_pdf_report():
    pdf_gen = PDFGenerator()
    pdf_data, accepted_candidates = pdf_gen.generate_pdf()

    if pdf_data and accepted_candidates is not None:
        st.download_button(
            label="Download PDF Report 📄",
            data=pdf_data,
            file_name="bias_detection_report.pdf",
            mime="application/pdf"
        )

        csv_buffer = io.StringIO()
        accepted_candidates.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label="Download Accepted Candidates Dataset 📊",
            data=csv_data,
            file_name="accepted_candidates.csv",
            mime="text/csv"
        )

        st.success("PDF report and accepted candidates dataset generated successfully! Click the buttons to download. ✅")

if __name__ == "__main__":
    st.title("Bias Detection Report Generator")
    generate_pdf_report()