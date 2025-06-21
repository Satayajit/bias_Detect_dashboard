import streamlit as st
import pandas as pd
from data_processor import DataProcessor

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'sensitive_cols' not in st.session_state:
    st.session_state.sensitive_cols = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'ml_score' not in st.session_state:
    st.session_state.ml_score = 0
if 'pii_columns' not in st.session_state:
    st.session_state.pii_columns = []
if 'fairness_metrics' not in st.session_state:
    st.session_state.fairness_metrics = {}
if 'bias_percentage' not in st.session_state:
    st.session_state.bias_percentage = 0

processor = DataProcessor()

st.markdown("<div class='card slide-in'><h3>1. Upload Dataset</h3></div>", unsafe_allow_html=True)
st.markdown("Upload your dataset to begin the analysis. Supported formats: CSV, Excel (up to 200MB).")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], help="Upload a CSV or Excel file to start the analysis.")

if uploaded_file:
    try:
        st.session_state.df = processor.load_data(uploaded_file)
        if st.session_state.df is not None:
            st.session_state.cleaned_df = st.session_state.df.copy()
            st.session_state.sensitive_cols = processor.detect_sensitive_columns(st.session_state.df)
            
            st.success("Dataset uploaded successfully! 🎉", icon="✅")
            
            # Data Overview
            st.markdown("<div class='card slide-in'><h3>Dataset Overview</h3></div>", unsafe_allow_html=True)
            with st.expander("View Dataset Preview", expanded=True):
                st.dataframe(st.session_state.df.head(), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", st.session_state.df.shape[0], help="Total number of records in the dataset.")
            with col2:
                st.metric("Columns", st.session_state.df.shape[1], help="Total number of features in the dataset.")
            with col3:
                st.metric("Sensitive Columns", len(st.session_state.sensitive_cols), help="Number of detected sensitive features.")

            st.markdown("<div class='section-title'>Detected Sensitive Columns</div>", unsafe_allow_html=True)
            st.write(st.session_state.sensitive_cols or "None detected", help="These columns may contain sensitive information like gender, age, etc.")

            # Data Cleaning
            st.markdown("<div class='card slide-in'><h3>Data Cleaning</h3></div>", unsafe_allow_html=True)
            cleaning_issues = processor.clean_data(st.session_state.cleaned_df)
            if cleaning_issues:
                st.warning("Issues detected and fixed: " + "; ".join(cleaning_issues), icon="🛠️")
                st.download_button(
                    label="Download Cleaned Dataset",
                    data=st.session_state.cleaned_df.to_csv(index=False),
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    help="Download the cleaned dataset after handling missing values and other issues."
                )
            else:
                st.info("No cleaning required; dataset is clean.", icon="ℹ️")

            st.session_state.analysis_done = True
            st.markdown("<div class='success bounce'>Proceed to the next steps using the sidebar navigation! 🚀</div>", unsafe_allow_html=True)
        else:
            st.error("Failed to load dataset. Please check file format and try again.", icon="❌")
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}. Please ensure the dataset is valid and try again.", icon="❌")
else:
    st.info("Please upload a dataset to start the analysis.", icon="ℹ️")