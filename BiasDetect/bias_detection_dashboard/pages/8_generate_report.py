import streamlit as st
from pdf_generator import generate_pdf_report

st.markdown("<div class='card slide-in'><h3>Generate Report</h3></div>", unsafe_allow_html=True)

# Check if all required session state variables are present
required_keys = ['df', 'cleaned_df', 'fairness_metrics', 'pii_columns', 'ml_score']
missing_keys = [key for key in required_keys if key not in st.session_state or st.session_state[key] is None]

if missing_keys:
    st.warning("Please complete the following steps before generating the report:", icon="⚠️")
    for key in missing_keys:
        if key == 'df' or key == 'cleaned_df':
            st.write("- Upload a dataset in the 'Upload' page.")
        elif key == 'fairness_metrics':
            st.write("- Perform bias analysis in the 'Bias Analysis' page.")
        elif key == 'pii_columns':
            st.write("- Check for PII in the 'Privacy Check' page.")
        elif key == 'ml_score':
            st.write("- Evaluate ML readiness in the 'ML Readiness' page.")
else:
    if st.button("Generate Report"):
        try:
            # Call generate_pdf_report without arguments
            generate_pdf_report()
        except Exception as e:
            st.error(f"Failed to generate PDF report: {str(e)}", icon="❌")