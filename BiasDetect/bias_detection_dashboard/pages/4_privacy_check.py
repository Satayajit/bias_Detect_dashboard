import streamlit as st
from privacy_checker import PrivacyChecker

checker = PrivacyChecker()

st.markdown("<div class='card slide-in'><h3>Privacy Check</h3></div>", unsafe_allow_html=True)

if st.session_state.df is not None:
    st.markdown("<div class='section-title'>Detect Personally Identifiable Information (PII)</div>", unsafe_allow_html=True)
    st.session_state.pii_columns = checker.detect_pii(st.session_state.cleaned_df)
    if st.session_state.pii_columns:
        st.markdown(f"<div class='alert pulse'>⚠️ Potential PII detected in columns: {', '.join(st.session_state.pii_columns)}</div>", unsafe_allow_html=True)
        with st.expander("What is PII and why remove it?"):
            st.markdown("""
            **Personally Identifiable Information (PII)** includes data that can identify individuals, such as names, emails, and phone numbers.
            Removing PII is crucial to protect user privacy and comply with data protection regulations like GDPR.
            """)
    else:
        st.success("No PII detected! ✅", icon="✅")
else:
    st.info("Please upload a dataset in the 'Upload' page to perform a privacy check.", icon="ℹ️")