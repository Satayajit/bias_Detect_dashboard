import streamlit as st
from bias_analyzer import BiasAnalyzer

analyzer = BiasAnalyzer()

st.markdown("<div class='card slide-in'><h3>Recommendations & Bias Mitigation</h3></div>", unsafe_allow_html=True)

if st.session_state.df is not None:
    st.markdown("<div class='section-title'>Actionable Recommendations</div>", unsafe_allow_html=True)
    recommendations = analyzer.get_recommendations(st.session_state.cleaned_df, st.session_state.sensitive_cols)
    for rec in recommendations:
        st.markdown(f"<div class='recommendation slide-in'>{rec}</div>", unsafe_allow_html=True)

    # Bias Mitigation
    if st.button("Apply Advanced Bias Mitigation", help="Apply reweighting to reduce bias in the dataset."):
        st.session_state.cleaned_df = analyzer.mitigate_bias(st.session_state.cleaned_df, st.session_state.sensitive_cols)
        st.success("Bias mitigation applied! Download the mitigated dataset below.", icon="✅")
        st.download_button(
            label="Download Mitigated Dataset",
            data=st.session_state.cleaned_df.to_csv(index=False),
            file_name="mitigated_dataset.csv",
            mime="text/csv",
            help="Download the dataset after applying bias mitigation techniques."
        )
else:
    st.info("Please upload a dataset in the 'Upload' page to view recommendations.", icon="ℹ️")