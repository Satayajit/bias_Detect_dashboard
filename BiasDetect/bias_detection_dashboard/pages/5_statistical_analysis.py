import streamlit as st
import pandas as pd  # Added missing import
from visualizer import Visualizer

visualizer = Visualizer()

st.markdown("<div class='card slide-in'><h3>Statistical Analysis</h3></div>", unsafe_allow_html=True)

if st.session_state.df is not None:
    st.markdown("<div class='section-title'>Statistical Insights</div>", unsafe_allow_html=True)

    # Correlation Heatmap
    visualizer.plot_correlation_heatmap(st.session_state.cleaned_df)

    # Statistical Summary
    st.markdown("<div class='section-title'>Statistical Summary</div>", unsafe_allow_html=True)
    stats = st.session_state.cleaned_df.describe().T
    st.write(stats)

    # Additional Statistics
    st.markdown("<div class='section-title'>Additional Statistics</div>", unsafe_allow_html=True)
    with st.expander("Skewness and Kurtosis"):
        skewness = st.session_state.cleaned_df.select_dtypes(include=['int64', 'float64']).skew()
        kurtosis = st.session_state.cleaned_df.select_dtypes(include=['int64', 'float64']).kurtosis()
        stats_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis})
        st.write(stats_df)
else:
    st.info("Please upload a dataset in the 'Upload' page to view statistical analysis.", icon="ℹ️")