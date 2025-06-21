import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from visualizer import Visualizer

visualizer = Visualizer()

st.markdown("<div class='card slide-in'><h3>Visualizations</h3></div>", unsafe_allow_html=True)

if st.session_state.df is not None:
    st.markdown("<div class='section-title'>Explore Your Data</div>", unsafe_allow_html=True)
    
    # Filter for numerical and categorical columns
    num_cols = st.session_state.cleaned_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = st.session_state.cleaned_df.select_dtypes(include=['object']).columns.tolist()

    # Tabs for different types of visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distributions", "Relationships", "Box Plots", "Violin Plots", "Data Flow"])

    with tab1:
        st.markdown("<div class='section-title'>Distributions</div>", unsafe_allow_html=True)
        col_to_plot = st.selectbox("Select a column to plot distribution", num_cols + cat_cols, key="dist_col")
        if col_to_plot:
            if st.session_state.cleaned_df[col_to_plot].dtype in ['int64', 'float64']:
                fig = px.histogram(st.session_state.cleaned_df, x=col_to_plot, title=f"Histogram of {col_to_plot}", nbins=30, color_discrete_sequence=['#3B82F6'])
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            else:
                fig = px.histogram(st.session_state.cleaned_df, x=col_to_plot, title=f"Bar Chart of {col_to_plot}", color=col_to_plot, color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B'])
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Relationships (Pair Plot)</div>", unsafe_allow_html=True)
        if num_cols:
            selected_cols = st.multiselect("Select numerical columns for pair plot", num_cols, default=num_cols[:3])
            if selected_cols:
                pair_plot_df = st.session_state.cleaned_df[selected_cols]
                sns.pairplot(pair_plot_df)
                st.pyplot(plt)
        else:
            st.warning("No numerical columns available for pair plot.", icon="⚠️")

    with tab3:
        st.markdown("<div class='section-title'>Box Plots</div>", unsafe_allow_html=True)
        num_col = st.selectbox("Select a numerical column", num_cols, key="box_num")
        cat_col = st.selectbox("Select a categorical column", cat_cols, key="box_cat")
        if num_col and cat_col:
            fig = px.box(st.session_state.cleaned_df, x=cat_col, y=num_col, title=f"Box Plot of {num_col} by {cat_col}", color=cat_col, color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("<div class='section-title'>Violin Plots</div>", unsafe_allow_html=True)
        num_col_violin = st.selectbox("Select a numerical column", num_cols, key="violin_num")
        cat_col_violin = st.selectbox("Select a categorical column", cat_cols, key="violin_cat")
        if num_col_violin and cat_col_violin:
            fig = px.violin(st.session_state.cleaned_df, x=cat_col_violin, y=num_col_violin, title=f"Violin Plot of {num_col_violin} by {cat_col_violin}", color=cat_col_violin, color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("<div class='section-title'>Data Flow Diagram</div>", unsafe_allow_html=True)
        visualizer.plot_data_flow(st.session_state.cleaned_df)
else:
    st.info("Please upload a dataset in the 'Upload' page to view visualizations.", icon="ℹ️")