import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

class Visualizer:
    def plot_distributions(self, df, sensitive_cols):
        for col in sensitive_cols:
            st.markdown(f"<div class='section-title slide-in'>Distribution of {col}</div>", unsafe_allow_html=True)
            if df[col].dtype in ['int64', 'float64']:
                fig = px.histogram(df, x=col, title=f"Histogram of {col}", nbins=30, color_discrete_sequence=['#3B82F6'])
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            else:
                fig = px.histogram(df, x=col, title=f"Bar Chart of {col}", color=col, color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B'])
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

    def plot_stacked_bar(self, df, sensitive_cols):
        if sensitive_cols:
            st.markdown("<div class='section-title slide-in'>Stacked Bar Chart</div>", unsafe_allow_html=True)
            target_col = st.selectbox("Select target column for stacked bar", df.columns, key="stacked_target")
            for col in sensitive_cols:
                if col != target_col:
                    grouped = df.groupby([col, target_col]).size().unstack().fillna(0)
                    fig = px.bar(grouped, title=f"Stacked Bar: {col} vs {target_col}", barmode='stack', color_discrete_sequence=['#3B82F6', '#10B981'])
                    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', transition_duration=500)
                    st.plotly_chart(fig, use_container_width=True)

    def plot_correlation_heatmap(self, df):
        st.markdown("<div class='section-title slide-in'>Correlation Heatmap</div>", unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale='RdBu_r')
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numerical columns for correlation analysis.", icon="⚠️")

    def plot_statistical_summary(self, df):
        st.markdown("<div class='section-title slide-in'>Statistical Summary</div>", unsafe_allow_html=True)
        stats = df.describe().T
        st.write(stats)

    def plot_data_flow(self, df):
        st.markdown("<div class='section-title slide-in'>Data Flow Diagram</div>", unsafe_allow_html=True)
        G = nx.DiGraph()
        for col in df.columns[:5]:  # Limit for visualization clarity
            G.add_node(col)
        for i in range(len(df.columns[:5])-1):
            G.add_edge(df.columns[i], df.columns[i+1])
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_x, node_y = [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2, color='#3B82F6')))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), marker=dict(size=20, color='#10B981')))
        fig.update_layout(
            title="Data Flow Diagram",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_fairness_metrics(self, metrics, sensitive_col):
        if not metrics or 'selection_rate' not in metrics.get('Selection Rate by Group', {}):
            st.warning(f"Cannot plot fairness metrics for {sensitive_col}: Selection rate data is missing.", icon="⚠️")
            return

        fig = go.Figure()
        selection_rates = metrics['Selection Rate by Group']['selection_rate']
        fig.add_trace(go.Bar(
            x=list(selection_rates.keys()),
            y=list(selection_rates.values()),
            name='Selection Rate',
            marker_color='#3B82F6'
        ))
        fig.add_hline(y=0.8, line_dash="dash", line_color="#EF4444", annotation_text="Fairness Threshold")
        fig.update_layout(
            title=f"Fairness Metrics for {sensitive_col}",
            xaxis_title=sensitive_col,
            yaxis_title="Selection Rate",
            plot_bgcolor='white',
            paper_bgcolor='white',
            transition_duration=500
        )
        st.plotly_chart(fig, use_container_width=True)