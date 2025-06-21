import streamlit as st
import plotly.graph_objects as go
from bias_analyzer import BiasAnalyzer
from visualizer import Visualizer

analyzer = BiasAnalyzer()
visualizer = Visualizer()

st.markdown("<div class='card slide-in'><h3>Bias & Fairness Analysis</h3></div>", unsafe_allow_html=True)

if st.session_state.df is not None:
    if st.session_state.sensitive_cols:
        binary_cols = [col for col in st.session_state.cleaned_df.columns if analyzer.is_binary(st.session_state.cleaned_df[col])]
        if binary_cols:
            target_col = st.selectbox("Select target column for bias analysis (must be binary: 0 or 1)", binary_cols, help="Choose the column representing the outcome (e.g., shortlisted).")
            
            # Calculate fairness metrics and bias percentage
            fairness_metrics = {}
            bias_scores = []
            for col in st.session_state.sensitive_cols:
                metrics = analyzer.calculate_fairness_metrics(st.session_state.cleaned_df, col, target_col)
                fairness_metrics[col] = metrics
                st.markdown(f"<div class='metric-card'>{col} Fairness Metrics</div>", unsafe_allow_html=True)
                st.write(metrics)
                visualizer.plot_fairness_metrics(metrics, col)
                
                # Calculate bias percentage based on disparate impact
                if metrics and 'Disparate Impact' in metrics:
                    di = metrics['Disparate Impact']
                    bias_score = abs(1 - di) * 100  # Bias percentage: deviation from 1 (perfect fairness)
                    bias_scores.append(bias_score)

            st.session_state.fairness_metrics = fairness_metrics
            st.session_state.bias_percentage = sum(bias_scores) / len(bias_scores) if bias_scores else 0

            # Display overall bias percentage
            st.markdown("<div class='section-title'>Overall Bias in Dataset</div>", unsafe_allow_html=True)
            st.metric("Bias Percentage", f"{st.session_state.bias_percentage:.2f}%", help="Average bias across sensitive features, calculated as the deviation of Disparate Impact from 1.")

            # Comparative Fairness Plot
            if len(fairness_metrics) > 1:
                st.markdown("<div class='section-title'>Comparative Fairness Metrics</div>", unsafe_allow_html=True)
                fig = go.Figure()
                for col, metrics in fairness_metrics.items():
                    if 'Selection Rate by Group' in metrics and 'selection_rate' in metrics['Selection Rate by Group']:
                        selection_rates = metrics['Selection Rate by Group']['selection_rate']
                        fig.add_trace(go.Bar(
                            x=list(selection_rates.values()),
                            y=list(selection_rates.keys()),
                            name=col,
                            orientation='h',
                            marker_color='#3B82F6' if col == st.session_state.sensitive_cols[0] else '#10B981'
                        ))
                fig.add_vline(x=0.8, line_dash="dash", line_color="#EF4444", annotation_text="Fairness Threshold")
                fig.update_layout(
                    title="Comparative Selection Rates Across Sensitive Features",
                    xaxis_title="Selection Rate",
                    yaxis_title="Groups",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No binary columns (0 or 1) detected for bias analysis. Please ensure your dataset includes a binary target column.", icon="⚠️")
    else:
        st.warning("No sensitive columns detected for bias analysis.", icon="⚠️")
else:
    st.info("Please upload a dataset in the 'Upload' page to perform bias analysis.", icon="ℹ️")