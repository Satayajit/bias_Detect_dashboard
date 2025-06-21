import streamlit as st
import plotly.graph_objects as go
from ml_predictor import MLPredictor

predictor = MLPredictor()

st.markdown("<div class='card slide-in'><h3>ML Readiness & Prediction</h3></div>", unsafe_allow_html=True)

if st.session_state.df is not None:
    st.markdown("<div class='section-title'>Machine Learning Readiness</div>", unsafe_allow_html=True)
    readiness, message, st.session_state.ml_score = predictor.check_ml_readiness(st.session_state.cleaned_df, st.session_state.sensitive_cols)
    
    # ML Readiness Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.ml_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ML Readiness Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3B82F6"},
            'steps': [
                {'range': [0, 50], 'color': "#EF4444"},
                {'range': [50, 75], 'color': "#F59E0B"},
                {'range': [75, 100], 'color': "#10B981"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    if readiness:
        st.markdown("<div class='success bounce'>🎉 Congratulations! Dataset is Ready for ML Modeling!</div>", unsafe_allow_html=True)
        st.image("assets/flower.gif", width=200, caption="Dataset is ML-ready!")
        st.balloons()
    else:
        st.markdown(f"<div class='alert pulse'>⚠️ Oops! Dataset needs fixing: {message}</div>", unsafe_allow_html=True)
        st.image("assets/error.gif", width=200, caption="Fix required before modeling.")

    # Sample Prediction
    binary_cols = [col for col in st.session_state.cleaned_df.columns if set(st.session_state.cleaned_df[col].dropna().unique()).issubset({0, 1})]
    if st.button("Run Sample Prediction", help="Train a simple ML model to predict the target variable."):
        if st.session_state.sensitive_cols and binary_cols:
            target_col = st.selectbox("Select target column for prediction", binary_cols, key="pred_target")
            prediction_results = predictor.predict(st.session_state.cleaned_df, target_col)
            if prediction_results is not None:
                st.markdown("<div class='section-title'>Sample Prediction Results</div>", unsafe_allow_html=True)
                st.write(prediction_results)
            else:
                st.error("Failed to generate predictions. Please check the dataset and target column.", icon="❌")
        else:
            st.warning("Select a binary target column for predictions.", icon="⚠️")
else:
    st.info("Please upload a dataset in the 'Upload' page to check ML readiness.", icon="ℹ️")