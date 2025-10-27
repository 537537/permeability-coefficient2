import streamlit as st
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import pandas as pd

# ========== 页面配置 ==========
st.set_page_config(
    page_title="Pervious Concrete Compressive Strength Prediction",
    page_icon="💧",
    layout="wide",
)

# ========== 自定义 CSS ==========
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e6f0ff, #ffffff);
}
h1 {
    text-align: center;
    color: #003366;
    font-family: 'Times New Roman', serif;
    font-weight: bold;
    font-size: 32px !important;
}
p, label {
    font-family: 'Times New Roman', serif;
    font-size: 16px;
}
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #007bff, #0056b3);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 0;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease;
    font-family: 'Times New Roman', serif;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #0056b3, #003d80);
    transform: scale(1.03);
}
.result-box {
    background-color: #f0f8ff;
    border: 2px solid #99ccff;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 64, 128, 0.1);
    margin-top: 25px;
}
.result-value {
    font-size: 26px;
    font-weight: bold;
    color: #003366;
}
</style>
""", unsafe_allow_html=True)

# ========== 页面标题 ==========
st.markdown("<h1>💧 Pervious Concrete Compressive Strength Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter the following 8 parameters to predict the compressive strength (MPa).</p>", unsafe_allow_html=True)

# ========== 模型和标准化器路径 ==========
MODEL_PATH = "final_catboost_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("⚠️ Model or scaler file is missing. Please check the file paths.")
else:
    @st.cache_resource
    def load_model_and_scaler():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    model, scaler = load_model_and_scaler()

    # ========== 输入参数布局 ==========
    col1, col2 = st.columns(2)
    with col1:
        W_C = st.number_input("W/C (Water–Cement Ratio)", min_value=0.0, value=0.3, step=0.01)
        Dmin = st.number_input("Dmin (Minimum Aggregate Size)", min_value=0.0, value=4.75, step=0.01)
        Porosity = st.number_input("Porosity", min_value=0.0, value=15.0, step=0.1)
        Diameter = st.number_input("Size (Cylinder diameter / Cube side)", min_value=0.0, value=100.0, step=1.0)
    with col2:
        A_C = st.number_input("A/C (Aggregate–Cement Ratio)", min_value=0.0, value=3.0, step=0.1)
        ASR = st.number_input("ASR (Aggregate Size Ratio)", min_value=0.0, value=0.5, step=0.01)
        shape_option = st.selectbox("Specimen Shape", ["Cylinder", "Cube"])
        Shape = 1 if shape_option == "Cylinder" else 2
        Height = st.number_input("Specimen Height", min_value=0.0, value=200.0, step=1.0)

    # ========== 预测按钮 ==========
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Compressive Strength")
    st.markdown("</div>", unsafe_allow_html=True)

    # ========== 执行预测 ==========
    if predict_button:
        try:
            feature_names = ["W/C", "A/C", "Dmin", "ASR", "Porosity", "Shape", "Diameter", "Height"]
            
            # 构造输入
            input_data = np.array([[W_C, A_C, Dmin, ASR, Porosity, Shape, Diameter, Height]])
            input_scaled = scaler.transform(input_data)
            input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

            # 预测
            prediction = model.predict(input_scaled)[0]

            # 显示预测结果
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-value'>Predicted Compressive Strength: {prediction:.2f} MPa</div>
            </div>
            """, unsafe_allow_html=True)

            # ========== SHAP Force Plot ==========
            st.markdown("### 🔹 SHAP Force Plot (Feature Contributions)")

            explainer = shap.Explainer(model)
            full_explanation = explainer(input_scaled_df)

            plot_explanation = shap.Explanation(
                values=full_explanation.values[0],
                base_values=full_explanation.base_values[0],
                data=None,
                feature_names=full_explanation.feature_names
            )

            force_plot_fig = shap.plots.force(
                plot_explanation,
                matplotlib=True,
                show=False,
                contribution_threshold=0
            )

            st.pyplot(force_plot_fig, bbox_inches='tight')
            plt.close(force_plot_fig)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction or SHAP computation: {e}")

# ========== 底部信息 ==========
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:14px; color:gray;'>Developed by Q.D. | Powered by Streamlit & CatBoost | SHAP Interpretation Enabled</p>", unsafe_allow_html=True)
