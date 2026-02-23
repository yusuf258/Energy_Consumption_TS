import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

TF_AVAILABLE = False
try:
    import tensorflow as tf
    import keras
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    pass

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="EnergyCast AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA & MODEL LAYER ---
@st.cache_resource
def load_ai_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'lstm_model.h5')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"Model veya Scaler dosyası bulunamadı! Yol: {model_path}")
        return None, None

    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Model yukleme hatasi: {e}")
        return None, None


def load_csv_data(uploaded_file):
    """Load and preprocess uploaded energy consumption data."""
    try:
        df = pd.read_csv(uploaded_file, sep=';', na_values=['?'],
                         infer_datetime_format=True,
                         parse_dates={'datetime': ['Date', 'Time']},
                         index_col='datetime')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df = df.astype('float32')
        df_hourly = df.resample('H').mean()
        return df_hourly
    except Exception:
        try:
            df = pd.read_csv(uploaded_file)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif df.index.dtype == 'object':
                df.index = pd.to_datetime(df.index)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            return df
        except Exception as e:
            st.error(f"Veri okuma hatasi: {e}")
            return None


def generate_simulation_data(seq_length=24, noise_level=0.2):
    """Generates simulation data - CLEARLY MARKED as synthetic."""
    x = np.linspace(0, 4 * np.pi, seq_length)
    base_pattern = 2.0 + np.sin(x)
    noise = np.random.normal(0, noise_level, seq_length)
    data = np.maximum(base_pattern + noise, 0.1)
    return data.reshape(-1, 1)


# --- 3. BUSINESS LOGIC ---
def make_prediction(model, scaler, input_data, seq_length=24):
    input_scaled = scaler.transform(input_data)
    model_input = input_scaled.reshape(1, seq_length, 1)
    pred_scaled = model.predict(model_input, verbose=0)
    prediction = scaler.inverse_transform(pred_scaled)[0][0]
    return prediction


# --- 4. UI LAYER ---
if not TF_AVAILABLE:
    st.error("TensorFlow yuklu degil. Lutfen 'pip install tensorflow' komutuyla yukleyin.")
    st.stop()

model, scaler = load_ai_assets()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933888.png", width=100)
    st.title("EnergyCast AI")
    st.markdown("---")

    st.subheader("📂 Veri Kaynagi")
    data_source = st.radio(
        "Veri kaynagini secin:",
        ["CSV Dosyasi Yukle", "Simulasyon (Demo)"],
        index=1
    )

    uploaded_file = None
    noise_level = 0.2
    if data_source == "CSV Dosyasi Yukle":
        uploaded_file = st.file_uploader(
            "Enerji tuketim verisi yukleyin",
            type=["csv", "txt"],
            help="household_power_consumption.txt formatinda veri bekleniyor"
        )
        target_col = st.text_input("Hedef kolon", value="Global_active_power")
    else:
        noise_level = st.slider("Gurultu Seviyesi", 0.0, 1.0, 0.2)
        st.warning("⚠️ Simulasyon modu: Yapay veri kullanilmaktadir.")

    st.markdown("---")
    if model:
        st.success("🟢 Model Online")
    else:
        st.error("🔴 Model Offline")
        st.stop()

    st.markdown("v1.1.0 | LSTM Architecture")

# Main Page
st.title("⚡ Smart Grid Load Forecasting System")
st.markdown("AI system predicting the next hour's energy demand by analyzing the past 24 hours of data.")

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- Data Loading ---
real_data = None
if data_source == "CSV Dosyasi Yukle" and uploaded_file:
    real_data = load_csv_data(uploaded_file)
    if real_data is not None:
        st.success(f"Veri yuklendi: {real_data.shape[0]} satir, {real_data.shape[1]} kolon")

        # Let user select a time window
        if target_col not in real_data.columns:
            target_col = real_data.columns[0]
            st.info(f"'{target_col}' kolonu kullaniliyor.")

        values = real_data[target_col].dropna().values
        total_hours = len(values)
        st.write(f"Toplam {total_hours} saatlik veri mevcut.")

        start_idx = st.slider("Baslangic saati secin (son 24 saat kullanilacak):",
                              24, total_hours, total_hours)
        window_data = values[start_idx-24:start_idx].reshape(-1, 1)

# --- Prediction ---
col_action, col_blank = st.columns([1, 4])
with col_action:
    predict_btn = st.button("🔄 Analyze & Predict", type="primary")

if predict_btn:
    with st.spinner("AI is processing data..."):
        time.sleep(0.3)

        if data_source == "CSV Dosyasi Yukle" and real_data is not None:
            input_data = window_data
            is_simulation = False
        else:
            input_data = generate_simulation_data(noise_level=noise_level)
            is_simulation = True

        prediction = make_prediction(model, scaler, input_data)
        last_val = input_data[-1][0]

        st.session_state['current_input'] = input_data
        st.session_state['current_pred'] = prediction
        st.session_state['last_val'] = last_val
        st.session_state['is_simulation'] = is_simulation
        st.session_state['history'].append(prediction)

# --- Results ---
if 'current_pred' in st.session_state:
    pred = st.session_state['current_pred']
    last = st.session_state['last_val']
    input_seq = st.session_state['current_input']
    is_sim = st.session_state.get('is_simulation', True)

    if is_sim:
        st.warning("⚠️ Bu tahmin SIMULASYON verisi ile yapilmistir. Gercek veri icin CSV yukleyin.")

    st.markdown("---")

    # KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="Next Hour Prediction", value=f"{pred:.4f} kW",
                  delta=f"{pred - last:.4f} kW", delta_color="inverse")
    with kpi2:
        st.metric(label="Last Measured Value", value=f"{last:.4f} kW")
    with kpi3:
        avg_24h = np.mean(input_seq)
        st.metric(label="24-Hour Average", value=f"{avg_24h:.4f} kW",
                  delta=f"{pred - avg_24h:.4f} kW (Diff from Avg)")

    # Chart
    st.subheader("📈 Trend Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(24)), y=input_seq.flatten(),
        mode='lines+markers', name='History (24h)',
        line=dict(color='#00C9A7', width=3),
        fill='tozeroy', fillcolor='rgba(0, 201, 167, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=[23, 24], y=[input_seq[-1][0], pred],
        mode='lines', line=dict(color='#FF8066', width=3, dash='dash'),
        name='Projection'
    ))
    fig.add_trace(go.Scatter(
        x=[24], y=[pred],
        mode='markers', marker=dict(color='#FF8066', size=15, symbol='diamond'),
        name='AI Prediction'
    ))
    fig.update_layout(
        xaxis_title="Time (Hours)", yaxis_title="Global Active Power (kW)",
        template="plotly_white", height=500, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Inspect Raw Data"):
        df_view = pd.DataFrame(input_seq, columns=["Consumption (kW)"])
        df_view.index.name = "Hour"
        st.dataframe(df_view.T)
else:
    st.info("👈 Click 'Analyze & Predict' button to start prediction.")
