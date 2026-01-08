import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import time

# --- 1. ตั้งค่า Icon และ UI ---
st.set_page_config(page_title="Pomelo Scan", page_icon="🍊", layout="centered")

# ซ่อนส่วนเกินของ Streamlit และตั้งค่า CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    
    .main .block-container { max-width: 450px; padding-top: 1rem; }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .splash-logo { font-size: 100px; animation: pulse 1.5s infinite; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. หน้าเปิดตัว (Splash Screen) ---
if 'initialized' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh; background-color: white;">
                <div class="splash-logo">🍊</div>
                <h1 style="color: #2e7d32; margin-top: 20px;">Pomelo Smart App</h1>
                <p style="color: #666;">กำลังโหลด AI และกล้อง...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3)
    st.session_state['initialized'] = True
    placeholder.empty()

# --- 3. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception:
    st.error("⚠️ ไฟล์โมเดลมีปัญหา กรุณาอัปโหลด best.pt ใหม่บน GitHub")

# --- 4. ระบบประมวลผลวิดีโอ (Real-time) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- 5. หน้าจอหลักแอป ---
st.markdown("<h2 style='text-align: center; color: #2e7d32; margin-bottom: 0;'>🍊 Pomelo Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 14px;'>Real-time AI Analysis</p>", unsafe_allow_html=True)

webrtc_streamer(
    key="pomelo-scan-final",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

st.markdown("<p style='text-align: center; color: #aaa; font-size: 10px; margin-top: 50px;'>พัฒนาโดย: ทีมคุณจิรัชญาณ์</p>", unsafe_allow_html=True)
