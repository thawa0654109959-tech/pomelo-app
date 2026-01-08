import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import time

# --- 1. ตั้งค่าหน้าตาแอป (Theme) ---
st.set_page_config(page_title="Pomelo AI Scanner", page_icon="🍊", layout="centered")

# CSS สำหรับตกแต่ง UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
    }
    .main {
        background-color: #f9fff9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #2e7d32;
        color: white;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .splash-logo {
        font-size: 100px;
        animation: pulse 1.5s infinite;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. หน้าเปิดตัว (Splash Screen) ---
if 'initialized' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh;">
                <div class="splash-logo">🍊</div>
                <h1 style="color: #2e7d32; margin-top: 20px;">Pomelo Smart App</h1>
                <p style="color: #666; font-size: 18px;">กำลังเข้าสู่ระบบ AI...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3) 
    st.session_state['initialized'] = True
    placeholder.empty()

# --- 3. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 4. ระบบวิเคราะห์วิดีโอ (เน้นตีกรอบแม่นๆ ไม่นับจำนวน) ---
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # รัน AI ตรวจจับ (ปรับ conf เป็น 0.5 เพื่อให้กรอบนิ่งขึ้น ไม่ขึ้นมั่ว)
        results = model.predict(img, conf=0.5, verbose=False)
        
        # วาดเฉพาะกรอบและชื่อคลาสลงบนภาพ
        annotated_frame = results[0].plot()
        
        return annotated_frame

# --- 5. หน้าจอหลัก ---
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🍊 Pomelo AI Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>สแกนสายพันธุ์และความสุกของส้มโอแบบเรียลไทม์</p>", unsafe_allow_html=True)

# ส่วนการใช้งานกล้อง
webrtc_streamer(
    key="pomelo-scan-only",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.divider()
st.markdown("<p style='text-align: center; color: #666;'>จัดทำโดย: ทีมคุณจิรัชญาณ์</p>", unsafe_allow_html=True)
