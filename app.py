import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import cv2
import time

# --- 1. ตั้งค่าหน้าเพจและไอคอน ---
st.set_page_config(page_title="Pomelo Scan", page_icon="🍊", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    
    /* หน้า Splash Screen */
    .splash-container {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: white;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        animation: fadeout 1s forwards;
        animation-delay: 2s;
    }
    @keyframes fadeout {
        from { opacity: 1; visibility: visible; }
        to { opacity: 0; visibility: hidden; }
    }
    .logo { font-size: 80px; animation: pulse 1.5s infinite; }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    </style>
    
    <div class="splash-container">
        <div class="logo">🍊</div>
        <h2 style="color: #2e7d32;">Pomelo Smart Scan</h2>
        <p style="color: #888;">AI Powered Detection</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 3. ระบบกล้องและนับจำนวน (Real-time Counter) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # รัน AI ตรวจจับ
    results = model.predict(img, conf=0.5, verbose=False)
    
    # นับจำนวนลูกส้มโอ
    count = len(results[0].boxes)
    
    # วาดกรอบสแกน
    annotated_frame = results[0].plot()
    
    # วาดตัวเลขจำนวนลูกลงบนจอ (มุมซ้ายบน)
    cv2.putText(annotated_frame, f"Count: {count}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- 4. ส่วนแสดงผลหลัก ---
st.markdown("<h3 style='text-align: center; color: #2e7d32;'>🍊 ตรวจสอบและนับจำนวนผลผลิต</h3>", unsafe_allow_html=True)

webrtc_streamer(
    key="pomelo-final-pro",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

st.markdown("<p style='text-align: center; color: #999; font-size: 12px;'>ระบบกำลังรันผ่าน Cloud Server</p>", unsafe_allow_html=True)
