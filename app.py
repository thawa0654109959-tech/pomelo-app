import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import time

# --- การตั้งค่าพื้นฐาน ---
st.set_page_config(page_title="Pomelo AI Scanner", page_icon="🍊", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .splash-logo { font-size: 100px; animation: pulse 1.5s infinite; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- หน้าเปิดตัว (Splash Screen) ---
if 'initialized' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh;">
                <div class="splash-logo">🍊</div>
                <h1 style="color: #2e7d32; margin-top: 20px;">Pomelo Smart App</h1>
                <p style="color: #666;">กำลังเริ่มต้นระบบ AI...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3)
    st.session_state['initialized'] = True
    placeholder.empty()

# --- โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- ฟังก์ชันจัดการวิดีโอ (Real-time Callback) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # รัน AI ตรวจจับ
    results = model.predict(img, conf=0.5, verbose=False)
    
    # วาดกรอบลงบนภาพ
    annotated_frame = results[0].plot()
    
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- ส่วนแสดงผลหลัก ---
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🍊 Pomelo AI Detector</h1>", unsafe_allow_html=True)

webrtc_streamer(
    key="pomelo-final",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.divider()
st.info("💡 คำแนะนำ: หากจอดำ ให้ลองรีเฟรชหน้าเว็บหรือสลับจาก WiFi มาใช้เน็ตมือถือครับ")
