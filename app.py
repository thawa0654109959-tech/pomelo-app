import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import time

# --- 1. ตั้งค่าหน้าตาแอป (Theme) ---
st.set_page_config(page_title="Pomelo AI Scanner", page_icon="🍊", layout="centered")

# CSS สำหรับตกแต่ง UI และ Animation หน้าเปิดตัว
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    .main { background-color: #f9fff9; }
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
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh;">
                <div class="splash-logo">🍊</div>
                <h1 style="color: #2e7d32; margin-top: 20px;">Pomelo Smart App</h1>
                <p style="color: #666; font-size: 18px;">กำลังเริ่มต้นระบบ AI...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3) 
    st.session_state['initialized'] = True
    placeholder.empty()

# --- 3. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    # มั่นใจว่าไฟล์ใน GitHub ชื่อ best.pt (ตัวพิมพ์เล็กหมด)
    return YOLO("best.pt")

model = load_model()

# --- 4. ฟังก์ชันจัดการวิดีโอ (เวอร์ชั่นใหม่) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # รัน AI ตรวจจับ (ปรับ conf=0.5 เพื่อความแม่นยำ)
    results = model.predict(img, conf=0.5, verbose=False)
    
    # วาดกรอบลงบนภาพ
    annotated_frame = results[0].plot()
    
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- 5. หน้าจอหลัก ---
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🍊 Pomelo AI Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>สแกนสายพันธุ์และความสุกของส้มโอแบบเรียลไทม์</p>", unsafe_allow_html=True)

# ส่วนการใช้งานกล้องแบบ Real-time
with st.container():
    webrtc_streamer(
        key="pomelo-scanner",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.divider()
st.markdown("<p style='text-align: center; color: #666;'>พัฒนาโดย: ทีมคุณจิรัชญาณ์</p>", unsafe_allow_html=True)
