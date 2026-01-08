import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import time

# --- 1. ตั้งค่าพื้นฐานแอป ---
st.set_page_config(page_title="Pomelo AI Scanner", page_icon="🍊", layout="centered")

# ตกแต่ง CSS
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

# --- 2. ระบบหน้าเปิดตัว (Splash Screen) ---
# ใช้ session_state เพื่อให้โชว์เฉพาะตอนเข้าแอปครั้งแรก หรือ Refresh ครั้งใหญ่
if 'initialized' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh;">
                <div class="splash-logo">🍊</div>
                <h1 style="color: #2e7d32; margin-top: 20px;">Pomelo Smart App</h1>
                <p style="color: #666; font-size: 18px;">กำลังเริ่มต้นระบบ AI อัจฉริยะ...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3) # แสดงหน้าเปิดตัว 3 วินาที
    st.session_state['initialized'] = True
    placeholder.empty()

# --- 3. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    # ตรวจสอบว่าชื่อไฟล์ใน GitHub ต้องเป็น best.pt เท่านั้น
    return YOLO("best.pt")

model = load_model()

# --- 4. ฟังก์ชันจัดการประมวลผลวิดีโอ (Callback) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # รัน AI ตรวจจับ (conf=0.5 ช่วยให้กรอบนิ่งขึ้น)
    results = model.predict(img, conf=0.5, verbose=False)
    
    # วาดกรอบลงบนภาพ
    annotated_frame = results[0].plot()
    
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- 5. ส่วนหน้าจอหลักของแอป ---
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🍊 Pomelo AI Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>สแกนสายพันธุ์และตรวจสอบคุณภาพส้มโอเรียลไทม์</p>", unsafe_allow_html=True)

# กล่องเครื่องมือกล้อง
with st.container():
    st.write("---")
    ctx = webrtc_streamer(
        key="pomelo-scanner-final",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # ทำให้วิดีโอไม่กระตุก
    )

st.write("---")

# --- 6. ส่วนข้อมูลเพิ่มเติม ---
with st.expander("ℹ️ ข้อมูลระบบและการใช้งาน"):
    st.write("""
    - **วิธีการใช้งาน:** กดปุ่ม 'Start' เพื่อเปิดกล้องและนำไปส่องที่ผลส้มโอ
    - **การประมวลผล:** ระบบใช้โมเดล YOLO11 ในการตรวจจับแบบ Real-time
    - **ผู้พัฒนา:** ทีมคุณ จิรัชญาณ์
    """)

st.markdown("<p style='text-align: center; color: #aaa; font-size: 12px;'>© 2024 Pomelo AI Technology</p>", unsafe_allow_html=True)
