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
                <p style="color: #666; font-size: 18px;">กำลังเชื่อมต่อระบบ AI...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3)  # โชว์หน้าเปิดตัว 3 วินาที
    st.session_state['initialized'] = True
    placeholder.empty()

# --- 3. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    # ตรวจสอบว่าไฟล์ชื่อ best.pt จริงไหม ถ้าเป็นชื่ออื่นให้แก้ตรงนี้ครับ
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("ไม่พบไฟล์โมเดล best.pt กรุณาตรวจสอบบน GitHub")

# --- 4. ระบบวิเคราะห์วิดีโอ (Real-time) ---
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # รัน AI ตรวจจับ
        results = model.predict(img, conf=0.45, verbose=False)
        
        # วาดกรอบและนับจำนวน
        annotated_frame = results[0].plot()
        
        return annotated_frame

# --- 5. หน้าจอหลักของแอป ---
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🍊 Pomelo Scanner Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>ระบบตรวจจับและนับจำนวนส้มโอเรียลไทม์</p>", unsafe_allow_html=True)

# ส่วนการใช้งานกล้อง
with st.container():
    st.info("💡 วิธีใช้: เปิดกล้องแล้วส่องไปที่ส้มโอ ระบบจะตีกรอบและนับจำนวนให้ทันที")
    
    webrtc_streamer(
        key="pomelo-pro",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

# --- 6. ส่วนสรุปผลและข้อมูลผู้พัฒนา ---
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.write("### 📊 ความสามารถระบบ")
    st.write("- ตรวจจับสายพันธุ์ส้มโอ")
    st.write("- วิเคราะห์ระดับความสุก")
    st.write("- นับจำนวนแบบ Real-time")

with col2:
    st.write("### 👤 ผู้จัดทำ")
    st.write("คุณจิรัชญาณ์ และคณะ")
    st.write("Project: AI Agriculture")

st.markdown("<br><p style='text-align: center; color: #aaa;'>© 2024 Pomelo AI Project. All Rights Reserved.</p>", unsafe_allow_html=True)
