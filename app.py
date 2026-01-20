import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import time
import cv2 # เพิ่มเพื่อวาดตัวเลขลงบนจอ

# --- 1. บังคับไอคอนและ UI ---
st.set_page_config(page_title="Pomelo Scan", page_icon="🍊", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    .main .block-container { max-width: 450px; padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 3. ระบบประมวลผลพร้อมตัวนับ (Counter) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # รัน AI ตรวจจับ
    results = model.predict(img, conf=0.5, verbose=False)
    
    # นับจำนวนลูกส้มโอที่เจอในเฟรมนี้
    count = len(results[0].boxes)
    
    # วาดกรอบสแกนปกติ
    annotated_frame = results[0].plot()
    
    # เพิ่มการเขียนข้อความ "Count: X" ลงบนมุมซ้ายบนของวิดีโอ
    cv2.putText(annotated_frame, f"Count: {count} Pomelos", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- 4. หน้าจอหลักแอป ---
st.markdown("<h2 style='text-align: center; color: #2e7d32;'>🍊 Pomelo Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Real-time Scanning & Counting</p>", unsafe_allow_html=True)

# ส่วนการแสดงผลจำนวนลูกแบบ Real-time บน UI ของ Streamlit (จะโชว์ล่างจอ)
webrtc_streamer(
    key="pomelo-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

st.info("💡 คำแนะนำ: ส่องกล้องไปที่ส้มโอเพื่อดูจำนวนและสายพันธุ์")
