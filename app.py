import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import time

# --- 1. การตั้งค่าพื้นฐานและไอคอนแอป ---
st.set_page_config(
    page_title="Pomelo Scanner", 
    page_icon="🍊", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS (ฉบับแก้ไข) ---
st.markdown("""
    <style>
    /* ซ่อนเมนู Streamlit และ Footer ทั้งหมด */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ปรับแต่งฟอนต์ */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    
    /* ล็อคความกว้างหน้าจอให้เหมือนแอปมือถือ */
    .main .block-container {
        max-width: 450px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Animation โลโก้เด้ง */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .splash-logo { font-size: 100px; animation: pulse 1.5s infinite; text-align: center; }
    
    /* ตกแต่งปุ่ม WebRTC ให้ดูทันสมัย */
    button[title="Start"] {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. หน้าเปิดตัว (Splash Screen) ---
if 'initialized' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 85vh;">
                <div class="splash-logo">🍊</div>
                <h1 style="color: #2e7d32; margin-top: 20px; font-weight: 500;">Pomelo Smart App</h1>
                <p style="color: #666; font-size: 16px;">System Initializing...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3)
    st.session_state['initialized'] = True
    placeholder.empty()

# --- 4. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 5. ระบบประมวลผลวิดีโอ ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- 6. ส่วนหน้าจอหลัก (UI) ---
st.markdown("<h2 style='text-align: center; color: #2e7d32; margin-bottom: 0;'>🍊 Pomelo Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 14px;'>Real-time AI Analysis</p>", unsafe_allow_html=True)

# กล่องวิดีโอ
with st.container():
    webrtc_streamer(
        key="pomelo-final-pro",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )

st.markdown("<p style='text-align: center; color: #aaa; font-size: 10px; margin-top: 50px;'>Version 1.0 | Secured by AI Cloud</p>", unsafe_allow_html=True)


