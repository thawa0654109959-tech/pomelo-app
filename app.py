import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2

# โหลดโมเดล
model = YOLO("best.pt")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # รัน AI ตรวจจับ (ลดขนาดภาพลงเล็กน้อยเพื่อให้ลื่นขึ้น)
        results = model.predict(img, conf=0.4, verbose=False)
        
        # วาดกรอบลงบนภาพสด
        annotated_frame = results[0].plot()
        
        return annotated_frame

st.title("🍊 สแกนส้มโอ Real-time")
st.write("เปิดกล้องแล้วส่องไปที่ส้มโอได้เลยครับ")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
