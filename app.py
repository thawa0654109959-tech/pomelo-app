import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Pomelo Real-time", layout="centered")
st.title("🍊 Pomelo Real-time Detection")

# 1. โหลดโมเดล
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 2. ใช้ฟีเจอร์กล้องของ Streamlit
img_file = st.camera_input("สแกนส้มโอแบบสด")

if img_file:
    # อ่านภาพ
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # 3. Predict
    results = model.predict(img_array, conf=0.25)
    
    # 4. แสดงผลทันที
    for r in results:
        res_plotted = r.plot()
        st.image(res_plotted, caption="วิเคราะห์ภาพล่าสุด", use_container_width=True)
        
        # แสดงจำนวนที่นับได้
        count = len(r.boxes)
        st.subheader(f"📊 จำนวนที่ตรวจพบ: {count} ลูก")
        
        # แยกคลาส
        detected_classes = [model.names[int(c)] for c in r.boxes.cls]
        for name in set(detected_classes):
            st.write(f"- {name}: {detected_classes.count(name)}")

# เพิ่มคำแนะนำการใช้
st.info("💡 ทริค: กดปุ่มถ่ายภาพรัวๆ เพื่อให้ระบบอัปเดตตำแหน่งส้มโอแบบต่อเนื่องครับ")
