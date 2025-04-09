# train.py
import torch
from ultralytics import YOLO

# تحقق من وجود GPU
print("GPU متاح:", torch.cuda.is_available())

# تحميل النموذج
model = YOLO('yolov8n.pt')  # تأكد من وجود الملف في نفس المجلد

# التدريب
results = model.train(
    data='data.yaml',
    epochs=50,
    batch=4,
    imgsz=200,  # صحح هذه السطر
    device='0' if torch.cuda.is_available() else 'cpu'
)