import gradio as gr 
from ultralytics import RTDETR
import cv2 

model = RTDETR("model/best.pt")

def detect(image):
    results = model.predict(image, conf=0.15, iou = 0.7)
    plotted = results[0].plot()
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title = "Real-Time Obstacle Detection RT-DETR",
    description= "Upload an image to detect obstacles across 25 classes").launch()