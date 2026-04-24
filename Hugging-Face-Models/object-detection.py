from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# Build object-detection pipeline from Hugging Face Hub
od_pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

def draw_boxes(image, predictions):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        label = pred["label"]
        score = pred["score"]
        box = pred["box"]

        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
        draw.text((xmin, max(0, ymin - 15)), f"{label} {score:.2f}", fill="red")

    return image

# Test on a local image
from PIL import Image
import requests
from io import BytesIO

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

response = requests.get(url)
raw_image = Image.open(BytesIO(response.content)).convert("RGB")
pipeline_output = od_pipe(raw_image)

print("Predictions:")
for item in pipeline_output:
    print(item)

processed_image = draw_boxes(raw_image, pipeline_output)
processed_image.save("detected_output.jpg")
print("Saved annotated image as detected_output.jpg")

# Optional Gradio demo
def get_pipeline_prediction(pil_image):
    pipeline_output = od_pipe(pil_image)
    return draw_boxes(pil_image, pipeline_output)

demo = gr.Interface(
    fn=get_pipeline_prediction,
    inputs=gr.Image(label="Input image", type="pil"),
    outputs=gr.Image(label="Output image with predicted instances", type="pil"),
    title="Object Detection with DETR"
)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)