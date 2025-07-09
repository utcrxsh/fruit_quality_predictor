import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms


@st.cache_resource
def load_model():
    try:
        session = ort.InferenceSession("models/fruit_quality_model.onnx", providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


label_map = {
    1: "🥀 Rotten Apple",
    2: "🍂 Slightly Rotten Apple",
    3: "🍎 Average Quality Apple",
    4: "🍏 Fresh Apple",
    5: "🍏✨ Very Fresh Apple"
}


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0).numpy()
    return tensor


def predict(image, session):
    try:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: image})[0]
        pred_class = int(np.argmax(output)) + 1  # +1 to shift from 0-based to 1-based
        confidence = float(np.max(output))
        return pred_class, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()


st.set_page_config(page_title="🍎 Fruit Quality Predictor", layout="centered")
st.title("🍎 Fruit Quality Predictor")
st.write("Upload an image of an apple to determine its quality (1 = Rotten, 5 = Very Fresh).")

session = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        input_tensor = preprocess(image)
        prediction, confidence = predict(input_tensor, session)
        label = label_map[prediction]
    
    st.success(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")
