import streamlit as st
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered"
)
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    max-width:1100px;
}

div[data-testid="metric-container"]{
    background-color:#1f2937;
    border:1px solid #374151;
    padding:20px;
    border-radius:15px;
}

div.stProgress > div > div > div{
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Image Classification")

st.caption(
    "Transfer Learning • ResNet18 • PyTorch • CIFAR-10 • Test Accuracy: 96.69%"

)

st.markdown(
    """
Upload a **JPG** or **PNG** image and let the trained deep learning model classify it into one of the **10 CIFAR-10 classes**.
"""
)

st.markdown("---")
# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.markdown("---")

    st.write("**Architecture:** ResNet18")
    st.write("**Dataset:** CIFAR-10")
    st.write("**Framework:** PyTorch")
    st.write("**Transfer Learning:** ✅ Yes")

    st.metric(
        "Test Accuracy",
        "96.69%"
    )

    st.markdown("---")

    st.write("**Developer**")
    st.write("Plabon Sarkar")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
EMOJIS = {
    "airplane": "✈️",
    "automobile": "🚗",
    "bird": "🐦",
    "cat": "🐱",
    "deer": "🦌",
    "dog": "🐶",
    "frog": "🐸",
    "horse": "🐴",
    "ship": "🚢",
    "truck": "🚚"
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

model = resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 10)
)

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
device = torch.device("cpu")
checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    return (
        prediction.item(),
        confidence.item(),
        probabilities.squeeze()
    )

# ==========================================================
# IMAGE UPLOADER
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, center, right = st.columns([1,2,1])

    with center:
        st.image(
            image,
            caption="🖼️ Uploaded Image",
            width=420
        )

    # Prediction
    prediction, confidence, probabilities = predict(image)

    with st.container(border=True):

        col1, col2 = st.columns(2)

        predicted_class = CLASS_NAMES[prediction]

        with col1:
            st.metric(
                "Prediction",
                f"{EMOJIS[predicted_class]} {predicted_class.capitalize()}"
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )
            st.progress(confidence)

    st.subheader("🏆 Top-5 Predictions")

    top5_prob, top5_idx = torch.topk(
        probabilities,
        k=5
    )

    for prob, idx in zip(top5_prob, top5_idx):
        class_name = CLASS_NAMES[idx]


        st.write( f"{EMOJIS[class_name]} **{class_name.capitalize()}** — {prob.item()*100:.2f}%")

        st.progress(float(prob))

        

    st.markdown("---")

st.subheader("📘 About This Model")

st.write("""
This application uses **Transfer Learning** with **ResNet18**
to classify images into one of the ten CIFAR-10 classes.

For best results, upload images belonging to:

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck
""")
st.markdown("---")

st.caption(
    "Built with ❤️ using PyTorch, Streamlit and Transfer Learning"
)