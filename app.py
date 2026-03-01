import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import gdown

st.set_page_config(
    page_title="DermaAI",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp { background-color: #F3FBEF; }
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #4F772D;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6A994E;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #D8F3DC;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #1B4332;
    }
    </style>
""", unsafe_allow_html=True)

device = torch.device("cpu")

model = models.resnet18(pretrained=False)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 7)
)

MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1rp3cEm2ymxI7KvOgx8PhQMdH5YIFL9Te"

if not os.path.exists(MODEL_PATH):
    st.write("Téléchargement du modèle en cours...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    st.error("Le modèle n'a pas pu être téléchargé.")
    st.stop()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

classes = [
    "Akiec",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Vascular Lesion"
]

st.markdown('<div class="main-title">DermaScan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Détection intelligente des lésions cutanées</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Télécharge une image dermatologique", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", width="stretch")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, 1).item()
        confidence = probabilities[0][prediction].item() * 100

    st.markdown(
        f'<div class="prediction-box">Classe prédite : {classes[prediction]} <br> Confiance : {confidence:.2f}%</div>',
        unsafe_allow_html=True
    )
