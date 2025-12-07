import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests

st.title("Detetor de Pneumonia por Radiografia")

# baixar modelo
url = "https://drive.google.com/uc?id=1xt7KvDXEmmiPCLWwYwfMk-QhH6RB8FVT"
r = requests.get(url)

with open("modelo_pneumonia.pth", "wb") as f:
    f.write(r.content)

# carregar modelo
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("modelo_pneumonia.pth", map_location="cpu"))
model.eval()

# transformar imagem
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

file = st.file_uploader("Escolhe uma radiografia", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    x = tf(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out,1)[0]

    st.write("Normal:", round(probs[0].item(),3))
    st.write("Pneumonia:", round(probs[1].item(),3))

    if probs[1] > probs[0]:
        st.error("POSSÍVEL PNEUMONIA")
    else:
        st.success("NORMAL")
