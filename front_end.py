from scipy.spatial.distance import cdist
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import streamlit as st

# Streamlit Page Config
st.set_page_config(page_title="AI Image Search", page_icon="🧠", layout="wide")

# ----- Custom CSS Styling -----
st.markdown("""
    <style>
        html, body {
            background-color: #f8f9fa;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #005f73;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #dbeafe;
            border-radius: 10px 10px 0px 0px;
            padding: 0.75rem 1rem;
            margin-right: 5px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563eb;
            color: white;
        }
        .uploaded-image, .similar-image {
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .success-box {
            font-size: 18px;
            color: green;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered Image Similarity Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #6c757d;'>Upload or search an image to discover visually similar images powered by deep learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# Image Transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
image_folder_text_search = r"C:\Users\kumar\Desktop\Intel project\train\archive (10)\Images"
image_folder_uploads = r"C:\Users\kumar\Desktop\Intel project\train\archive (10)\Images"

# Load Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Load Saved Features
feature_vectors = np.load("feature_vectors.npy", allow_pickle=True)
image_names = np.load("image_names.npy", allow_pickle=True)

# Extract Features
def extract_features(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img).squeeze().numpy()
    return features

# Find Similar Images
def find_similar_images(query_features, top_n=5):
    distances = cdist(query_features.reshape(1, -1), feature_vectors, metric="cosine").squeeze()
    top_matches = distances.argsort()[1:top_n+1]
    return [image_names[i] for i in top_matches]

# Tabs
tab1, tab2 = st.tabs(["📂 Upload Image", "🔎 Search by Name"])

# ----- Tab 1: Upload -----
with tab1:
    st.markdown("<h2>📤 Upload an Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image to find similar ones", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        uploaded_path = os.path.join(image_folder_uploads, uploaded_file.name)
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(uploaded_path, caption="Uploaded Image", use_column_width=True, output_format="JPEG", channels="RGB")
        with col2:
            st.success("✅ Image Uploaded and Processed")

        with st.spinner("Analyzing and retrieving similar images..."):
            query_features = extract_features(uploaded_path)
            similar_images = find_similar_images(query_features, top_n=5)

        st.markdown("<h3>🎯 Top 5 Similar Images</h3>", unsafe_allow_html=True)
        cols = st.columns(5)
        for col, img_name in zip(cols, similar_images):
            img_path = os.path.join(image_folder_text_search, img_name)
            col.image(img_path, caption=img_name, use_column_width=True)

# ----- Tab 2: Search by Name -----
with tab2:
    st.markdown("<h2>🔍 Search by Image Name</h2>", unsafe_allow_html=True)
    search_query = st.text_input("Enter the image name (without extension):")

    if search_query:
        file_name = f"{search_query}.jpg"
        search_path = os.path.join(image_folder_text_search, file_name)

        if os.path.exists(search_path):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(search_path, caption="Query Image", use_column_width=True)
            with col2:
                st.success("✅ Image Found in Database")

            with st.spinner("Retrieving similar images..."):
                query_features = extract_features(search_path)
                similar_images = find_similar_images(query_features, top_n=5)

            st.markdown("<h3>🎯 Top 5 Similar Images</h3>", unsafe_allow_html=True)
            cols = st.columns(5)
            for col, img_name in zip(cols, similar_images):
                img_path = os.path.join(image_folder_text_search, img_name)
                col.image(img_path, caption=img_name, use_column_width=True)
        else:
            st.error(f"❌ '{file_name}' not found in the dataset.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>🚀 Built with ❤️ using <b>PyTorch</b> & <b>Streamlit</b></p>", unsafe_allow_html=True)
