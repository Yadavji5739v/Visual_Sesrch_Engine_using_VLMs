# 🔍 Visual Search Engine using VLMs

A Visual Search Engine that uses Vision-Language Models (VLMs) for AI-powered image retrieval based on textual queries or sample images.

---

## 📌 Project Overview

### 🧠 Problem Statement

Develop a visual search engine that leverages vision-language models (VLMs) to retrieve 
relevant images based on textual queries or sample images. The system should embed both text 
and images into a shared representation space, allowing users to search via keywords, natural 
language descriptions, or example images. 

---

## 🛠 Technologies Used

- **Programming Language**: Python  
- **Deep Learning Framework**: PyTorch, Torchvision  
- **Feature Extraction**: ResNet-18  
- **Web UI**: Streamlit  
- **Libraries**: NumPy, SciPy, PIL  
- **Version Control**: Git & GitHub

---

## 📌 Features

✅ **Image-based search** – Find visually similar images from a dataset  
✅ **Text-based search** – Retrieve images using keywords or descriptions  
✅ **Deep Learning-powered feature extraction** – Uses ResNet-18 for encoding  
✅ **Fast similarity search** – Powered by precomputed image embeddings  
✅ **Streamlit UI** – Clean, user-friendly and interactive interface


---

## 📁 Project Structure

<pre>

📦 Visual_Search_Engine_using_VLM
├── 📂 images/                # Sample image dataset for search
├── 📂 models/                # Pre-trained models or saved embeddings
├── 📜 app.py                 # Streamlit app to run the visual search engine
├── 📜 feature_extraction.py  # Script to extract features from images using ResNet-18
├── 📜 search.py              # Handles similarity search (image & text-based)
├── 📜 requirements.txt       # Python dependencies
├── 📜 README.md              # Project documentation


