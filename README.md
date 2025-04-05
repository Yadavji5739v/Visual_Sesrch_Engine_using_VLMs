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
├── 📜 front_end.py                 # Streamlit app to run the visual search engine
├── 📜 main.py  # Script to extract features from images using ResNet-18
├── 📜 requirements.txt       # Python dependencies
├── 📜 README.md              # Project documentation

</pre>

# 🔍 Visual Search Engine using Vision-Language Models (VLMs)

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Krishnandu-Halder/Visual_Search_Engine_using_VLM.git
cd Visual_Search_Engine_using_VLM

🔹 Windows
python -m venv venv
venv\Scripts\activate

🔹 Linux & Mac
python3 -m venv venv
source venv/bin/activate


3️⃣ Install Dependencies
pip install -r requirements.txt

🚀 Usage
1️⃣ Extract Image Features
python feature_extraction.py


2️⃣ Run the Web Application
streamlit run app.py

3️⃣ Search via Command Line
🔹 Find similar images using a query image:
python search.py --query path/to/query_image.jpg


🔹 Find images using text descriptions:

python search.py --text "A cat sitting on a chair"


🔧 Managing Virtual Environment
🔹 Activate Virtual Environment
Windows
venv\Scripts\activate


Linux & Mac
source venv/bin/activate


🔹 Deactivate Virtual Environment
deactivate


---

 

Happy coding!
