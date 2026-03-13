# 📚 VisionXplain : LLM for Fake Image Detection
 
This project develops an **AI-based fake image detection system** that analyzes uploaded images and classifies them as **REAL or FAKE**. The system processes images through a structured deep learning pipeline and uses **CLIP-based visual feature extraction, neural classification, attention heatmaps, and LLM-based reasoning** to provide both predictions and explanations.

The goal of the system is to **detect AI-generated or manipulated images and reduce the spread of synthetic media and visual misinformation by providing an explainable and trustworthy image verification framework.**

---

# How the System Works (Step-by-Step)

The system follows a **multi-stage AI pipeline**. Each module processes the image sequentially before producing the final prediction and explanation.

---

## Step 1: User Input

The process begins when the user uploads an image through the **VisionXplain web interface**.

Supported input types include images such as:

- Social media images
- News images
- AI-generated artwork
- Potential deepfake images

Supported formats include:

- JPG
- PNG
- JPEG

Once uploaded, the image is sent to the backend **analysis pipeline**.

---

## Step 2: Image Preprocessing

Before analysis, the image is standardized to ensure compatibility with the deep learning models.

Preprocessing includes:

- Resizing the image to **224 × 224 pixels**
- Normalizing pixel values
- Converting the image to **PyTorch tensor format**

These steps prepare the image for processing by the **CLIP Vision Transformer model**.

---

## Step 3: Visual Feature Extraction

The system uses **CLIP (Contrastive Language–Image Pretraining)** to extract semantic features from the image.

Model used:

**CLIP ViT-L/14**

CLIP converts the image into a **high-dimensional feature embedding** that captures semantic patterns such as:

- object structure
- lighting inconsistencies
- texture anomalies
- generative artifacts

The transformation pipeline is:
# 📚 VisionXplain : Explainable Fake Image Detection using CLIP and LLM

This project develops an **AI-based fake image detection system** that analyzes uploaded images and classifies them as **REAL or FAKE**. The system processes images through a structured deep learning pipeline and uses **CLIP-based visual feature extraction, neural classification, attention heatmaps, and LLM-based reasoning** to provide both predictions and explanations.

The goal of the system is to **detect AI-generated or manipulated images and reduce the spread of synthetic media and visual misinformation by providing an explainable and trustworthy image verification framework.**

---

# How the System Works (Step-by-Step)

The system follows a **multi-stage AI pipeline**. Each module processes the image sequentially before producing the final prediction and explanation.

---

## Step 1: User Input

The process begins when the user uploads an image through the **VisionXplain web interface**.

Supported input types include images such as:

- Social media images
- News images
- AI-generated artwork
- Potential deepfake images

Supported formats include:

- JPG
- PNG
- JPEG

Once uploaded, the image is sent to the backend **analysis pipeline**.

---

## Step 2: Image Preprocessing

Before analysis, the image is standardized to ensure compatibility with the deep learning models.

Preprocessing includes:

- Resizing the image to **224 × 224 pixels**
- Normalizing pixel values
- Converting the image to **PyTorch tensor format**

These steps prepare the image for processing by the **CLIP Vision Transformer model**.

---

## Step 3: Visual Feature Extraction

The system uses **CLIP (Contrastive Language–Image Pretraining)** to extract semantic features from the image.

Model used:

**CLIP ViT-L/14**

CLIP converts the image into a **high-dimensional feature embedding** that captures semantic patterns such as:

- object structure
- lighting inconsistencies
- texture anomalies
- generative artifacts

The transformation pipeline is:
Image → CLIP Encoder → Feature Embedding

These embeddings represent the **semantic representation of the image**.

---

## Step 4: Fake Image Classification

The extracted feature embedding is passed to a **trained neural network classifier**.

Classifier architecture:
Input Layer : CLIP Feature Embedding
Fully Connected Layer
Output Layer : 2 classes


The classifier predicts:

- **REAL**
- **FAKE**

The output includes:

- Prediction label
- Confidence score


---

## Step 5: Attention Heatmap Generation

To improve interpretability, the system generates **attention heatmaps**.

The heatmap is derived from the **attention layers of the CLIP Vision Transformer**.

The process includes:

- Extracting attention weights
- Averaging attention across transformer heads
- Mapping attention values to image patches
- Generating a heatmap overlay

This highlights the **regions of the image that influenced the model’s decision**.

Common detected artifacts include:

- distorted facial regions
- unnatural lighting
- inconsistent textures
- rendering artifacts


---

## Step 6: Visual Evidence Extraction

The system uses the **BLIP Vision-Language Model** to generate a description of the image.

Model used:

**Salesforce BLIP Image Captioning**

BLIP generates a natural language caption describing the image content.



This caption provides **contextual visual evidence** for the reasoning module.

The system analyzes the heatmap and converts visual attention patterns into **human-readable explanations**.


---

## Step 7: LLM Explanation

The system generates a **final explanation using an LLM reasoning module**.

Inputs to the reasoning module include:

- Prediction result
- Confidence score
- Visual caption (BLIP)
- Heatmap interpretation

The LLM generates a structured explanation such as:



This ensures **transparent and explainable AI predictions**.

---

## Step 8: Report Generation

The system automatically generates **analysis reports** after processing the image.

Two report formats are supported:

### TXT Report

Includes:

- prediction
- confidence score
- explanation
- model details

### PDF Report

Includes:

- original image
- attention heatmap
- classification result
- AI explanation

These reports allow users to **download and share verification results**.

---

## Step 9: Output to User

The system displays the results through the web interface.

The output includes:

- classification label
- confidence score
- visual heatmap
- explanation
- downloadable reports



---

# Final Workflow Summary

The complete pipeline is:
User Upload
→ Image Preprocessing
→ CLIP Feature Extraction
→ Neural Network Classification
→ Attention Heatmap Generation
→ Visual Evidence Extraction
→ LLM Explanation
→ Report Generation
→ Final Output

This architecture ensures **accurate, explainable, and scalable fake image detection.**

---


## 🗂️ Project Structure

LLM_Fake_Image_Detection/
│
├── backend/                                   # Backend system
│
│   ├── app.py                                 # Flask backend server
│   ├── main.py                                # CLI testing script
│
│   ├── pipeline/
│   │   └── run_pipeline.py                    # Complete fake image detection pipeline
│
│   ├── preprocessing/
│   │   └── preprocess.py                      # Image preprocessing (resize, normalize)
│
│   ├── feature_extraction/
│   │   └── clip_encoder.py                    # CLIP feature extraction module
│
│   ├── classification/
│   │   └── classifier.py                      # Neural network classifier model
│
│   ├── localization/
│   │   └── attention_localization.py          # Attention heatmap generation
│
│   ├── explainability/
│   │   ├── blip_explainer.py                  # BLIP visual caption generation
│   │   ├── heatmap_analyzer.py                # Heatmap interpretation
│   │   └── llm_reasoner.py                    # LLM-based reasoning and explanation
│
│   ├── static/
│   │   ├── uploads/                           # Uploaded images from users
│   │   └── heatmaps/                          # Generated attention heatmaps
│
│   ├── outputs/
│   │   └── reports/                           # Generated TXT and PDF reports
│
│   └── fake_image_classifier.pth              # Trained fake image classification model
│
├── frontend/                                  # Web interface
│
│   ├── server.js                              # Node.js frontend server
│   ├── package.json                           # Node.js dependencies
│   ├── package-lock.json
│
│   ├── public/
│   │   └── index.html                         # Main frontend UI page
│
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css                      # UI styling
│   │   │
│   │   └── js/
│   │       └── script.js                      # Frontend interaction logic
│
│   └── node_modules/                          # Installed frontend packages
│
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation



---

# 💻 Tech Stack

## System Components and Technologies

| Component | Technology Used |
|-----------|----------------|
| Web Interface | Node.js Frontend |
| Backend API | Flask |
| Deep Learning Framework | PyTorch |
| Image Encoder | CLIP (ViT-L/14) |
| Vision-Language Model | BLIP |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Explainability | Transformer Attention Heatmaps |
| Report Generation | ReportLab |
| Model Integration | HuggingFace Transformers |

---

# Algorithms

| Category | Algorithm | Module Applied |
|----------|-----------|----------------|
| Feature Extraction | CLIP Vision Transformer | Feature Extraction Module |
| Classification | Neural Network | Classification Module |
| Explainability | Attention Heatmaps | Localization Module |
| Visual Captioning | BLIP Image Captioning | Explainability Module |
| Reasoning | LLM Prompt-Based Explanation | LLM Reasoning Module |
| Image Processing | OpenCV Image Processing | Preprocessing Module |

---


## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone <https://github.com/kpjahnavi/LLM_for_Fake_Image_Detection>
cd LLM_Fake_Image_Detection
```

### 2. Create and activate Python virtual environment
```bash
python -m venv .venv
```

Activate the environment:

Windows
```bash
.venv\Scripts\activate
```

macOS / Linux
```bash
source .venv/bin/activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Node.js dependencies (Frontend)
Navigate to the frontend directory and install the required packages.

```bash
cd frontend
npm install
```

### 5. Run the backend server
```bash
python app.py
```

The backend server will start at:

```
http://localhost:5000
```

### 6. Run the frontend application
```bash
npm start
```

Visit the application in your browser:

```
http://localhost:3000
```


## 🧪 Features Summary

- ✅ AI-powered fake image detection (REAL vs FAKE classification)
- ✅ CLIP-based semantic visual feature extraction
- ✅ Neural network classifier for image authenticity prediction
- ✅ Attention heatmap visualization for explainable AI
- ✅ BLIP-based visual caption generation for image understanding
- ✅ LLM-based reasoning to explain prediction results
- ✅ Automated TXT and PDF report generation
- ✅ Web-based interface for uploading and analyzing images

---

## 🧠 Example Use Cases

- Detect **AI-generated or deepfake images**
- Verify **authenticity of images shared on social media**
- Analyze **suspicious images in news or online content**
- Upload **screenshots or digital media for authenticity verification**
- Assist journalists and researchers in **identifying synthetic media**
- Provide **explainable AI insights for digital media analysis**

---


## 🔗 GitHub Repository

Project Source Code: [VisionXplain – LLM for Fake Image Detection](https://github.com/kpjahnavi/LLM_for_Fake_Image_Detection)

---

## 📽️ Demo Video

Watch the project demo video here:  
▶️ [Project Demo – YouTube](https://youtu.be/Xb4AYXSxpag)

---

## 📄 Project Report

Read the full project report here:  
📘 [Project Report – Google Drive](https://drive.google.com/file/d/1i00vzCnQhNVUyvkQRqzDwov_vymOkh_V/view?usp=sharing)

---

## 🤝 Contributors

- [*K P Jahnavi*](https://github.com/kpjahnavi)
- [*Himanth Reddy*]
- [*CH Srikanth Reddy*]
- [*Mohammad Farooq*]

---

## 📄 License

This project is developed for *academic, research, and hackathon purposes* as part of an *AI-based Fake Image Detection system using Vision-Language Models and Explainable AI*.
 
