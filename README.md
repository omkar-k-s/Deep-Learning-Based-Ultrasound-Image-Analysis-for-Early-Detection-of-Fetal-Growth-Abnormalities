#  Deep-Learning-Based-Ultrasound-Image-Analysis-for-Early-Detection-of-Fetal-Growth-Abnormalities

A comprehensive, production-ready AI system for diagnosing fetal growth restriction and other abnormalities from ultrasound images. 

##  Screenshots

*(You can drag and drop your project screenshots here to replace these placeholders!)*

![Dashboard Interface](https://via.placeholder.com/800x400.png?text=Dashboard+Interface)
![Diagnosis Report](https://via.placeholder.com/800x400.png?text=Diagnosis+Report)
![Grad-CAM Visualization](https://via.placeholder.com/800x400.png?text=Grad-CAM+Visualization)

---

##  Overview

The system combines state-of-the-art technologies to assist medical professionals:

- **CNN (Convolutional Neural Networks)**: Image classification using ResNet50
- **RAG (Retrieval-Augmented Generation)**: Medical knowledge base retrieval using FAISS + Sentence Transformers
- **LLM Report Generation**: Structured clinical reports with medical context
- **Grad-CAM Visualization**: Attention maps showing model decision regions
- **MongoDB Atlas Integration**: Persistent cloud storage for all patient diagnostic records
- **Flask Web Interface**: Professional healthcare application interface

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- MongoDB Atlas Account (Free tier)
- 2GB RAM minimum (4GB+ recommended)

### Installation

1. **Clone or navigate to project directory**:
```bash
cd majorproject
```

2. **Set up MongoDB Connection**:
Create a `.env` file in the root directory by copying `.env.example`, and paste your MongoDB connection string:
```bash
cp .env.example .env
# Edit .env and replace <db_username> with your MongoDB username and insert your password
```

3. **Install dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

4. **Run the application**:
```bash
python app.py
```

5. **Open in browser**:
Navigate to `http://localhost:5000`

---

## 📁 Project Structure

```
majorproject/
├── backend/
│   ├── app.py                          # Main Flask application
│   ├── db.py                           # MongoDB Cloud Integration
│   ├── requirements.txt                # Python dependencies
│   ├── model/
│   │   ├── cnn_model.py               # CNN classifier (ResNet50)
│   │   ├── inference.py               # Inference pipeline
│   │   └── gradcam.py                 # Grad-CAM visualization
│   ├── rag/
│   │   ├── retriever.py               # Medical knowledge retriever
│   │   └── knowledge_base.json        # Medical knowledge database
│   ├── llm/
│   │   └── report_generator.py        # Clinical report generator
│   └── uploads/                       # Temporary image/PDF storage
├── frontend/
│   ├── templates/
│   │   └── index.html                 # Frontend interface
│   └── static/                        # CSS, JS, images
├── Dockerfile                         # Docker container setup
├── .env.example                       # Template for environment variables
└── README.md                          # This file
```

---

##  System Architecture

### 1. CNN Image Analysis
- Uses transfer learning (ResNet50/MobileNet) to classify images into **Normal Fetus**, **Fetal Growth Restriction (FGR)**, or **Other Abnormalities**.

### 2. RAG (Retrieval-Augmented Generation)
- Uses FAISS and Sentence Transformers to retrieve relevant medical conditions, maternal risk factors, and biometric standards based on the CNN's prediction.

### 3. Automated Report Generation
- Generates a structured 9-section clinical report containing technical details, risk assessment, findings, and recommendations. Outputs to highly professional PDF reports for both Doctor and Patient.

### 4. Advanced Visualization
- Employs **Grad-CAM Heatmaps** overlaid on the original ultrasound to visually explain *why* the model made its decision, improving clinical trust.

### 5. Persistent Cloud Storage
- Integrates directly with **MongoDB Atlas** to save patient info, diagnosis results, and file paths to a secure cloud database, allowing a history of diagnoses to be tracked over time.

---

##  API Endpoints

- **`POST /api/diagnose`**: Main diagnosis endpoint. Accepts image upload and patient details.
- **`GET /api/history`**: Retrieves recent diagnostic records from MongoDB.
- **`GET /api/health`**: System health check.
- **`GET /api/model-info`**: Model information and statistics.
- **`POST /api/batch-diagnose`**: Process multiple images.

---

##  Security & Best Practices

1. **Input Validation**: Robust file type/size checking for ultrasound images. Rejects non-medical photography.
2. **Privacy**: Images are stored with timestamps.
3. **Database Security**: Uses authenticated MongoDB Atlas connections over TLS.

---

##  License
Educational use - modify and distribute freely with attribution.
