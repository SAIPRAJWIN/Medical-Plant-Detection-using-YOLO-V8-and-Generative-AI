<<<<<<< HEAD
# Medical-Plant-Detection-using-YOLO-V8-and-Generative-AI
This project focuses on the development of an intelligent and user-friendly web application that can accurately detect medicinal plants using deep learning techniques. By leveraging the power of YOLOv8 (You Only Look Once) object detection model and Streamlit, this application allows users to upload images of plants and receive real-time prediction
=======
# 🌿 Medical Plant Detection using YOLOv8 and Generative AI

## 🧠 Project Overview

This project focuses on the development of an intelligent and user-friendly web application that can accurately detect **medicinal plants** using deep learning techniques. By leveraging the power of **YOLOv8 (You Only Look Once)** object detection model and **Streamlit**, this application allows users to upload images of plants and receive real-time predictions on the identified medicinal species.

This tool is designed to support **botanical research**, **herbal medicine identification**, and **educational purposes**, especially in rural and tribal areas where access to professional botanical expertise may be limited.

---

## 🎯 Objectives

- Build a lightweight and accurate object detection model capable of recognizing multiple medicinal plants.
- Deploy an interactive web interface using Streamlit for ease of use.
- Enable image upload, display the most confident plant classification, and provide downloadable reports.
- Maintain user-specific logs with timestamped detection data (name, age, and results).

---

## 🛠️ Technology Stack

| Component            | Description                                 |
|----------------------|---------------------------------------------|
| **Model**            | YOLOv8 (Ultralytics) pre-trained + custom   |
| **Framework**        | PyTorch, Streamlit                          |
| **Language**         | Python                                      |
| **Interface**        | Streamlit-based UI                          |
| **Tools & Libraries**| OpenCV, NumPy, Pandas, PIL, os, datetime    |

---

## 🔍 Model Training & Optimization

- The YOLOv8 model was trained using a **custom dataset** consisting of labeled medicinal plant images.
- Training involves:
  - Data preprocessing and augmentation
  - Bounding box annotations in YOLO format
  - Model fine-tuning and validation
- The final model is exported as `best.pt`, ready for inference.

---

## 📷 Streamlit Application Features

- 🌱 Upload an image for medicinal plant detection.
- 🔎 Detect and show **only the most confident** plant name from the image.
- 🧾 Display the result clearly below the image.
- 🧑‍💼 Collect user name and age for logging purposes.
- ⏰ Show a **real-time clock** on the interface.
- 📥 Download detection results and image as a `.txt` log file.
- ✨ Custom UI with background styling for an enhanced experience.

---

## 📁 File Structure

```
├── Medical Plant Detection.ipynb   # Jupyter Notebook for model loading and Streamlit app
├── best.pt                         # Trained YOLOv8 model
├── plant_images/                   # Folder with test or sample plant images
├── requirements.txt                # Required Python packages
├── utils/                          # Helper functions (optional)
└── README.md                       # Project documentation
```

---

## 🧪 Sample Use Case

1. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

2. Upload a plant image.

3. Fill in your name and age in the provided fields.

4. View the detection result and download the output file.

---

## 📥 Sample Output File (TXT)

```
Name: Rohith
Age: 22
Timestamp: 2025-05-29 17:05:23
Detected Plant: Ocimum tenuiflorum (Tulsi)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- YOLOv8 (`ultralytics` package)
- Streamlit
- OpenCV, Pillow, Pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This project utilizes a comprehensive medical plant detection dataset hosted on Roboflow Universe. The dataset contains high-quality annotated images of various medicinal plants, specifically curated for object detection and classification tasks.

**Dataset URL:** [https://universe.roboflow.com/pkm-cj9yb/plant-detection-iqmnt](https://universe.roboflow.com/pkm-cj9yb/plant-detection-iqmnt)

### Dataset Features
- High-resolution images of medicinal plants
- Accurate bounding box annotations
- Multiple export formats (YOLO, COCO, Pascal VOC)
- Pre-processing and augmentation options
- Suitable for training custom plant detection models

### Usage
The dataset is used to train our deep learning models for accurate plant identification and classification, enabling the system to recognize various medicinal plants with high precision.

### Run the Application

```bash
streamlit run Medical_Plant_Detection.py
```

---

## 🎓 Applications

- Medicinal plant identification in remote areas.
- Assisting researchers and herbal practitioners.
- Building botanical datasets for AI.
- Educational tools for plant taxonomy.

---

## Screenshots

### Authentication

![image](https://github.com/user-attachments/assets/de7e3869-2aaa-4564-8724-feb82c51e585)


*User login interface for accessing the medical plant detection system*

### Plant Detection

![image](https://github.com/user-attachments/assets/093202ab-b91a-4c7d-833f-5f01728efc8e)


*Upload image interface for plant identification*

![image](https://github.com/user-attachments/assets/2b55bdd3-4e98-474f-a60c-2c73a48d2fb2)


*Real-time plant detection using webcam*

### AI Assistant

![image](https://github.com/user-attachments/assets/9bff265b-1d06-49e0-b85d-7afd4bb5a983)


*Interactive AI chatbot for plant-related queries*

### Detection History

![image](https://github.com/user-attachments/assets/605bcf41-e59c-44ed-8cc9-f08867215cd6)


*View previous plant detection results and analysis*

## Additional Features

### User Feedback

![image](https://github.com/user-attachments/assets/08df9d54-9e36-4df8-b75d-27a61e22c164)


*User feedback interface for system improvement*

### About Section

![image](https://github.com/user-attachments/assets/29412f48-d4df-4eb7-8e05-7514f4da2938)


*Information about the medical plant detection system*

---
---


## 🔗 Let’s Connect

📫 **LinkedIn**  
Let’s connect and grow professionally:  
[linkedin.com/in/anisetty-sai-prajwin-536742306/](https://www.linkedin.com/in/anisetty-sai-prajwin-536742306/)



---


> 💡 _“Final-year student, forever learner — building the future, one project at a time.”_

Feel free to explore my repositories and reach out for **collaborations**, **internships**, or to discuss **innovative ideas**!

>>>>>>> source/main
