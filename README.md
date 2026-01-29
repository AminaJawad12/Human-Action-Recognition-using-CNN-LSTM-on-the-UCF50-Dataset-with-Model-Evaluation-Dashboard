# Human Action Recognition using CNN–LSTM on the UCF50 Dataset with Model Evaluation Dashboard

Human Action Recognition using **CNN + LSTM** is an end-to-end deep learning project built on the **UCF50 dataset** to classify human activities from video sequences.  
A **CNN** extracts spatial features from individual frames, while an **LSTM** captures temporal motion patterns across frame sequences.  
A **model evaluation dashboard** presents accuracy and other performance metrics.

---

## 📊 Dataset
- **Name:** UCF50 – Action Recognition Dataset  
- **Total Classes:** 50 human action categories  
- **Type:** Video-based dataset  
- **Official Link:** https://www.crcv.ucf.edu/data/UCF50.php  

⚠️ **Note:**  
The dataset is **not included** in this repository due to its large size and copyright restrictions.  
Please download it manually and place it inside the `dataset/` folder before running the notebook.

---

## 🧠 Model Architecture
- **CNN (Convolutional Neural Network)**  
  Extracts spatial features from individual video frames.

- **LSTM (Long Short-Term Memory)**  
  Learns temporal dependencies and motion patterns across frame sequences.

- **Fully Connected Layers**  
  Perform final classification into one of the 50 action categories.

---

## 🛠 Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Jupyter Notebook / Google Colab  

---

## 🚀 How to Run the Project (Google Colab)

Follow the steps below to run this project using **Google Colab**.

### 1️⃣ Clone the Repository
Open a new notebook in **Google Colab** and run the following commands:

    git clone https://github.com/AminaJawad12/Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard.git
    cd Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard

---

### 2️⃣ Download and Prepare the Dataset
- Download the **UCF50 dataset** from the official website.
- Extract the dataset.
- Upload it to Colab and place it in the following structure:

    dataset/
     └── UCF50/

⚠️ Make sure the folder structure is correct, otherwise the notebook will not locate the videos.

---

### 3️⃣ Open the Notebook
Open the following notebook in Google Colab:

    Human_Action_Recognition_CNN_LSTM.ipynb

---

### 4️⃣ Install Required Dependencies
Run the installation cell in the notebook, or install manually:

    pip install tensorflow opencv-python numpy matplotlib scikit-learn

---

### 5️⃣ Run the Project
- Run **each cell sequentially** from top to bottom.
- The notebook will:
  - Preprocess video frames  
  - Train the CNN–LSTM model  
  - Evaluate performance  
  - Display accuracy, confusion matrix, and results dashboard  

---

## 📈 Result Screenshot

![Result Screenshot](https://github.com/AminaJawad12/Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard/blob/main/Result%20image.jpeg?raw=true)

---

## ✅ Notes
- Training time depends on hardware availability.
- Using **Google Colab GPU** is strongly recommended.
- Always run all cells **in order** to avoid errors.

---

⭐ If you find this project helpful, consider giving the repository a star!
