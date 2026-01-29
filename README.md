# Human Action Recognition using CNN–LSTM on the UCF50 Dataset with Model Evaluation Dashboard

Human Action Recognition using **CNN + LSTM** is an end-to-end deep learning project on the **UCF50 dataset** that classifies human activities from video sequences. **CNN** extracts spatial features from frames, while **LSTM** captures temporal motion patterns. A **results dashboard** presents final accuracy and evaluation outputs.

---

## 📊 Dataset
- **Name:** UCF50 – Action Recognition Dataset  
- **Total Classes:** 50 Human Action Categories  
- **Type:** Video-based dataset  
- **Official Link:** [UCF50 Dataset](https://www.crcv.ucf.edu/data/UCF50.php)  

⚠️ *Note:* The dataset is **not included** in this repository due to its large size and copyright restrictions. Please download and place it in the `dataset/` folder before running the notebook.

---

## 🧠 Model Architecture
- **CNN (Convolutional Neural Network):**  
  Extracts spatial features from individual video frames.  

- **LSTM (Long Short-Term Memory):**  
  Captures temporal motion patterns across frame sequences.  

- **Fully Connected Layers:**  
  Performs final action classification.  

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

## 📁 Project Structure

Human_Action_Recognition_CNN_LSTM/
│
├── notebooks/ # Jupyter notebook implementation
├── dataset/ # Place your UCF50 dataset here
├── models/ # Saved trained model (optional)
├── results/ # Accuracy and evaluation outputs
├── report/ # Project report (PDF)
├── requirements.txt
└── README.md


---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/AminaJawad12/Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard.git
cd Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard

###2. Install Dependencies
Make sure you have Python 3.8+ installed, then run:
pip install -r requirements.txt

###3. Run the Notebook
Open Jupyter Notebook:
jupyter notebook
Navigate to:

notebooks/Human_Action_Recognition_CNN_LSTM.ipynb

Execute the cells sequentially to:

Preprocess the dataset

Train the CNN-LSTM model

Evaluate the model

Visualize predictions on videos

## 📊 Results

<img width="1287" height="779" alt="image" src="https://github.com/user-attachments/assets/dbfb98b3-bc55-4b4a-9f5b-060f07ee53f4" />

  
