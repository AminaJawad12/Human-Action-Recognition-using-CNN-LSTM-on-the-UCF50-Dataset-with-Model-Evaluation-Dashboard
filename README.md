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
## 🚀 How to Run the Project

```bash
# 1. Clone the repo
git clone https://github.com/AminaJawad12/Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard.git
cd Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard

# 2. Download UCF50 dataset
# Place extracted dataset in 'dataset/' folder

# 3. Open notebook in Google Colab or Jupyter
# File: Human_Action_Recognition_CNN_LSTM.ipynb

# 4. Install dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn

# 5. Run all cells in order
# Model trains and dashboard shows results

---

## 📈 Result Screenshot

![Result Screenshot](https://github.com/AminaJawad12/Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard/blob/main/Result%20image.jpeg?raw=true)






