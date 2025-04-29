Brain Tumor Detection and Segmentation

Project Overview : 

This project implements a two-stage system for brain tumor detection and segmentation using:

Quantum Convolutional Neural Networks (QCNN) for tumor detection (classification),

3D U-Net for precise tumor segmentation.

We utilize the BraTS 2020 dataset for training and evaluation, aiming to improve diagnostic processes in medical imaging.

Technologies Used
Python 3.x

TensorFlow / Keras

PyTorch (for U-Net)

Pennylane / Qiskit (for QCNN)

Scikit-learn

Numpy, Pandas

Matplotlib, Seaborn

Streamlit (for simple web demo)

Google Colab (for model training)

Dataset
BraTS 2020 (Brain Tumor Segmentation Challenge)

MRI modalities: T1, T1Gd, T2, FLAIR

Labels: Enhancing tumor, peritumoral edema, necrotic core

Available at: BraTS Dataset Link

Project Architecture
1. Tumor Detection (QCNN)
Preprocessing: Normalize MRI scans, resize.

Model: Quantum Convolutional Neural Network (QCNN)

Output: Binary classification (Tumor / No Tumor)

2. Tumor Segmentation (U-Net)
Preprocessing: Volume slicing, normalization.

Model: 3D U-Net with enhancements.

Output: Segmentation mask highlighting tumor regions.

Setup Instructions
Clone the repository:


git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Install dependencies:


pip install -r requirements.txt
(Optional) For quantum components, install Pennylane:


pip install pennylane
Run the training notebooks or inference script:

cd notebooks
# Open and run QCNN and U-Net training notebooks
(Optional) Launch the Streamlit app:


streamlit run app/app.py
Key Results
Detection Accuracy (QCNN): ~96%

Segmentation Dice Score (U-Net): ~0.91

Inference Time: Optimized for near real-time analysis.

