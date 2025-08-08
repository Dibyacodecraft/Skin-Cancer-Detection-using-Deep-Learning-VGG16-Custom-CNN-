\# **Skin Cancer Detection using Deep Learning**



This project classifies skin lesions as \*\*benign\*\* or \*\*malignant\*\* using a combination of \*\*VGG16 (pre-trained)\*\* and \*\*Custom CNN models\*\*. It supports training, validation, testing, and image-based prediction.



\## 📊 **Overview**



\- Utilizes image metadata in CSV format (`train.csv`, `valid.csv`, `test.csv`)

\- Preprocesses input images (resize, normalize)

\- Trains and evaluates deep learning models

\- Outputs confusion matrix, ROC curve, accuracy, sensitivity \& specificity



\## 🧠 **Models Used**



\- Pre-trained VGG16 from Keras

\- Custom CNN trained on melanoma dataset



\## **📦 Files Included**



\- `SkinCancerDetection.py`: Main script

\- `SkinCancerDetectionFinal.ipynb`: Notebook version

\- `labels.csv`, `train.csv`, `valid.csv`, `test.csv`: Dataset metadata

\- `Explaination.txt`: Full breakdown and code walkthrough

\- `settings.json`: VSCode UI customization



**## 🔧 Setup \& Requirements**



```bash



pip install tensorflow keras tensorflow\_hub matplotlib seaborn numpy pandas scikit-learn imbalanced-learn











**Running the Script**



bash



python SkinCancerDetection.py



You'll be prompted to enter the number of images and their file paths.

The model will predict and display results using VGG16 and CNN (if available).







**Evaluation**



Confusion Matrix

ROC-AUC Score

Accuracy, Sensitivity, Specificity

Visual Predictions (blue = correct, red = incorrect)



