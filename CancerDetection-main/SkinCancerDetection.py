import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.utils import get_file
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import os
import glob
import random

# Set random seeds for reproducibility
tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)

# Class names
class_names = ["benign", "malignant"]

# Function to generate CSV metadata file
def generate_csv(folder, label2int):
    folder_name = os.path.basename(folder)
    labels = list(label2int)
    df = pd.DataFrame(columns=["filepath", "label"])
    i = 0
    for label in labels:
        path = os.path.join(folder, label, "*")
        print("Reading", path)
        files = glob.glob(path)
        print(f"Found {len(files)} files for {label}")
        for filepath in files:
            df.loc[i] = [filepath, label2int[label]]
            i += 1
    if i == 0:
        print(f"Error: No images found in {folder}. Check directory structure and image files.")
    output_file = f"{folder_name}.csv"
    print("Saving", output_file)
    df.to_csv(output_file)
    return df

# Generate CSV files for train, validation, and test sets
label_mapping = {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1}
train_df = generate_csv(r"E:\Project file\dataset\train", label_mapping)
valid_df = generate_csv(r"E:\Project file\dataset\valid", label_mapping)
test_df = generate_csv(r"E:\Project file\dataset\test", label_mapping)

# Load data
train_metadata_filename = "train.csv"
valid_metadata_filename = "valid.csv"
df_train = pd.read_csv(train_metadata_filename)
df_valid = pd.read_csv(valid_metadata_filename)
n_training_samples = len(df_train)
n_validation_samples = len(df_valid)
print("Number of training samples:", n_training_samples)
print("Number of validation samples:", n_validation_samples)

# Check if datasets are empty
if n_training_samples == 0 or n_validation_samples == 0:
    print("Error: Dataset is empty. Please populate the dataset directories with images.")
    exit(1)

train_ds = tf.data.Dataset.from_tensor_slices((df_train["filepath"], df_train["label"]))
valid_ds = tf.data.Dataset.from_tensor_slices((df_valid["filepath"], df_valid["label"]))

# Preprocess data
def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [224, 224])  # VGG16 default input size

def process_path(filepath, label):
    img = tf.io.read_file(filepath)
    img = decode_img(img)
    return img, label

valid_ds = valid_ds.map(process_path)
train_ds = train_ds.map(process_path)

# Print sample image shape and label
for image, label in train_ds.take(1):
    print("Image shape:", image.shape)
    print("Label:", label.numpy())

# Training parameters
batch_size = 64
optimizer = "rmsprop"

# Prepare dataset for training
def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

valid_ds = prepare_for_training(valid_ds, batch_size=batch_size, cache="valid-cached-data")
train_ds = prepare_for_training(train_ds, batch_size=batch_size, cache="train-cached-data")

# Load models and make predictions
try:
    vgg16_model = load_model(r'E:\AI ML\Vgn_Weight_tf\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
except FileNotFoundError:
    print("VGG16 model file not found. Loading from Keras applications...")
    from tensorflow.keras.applications import VGG16
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

cnn_model = None
try:
    cnn_model = load_model(r'E:\AI ML\benign-vs-malignant_64_rmsprop_0.373.h5')
except FileNotFoundError:
    print("CNN model file not found. Proceeding with VGG16 predictions only.")

# Get user input for image paths
n = int(input("Enter the number of images: "))
img_paths = [input(f"Enter the path of image {i+1}: ") for i in range(n)]

# Create subplots for visualization
fig, axs = plt.subplots(n, 1 if cnn_model is None else 2, figsize=(10, n * 5))

# Process and predict for each image
for i, img_path in enumerate(img_paths):
    img = image.load_img(img_path, target_size=(224, 224))  # Match VGG16 input size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    vgg16_preds = vgg16_model.predict(x)
    vgg16_preds = np.argmax(vgg16_preds, axis=1)

    if cnn_model is None:
        axs[i].imshow(img)
        axs[i].set_title(f'VGG16 Prediction: {class_names[vgg16_preds[0]]}')
    else:
        cnn_preds = cnn_model.predict(x)
        cnn_preds = np.argmax(cnn_preds, axis=1)

        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f'VGG16 Prediction: {class_names[vgg16_preds[0]]}')
        axs[i, 1].imshow(img)
        axs[i, 1].set_title(f'CNN Prediction: {class_names[cnn_preds[0]]}')

plt.tight_layout()
plt.show()