import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LABEL_MAPPING = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3,
    'akiec': 4, 'vasc': 5, 'df': 6
}

REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def load_and_preprocess_data(metadata_path, image_folders, target_size=(64, 64)):
    """
    Membaca metadata HAM10000 dan memuat gambar dengan koordinasi label.
    """
    metadata = pd.read_csv(metadata_path)
    label_dict = dict(zip(metadata['image_id'], metadata['dx']))
    
    images = []
    labels = []
    
    for folder in image_folders:
        if not os.path.exists(folder):
            continue
        for img_name in tqdm(os.listdir(folder), desc=f"Loading from {folder}"):
            if img_name.endswith('.jpg'):
                img_id = os.path.splitext(img_name)[0]
                label_str = label_dict.get(img_id)
                
                if label_str in LABEL_MAPPING:
                    img_path = os.path.join(folder, img_name)
                    img = load_img(img_path, target_size=target_size)
                    img_array = img_to_array(img) / 255.0  # Normalisasi [0, 1]
                    
                    images.append(img_array)
                    labels.append(LABEL_MAPPING[label_str])
                    
    X = np.array(images, dtype='float32')
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=7)
    
    return X, y

def get_train_test_splits(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Membagi dataset menjadi train, validation, dan test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=np.argmax(y, axis=1)
    )
    
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_relative_size, random_state=random_state, stratify=np.argmax(y_train, axis=1)
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
