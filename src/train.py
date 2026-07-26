import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_and_preprocess_data, get_train_test_splits, REVERSE_LABEL_MAPPING
from model import build_vgg19_model

def main():
    # 1. Unduh / Tentukan lokasi dataset
    import kagglehub
    dataset_path = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
    
    metadata_path = os.path.join(dataset_path, 'HAM10000_metadata.csv')
    image_folders = [
        os.path.join(dataset_path, 'HAM10000_images_part_1'),
        os.path.join(dataset_path, 'HAM10000_images_part_2')
    ]
    
    # 2. Muat data
    print("Memuat dan memproses gambar...")
    X, y = load_and_preprocess_data(metadata_path, image_folders, target_size=(64, 64))
    
    # 3. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_splits(X, y)
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 4. Build & Compile Model
    model = build_vgg19_model(input_shape=(64, 64, 3), num_classes=7)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('vgg19_skin_cancer_best.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    # 5. Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )
    
    # 6. Evaluasi Test Set
    test_loss, test_acc, test_recall, test_precision = model.evaluate(X_test, y_test)
    print(f"\n[Test Result] Accuracy: {test_acc:.4f} | Recall: {test_recall:.4f} | Precision: {test_precision:.4f}")
    
    # 7. Classification Report & Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    target_names = [REVERSE_LABEL_MAPPING[i] for i in range(7)]
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

if __name__ == "__main__":
    main()
