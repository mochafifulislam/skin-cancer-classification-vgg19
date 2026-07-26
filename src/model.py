import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D

def build_vgg19_model(input_shape=(64, 64, 3), num_classes=7, fine_tune_at=None):
    """
    Membangun model Transfer Learning VGG19.
    """
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers
    base_model.trainable = False
    
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="VGG19_SkinCancerClassifier")
    return model
