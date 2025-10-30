# src/imaging/cnn_tf/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def get_cnn_model(input_shape=(64, 64, 1), num_classes=1):
    """
    Returns a small CNN for binary classification.
    Binary classification uses sigmoid output (num_classes=1).
    For multi-class, set num_classes>1 and adjust activation to softmax.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

if __name__ == "__main__":
    m = get_cnn_model()
    m.summary()
