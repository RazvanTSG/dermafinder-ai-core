import os
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications import EfficientNetB0

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
train_dir = "dataset/Data/train"
test_dir = "dataset/Data/test"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax', dtype='float32')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1)

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

class_weights_dict = {0: 1.0, 1: 1.0, 2: 3.0, 3: 0.8}
model.fit(train_ds, epochs=20, initial_epoch=10, validation_data=val_ds, verbose=1, class_weight=class_weights_dict)

model.save("models/derma_model_final.h5")
