import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
test_dir = "dataset/Data/test"

model = tf.keras.models.load_model("models/derma_model_final.h5")
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

tta_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

y_true_tta = []
y_pred_tta = []

for images, labels in tqdm(val_ds, desc="Evaluating"):
    y_true_tta.extend(np.argmax(labels.numpy(), axis=1))
    batch_stack = [model.predict(images, verbose=0)]
    for _ in range(4):
        aug_images = tta_layers(images, training=True)
        batch_stack.append(model.predict(aug_images, verbose=0))
    avg_probs = np.mean(batch_stack, axis=0)
    y_pred_tta.extend(np.argmax(avg_probs, axis=1))

class_names = ['BCC', 'BKL', 'MEL', 'NV']
print(classification_report(y_true_tta, y_pred_tta, target_names=class_names))
