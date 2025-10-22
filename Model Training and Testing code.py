import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
import zipfile
import os
from tensorflow.keras.utils import img_to_array, load_img

# ---------------------------
# Step 1: Upload Dataset
# ---------------------------
uploaded = files.upload()  # Upload dataset.zip

with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

os.listdir("./dataset")  # Should show like Cat, Dog, Horse

# ---------------------------
# Step 2: Load Dataset
# ---------------------------
IMG_SIZE = (224, 224)  # Pretrained model input size
BATCH_SIZE = 4

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Save class names

class_names = train_ds_raw.class_names
print("Classes:", class_names)

# Split into train/test
train_size = int(0.8 * len(train_ds_raw))
train_ds = train_ds_raw.take(train_size)
test_ds = train_ds_raw.skip(train_size)

# ---------------------------
# Step 3: Data Augmentation
# ---------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x)/255.0, y))
test_ds = test_ds.map(lambda x, y: (x/255.0, y))

# ---------------------------
# Step 4: Build Transfer Learning Model
# ---------------------------
base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze pretrained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# Step 5: Train Model
# ---------------------------
history = model.fit(train_ds, validation_data=test_ds, epochs=20)

# ---------------------------
# Step 6: Test Any Uploaded Image
# ---------------------------
uploaded = files.upload()  # Upload a test image

for filename in uploaded.keys():
    path = filename
    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()
