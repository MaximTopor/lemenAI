import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 4

TRAIN_DIR = "train_images"
TRAIN_CSV = "train_images.csv"

def file_exists(path):
    return os.path.isfile(path)

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.resize((IMG_WIDTH, IMG_HEIGHT))
        return True
    except Exception:
        return False

def predict_one_image(model, image_path, class_names):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        print(f"\nSúbor: {image_path}")
        print("Pravdepodobnosti:", prediction[0])
        print("Predikovaná trieda:", predicted_class)

    except Exception as e:
        print(f"\nChyba pri spracovaní {image_path}: {e}")


train_df = pd.read_csv(TRAIN_CSV)

train_df["id"] = train_df["id"].astype(str)
train_df["class_num"] = train_df["class_num"].astype(str)
train_df["filepath"] = train_df["id"].apply(lambda x: os.path.join(TRAIN_DIR, x))

train_df = train_df[train_df["filepath"].apply(file_exists)].copy()
train_df = train_df[train_df["filepath"].apply(is_valid_image)].copy()

print("Počet validných obrázkov:", len(train_df))
print("\nRozdelenie tried:")
print(train_df["class_num"].value_counts())

if len(train_df) == 0:
    raise ValueError("Žiadne validné obrázky.")


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="class_num",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="class_num",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\nValidation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")


class_names = list(train_generator.class_indices.keys())
print("\nTriedy:", class_names)

images_to_check = [
    "test_images/test_0018.jpg",
    "test_images/test_0001.jpg",
    "test_images/test_0002.jpg"
]


for image_path in images_to_check:
    predict_one_image(model, image_path, class_names)