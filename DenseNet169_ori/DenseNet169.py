'''
screen -S CV_Final

screen -r CV_Final

安裝相關環境:

conda create -n resnet50_env python=3.8 -y

conda activate resnet50_env

conda install tensorflow-gpu

conda install matplotlib

conda install pandas

conda install pillow
'''

import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import pandas as pd

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CURRENT_HOME_PATH = os.path.dirname(CURRENT_DIR_PATH)

# 設定
image_size = (224, 224)
batch_size = 16
epochs = 10

# 讀取 name.txt 和 query.txt
with open(CURRENT_HOME_PATH + f"/name.txt", "r") as f:
    name_lines = f.readlines()

with open(CURRENT_HOME_PATH + f"/query.txt", "r") as f:
    query_indices = [int(line.strip()) for line in f.readlines()]

# 解析 name.txt
all_data = []
for idx, line in enumerate(name_lines):
    img_file, label = line.strip().split()
    all_data.append((img_file, label))

# 分割成測試集和訓練集
query_set = {all_data[i - 1] for i in query_indices}  # 減1以配對行數
train_set = set(all_data) - query_set

# 建立資料路徑
base_path = CURRENT_HOME_PATH + f"/pic01/pic"

def create_dataframe(data_set):
    data = {
        "filepath": [os.path.join(base_path, label, img_file) for img_file, label in data_set],
        "label": [label for _, label in data_set],
    }
    return pd.DataFrame(data)

train_df = create_dataframe(train_set)
test_df = create_dataframe(query_set)

# 建立 label map
label_map = {label: idx for idx, label in enumerate(sorted(train_df["label"].unique()))}

# 建立資料生成器
def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size) / 255.0
    return img, tf.one_hot(label, depth=len(label_map))

# 轉換 DataFrame 為 tf.data.Dataset
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_df["filepath"].values, train_df["label"].map(label_map).values))
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=len(train_df))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_df["filepath"].values, test_df["label"].map(label_map).values))
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# 模型架構
input_shape = (image_size[0], image_size[1], 3)
feature_model = tf.keras.applications.DenseNet169(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape,
)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer = tf.keras.layers.Dense(len(label_map), activation="softmax")
dropout_layer = tf.keras.layers.Dropout(0.5)

'''
dense_layer = tf.keras.layers.Dense(10,activation='softmax')
dropout_layer = tf.keras.layers.Dropout(0.5)
'''
model = models.Sequential([
    feature_model,
    global_average_layer,
    dense_layer,
    dropout_layer
])

# 模型編輯
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 模型訓練
history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)

# 儲存模型
model.save(filepath=CURRENT_DIR_PATH + f"/model/", overwrite=True, save_format="tf")

# 繪製結果
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.savefig(CURRENT_DIR_PATH + f"/accuracy.png")

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.savefig(CURRENT_DIR_PATH + f"/loss.png")