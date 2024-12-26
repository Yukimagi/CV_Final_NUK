import os

import pandas as pd
from tensorflow.keras import models
import tensorflow as tf
from sklearn.metrics import accuracy_score
image_size = (224, 224)
batch_size = 32

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CURRENT_HOME_PATH = os.path.dirname(CURRENT_DIR_PATH)
# 模型路徑
load_path = os.path.join(CURRENT_DIR_PATH, "model")

# 加載模型
loaded_model = models.load_model(load_path)

# 驗證加載的模型
loaded_model.summary()
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
print(label_map)
# 建立資料生成器
def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size) / 255.0
    return img, tf.one_hot(label, depth=len(label_map))

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_df["filepath"].values, test_df["label"].map(label_map).values))
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
#print(test_ds)
# 預測並計算準確率
y_true = test_df["label"].map(label_map).values  # 真實標籤
y_pred = []

for images, _ in test_ds:
    predictions = loaded_model.predict(images)
    y_pred.extend(tf.argmax(predictions, axis=1).numpy())

# 計算準確率
accuracy = accuracy_score(y_true, y_pred)
print(f"準確率: {accuracy:.2%}")