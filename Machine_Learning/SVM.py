import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 設定路徑
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CURRENT_HOME_PATH = os.path.dirname(CURRENT_DIR_PATH)

# 讀取檔案
with open(CURRENT_HOME_PATH + f"/name.txt", "r") as f:
    name_lines = f.readlines()

with open(CURRENT_HOME_PATH + f"/query.txt", "r") as f:
    query_indices = [int(line.strip()) for line in f.readlines()]

with open(CURRENT_HOME_PATH + f"/fnormal.txt", "r") as f:
    fnormal_lines = f.readlines()

# 解析 name.txt
all_data = []
for idx, line in enumerate(name_lines):
    img_file, label = line.strip().split()
    all_data.append((fnormal_lines[idx].strip(), label))

# 分割成測試集和訓練集
query_set = {all_data[i - 1] for i in query_indices}  # 減1以配對行數
train_set = set(all_data) - query_set

# 建立 DataFrame
def create_dataframe(data_set):
    data = {
        "filepath": [img_file.split() for img_file, label in data_set],  # 不需要 eval，直接處理成列表
        "label": [label for _, label in data_set],
    }
    return pd.DataFrame(data)

train_df = create_dataframe(train_set)
test_df = create_dataframe(query_set)

# 將 'filepath' 欄位轉換為數值格式 (這裡假設 'filepath' 已經是列表格式)
X_train = [list(map(float, feat)) for feat in train_df['filepath']]  # 轉換為數字列表
y_train = train_df['label']

X_test = [list(map(float, feat)) for feat in test_df['filepath']]  # 轉換為數字列表
y_test = test_df['label']

# 進行標籤編碼（如果標籤是文字的話）
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 初始化 SVM 模型並訓練
svm_model = SVC(kernel='linear')  # 你也可以選擇其他 kernel，如 'rbf'
svm_model.fit(X_train, y_train_encoded)

# 使用訓練好的模型進行預測
y_pred = svm_model.predict(X_test)

# 計算準確度
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 可以反向編碼將預測的數字轉回原本的標籤
y_pred_labels = label_encoder.inverse_transform(y_pred)
print("Predictions:", y_pred_labels)
