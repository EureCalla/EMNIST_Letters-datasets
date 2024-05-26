# !pip install emnist -U

import tensorflow as tf
from emnist import list_datasets

list_datasets()  # 這幾種類型的數據集可以使用

# 載入 EMNIST 資料
from emnist import extract_test_samples, extract_training_samples

(x_train, y_train), (x_test, y_test) = extract_training_samples(
    "letters"
), extract_test_samples("letters")

# 訓練/測試資料的 X/y 維度
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# 訓練/測試資料的編碼要-1才能對應字母
y_train = y_train - 1
y_test = y_test - 1

# 顯示第1張圖片圖像
import matplotlib.pyplot as plt

# 第一筆資料
X2 = x_train[0, :, :]  # 0代表第一個字母，後兩個分別代表圖像高度和圖像寬度
# 繪製點陣圖，cmap='gray':灰階
plt.imshow(X2.reshape(28, 28), cmap="gray")
# 隱藏刻度
# plt.axis('off')
# 顯示圖形
plt.show()


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 特徵縮放常態化(Normalization)
x_train_norm = x_train.astype("float32") / 255
x_test_norm = x_test.astype("float32") / 255

# 建立模型
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(26, activation="softmax"),
    ]
)

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 模型訓練
history = model.fit(
    x_train_norm, y_train, epochs=15, validation_split=0.2, batch_size=100
)
# 檢查 history 所有鍵值
history.history.keys()
# 對訓練過程的準確率繪圖
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], "blue", label="Train data")
plt.plot(history.history["val_accuracy"], "green", label="Validation data")
plt.legend()
# 對訓練過程的準確率繪圖
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], "blue", label="Train data")
plt.plot(history.history["val_loss"], "green", label="Validation data")
plt.legend()


# 評分(Score Model)
score = model.evaluate(x_test_norm, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], ":", score[i] * 100, "%")

predictions = np.argmax(model.predict(x_test_norm), axis=-1)
print(predictions)
# 亂序比對
random_list = np.random.randint(0, x_test_norm.shape[0], 20)
print("actual    :", [y_test[random_list]])
print("prediction:", [predictions[random_list]])

# 模型存檔
model.save("my_emnist_model.h5")
