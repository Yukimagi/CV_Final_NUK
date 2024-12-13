import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras

#設定
image_size = (224,224)

batch_size = 32
epochs = 50
num_classes = 10
train_root='./train/'

#載入資料並切割為訓練資料及驗證資料
tdata = tf.keras.preprocessing.image_dataset_from_directory(
    './train/',
    validation_split=0.2,
    subset="training",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    #color_mode="grayscale",
    label_mode='categorical'
)
vdata = tf.keras.preprocessing.image_dataset_from_directory(
    './train/',
    validation_split=0.2,
    subset="validation",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    #color_mode="grayscale",
    label_mode='categorical'
)

train_class_names = tdata.class_names
print("train_label:",train_class_names)

val_class_names = vdata.class_names
print("val_label:",val_class_names)


#預處理資料
tdata = tdata.prefetch(buffer_size=batch_size)
vdata = vdata.prefetch(buffer_size=batch_size)



input_shape = (image_size[0],image_size[1],3)


#===================================模型宣告=================================#+
from tensorflow.keras import layers
from tensorflow.keras import models

feature_model = tf.keras.applications.ResNet101(
    include_top=False,#是否包含全連階層
    weights='imagenet',#是否載入imagenet權重
    input_shape=input_shape, #輸入圍度
    classes = num_classes, #類別數量
    #classifier_activation="softmax",
)

#feature_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer = tf.keras.layers.Dense(10,activation='softmax')

model = models.Sequential()
model.add(feature_model)
model.add(global_average_layer)
model.add(dense_layer)
print("網路架構")
model.summary()

#優化器
Adam = tf.keras.optimizers.Adam
optimizer = Adam(lr=1e-3)
#建立模型
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',#categorical_crossentropy #binary_crossentropy
    metrics=['accuracy'],
)
#============================================================================#

history = model.fit(
    tdata,
    epochs=epochs,
    validation_data=vdata,
    batch_size=batch_size
)

model.save(
    filepath='model/',
    overwrite=True,
    save_format='tf',
)

model.summary()
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("acc.png")
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss.png")