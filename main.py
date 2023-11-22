from keras.utils import plot_model
from keras.models import load_model
import MSiT
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf


batch_size =64
epochs = 20


#读取数据......

input_shape =x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

model = MSiT.MSiT(input_shape)

adam = Adam(lr=0.001)
bin_accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
metrics_acc = [bin_accuracy]
model.compile(optimizer=adam, loss='binary_crossentropy',
              metrics=metrics_acc)
callback_list = [ModelCheckpoint(filepath='MSiT.hdf5', verbose=1, save_best_only=True,monitor='val_loss')]

# 训练模型
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
          callbacks=callback_list)

model.load_weights('MSiT.hdf5')
# 评估模型

loss, bin_accuracy = model.evaluate(x_test, y_test)
print("loss:", loss)
print("accuracy", bin_accuracy)





