from keras.utils import plot_model
from keras.models import load_model
import MSiT
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import dataprocess

batch_size =64
epochs = 20
num_classes = 
length = 5120
number = 1000# 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例

#读取数据......
path = r""
path_test = r""

x_train, y_train, x_valid, y_valid, x_test, y_test = datapreprocess.prepro(d_path=path,length=length,number=number,normal=normal,rate=rate,enc=True, enc_step=28)
x_train1, y_train1, x_valid1, y_valid1, x_test1, y_test1 = datapreprocess.prepro(d_path=path,length=length,number=number,normal=normal,rate=rate,enc=True, enc_step=28)

x_train, x_valid, x_test = x_train[:, :, :,  np.newaxis], x_valid[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
x_train1, x_valid1, x_test1 = x_train1[:, :, :,  np.newaxis], x_valid1[:, :, :, np.newaxis], x_test1[:, :, :, np.newaxis]
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

#跨个体评估
loss1, bin_accuracy1 = model.evaluate(x_test1, y_test1)
print("loss:", loss1)
print("accuracy", bin_accuracy1)





