""" 基于CNN(卷积神经网络)的手写数字识别
    机器学习导论，2019-11-15
"""

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
import xlwt
from matplotlib import pyplot as plt
import os

# 绘图预处理
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 构建卷积神经网络
def create_cnn():
    recognizer = Sequential()  # 搭建网络,定义顺序模型
    recognizer.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                          input_shape=(28, 28, 1), activation='relu'))  # 第一个卷积层
    recognizer.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # 第一个池化层
    recognizer.add(Dropout(0.05))  # 正则化

    recognizer.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))  # 第二个卷积层
    recognizer.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # 第二个池化层
    recognizer.add(Dropout(0.05))  # 第二个池化层正则化

    recognizer.add(Flatten())  # 池化层扁平化
    recognizer.add(Dense(256, activation='relu'))  # 全连接层激活函数relu最合适，batch_size256最合适
    recognizer.add(Dropout(0.5))  # 正则化
    recognizer.add(Dense(10, activation='softmax'))  # 全连接层输出
    return recognizer


# 将数据输出到表格
def print_to_file(data, path):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('test')
    ws.write(0, 0, 'data')  # 写入表头
    for i in range(len(data)):
        d = data[i]
        ws.write(i + 1, 0, str(d))
    wb.save(path)


# 特征向量二值化
def vec_two_value(data):
    for i in range(data.shape[0]):
        row_data = data[i]
        for j in range(len(row_data)):
            if row_data[j] > 15:  # 多个阈值进行测试，最简单的噪声去除
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data


# 可视化数据集
def display_image(data, img_size):
    plt.figure(figsize=(8, 7))  # 画布大小
    data = data.reshape(img_size, img_size)
    plt.subplot(1, 1, 1)
    plt.imshow(data)
    plt.show()


# 数据输入
def load_data():
    file = '../data/mnist.npz'
    f = np.load(file)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# main()
# X_data, y_data, X_test_data = data_input()  # 读取三个数据文件
generate_file_path = '../data/y_test_data.xls'

(X_data, y_data), (X_test_data, y_test_val) = load_data()

print_to_file(y_test_val, '../data/val.xls')

plt.subplot(221)
plt.imshow(X_data[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_data[1], cmap=plt.get_cmap('gray'))
plt.show()

# 将总数据变为四维数据 60000 * 28 * 28 -> 60000 * 28 * 28 * 1，每个图像为三维数据
# print(X_data.shape[2])
print(X_data)
X_train = np.reshape(X_data, (60000, 28, 28, 1))
X_test = np.reshape(X_test_data, (10000, 28, 28, 1))

# 数据可视化
# display_image(X_train[2], 28)  # 可视化一张图
# display_image(X_data[1], 28)

# 标准化
X_train_nor = X_train / 255
X_test_nor = X_test / 255
y_data = np_utils.to_categorical(y_data, 10)  # 标签转化为类，独热编码

recognizer = create_cnn()
# 优化
adam = Adam(lr=1e-4)  # 定义优化
# 训练，计算loss rate，准确率
recognizer.compile(loss='categorical_crossentropy', optimizer='adam',
                   metrics=['accuracy'])

recognizer.fit(x=X_train_nor, y=y_data, validation_split=0.2, batch_size=64,
               epochs=20, verbose=2)  # verbose=2输出每一个epoch日志

# 测试
submission_data = recognizer.predict_classes(X_test_nor)  # 预测得到类别
# 反编码
# submission_data = np.argmax(submission_data, axis=1)
# inverted = encoder.inverse_transform([submission_data])

# 拟合率
'''
loss, accuracy = recognizer.evaluate(true_Test, y_test)
print(loss)
print(accuracy)'''

# 输出到文件

print_to_file(submission_data, generate_file_path)
